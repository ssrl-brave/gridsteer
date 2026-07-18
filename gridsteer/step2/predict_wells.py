#!/usr/bin/env python3
"""
Predict the locations of wells that never entered the captured frames.

The scan often stops partway across the tray, so map_wells.py only
sees the first wells of each row. Wells sit on two straight rows with
equal spacing (row 1: 9 wells, row 2: 10 wells) and the scan always
starts at the C1 end and travels left, so each row is completed by
stepping leftward from the last detection by the row's average spacing
vector until the expected count is reached.

This module also learns the motor-to-pixel mapping: pairing recorded
stage deltas with registered pixel deltas trains a linear ridge
regression from pixel shift (dx, dy) to motor shift (dx, dy, dz), which
answers "what absolute motor coordinates center well R{r}C{c} in the
frame?" (written to well_centering_positions.json).

Pure geometry only; imported and driven by scripts/map_wells.py.
"""

import re
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

# Expected number of wells per row for this tray (row index -> count).
EXPECTED_WELLS_PER_ROW = {1: 9, 2: 10}


@dataclass
class PredictedWell:
    label: str
    cy: float          # tray (mosaic) coordinates
    cx: float
    r: float           # rim radius in mosaic pixels


def _row_index(label: str) -> int:
    m = re.match(r"R(\d+)C\d+", label)
    if not m:
        raise ValueError(f"unrecognized well label {label!r}")
    return int(m.group(1))


def predict_missing_wells(wells, expected=None):
    """Extend each row of detected wells leftward to its expected count.

    wells: detected wells (.label "R{row}C{col}", .cx, .cy, .r) as
    produced by map_wells.label_wells. Returns a list of
    PredictedWell for the positions the scan never reached.
    """
    if expected is None:
        expected = EXPECTED_WELLS_PER_ROW

    rows = {}
    for w in wells:
        rows.setdefault(_row_index(w.label), []).append(w)

    predicted = []
    for ri, row_wells in sorted(rows.items()):
        n_expected = expected.get(ri)
        n = len(row_wells)
        if n_expected is None or n > n_expected:
            print(f"  Warning: Row {ri} Has {n} Detections but Expects "
                  f"{n_expected}; Skipping Prediction", file=sys.stderr)
            continue
        if n == n_expected:
            continue
        if n < 2:
            print(f"  Warning: Row {ri} Has {n} Detection(s); Need at "
                  f"Least 2 to Estimate Spacing", file=sys.stderr)
            continue

        # C1 is rightmost; the scan moves left, so extend past the
        # leftmost detection by the row's average spacing vector.
        row_wells = sorted(row_wells, key=lambda w: -w.cx)
        first, last = row_wells[0], row_wells[-1]
        dx = (last.cx - first.cx) / (n - 1)
        dy = (last.cy - first.cy) / (n - 1)
        r = float(np.median([w.r for w in row_wells]))
        for i in range(n + 1, n_expected + 1):
            predicted.append(PredictedWell(
                label=f"R{ri}C{i}", r=r,
                cx=last.cx + (i - n) * dx,
                cy=last.cy + (i - n) * dy))
    return predicted


# --------------------------------------------------------------------------
# Motor-to-pixel mapping: ridge regression from pixel shift to motor shift
# --------------------------------------------------------------------------

def _row_col(label: str):
    m = re.match(r"R(\d+)C(\d+)", label)
    if not m:
        raise ValueError(f"unrecognized well label {label!r}")
    return int(m.group(1)), int(m.group(2))


def _project(H, x, y):
    """Apply a 3x3 homography to one point (pure numpy, no cv2)."""
    p = H @ np.array([x, y, 1.0])
    return p[:2] / p[2]


@dataclass
class MotorPixelMap:
    """Linear ridge model: pixel shift (dx, dy) -> motor shift (dx, dy, dz).

    One sklearn Ridge model per motor axis, as in motor_prediction.py.
    Trained per run; phi is assumed constant (the fit stands down when
    it varies).
    """
    model_x: Ridge
    model_y: Ridge
    model_z: Ridge
    r2: dict              # per-axis training R^2 (None if the axis never moved)
    n_pairs: int
    alpha: float

    def motor_shift(self, pixel_delta):
        """Motor (dx, dy, dz) that moves the tray content by pixel_delta."""
        X = np.asarray(pixel_delta, float).reshape(1, -1)
        return np.array([m.predict(X)[0]
                         for m in (self.model_x, self.model_y, self.model_z)])


def fit_motor_pixel_map(Gs, meta, ref_xy, alpha=1.0):
    """Fit the pixel-shift -> motor-shift ridge regression for one run.

    Gs: per-frame homographies (frame pixels -> mosaic pixels).
    meta: per-frame stage metadata dicts with keys x, y, z (and phi).
    ref_xy: a tray point in mosaic coordinates whose per-frame image
    position measures how the content moved.

    Training pairs come from ALL frame pairs (i < j), not just
    consecutive ones: with a constant per-axis step, one-gap pairs make
    that axis indistinguishable from the intercept; mixed gaps break
    the degeneracy. Returns a MotorPixelMap, or None when the metadata
    cannot support a fit (missing keys, varying phi, < 3 usable pairs).
    """
    if not meta or not all(all(k in m for k in ("x", "y", "z")) for m in meta):
        print("  Warning: Incomplete Stage Metadata; Cannot Fit Motor Map",
              file=sys.stderr)
        return None
    phis = [m.get("phi", 0.0) for m in meta]
    if max(phis) - min(phis) > 1e-6:
        print("  Warning: Phi Varies Across the Run; the Translation-Only "
              "Motor Map Cannot Model Rotation", file=sys.stderr)
        return None

    pts = np.array([_project(np.linalg.inv(G), *ref_xy) for G in Gs])
    motors = np.array([[m["x"], m["y"], m["z"]] for m in meta])
    ii, jj = np.triu_indices(len(pts), k=1)
    dpix = pts[jj] - pts[ii]
    dmot = motors[jj] - motors[ii]

    moved = np.linalg.norm(dmot, axis=1) > 1e-9
    X, Y = dpix[moved], dmot[moved]
    if len(X) < 3:
        print(f"  Warning: Only {len(X)} Usable Frame Pairs; Need at Least 3 "
              f"to Fit the Motor Map", file=sys.stderr)
        return None

    models = {}
    r2 = {}
    for i, axis in enumerate(("x", "y", "z")):
        model = Ridge(alpha=alpha)
        model.fit(X, Y[:, i])
        models[axis] = model
        # R^2 is undefined for an axis the stage never moved.
        var = float(((Y[:, i] - Y[:, i].mean()) ** 2).sum())
        r2[axis] = None if var < 1e-18 else round(model.score(X, Y[:, i]), 4)
    return MotorPixelMap(model_x=models["x"], model_y=models["y"],
                         model_z=models["z"], r2=r2,
                         n_pairs=len(X), alpha=alpha)


def motor_centering_positions(mapping, wells, predicted, Gs, meta,
                              frame_shape, ref_frame=0):
    """Absolute motor coordinates that center each well in the frame.

    For every well (detected and predicted) the well's position in the
    reference frame is projected through that frame's homography; the
    pixel offset from there to the frame center goes through the fitted
    mapping to get the motor shift, which is applied to the reference
    frame's recorded motor position. Returns a JSON-ready dict.
    """
    h, w = frame_shape
    center = np.array([w / 2.0, h / 2.0])
    Ginv = np.linalg.inv(Gs[ref_frame])
    ref_motor = meta[ref_frame]

    out = {
        "frame_center": [center[0], center[1]],
        "reference_frame_number": ref_frame,
        "reference_motor_position": {
            "x": ref_motor["x"], "y": ref_motor["y"],
            "z": ref_motor["z"], "phi": ref_motor.get("phi", 0.0)},
        "calibration": {
            "method": f"linear ridge regression (alpha={mapping.alpha})",
            "mapping_direction": "pixel shift (dx,dy) -> motor shift (dx,dy,dz)",
            "n_training_pairs": mapping.n_pairs,
            "r2_scores": mapping.r2},
        "well_centering_positions": {},
    }

    predicted_labels = {p.label for p in predicted}
    for well in sorted(list(wells) + list(predicted),
                       key=lambda w: _row_col(w.label)):
        row, col = _row_col(well.label)
        px = _project(Ginv, well.cx, well.cy)
        # Centering needs a content shift of (center - current), the
        # same direction the training deltas were measured in.
        dpix = center - px
        dmot = mapping.motor_shift(dpix)
        out["well_centering_positions"][f"({row},{col})"] = {
            "label": well.label,
            "row": row,
            "column": col,
            "predicted": well.label in predicted_labels,
            "pixel_position_in_reference_frame": {
                "x": round(float(px[0]), 2), "y": round(float(px[1]), 2)},
            "pixel_offset_to_center": {
                "dx": round(float(dpix[0]), 2), "dy": round(float(dpix[1]), 2)},
            "predicted_motor_shift": {
                "dx": float(dmot[0]), "dy": float(dmot[1]),
                "dz": float(dmot[2])},
            "motor_position": {
                "x": float(ref_motor["x"] + dmot[0]),
                "y": float(ref_motor["y"] + dmot[1]),
                "z": float(ref_motor["z"] + dmot[2]),
                "phi": ref_motor.get("phi", 0.0)},
        }
    return out
