#!/usr/bin/env python3
"""
Predict the locations of wells that never entered the captured frames,
and fit the motor-to-pixel mapping.

The scan often stops partway across the tray, so each row is completed
by stepping leftward from the last detection by the row's average
spacing vector until its expected count is reached. The motor mapping
is a linear ridge regression from pixel shift (dx, dy) to motor shift
(dx, dy, dz), used to compute the absolute motor coordinates that
center each well in the frame.

Pure geometry only; imported and driven by map_wells.py.
"""

import re
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

# Physical layout (row index -> well count); row 2 sticks out by half
# a pitch on both ends. Shared by well_template.py and the legacy
# extrapolation below so the two cannot drift apart.
ROW_COUNTS = {1: 9, 2: 10}


@dataclass
class PredictedWell:
    label: str
    cy: float          # tray (mosaic) coordinates
    cx: float
    r: float           # rim radius in mosaic pixels


def row_col(label: str):
    """(row, column) parsed from a well label such as "R2C7"."""
    m = re.match(r"R(\d+)C(\d+)", label)
    if not m:
        raise ValueError(f"unrecognized well label {label!r}")
    return int(m.group(1)), int(m.group(2))


def predict_missing_wells(wells):
    """Extend each row of detected wells leftward to its expected count.

    wells: detected wells (.label "R{row}C{col}", .cx, .cy, .r).
    Returns PredictedWell entries for positions the scan never reached.
    """
    rows = {}
    for w in wells:
        rows.setdefault(row_col(w.label)[0], []).append(w)

    predicted = []
    for ri, row_wells in sorted(rows.items()):
        n_expected = ROW_COUNTS.get(ri)
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

        # C1 is rightmost and the scan moves left: extend past the
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


def stage_metadata_issue(meta):
    """Why the stage metadata cannot support a translation-only fit:
    "incomplete" (a frame lacks x/y/z), "phi varies" (the stage
    rotated), or None when it can. One predicate gates both
    apply_trajectory and fit_motor_pixel_map so they cannot drift
    apart."""
    if not meta or not all(all(k in m for k in ("x", "y", "z")) for m in meta):
        return "incomplete"
    phis = [m.get("phi", 0.0) for m in meta]
    if max(phis) - min(phis) > 1e-6:
        return "phi varies"
    return None


# --------------------------------------------------------------------------
# Motor-to-pixel mapping: ridge regression from pixel shift to motor shift
# --------------------------------------------------------------------------

def _project(H, x, y):
    """Apply a 3x3 homography to one point (pure numpy, no cv2)."""
    p = H @ np.array([x, y, 1.0])
    return p[:2] / p[2]


@dataclass
class MotorPixelMap:
    """Linear ridge model: pixel shift (dx, dy) -> motor shift (dx, dy, dz).

    One Ridge model per motor axis, trained per run; phi is assumed
    constant (the fit stands down when it varies).
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
    ref_xy: tray point (mosaic coords) whose per-frame image position
    measures how the content moved.

    Training pairs come from ALL frame pairs (i < j): with a constant
    per-axis step, consecutive pairs alone make that axis
    indistinguishable from the intercept. Returns a MotorPixelMap, or
    None when the metadata cannot support a fit.
    """
    issue = stage_metadata_issue(meta)
    if issue == "incomplete":
        print("  Warning: Incomplete Stage Metadata; Cannot Fit Motor Map",
              file=sys.stderr)
        return None
    if issue == "phi varies":
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

    Each well's pixel offset from its reference-frame position to the
    frame center goes through the fitted mapping; the resulting motor
    shift is applied to the reference frame's recorded motor position.
    Returns a JSON-ready dict.
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
                       key=lambda w: row_col(w.label)):
        row, col = row_col(well.label)
        px = _project(Ginv, well.cx, well.cy)
        # Content shift of (center - current), matching the direction
        # the training deltas were measured in.
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
