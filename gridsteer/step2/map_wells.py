#!/usr/bin/env python3
"""
Identify the wells of a moving tray: register the frames, fuse a mosaic,
detect and label wells, track them per frame, and compute the motor
coordinates that center each well. See the step2 README for the full
pipeline description.

Detection matches the whole known well layout as one constellation
(well_template.py, the default); --no-template runs the legacy per-well
ring detector followed by row extrapolation (predict_wells.py).

Usage:
    python -m gridsteer.step2.map_wells data/mapper15 --out output_tracks

Outputs (in <out>/<dataset-name>/):
    mosaic_labeled.png   mosaic with detected (solid) and predicted
                         (dashed) wells
    frames_overlay.png   montage of frames with projected well tracks
    tracks.csv           frame, well id, x, y, radius, visibility
    pose.csv             per-frame planar pose parameters
    wells.json           well positions/radii in tray coordinates;
                         "observed" separates detections from predictions
    well_centering_positions.json
                         absolute motor coordinates that center each well
"""

import argparse
import csv
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .predict_wells import (predict_missing_wells, fit_motor_pixel_map,
                            motor_centering_positions, row_col,
                            stage_metadata_issue)
from .well_template import (DEFAULT_TEMPLATE_PATH, load_template,
                            match_layout)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def numeric_key(path: Path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def load_frames(data_dir: Path, key: str = "sample"):
    """Load all .npz frames in a directory, sorted by the number in the name."""
    files = sorted(data_dir.glob("*.npz"), key=numeric_key)
    if not files:
        raise FileNotFoundError(f"no .npz files in {data_dir}")
    frames, meta = [], []
    for f in files:
        d = np.load(f)
        frames.append(np.asarray(d[key]))
        meta.append({k: float(np.atleast_1d(d[k])[0])
                     for k in ("x", "y", "z", "phi") if k in d})
    return frames, meta, [f.name for f in files]


# --------------------------------------------------------------------------
# Registration: frame -> tray (frame 0) coordinates
# --------------------------------------------------------------------------

def match_pairs(frames):
    """SIFT+RANSAC homography (frame t -> t-1) for each consecutive pair.

    Returns a list of (H, n_inliers), H being None when matching failed.
    Weak pairs are kept; apply_trajectory() decides whether to bridge.
    """
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    kps, descs = [], []
    for f in frames:
        kp, de = sift.detectAndCompute(f, None)
        kps.append(kp)
        descs.append(de)

    pairs = []
    for t in range(1, len(frames)):
        if descs[t] is None or descs[t - 1] is None:
            pairs.append((None, 0))
            continue
        matches = matcher.match(descs[t], descs[t - 1])
        if len(matches) < 4:
            pairs.append((None, len(matches)))
            continue
        src = np.float32([kps[t][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kps[t - 1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC)
        pairs.append((H, int(inl.sum()) if inl is not None else 0))
    return pairs


def fit_stage_map(dstage, dpix, healthy, tol=3.0, iters=300):
    """RANSAC fit of pixel motion vs stage motion (dpix ~ dstage @ A).

    tol sits between the ~1-2px calibration accuracy and the aliasing
    errors this catches (a well pitch, hundreds of px); rcond keeps
    collinear scans at a minimum-norm fit. Returns
    (A, median_residual_px), or (None, None) with < 3 healthy pairs.
    """
    idx = np.where(healthy)[0]
    if len(idx) < 3:
        return None, None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iters):
        sample = rng.choice(idx, 3, replace=False)
        A, *_ = np.linalg.lstsq(dstage[sample], dpix[sample], rcond=1e-3)
        resid = np.hypot(*(dstage[idx] @ A - dpix[idx]).T)
        inliers = idx[resid < tol]
        if best is None or len(inliers) > len(best):
            best = inliers
    if len(best) < 3:
        return None, None
    A, *_ = np.linalg.lstsq(dstage[best], dpix[best], rcond=1e-3)
    resid = np.hypot(*(dstage[best] @ A - dpix[best]).T)
    return A, float(np.median(resid))


def ecc_refine(frames, t, seed_txy):
    """Refine a predicted pair translation with ECC.
    Returns a 3x3 H (frame-t -> frame-(t-1)) or None."""
    warp = np.array([[1, 0, seed_txy[0]], [0, 1, seed_txy[1]]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        cv2.findTransformECC(frames[t].astype(np.float32),
                             frames[t - 1].astype(np.float32),
                             warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    except cv2.error:
        return None
    return np.vstack([warp, [0, 0, 1]]).astype(float)


def apply_trajectory(pairs, frames, meta, min_inliers):
    """Validate/bridge pairwise transforms against the stage trajectory.

    Agreeing pairs pass through; weak pairs (defocus) are bridged with
    the stage prediction + ECC; confident-but-contradicting pairs
    (well-lattice aliasing) are rejected and rebuilt the same way.
    Returns (pairs, report); without usable metadata, pairs are
    returned unchanged.
    """
    report = {"calibrated": False, "residual_px": None,
              "bridged": [], "rejected": []}

    if stage_metadata_issue(meta):
        return pairs, report

    dstage = np.diff(np.array([[m["x"], m["y"], m["z"]] for m in meta]), axis=0)
    healthy = np.array([H is not None and n >= min_inliers for H, n in pairs])
    dpix = np.array([[H[0, 2], H[1, 2]] if H is not None else [np.nan, np.nan]
                     for H, _ in pairs])

    A, resid = fit_stage_map(dstage, dpix, healthy)
    if A is None:
        return pairs, report
    # The in-plane gain |dpix|/|dstage_xy| tracks the image zoom, so it
    # doubles as a template-free scale prior for detection.
    report.update(calibrated=True, residual_px=round(resid, 2),
                  stage_gain=round(float(np.linalg.norm(A[:2])), 2))

    pred = dstage @ A
    # Between the ~2px calibration accuracy and one-well-pitch aliasing jumps.
    threshold = max(10.0, 10 * resid)

    out = []
    for i, (H, n) in enumerate(pairs):
        deviation = (np.hypot(H[0, 2] - pred[i, 0], H[1, 2] - pred[i, 1])
                     if H is not None else np.inf)
        if H is not None and n >= min_inliers and deviation <= threshold:
            out.append((H, n))
            continue
        kind = "rejected" if (H is not None and n >= min_inliers) else "bridged"
        refined = ecc_refine(frames, i + 1, pred[i])
        if refined is None:
            refined = np.array([[1, 0, pred[i, 0]],
                                [0, 1, pred[i, 1]],
                                [0, 0, 1.0]])
        report[kind].append(i + 1)
        out.append((refined, n))
    return out, report


def register_frames(frames, meta=None, min_inliers: int = 15):
    """Chained pairwise registration with trajectory validation.

    Returns (Hs, report) where Hs[t] maps frame-t -> frame-0 and report
    describes any trajectory interventions (see apply_trajectory).
    """
    pairs = match_pairs(frames)
    pairs, report = apply_trajectory(pairs, frames, meta, min_inliers)

    Hs = [np.eye(3)]
    for t, (H, n) in enumerate(pairs, start=1):
        if H is None or (not report["calibrated"] and n < min_inliers):
            raise RuntimeError(
                f"registration failed at frame {t}: {n} inliers and no "
                f"usable stage metadata to bridge with")
        Hs.append(Hs[-1] @ H)  # compose: t -> t-1 -> ... -> 0
    return Hs, report


def available_memory():
    """Bytes of memory the OS reports as allocatable, or None if unknown."""
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def mosaic_transforms(frames, Hs):
    """Global transforms G[t]: frame-t -> mosaic canvas (all coords >= 0)."""
    corners = []
    for f, H in zip(frames, Hs):
        h, w = f.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, H).reshape(-1, 2))
    corners = np.concatenate(corners)
    mn, mx = corners.min(axis=0), corners.max(axis=0)
    T = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1.0]])
    size = np.ceil(mx - mn).astype(int) + 1  # (w, h)
    # Memory rail: a runaway canvas almost always means a bad homography.
    need = (len(frames) + 2) * int(size[0]) * int(size[1]) * 4
    avail = available_memory()
    if avail is not None and need > avail:
        raise RuntimeError(
            f"aligned stack needs {need / 1e9:.1f} GB for a "
            f"{size[1]}x{size[0]} canvas but only {avail / 1e9:.1f} GB is "
            f"available -- likely a registration failure (check the "
            f"per-frame homographies) or too little memory")
    return [T @ H for H in Hs], (int(size[1]), int(size[0]))


def build_mosaic(frames, Gs, canvas_shape):
    """Warp all frames to the canvas; per-pixel temporal median + coverage."""
    h, w = canvas_shape
    stack = np.full((len(frames), h, w), np.nan, np.float32)
    for t, (f, G) in enumerate(zip(frames, Gs)):
        stack[t] = cv2.warpPerspective(
            f.astype(np.float32), G.astype(np.float64), (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"))
    count = (~np.isnan(stack)).sum(axis=0)
    # Unobserved pixels are all-NaN stacks; silence nanmedian's warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mosaic = np.nanmedian(stack, axis=0)
    return mosaic, count


# --------------------------------------------------------------------------
# Detection on the mosaic: ring matched filter, radius measured by sweep
# --------------------------------------------------------------------------

@dataclass
class Well:
    label: str
    cy: float          # tray (mosaic) coordinates
    cx: float
    r: float           # rim radius in mosaic pixels
    score: float       # NCC response
    contrast: float    # interior-minus-rim intensity margin
    completeness: float  # fraction of rim that is a closed dark circle


SMOOTH_SIGMA = 2.0  # pre-detection Gaussian; pure noise suppression


def ring_template(r: float, thickness: float = None, pad: int = None):
    """Zero-mean annulus template.

    Defaults build a blur-limited thin ring used to bootstrap-measure
    the radius and rim width; detection re-runs with measured values.
    """
    if thickness is None:
        thickness = 3 * SMOOTH_SIGMA
    if pad is None:
        pad = int(3 * SMOOTH_SIGMA)
    n = int(r + thickness + pad)
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(yy, xx)
    t = ((d > r - thickness / 2) & (d < r + thickness / 2)).astype(np.float32)
    return t - t.mean()


def _ncc(image, template, valid):
    """NCC response with degenerate regions zeroed.

    Flat areas (unobserved canvas, warp borders) have ~zero variance
    and produce garbage peaks; the response is zeroed where the window
    is mostly unobserved, variance is negligible, or |NCC| > 1.
    """
    from skimage.feature import match_template
    resp = match_template(image, template, pad_input=True)
    n = template.shape[0]
    m1 = cv2.boxFilter(image.astype(np.float32), -1, (n, n))
    m2 = cv2.boxFilter((image * image).astype(np.float32), -1, (n, n))
    local_var = np.maximum(m2 - m1 * m1, 0.0)
    floor = (0.01 * float(image.std())) ** 2
    resp[local_var < floor] = 0.0
    resp[np.abs(resp) > 1.0] = 0.0
    support = cv2.boxFilter(valid.astype(np.float32), -1, (n, n))
    resp[support < 0.5] = 0.0
    return resp


def measure_radius(inv, valid, r_min=50, r_max=200, step=5):
    """1-D sweep: the well radius is measured from the data, not supplied."""
    best_r, best_v = r_min, -np.inf
    for r in range(r_min, r_max + 1, step):
        v = _ncc(inv, ring_template(r), valid).max()
        if v > best_v:
            best_r, best_v = r, v
    # refine around the coarse winner
    for r in range(best_r - step + 1, best_r + step):
        if r <= 0:
            continue
        v = _ncc(inv, ring_template(r), valid).max()
        if v > best_v:
            best_r, best_v = r, v
    return best_r


def measure_rim_width(smooth, inv, r_star, valid):
    """Rim thickness: FWHM of the dark rim trough in the radial profile
    around the strongest ring response."""
    resp = _ncc(inv, ring_template(r_star), valid)
    py, px = np.unravel_index(np.nanargmax(resp), resp.shape)
    rs = np.arange(max(1.0, 0.5 * r_star), 1.5 * r_star, 1.0)
    prof = radial_profile(smooth, py, px, rs)
    i_min = int(np.nanargmin(prof))
    baseline = float(np.nanmedian(prof))
    half_depth = (baseline + float(prof[i_min])) / 2
    below = prof < half_depth
    lo = hi = i_min
    while lo > 0 and below[lo - 1]:
        lo -= 1
    while hi < len(rs) - 1 and below[hi + 1]:
        hi += 1
    return float(rs[hi] - rs[lo] + 1)


def radial_profile(img, cy, cx, radii, n_theta=360):
    """Mean intensity on circles of the given radii (NaN-aware)."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    out = []
    for r in radii:
        ys = np.round(cy + r * np.sin(th)).astype(int)
        xs = np.round(cx + r * np.cos(th)).astype(int)
        ok = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
        vals = img[ys[ok], xs[ok]]
        out.append(np.nanmean(vals) if len(vals) else np.nan)
    return np.array(out)


def rim_stats(img, cy, cx, r, n_theta=360):
    """(contrast, completeness) of the rim at radius r.

    contrast: interior brightness minus mean rim brightness.
    completeness: fraction of rim angles darker than the interior by
    half the contrast -- ~1.0 for a closed rim, low for partial arcs.
    """
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ys = np.round(cy + r * np.sin(th)).astype(int)
    xs = np.round(cx + r * np.cos(th)).astype(int)
    ok = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
    rim = img[ys[ok], xs[ok]]
    interior = radial_profile(img, cy, cx, np.arange(2, max(3, 0.6 * r), 4))
    mi = np.nanmean(interior)
    contrast = float(mi - np.nanmean(rim))
    completeness = float(np.nanmean(rim < mi - 0.5 * contrast))
    return contrast, completeness


def detect_wells(mosaic, count, core_frac=0.5, candidate_frac=0.25,
                 contrast_frac=0.5, min_completeness=0.75):
    """Ring matched filter on the fused mosaic.

    Peaks >= core_frac * max are accepted outright. Weaker peaks pass
    only with a nearly complete dark rim and sufficient contrast --
    relative, self-calibrating tests that lighting cannot shift.
    """
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian

    valid = count > 0
    fill = np.nanmax(mosaic)
    smooth = gaussian(np.where(valid, mosaic, fill), 2)
    inv = smooth.max() - smooth

    r_star = measure_radius(inv, valid)
    rim_w = measure_rim_width(smooth, inv, r_star, valid)

    # Fine sweep +/- rim_w/4; exact per-well radii are re-measured from
    # the radial profile below.
    step = max(1.0, rim_w / 8)
    best, best_r = None, None
    for r in np.arange(r_star - rim_w / 4, r_star + rim_w / 4 + step / 2, step):
        resp = _ncc(inv, ring_template(r, thickness=rim_w, pad=int(rim_w)), valid)
        if best is None:
            best, best_r = resp, np.full(resp.shape, float(r))
        else:
            upd = resp > best
            best[upd] = resp[upd]
            best_r[upd] = r

    # Suppression radius between ~1*r (duplicate peaks of one well) and
    # 2*r (the closest two centers can be).
    peaks = peak_local_max(best, min_distance=int(1.5 * r_star),
                           threshold_abs=candidate_frac * best.max(),
                           exclude_border=False)

    raw = []
    for py, px in peaks:
        r0 = best_r[py, px]
        rs = np.arange(0.75 * r0, 1.25 * r0, 1.0)
        prof = radial_profile(smooth, py, px, rs)
        r_fit = float(rs[np.nanargmin(prof)])
        contrast, completeness = rim_stats(smooth, py, px, r_fit)
        raw.append(dict(cy=float(py), cx=float(px), r=r_fit,
                        score=float(best[py, px]),
                        contrast=contrast, completeness=completeness))

    # Null-hypothesis gate: extreme-value statistics bounds the max NCC
    # of pure noise; requiring 1.5x that makes "zero wells" possible.
    med = float(np.median(best[valid]))
    mad = float(np.median(np.abs(best[valid] - med)))
    null_max = med + np.sqrt(2 * np.log(max(int(valid.sum()), 2))) * 1.4826 * mad

    strong = core_frac * best.max()
    core = [w for w in raw if w["score"] >= strong
            and w["score"] >= 1.5 * null_max]
    if not core:
        return [], r_star, rim_w
    min_contrast = contrast_frac * float(np.median([w["contrast"] for w in core]))
    wells = [w for w in raw
             if w["score"] >= strong
             or (w["completeness"] >= min_completeness
                 and w["contrast"] >= min_contrast)]
    return wells, r_star, rim_w


def label_wells(wells, r_star):
    """Row/column labels in tray coordinates (stable across the sequence)."""
    wells = sorted(wells, key=lambda w: w["cy"])
    rows, current = [], [wells[0]]
    for w in wells[1:]:
        if w["cy"] - current[-1]["cy"] < r_star:
            current.append(w)
        else:
            rows.append(current)
            current = [w]
    rows.append(current)
    out = []
    for ri, row in enumerate(rows):
        # Columns are numbered right-to-left: the rightmost well is C1.
        for ci, w in enumerate(sorted(row, key=lambda w: -w["cx"]), start=1):
            out.append(Well(label=f"R{ri + 1}C{ci}", **w))
    return out


# --------------------------------------------------------------------------
# Tracking: project tray-coordinate wells into every frame
# --------------------------------------------------------------------------

def local_scale(Ginv, x, y):
    """Isotropic scale of the mosaic->frame map at a point (for radii)."""
    p = np.float32([[x, y], [x + 1, y], [x, y + 1]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(p, Ginv).reshape(-1, 2)
    j = np.array([q[1] - q[0], q[2] - q[0]]).T
    return float(np.sqrt(abs(np.linalg.det(j))))


def project_tracks(wells, Gs, frame_shape):
    """Per-frame well positions = tray positions through each frame's warp."""
    h, w = frame_shape
    records = []
    for t, G in enumerate(Gs):
        Ginv = np.linalg.inv(G)
        for well in wells:
            p = cv2.perspectiveTransform(
                np.float32([[well.cx, well.cy]]).reshape(-1, 1, 2),
                Ginv).reshape(2)
            s = local_scale(Ginv, well.cx, well.cy)
            r = well.r * s
            x, y = float(p[0]), float(p[1])
            fully = (r <= x <= w - r) and (r <= y <= h - r)
            partly = (-r < x < w + r) and (-r < y < h + r)
            records.append(dict(
                frame=t, well=well.label, x=round(x, 2), y=round(y, 2),
                r=round(r, 2),
                visibility="full" if fully else ("partial" if partly else "out")))
    return records


def planar_pose(Gs):
    """Per-frame planar pose from the homography (frame->tray).

    Reports in-plane rotation, isotropic scale, translation, and the
    perspective-row norm as a tilt indicator (0 when fronto-parallel).
    Units keep the signal through 2-decimal rounding: scale as percent
    deviation from 1, perspective in 1e-6.
    """
    rows = []
    for t, G in enumerate(Gs):
        H = G / G[2, 2]
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]
        rot = float(np.degrees(np.arctan2(c, a)))
        scale = float(np.sqrt(abs(a * d - b * c)))
        persp = float(np.hypot(H[2, 0], H[2, 1]))
        rows.append(dict(frame=t, rot_deg=round(rot, 2),
                         scale_pct=round(100 * (scale - 1), 2),
                         tx=round(float(H[0, 2]), 2), ty=round(float(H[1, 2]), 2),
                         persp_x1e6=round(1e6 * persp, 2)))
    return rows


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def save_mosaic_figure(mosaic, wells, path, predicted=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # Widen the view so predicted wells beyond the mosaic are not clipped.
    x0, x1, y0, y1 = 0, mosaic.shape[1], 0, mosaic.shape[0]
    if predicted:
        all_wells = list(wells) + list(predicted)
        pad = 1.5 * max(w.r for w in all_wells)
        x0 = min(x0, min(w.cx - w.r for w in all_wells) - pad)
        x1 = max(x1, max(w.cx + w.r for w in all_wells) + pad)
        y0 = min(y0, min(w.cy - w.r for w in all_wells) - pad)
        y1 = max(y1, max(w.cy + w.r for w in all_wells) + pad)

    fig, ax = plt.subplots(figsize=(12, 12 * (y1 - y0) / (x1 - x0)))
    ax.imshow(mosaic, cmap="gray")
    for w in wells:
        ax.add_patch(Circle((w.cx, w.cy), w.r, fill=False, color="lime", lw=2))
        ax.text(w.cx, w.cy, w.label, color="red", fontsize=14,
                ha="center", va="center", weight="bold")
    for w in predicted:
        ax.add_patch(Circle((w.cx, w.cy), w.r, fill=False, color="cyan",
                            lw=2, linestyle="--"))
        ax.text(w.cx, w.cy, w.label, color="cyan", fontsize=14,
                ha="center", va="center", weight="bold")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    if predicted:
        ax.plot([], [], color="lime", lw=2, label="Detected")
        ax.plot([], [], color="cyan", lw=2, linestyle="--", label="Predicted")
        ax.legend(loc="upper right", fontsize=12)
    if not wells and not predicted:
        ax.set_title("Fused Tray Mosaic (No Wells Detected)")
    else:
        ax.set_title("Fused Tray Mosaic with Labeled Wells"
                     + (" (Dashed = Predicted, Not Observed)" if predicted else ""))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def save_overlay_montage(frames, records, path, max_panels=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    step = max(1, len(frames) // max_panels)
    idx = list(range(0, len(frames), step))[:max_panels]
    ncol = min(6, len(idx))
    nrow = int(np.ceil(len(idx) / ncol))
    panel_w = 3.0                                    # inches per panel
    title_pad = 0.2                                  # headroom for panel titles
    panel_h = panel_w * frames[0].shape[0] / frames[0].shape[1] + title_pad
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(panel_w * ncol, panel_h * nrow))
    axes = np.atleast_1d(axes).ravel()
    by_frame = {}
    for rec in records:
        by_frame.setdefault(rec["frame"], []).append(rec)
    for ax, t in zip(axes, idx):
        ax.imshow(frames[t], cmap="gray")
        for rec in by_frame.get(t, []):
            if rec["visibility"] == "out":
                continue
            color = "lime" if rec["visibility"] == "full" else "orange"
            circ = Circle((rec["x"], rec["y"]), rec["r"],
                          fill=False, color=color, lw=1.5)
            ax.add_patch(circ)
            h, w = frames[t].shape[:2]
            if 0 <= rec["x"] < w and 0 <= rec["y"] < h:
                ax.text(rec["x"], rec["y"], rec["well"], color="red",
                        fontsize=8, ha="center", va="center",
                        weight="bold", clip_on=True)
        ax.set_title(f"Frame {t}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="directory of .npz frames")
    ap.add_argument("--key", default="sample", help="npz key holding the image")
    ap.add_argument("--out", type=Path, default=Path("output_tracks"),
                    help="output root directory")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="exact output directory (overrides --out; no "
                         "subdirectory is appended)")
    ap.add_argument("--template", type=Path, default=None,
                    help="well layout template JSON (default: the bundled "
                         f"{DEFAULT_TEMPLATE_PATH.name})")
    ap.add_argument("--no-template", action="store_true",
                    help="use the legacy per-well ring detector instead of "
                         "whole-layout template matching")
    args = ap.parse_args()

    if args.outdir is not None:
        out = args.outdir
    else:
        out = args.out / args.data_dir.name
    out.mkdir(parents=True, exist_ok=True)

    def print_observed(ws):
        for w in ws:
            print(f"    {w.label}: Tray ({w.cy:7.1f},{w.cx:7.1f})"
                  f"  r={w.r:5.1f}px  Score={w.score:.2f}"
                  f"  Contrast={w.contrast:5.1f}"
                  f"  Completeness={w.completeness:.2f}")

    def print_predicted(ws):
        for w in ws:
            print(f"    {w.label}: Tray ({w.cy:7.1f},{w.cx:7.1f})"
                  f"  r={w.r:5.1f}px  (Predicted)")

    print(f"Loading Frames from {args.data_dir} ...")
    frames, meta, names = load_frames(args.data_dir, args.key)
    print(f"  {len(frames)} Frames of Shape {frames[0].shape}")

    print("Registering Frames (SIFT + RANSAC Homography) ...")
    Hs, reg = register_frames(frames, meta)
    if reg["calibrated"]:
        print(f"  Stage-to-Pixel Calibration Residual: {reg['residual_px']}px")
        for kind in ("bridged", "rejected"):
            if reg[kind]:
                print(f"  {kind.title()} Pairs (Frame Indices): {reg[kind]}")
    else:
        print("  No Usable Stage Metadata; Image-Only Registration")
    Gs, canvas_shape = mosaic_transforms(frames, Hs)
    print(f"  Mosaic Canvas: {canvas_shape}")

    print("Fusing Temporal Median Mosaic ...")
    mosaic, count = build_mosaic(frames, Gs, canvas_shape)

    def no_wells_exit():
        save_mosaic_figure(mosaic, [], out / "mosaic_labeled.png")
        print(f"  Mosaic Saved to {out / 'mosaic_labeled.png'}")
        print("No Wells Detected", file=sys.stderr)
        sys.exit(1)

    template = None
    if not args.no_template:
        tpath = args.template if args.template else DEFAULT_TEMPLATE_PATH
        if tpath.exists():
            template = load_template(tpath)
        elif args.template:
            ap.error(f"template file not found: {tpath}")
        else:
            print(f"  No Bundled Template ({tpath.name}); "
                  f"Falling Back to the Legacy Per-Well Detector")

    if template is not None:
        print("Detecting Wells (Whole-Layout Template Matching) ...")
        expected = None
        if reg.get("stage_gain") and template.get("stage_gain"):
            # The run's own stage-to-pixel gain over the template
            # scan's pins the zoom, so detection searches around the
            # scale this scan is actually at.
            expected = reg["stage_gain"] / template["stage_gain"]
            print(f"  Zoom Prior from the Stage Calibration: "
                  f"Expected Scale {expected:.2f}")
        wells, predicted, tinfo = match_layout(mosaic, count, template,
                                               expected_scale=expected)
        if not wells:
            no_wells_exit()
        print(f"  Matched Layout: Scale={tinfo['scale']:.3f}, "
              f"Rotation={tinfo['rotation_deg']:+.1f} deg, "
              f"Orientation={tinfo['orientation'].title()}, "
              f"Well Radius ~{tinfo['feature_radius']:.0f}px")
        if tinfo.get("lattice_placed"):
            print(f"  Wells Cut Off by the Mosaic Edge, Placed on the "
                  f"Fitted Lattice: {', '.join(tinfo['lattice_placed'])}")
        print(f"  {len(wells)} Observed Wells:")
        print_observed(wells)
        if predicted:
            print(f"  {len(predicted)} Wells Placed by the Fitted Layout"
                  f" (Not Observed):")
            print_predicted(predicted)
    else:
        print("Detecting Wells (Ring Matched Filter) ...")
        raw_wells, r_star, rim_w = detect_wells(mosaic, count)
        if not raw_wells:
            no_wells_exit()
        wells = label_wells(raw_wells, r_star)
        print(f"  Measured Rim Radius ~{r_star}px, Rim Width ~{rim_w:.0f}px;"
              f" {len(wells)} Wells:")
        print_observed(wells)

        print("Predicting Missing Wells (Row Line + Spacing) ...")
        predicted = predict_missing_wells(wells)
        if predicted:
            print(f"  {len(predicted)} Wells Extrapolated from the "
                  f"Detected Rows:")
            print_predicted(predicted)
        else:
            print("  All Expected Wells Detected; Nothing to Predict")

    print("Projecting Per-Frame Tracks ...")
    records = project_tracks(wells, Gs, frames[0].shape)
    pose = planar_pose(Gs)

    print("Learning Motor-to-Pixel Mapping (Linear Ridge Regression) ...")
    ref_xy = (float(np.mean([w.cx for w in wells])),
              float(np.mean([w.cy for w in wells])))
    mapping = fit_motor_pixel_map(Gs, meta, ref_xy)
    centering = None
    if mapping is not None:
        print(f"  Fitted on {mapping.n_pairs} Frame Pairs; "
              f"R^2 per Motor Axis: {mapping.r2}")
        centering = motor_centering_positions(
            mapping, wells, predicted, Gs, meta, frames[0].shape)
        n = len(centering["well_centering_positions"])
        print(f"  Motor Centering Coordinates Computed for {n} Wells")
    else:
        print("  Skipped: Stage Metadata Cannot Support the Fit")

    combined = sorted(
        [{**asdict(w), "observed": True} for w in wells]
        + [{**asdict(w), "observed": False} for w in predicted],
        key=lambda w: row_col(w["label"]))
    with open(out / "wells.json", "w") as f:
        json.dump(combined, f, indent=2)
    if centering is not None:
        with open(out / "well_centering_positions.json", "w") as f:
            json.dump(centering, f, indent=2)
        with open(out / "mapping.json", "w") as f:
            json.dump(centering, f, indent=2)
    with open(out / "tracks.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wr.writeheader()
        wr.writerows(records)
    with open(out / "pose.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(pose[0].keys()))
        wr.writeheader()
        wr.writerows(pose)

    save_mosaic_figure(mosaic, wells, out / "mosaic_labeled.png",
                       predicted=predicted)
    save_overlay_montage(frames, records, out / "frames_overlay.png")

    n_vis = sum(r["visibility"] != "out" for r in records)
    print(f"Done. {n_vis} Well-Frame Observations Written to {out}/")


if __name__ == "__main__":
    main()
