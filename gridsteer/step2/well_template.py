#!/usr/bin/env python3
"""
Build and match a whole-layout well template.

Per-well detection silently drops wells whose rims are weakened by
lighting or appearance (shadows, dark sample), so the *entire* known
layout -- two staggered rows of 9 + 10 wells -- is matched as one rigid
constellation against the fused mosaic, searching over zoom and
mounting orientation. See the step2 README for the full pipeline.

build_template (offline, once): fit a staggered-lattice model to the
legacy ring detector's labeled output and save all 19 canonical well
positions to JSON.

match_layout (default detection path): score circular-boundary
evidence per pixel with a lighting- and polarity-robust rim-coverage
response, sweep the layout over scale and orientation with FFT
translation scoring, polish the top candidates, and pick the winner by
verified evidence.

Usage (template building):
    python -m gridsteer.step2.well_template data/Step2TestData/mapper15 \
        --out gridsteer/step2/xmed1_layout_template.json
"""

import argparse
import json
import sys
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

from .predict_wells import ROW_COUNTS, PredictedWell, row_col

DEFAULT_TEMPLATE_PATH = Path(__file__).with_name("xmed1_layout_template.json")


def _mirror_columns(label: str) -> str:
    """Same well with its row's columns numbered from the other end."""
    r, c = row_col(label)
    return f"R{r}C{ROW_COUNTS[r] + 1 - c}"


def _flip_label(label: str) -> str:
    """Physical label of an image-labeled well on a 180-rotated mount
    (rows swap and columns reverse)."""
    r, c = row_col(label)
    other = 2 if r == 1 else 1
    return f"R{other}C{ROW_COUNTS[other] + 1 - c}"


def _is_flipped(a, b) -> bool:
    """True when the similarity [[a, -b], [b, a]] turns the layout past
    90 degrees, i.e. the tray was mounted 180 rotated."""
    return abs(np.degrees(np.arctan2(b, a))) > 90.0


# --------------------------------------------------------------------------
# Template building: canonicalize the ring-fitting pipeline's output
# --------------------------------------------------------------------------

def _lattice_offset(row: int, col: int) -> float:
    """Column position in pitch units: row 1 (9 wells) is inset by half
    a pitch relative to row 2 (10 wells) at both ends."""
    return (col - 1) + (0.5 if row == 1 else 0.0)


def build_template(wells, predicted=(), source="", stage_gain=None):
    """Canonical layout template from labeled wells.

    wells/predicted: objects with .label, .cx, .cy (wells also .r);
    together they must cover the full layout. The lattice model
    pos(row, col) = T + D * _lattice_offset(row, col) + N * (row - 1)
    is fit to the labeled *detections* only (extrapolated wells just
    complete the row-count check), then all 19 template wells are
    generated from the fit.
    stage_gain: the source scan's in-plane stage-to-pixel gain -- the
    gain at which the tray appears at template scale 1.0; a later run's
    own gain divided by it predicts that run's scale.

    Returns a JSON-ready dict with centroid-centered coordinates, rows
    horizontal, R1 (the 9-well row) on top, C1 to the right.
    """
    counts = {}
    for label in {w.label for w in list(wells) + list(predicted)}:
        r, _ = row_col(label)
        counts[r] = counts.get(r, 0) + 1
    flipped_build = counts == {1: ROW_COUNTS[2], 2: ROW_COUNTS[1]}
    if flipped_build:
        # Build data was captured on a flipped mount; convert image
        # labels to physical ones (the 9-well row is always R1).
        counts = {1: ROW_COUNTS[1], 2: ROW_COUNTS[2]}
    if counts != ROW_COUNTS:
        raise ValueError(
            f"cannot build a template from an incomplete layout: got "
            f"{counts} wells per row, expected {ROW_COUNTS}")

    observed = {(_flip_label(w.label) if flipped_build else w.label):
                (float(w.cx), float(w.cy)) for w in wells}
    A, Y = [], []
    for label, (x, y) in observed.items():
        r, c = row_col(label)
        k, m = _lattice_offset(r, c), r - 1
        A.append([1.0, 0.0, k, 0.0, m, 0.0])
        A.append([0.0, 1.0, 0.0, k, 0.0, m])
        Y.extend([x, y])
    A, Y = np.array(A), np.array(Y)
    sol, _, rank, _ = np.linalg.lstsq(A, Y, rcond=None)
    if rank < 6:
        raise ValueError("degenerate layout: detected wells must span "
                         "both rows and at least two columns")
    T, D, N = sol[0:2], sol[2:4], sol[4:6]
    # k() already encodes the machined half-pitch stagger, so N's
    # along-row component is measurement noise; keep only the row gap.
    d_hat = D / np.hypot(*D)
    perp = np.array([-d_hat[1], d_hat[0]])
    N = float(N @ perp) * perp
    resid = A @ np.concatenate([T, D, N]) - Y
    rms = float(np.sqrt(np.mean(resid ** 2)))

    labels, P = [], []
    for r, n in sorted(ROW_COUNTS.items()):
        for c in range(1, n + 1):
            labels.append(f"R{r}C{c}")
            P.append(T + D * _lattice_offset(r, c) + N * (r - 1))
    P = np.array(P)
    P -= P.mean(axis=0)

    # Rotate D (direction of increasing columns) onto -u: rows
    # horizontal, C1 to the right.
    phi = np.pi - np.arctan2(D[1], D[0])
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi), np.cos(phi)]])
    P = P @ R.T
    # With C1 to the right, R1 on top requires the row-1-to-row-2
    # vector to point down (+v); no physical mounting violates this.
    if (R @ N)[1] <= 0:
        raise ValueError("input labels are mirrored (R1 is not the top "
                         "row when C1 is to the right)")

    radius = float(np.median([w.r for w in wells]))
    gain = {} if stage_gain is None else {"stage_gain": round(stage_gain, 2)}
    return {
        **gain,
        "wells": [{"label": l, "u": round(float(P[i, 0]), 2),
                   "v": round(float(P[i, 1]), 2)}
                  for i, l in enumerate(labels)],
        "radius": round(radius, 2),
        "row_counts": {str(k): v for k, v in ROW_COUNTS.items()},
        "pitch": round(float(np.hypot(*D)), 2),
        "row_gap": round(float((R @ N)[1]), 2),
        "fit_rms": round(rms, 2),
        "n_observed": len(list(wells)),
        "source": str(source),
    }


def save_template(template: dict, path):
    with open(path, "w") as f:
        json.dump(template, f, indent=2)


def load_template(path):
    with open(path) as f:
        t = json.load(f)
    if not t.get("wells") or "radius" not in t:
        raise ValueError(f"{path} is not a well layout template")
    return t


# --------------------------------------------------------------------------
# Rim-coverage response: lighting- and polarity-robust well evidence
# --------------------------------------------------------------------------

def _shift(a, sy: int, sx: int):
    """Integer shift with zero fill: out[y, x] = a[y - sy, x - sx]."""
    out = np.zeros_like(a)
    h, w = a.shape
    ys0, ys1 = max(0, -sy), min(h, h - sy)
    xs0, xs1 = max(0, -sx), min(w, w - sx)
    if ys0 < ys1 and xs0 < xs1:
        out[ys0 + sy:ys1 + sy, xs0 + sx:xs1 + sx] = a[ys0:ys1, xs0:xs1]
    return out


def _coverage_response(smooth, valid, r: float, n_sectors: int = 16):
    """Per-pixel evidence of a circular boundary of radius r, z-scored.

    Radially aligned gradient energy (|grad| * cos 2*delta) is sampled
    on a ring of radius r in n_sectors directions; the interquartile
    mean over sectors rejects straight edges and speckle while scoring
    closed boundaries of either polarity alike. High-pass filtered and
    z-scored (median/MAD) over the observed region, so responses are
    comparable across images, radii, and lighting.
    """
    gy, gx = np.gradient(smooth.astype(np.float32))
    mag = np.hypot(gx, gy)
    mag[~valid] = 0.0  # warp borders / unobserved canvas are not edges
    phase = np.arctan2(gy, gx)
    sigma = max(2.0, 0.08 * r)

    S = np.empty((n_sectors,) + smooth.shape, np.float32)
    for k in range(n_sectors):
        th = 2 * np.pi * k / n_sectors
        aligned = mag * np.cos(2 * (phase - th))
        aligned = cv2.GaussianBlur(aligned, (0, 0), sigma)
        dy, dx = int(round(r * np.sin(th))), int(round(r * np.cos(th)))
        S[k] = _shift(aligned, -dy, -dx)
    S.sort(axis=0)
    lo, hi = n_sectors // 4, (3 * n_sectors) // 4
    resp = S[lo:hi].mean(axis=0)
    resp -= cv2.GaussianBlur(resp, (0, 0), 1.5 * r)

    vals = resp[valid]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    z = (resp - med) / (1.4826 * mad + 1e-9)
    z[~valid] = 0.0
    return z


# --------------------------------------------------------------------------
# Layout matching: scale + orientation sweep, FFT translation scoring
# --------------------------------------------------------------------------

def _transform(P, a, b, t):
    """Apply the similarity [[a, -b], [b, a]] + t to (u, v) points."""
    return P @ np.array([[a, b], [-b, a]]) + t


def _fit_similarity(P, Q, w):
    """Weighted least-squares similarity q ~ s R p + t.

    Returns (a, b, tx, ty) with s R = [[a, -b], [b, a]]; needs >= 2
    distinct points.
    """
    w = np.asarray(w, float)
    W = w.sum()
    pm = (w[:, None] * P).sum(axis=0) / W
    qm = (w[:, None] * Q).sum(axis=0) / W
    Pc, Qc = P - pm, Q - qm
    den = float((w * (Pc ** 2).sum(axis=1)).sum())
    a = float((w * (Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1])).sum()) / den
    b = float((w * (Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])).sum()) / den
    t = qm - np.array([a * pm[0] - b * pm[1], b * pm[0] + a * pm[1]])
    return a, b, t


def _best_translation(scored, P, a, b):
    """Best translation of the layout at fixed scale/rotation (a, b):
    every placement is scored at once by FFT cross-correlation of the
    scored map with a kernel of ones at the layout's well positions.
    Returns (score, t)."""
    from scipy.signal import fftconvolve

    pts = _transform(P, a, b, np.zeros(2))
    mn = pts.min(axis=0)
    ij = np.round(pts - mn).astype(int)                  # (x, y)
    kern = np.zeros((ij[:, 1].max() + 1, ij[:, 0].max() + 1), np.float32)
    kern[ij[:, 1], ij[:, 0]] += 1.0
    score = fftconvolve(scored, kern[::-1, ::-1], mode="full")
    idx = np.unravel_index(np.argmax(score), score.shape)
    oy = idx[0] - (kern.shape[0] - 1)
    ox = idx[1] - (kern.shape[1] - 1)
    return float(score[idx]), np.array([ox, oy]) - mn


def _snap(resp, valid, x, y, radius):
    """(peak value, x, y) of resp within a disk around (x, y), or None
    if the window falls outside the observed mosaic. The position is
    the centroid of the above-half-peak mass, which localizes the broad
    coverage blob better than its strongest pixel."""
    h, w = resp.shape
    r = int(np.ceil(radius))
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    win = resp[y0:y1, x0:x1].copy()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    win[(np.hypot(xx - x, yy - y) > radius) | ~valid[y0:y1, x0:x1]] = -np.inf
    if not np.isfinite(win).any():
        return None
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    peak = float(win[iy, ix])
    mass = np.where(np.isfinite(win), win - 0.5 * peak, 0.0).clip(min=0.0)
    near = np.hypot(xx - (x0 + ix), yy - (y0 + iy)) <= radius
    mass[~near] = 0.0
    total = mass.sum()
    if peak > 0 and total > 0:
        return (peak, float((mass * xx).sum() / total),
                float((mass * yy).sum() / total))
    return peak, float(x0 + ix), float(y0 + iy)


def _snap_noise_floor(z, valid, radius, n_samples=64):
    """Median peak _snap finds at random observed positions -- what a
    snap collects from noise alone, since maximizing z over a disk
    yields a positive peak even on well-free tray. Only credit above
    this floor is evidence of structure."""
    rng = np.random.default_rng(0)
    ys, xs = np.where(valid)
    if not len(ys):
        return 0.0
    idx = rng.choice(len(ys), size=min(n_samples, len(ys)), replace=False)
    vals = [hit[0] for i in idx
            if (hit := _snap(z, valid, float(xs[i]), float(ys[i]), radius))
            is not None]
    return max(0.0, float(np.median(vals))) if vals else 0.0


def _rim_occlusion(valid, x, y, r, n_theta=180):
    """(visible fraction, unit vector toward the missing arc) for the
    rim of radius r around (x, y). Sampled on the rim itself because
    the valid region is irregular (a union of warped frames). The
    direction is None when the rim is fully visible or the missing
    angles do not concentrate in a single arc."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    cs, sn = np.cos(th), np.sin(th)
    xs = np.round(x + r * cs).astype(int)
    ys = np.round(y + r * sn).astype(int)
    h, w = valid.shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    vis = np.zeros(n_theta, bool)
    vis[ok] = valid[ys[ok], xs[ok]]
    miss = ~vis
    if not miss.any():
        return 1.0, None
    d = np.array([cs[miss].sum(), sn[miss].sum()])
    n = float(np.hypot(*d))
    if n < 0.5 * miss.sum():
        return float(vis.mean()), None
    return float(vis.mean()), d / n


def _rim_visibility(valid, x, y, r, n_theta=180):
    """Fraction of the rim of radius r around (x, y) that lies on the
    observed mosaic (see _rim_occlusion)."""
    return _rim_occlusion(valid, x, y, r, n_theta)[0]


# Rims at least this visible anchor the similarity fit; a missing arc
# biases the snap, so the bar sits just below fully visible (slack for
# rounding at the mosaic border).
FULL_RIM_VISIBILITY = 0.98

# Penalty for predicting a well on visibly empty tray (fraction of the
# evidence ceiling): large enough that a one-pitch row shift always
# loses, small enough that one absent well cannot veto a placement.
EMPTY_TRAY_PENALTY = 0.15

# Real rims stay radially aligned over most of their visible arc;
# random texture and partial arcs (tray corners) land at or below ~0.5.
# On arc_verified's minimum-completeness statistic, measured corner
# impostors reach at most 0.50 while real wells stay at or above 0.56.
MIN_ARC_COMPLETENESS = 0.55

# Below this, even the snap's along-arc component is unreliable and the
# well contributes nothing to the fit.
MIN_PARTIAL_RIM_VISIBILITY = 0.4


def _evidence_cap(z, valid):
    """Evidence ceiling: the map's top-0.1% response, floored at 1 z so
    a well-free map cannot promote noise. Saturating here stops one
    outlier blob (dark sample disk, debris) from outscoring several
    real wells."""
    return max(1.0, float(np.percentile(z[valid], 99.9)))


def _refine_partial(P, Q, w, full, partial, arc_dir, a, b, t, rcond=0.1,
                    translation_pref=4.0):
    """Correct the similarity (a, b, t) from mixed constraints.

    Fully visible anchors constrain both components of their snap
    residual; truncated anchors only the unbiased component along
    arc_dir (their visible arc). Solved as a truncated-SVD delta so
    undetermined parameter directions keep their coarse-sweep values;
    ambiguities resolve toward translation (translation_pref), since
    scale/rotation errors amplify across the tray span. Returns the
    updated (a, b, t)."""
    L = float(np.sqrt(np.mean(P ** 2)))
    X = _transform(P, a, b, t)
    rows, rhs = [], []
    for i in np.where(full | partial)[0]:
        u, v = P[i] / L
        M = np.array([[u, -v, 1.0, 0.0], [v, u, 0.0, 1.0]])
        r = Q[i] - X[i]
        s = np.sqrt(w[i])
        if full[i]:
            rows.extend(s * M)
            rhs.extend(s * r)
        else:
            rows.append(s * (arc_dir[i] @ M))
            rhs.append(s * float(arc_dir[i] @ r))
    A, y = np.array(rows), np.array(rhs)
    A[:, :2] /= translation_pref
    sol, *_ = np.linalg.lstsq(A, y, rcond=rcond)
    da, db = sol[0] / (translation_pref * L), sol[1] / (translation_pref * L)
    return a + da, b + db, t + sol[2:4]


def _refine_placement(z, valid, P, a, b, t, r_t, anchor_z):
    """Snap wells to local response peaks and re-fit the similarity,
    iterated with a shrinking snap radius (see match_layout step 3).
    Returns the updated (a, b, t) plus the final snap values and
    positions."""
    for frac in (0.5, 0.4, 0.3):
        r_px = np.hypot(a, b) * r_t
        X = _transform(P, a, b, t)
        snap_vals = np.full(len(P), -np.inf)
        snap_pos = X.copy()
        for i, (x, y) in enumerate(X):
            hit = _snap(z, valid, x, y, frac * r_px)
            if hit is not None:
                snap_vals[i], snap_pos[i, 0], snap_pos[i, 1] = hit
        anchors = snap_vals >= anchor_z
        if not anchors.any():
            break
        # Truncated rims bias their snaps, so fully visible wells alone
        # anchor the fit; with fewer than two, truncated wells add only
        # their unbiased along-arc component (_refine_partial).
        vis = np.zeros(len(P))
        arc_dir = np.zeros((len(P), 2))
        for i in np.where(anchors)[0]:
            vis[i], cut = _rim_occlusion(valid, X[i, 0], X[i, 1], r_px)
            if cut is not None:
                arc_dir[i] = -cut[1], cut[0]  # along the visible arc
        whole = anchors & (vis >= FULL_RIM_VISIBILITY)
        partial = (anchors & ~whole & (vis >= MIN_PARTIAL_RIM_VISIBILITY)
                   & (np.hypot(*arc_dir.T) > 0))
        if whole.sum() >= 2:
            a, b, t = _fit_similarity(P[whole], snap_pos[whole],
                                      snap_vals[whole])
        elif whole.any() or partial.any():
            sv = np.where(np.isfinite(snap_vals), snap_vals, 0.0)
            weights = sv * np.where(whole, 1.0, vis)
            a, b, t = _refine_partial(P, snap_pos, weights, whole, partial,
                                      arc_dir, a, b, t)
        else:
            break
    return a, b, t, snap_vals, snap_pos


def _arc_completeness(smooth, valid, x, y, r, n_sectors=18):
    """Fraction of visible ring sectors with positive mean radially
    aligned gradient energy: ~1 for a real rim, ~0.5 or below for
    random texture and partial arcs (rounded tray corners). smooth is
    the (mag, phase) pair from _gradients. Returns 0.0 when no part of
    the ring is observed."""
    mag, phase = smooth
    n = n_sectors * 10
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.round(x + r * np.cos(th)).astype(int)
    ys = np.round(y + r * np.sin(th)).astype(int)
    h, w = mag.shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    ok[ok] &= valid[ys[ok], xs[ok]]
    a = np.zeros(n)
    a[ok] = (mag[ys[ok], xs[ok]]
             * np.cos(2 * (phase[ys[ok], xs[ok]] - th[ok])))
    n_ok = ok.reshape(n_sectors, -1).sum(axis=1)
    sect = a.reshape(n_sectors, -1).sum(axis=1) / np.maximum(n_ok, 1)
    vis = n_ok > 0
    if not vis.any():
        return 0.0
    return float((sect[vis] > 0).mean())


def _gradients(smooth):
    """(magnitude, direction) of the smoothed mosaic's gradient, shared
    by every _arc_completeness call of one match."""
    gy, gx = np.gradient(smooth.astype(np.float32))
    return np.hypot(gx, gy), np.arctan2(gy, gx)


# One polished candidate placement. score/coarse_score lead the tuple
# so p[:2] ranks candidates: verified score first, sweep score tie-break.
Placement = namedtuple("Placement",
                       "score coarse_score a b t z snap_vals snap_pos slid")


def match_layout(mosaic, count, template, observed_z=3.0, anchor_z=2.5,
                 scale_span=(0.6, 1.6), scale_step=1.08,
                 theta_deg_offsets=(-3.0, 0.0, 3.0),
                 min_observed=2, top_k=5, expected_scale=None):
    """Match the full well layout against a fused mosaic.

    expected_scale: optional zoom prior from the run's own stage
    calibration (the run's in-plane stage gain over the template's).
    When given, the scale search is confined to a factor-of-1.5 window
    around it -- several times the prior's observed error, yet
    excluding the half- and double-scale aliases. Without it, the
    search covers scale_span.

    Stages:
    1. Sweep the layout over scale and orientation (0/180 degrees plus
       theta_deg_offsets), scoring every translation at once on a
       2x-downsampled rim-coverage response by FFT cross-correlation.
    2. Polish the top_k candidates (local score maxima along each
       orientation's scale axis, each orientation's best always
       included): snap/fit refinement, scale continuation, column
       anchoring (see polish()).
    3. Pick the winner: scale basins within an orientation by
       verified_score, the orientation by the symmetric difference of
       the champions' placements (exclusive_score).

    Returns (wells, predicted, info): wells are map_wells.Well entries
    for positions whose response z-score reaches observed_z, at their
    snapped positions -- except truncated wells, placed on the fitted
    lattice instead (labels in info["lattice_placed"]); predicted are
    PredictedWell entries for the rest, at the fitted template
    positions. Rows are labeled physically (R1 is always the 9-well
    row); columns right-to-left in the image (C1 rightmost). Returns
    ([], [], info) when no placement yields min_observed observed
    wells.
    """
    from skimage.filters import gaussian
    from .map_wells import SMOOTH_SIGMA, Well, rim_stats

    P = np.array([[w["u"], w["v"]] for w in template["wells"]], float)
    labels = [w["label"] for w in template["wells"]]
    r_t = float(template["radius"])

    valid = (count > 0) & np.isfinite(mosaic)
    fill = np.nanmax(mosaic)
    smooth = gaussian(np.where(valid, mosaic, fill), SMOOTH_SIGMA)

    # Radius rails: below 8px a rim is unresolvable, above ~half the
    # mosaic no layout fits.
    r_lo, r_hi = 8.0, 0.45 * min(mosaic.shape)
    n_scales = max(1, int(np.ceil(np.log(scale_span[1] / scale_span[0])
                                  / np.log(scale_step))) + 1)
    scale_factors = np.geomspace(scale_span[0], scale_span[1], n_scales)
    if expected_scale:
        # The prior's measured error reaches ~8%; a factor-of-1.5
        # window leaves several times that margin while excluding the
        # half-/double-scale look-alike lattices. The same geometric
        # grid is kept and filtered to the window; the radius rails
        # tighten too, so scale continuation stays inside it.
        window = (expected_scale / 1.5, expected_scale * 1.5)
        step = (scale_factors[1] / scale_factors[0] if n_scales > 1
                else scale_step)
        k_lo = int(np.ceil(np.log(window[0] / scale_span[0]) / np.log(step)))
        k_hi = int(np.floor(np.log(window[1] / scale_span[0]) / np.log(step)))
        scale_factors = scale_span[0] * step ** np.arange(k_lo, k_hi + 1)
        r_lo = max(r_lo, window[0] * r_t)
        r_hi = min(r_hi, window[1] * r_t)

    cells = {}  # (scale index, base orientation) -> (score, f, theta, t)
    for fi, f in enumerate(scale_factors):
        r_px = f * r_t
        if not (r_lo <= r_px <= r_hi):
            continue
        z = _coverage_response(smooth, valid, r_px)
        zhalf = cv2.resize(z, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        vhalf = cv2.resize(valid.astype(np.uint8),
                           (zhalf.shape[1], zhalf.shape[0]),
                           interpolation=cv2.INTER_AREA) > 0
        # Saturate at the evidence ceiling so one outlier blob cannot
        # outvote several real wells (_evidence_cap).
        cap = _evidence_cap(zhalf, vhalf)
        scored = (np.minimum(zhalf, cap)
                  - EMPTY_TRAY_PENALTY * cap * vhalf.astype(np.float32))
        for base in (0.0, 180.0):
            for off in theta_deg_offsets:
                th = np.radians(base + off)
                a, b = f * np.cos(th) / 2, f * np.sin(th) / 2
                v, t = _best_translation(scored, P, a, b)
                cell = (fi, base)
                if np.isfinite(v) and (cell not in cells
                                       or v > cells[cell][0]):
                    cells[cell] = (v, f, base + off, t * 2.0)

    info = {}
    if expected_scale:
        info["expected_scale"] = round(float(expected_scale), 4)
    if not cells:
        return [], [], info

    # Candidate selection: the raw sweep score cannot pick one winner,
    # so carry top_k structurally distinct candidates -- the local
    # score maxima along each orientation's scale axis (one slot per
    # basin), plus both orientations' best -- and let the verified
    # score choose after polish.
    maxima = {0.0: [], 180.0: []}
    for base in (0.0, 180.0):
        fis = sorted(fi for fi, b in cells if b == base)
        for j, fi in enumerate(fis):
            s = cells[(fi, base)][0]
            if ((j == 0 or s >= cells[(fis[j - 1], base)][0])
                    and (j == len(fis) - 1
                         or s >= cells[(fis[j + 1], base)][0])):
                maxima[base].append(cells[(fi, base)])
    chosen = [max(m, key=lambda c: c[0]) for m in maxima.values() if m]
    taken = {id(c) for c in chosen}
    rest = sorted((c for m in maxima.values() for c in m
                   if id(c) not in taken), key=lambda c: -c[0])
    chosen += rest[:max(0, top_k - len(chosen))]

    grads = _gradients(smooth)

    def arc_verified(snap_vals, X, snap_pos, r_px):
        """Per-position mask: True unless an observed-level snap fails
        the arc-completeness test (a corner-like impostor). Takes the
        *minimum* completeness over one grid step of radius slack and
        both the lattice and snapped positions: a real rim clears the
        bar everywhere, accidental alignment does not."""
        def ok(i):
            pts = [tuple(X[i])]
            if np.isfinite(snap_pos[i]).all():
                pts.append(tuple(snap_pos[i]))
            return min(_arc_completeness(grads, valid, x, y, rr)
                       for x, y in pts
                       for rr in (r_px / scale_step, r_px,
                                  r_px * scale_step)) >= MIN_ARC_COMPLETENESS
        return np.array([sv < observed_z or ok(i)
                         for i, sv in enumerate(snap_vals)])

    def score_terms(z, a, b, t, snap_vals, snap_pos):
        """Per-layout-position scoring terms of one placement, shared
        by verified_score and exclusive_score.

        Returns (X, in_view, contrib, vis, cap): placed positions,
        which of them the mosaic shows, their evidence, their
        empty-tray-penalty weight, and the evidence ceiling.

        contrib is the *snap* value saturated at cap (rival placements
        covering the same wells then claim the same peaks) minus the
        snap noise floor (otherwise a position on well-free tray could
        collect more noise credit than its penalty costs). A position
        failing arc verification contributes nothing but still pays
        the empty-tray penalty; vis weights that penalty by how much
        of the ring the mosaic shows.
        """
        cap = _evidence_cap(z, valid)
        r_px = np.hypot(a, b) * r_t
        floor = _snap_noise_floor(z, valid, 0.3 * r_px)
        X = _transform(P, a, b, t)
        ok = arc_verified(snap_vals, X, snap_pos, r_px)
        h, w = z.shape
        n = len(X)
        in_view = np.zeros(n, bool)
        contrib, vis = np.zeros(n), np.zeros(n)
        for i, (x, y) in enumerate(X):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h and valid[yi, xi]):
                continue
            in_view[i] = True
            if not ok[i]:
                contrib[i] = 0.0
            elif np.isfinite(snap_vals[i]):
                contrib[i] = max(0.0, min(float(snap_vals[i]), cap) - floor)
            else:
                contrib[i] = max(0.0, min(float(z[yi, xi]), cap))
            vis[i] = _rim_visibility(valid, x, y, r_px)
        return X, in_view, contrib, vis, cap

    def verified_score(z, a, b, t, snap_vals, snap_pos):
        """Sweep objective at a refined placement: evidence of every
        in-view position minus its empty-tray penalty. Kept in robust-z
        units, NOT normalized by the per-map ceiling, which would
        inflate a wrong-radius response's mediocre blobs."""
        _, in_view, contrib, vis, cap = score_terms(z, a, b, t, snap_vals,
                                                    snap_pos)
        return float((contrib - EMPTY_TRAY_PENALTY * cap * vis)[in_view].sum())

    z_init = {}  # scale factor -> response; shared across candidates

    def polish(cand):
        """Refine one sweep candidate to its final placement: snap/fit
        refinement, then scale continuation, then column anchoring --
        all judged by the verified sweep objective."""
        coarse_score, f, theta0, t = cand
        th = np.radians(theta0)
        a, b = f * np.cos(th), f * np.sin(th)
        if f not in z_init:
            z_init[f] = _coverage_response(smooth, valid, f * r_t)
        z = z_init[f]
        a, b, t, snap_vals, snap_pos = _refine_placement(z, valid, P, a, b,
                                                         t, r_t, anchor_z)
        score = verified_score(z, a, b, t, snap_vals, snap_pos)

        # Scale continuation: the sweep quantizes scale, and an
        # anchor-poor fit cannot correct it, so climb the scale grid
        # one step at a time (rebuild the response, re-refine) while
        # the objective improves.
        ratios = [1.0 / scale_step, scale_step]
        for _ in range(len(scale_factors)):
            improved = False
            for ratio in list(ratios):
                r_new = np.hypot(a, b) * ratio * r_t
                if not (r_lo <= r_new <= r_hi):
                    continue
                anchors = snap_vals >= anchor_z
                keep = anchors if anchors.any() else np.ones(len(P), bool)
                X = _transform(P, a, b, t)
                a2, b2 = a * ratio, b * ratio
                # Rescale about the anchors' centroid so the matched
                # wells stay put while the layout breathes.
                t2 = (X[keep].mean(axis=0)
                      - _transform(P[keep], a2, b2, np.zeros(2)).mean(axis=0))
                z2 = _coverage_response(smooth, valid, r_new)
                a2, b2, t2, sv2, sp2 = _refine_placement(z2, valid, P, a2,
                                                         b2, t2, r_t,
                                                         anchor_z)
                score2 = verified_score(z2, a2, b2, t2, sv2, sp2)
                if score2 > score:
                    z, score = z2, score2
                    a, b, t, snap_vals, snap_pos = a2, b2, t2, sv2, sp2
                    if ratio != ratios[0]:
                        ratios.reverse()  # keep trying this direction
                    improved = True
                    break
            if not improved:
                break

        # Column anchoring: when the scan's starting margin never
        # entered the mosaic, whole-pitch slides along the rows are
        # indistinguishable to the image, so slide toward C1 (where
        # every scan starts) while each slide costs less than half the
        # empty-tray penalty; a slide the image opposes always costs
        # more and is never taken.
        slid = 0
        i1, i2 = labels.index("R2C1"), labels.index("R2C2")
        toward_c10 = _is_flipped(a, b)
        cap = _evidence_cap(z, valid)
        for _ in range(max(ROW_COUNTS.values()) - 1):
            obs = ((snap_vals >= observed_z)
                   & arc_verified(snap_vals, _transform(P, a, b, t),
                                  snap_pos, np.hypot(a, b) * r_t))
            if not obs.any():
                break
            # Slack before an observed well's image column passes C1
            # (flipped mounts reverse the template column order).
            room = min((ROW_COUNTS[r] - c if toward_c10 else c - 1)
                       for r, c in (row_col(labels[i])
                                    for i in np.where(obs)[0]))
            if room < 1:
                break
            d = (P[i1] - P[i2]) @ np.array([[a, b], [-b, a]])
            t2 = t + d if toward_c10 else t - d
            a2, b2, t2, sv2, sp2 = _refine_placement(z, valid, P, a, b, t2,
                                                     r_t, anchor_z)
            score2 = verified_score(z, a2, b2, t2, sv2, sp2)
            if score2 < score - 0.5 * EMPTY_TRAY_PENALTY * cap:
                break
            a, b, t, snap_vals, snap_pos, score = a2, b2, t2, sv2, sp2, score2
            slid += 1
        return Placement(score, coarse_score, a, b, t, z, snap_vals,
                         snap_pos, slid)

    # Decide in two stages: within an orientation, scale basins are
    # separated by absolute verified evidence; between the orientations
    # (the same wells relabeled), only the symmetric difference of the
    # two placements is scored, where the mounts genuinely disagree.
    polished = [polish(cand) for cand in chosen]

    champs = {}
    for p in polished:
        k = _is_flipped(p.a, p.b)
        if k not in champs or p[:2] > champs[k][:2]:
            champs[k] = p

    def exclusive_score(p, rival):
        """Normalized verified contributions of p's in-view positions
        lying farther than half a pitch from every rival position
        (positions the mounts agree on cancel out of the duel).

        A disputed position is judged at the midpoint of the two
        claims: contrib / cap > 0.5 supports "well", less supports
        "empty tray". A corner arc scores structurally below a full
        ring, so the wrong mount's phantom end well costs it the
        duel."""
        X, in_view, contrib, vis, cap = score_terms(p.z, p.a, p.b, p.t,
                                                    p.snap_vals, p.snap_pos)
        Xr = _transform(P, rival.a, rival.b, rival.t)
        pitch = (np.hypot(*(P[labels.index("R2C1")]
                            - P[labels.index("R2C2")]))
                 * min(np.hypot(p.a, p.b), np.hypot(rival.a, rival.b)))
        disputed = np.hypot(X[:, None, 0] - Xr[None, :, 0],
                            X[:, None, 1] - Xr[None, :, 1]).min(axis=1)
        mine = in_view & (disputed >= 0.5 * pitch)
        return float((contrib / cap - 0.5 * vis)[mine].sum())

    if len(champs) == 2:
        pc, pf = champs[False], champs[True]
        ratio = np.hypot(pf.a, pf.b) / np.hypot(pc.a, pc.b)
        if max(ratio, 1.0 / ratio) <= scale_step:
            # Commensurate scales: the same placement relabeled, so
            # the duel over exclusive claims is meaningful.
            excl = {False: exclusive_score(pc, pf),
                    True: exclusive_score(pf, pc)}
            # Ties (no disputed spots in view) fall back to the
            # champions' z scores.
            flip_won = max(excl, key=lambda k: (excl[k], champs[k].score))
            info["orientation_exclusive_scores"] = {
                ("flipped" if k else "canonical"): round(v, 2)
                for k, v in excl.items()}
        else:
            # Different scale basins void the duel's premise; compare
            # absolute evidence strength instead.
            flip_won = max(champs, key=lambda k: champs[k][:2])
    else:
        flip_won = next(iter(champs))
    won = champs[flip_won]
    a, b, t, z, snap_vals, snap_pos = (won.a, won.b, won.t, won.z,
                                       won.snap_vals, won.snap_pos)
    info["coarse_score"] = round(won.coarse_score, 2)
    info["placement_score"] = round(won.score, 2)
    if won.slid:
        info["column_anchor_slides"] = won.slid
    info["candidates"] = [
        {"orientation": "flipped" if _is_flipped(p.a, p.b) else "canonical",
         "scale": round(float(np.hypot(p.a, p.b)), 3),
         "verified_score": round(p.score, 2)}
        for p in sorted(polished, key=lambda p: -p.score)]

    scale = float(np.hypot(a, b))
    theta = float(np.degrees(np.arctan2(b, a)))
    flipped = _is_flipped(a, b)
    r_out = scale * r_t
    X = _transform(P, a, b, t)

    if flipped:
        # Columns are numbered right-to-left in the *image* (C1
        # rightmost), so a flipped mount reverses the template order.
        labels = [_mirror_columns(l) for l in labels]

    # Positions that clear observed_z but fail arc verification are
    # reported as predicted: whatever they responded to is not a rim.
    verified = arc_verified(snap_vals, X, snap_pos, r_out)

    wells, predicted, lattice_placed = [], [], []
    for i, label in enumerate(labels):
        if snap_vals[i] >= observed_z and verified[i]:
            px, py = snap_pos[i]
            if _rim_visibility(valid, px, py, r_out) < FULL_RIM_VISIBILITY:
                # The mosaic cuts this rim, biasing the snap; the
                # fitted lattice places it better.
                px, py = X[i]
                lattice_placed.append(label)
            contrast, completeness = rim_stats(smooth, py, px, r_out)
            wells.append(Well(label=label, cy=float(py), cx=float(px),
                              r=float(r_out), score=float(snap_vals[i]),
                              contrast=contrast, completeness=completeness))
        else:
            predicted.append(PredictedWell(label=label, cy=float(X[i, 1]),
                                           cx=float(X[i, 0]), r=float(r_out)))

    h, w = mosaic.shape
    in_view = [0 <= xi < w and 0 <= yi < h and valid[int(yi), int(xi)]
               for xi, yi in X]
    info.update(scale=round(scale, 4),
                rotation_deg=round(theta - (180.0 if flipped else 0.0)
                                   + (360.0 if theta < -90 else 0.0), 2),
                orientation="flipped" if flipped else "canonical",
                feature_radius=round(r_out, 1),
                n_observed=len(wells), n_in_view=int(sum(in_view)),
                lattice_placed=lattice_placed,
                observed_z=[round(float(v), 1)
                            for v, ok in zip(snap_vals, verified)
                            if v >= observed_z and ok])
    # Acceptance: enough observed wells and a decent share of in-view
    # positions -- a mostly-empty placement is coincidence, not a match.
    if len(wells) < min_observed or len(wells) < 0.4 * sum(in_view):
        return [], [], info
    return wells, predicted, info


# --------------------------------------------------------------------------
# CLI: build a template by running the ring-fitting pipeline
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build a whole-layout well template by running the "
                    "ring-fitting pipeline on a dataset it detects well.")
    ap.add_argument("data_dir", type=Path, help="directory of .npz frames")
    ap.add_argument("--key", default="sample", help="npz key holding the image")
    ap.add_argument("--out", type=Path, default=DEFAULT_TEMPLATE_PATH,
                    help="output template JSON path")
    args = ap.parse_args()

    from .map_wells import (load_frames, register_frames, mosaic_transforms,
                            build_mosaic, detect_wells, label_wells)
    from .predict_wells import predict_missing_wells

    print(f"Loading Frames from {args.data_dir} ...")
    frames, meta, _ = load_frames(args.data_dir, args.key)
    print("Registering Frames ...")
    Hs, reg = register_frames(frames, meta)
    Gs, canvas_shape = mosaic_transforms(frames, Hs)
    print("Fusing Mosaic ...")
    mosaic, count = build_mosaic(frames, Gs, canvas_shape)
    print("Detecting Wells (Ring Matched Filter) ...")
    raw_wells, r_star, _ = detect_wells(mosaic, count)
    if not raw_wells:
        print("No Wells Detected; Cannot Build a Template", file=sys.stderr)
        sys.exit(1)
    wells = label_wells(raw_wells, r_star)
    predicted = predict_missing_wells(wells)
    print(f"  {len(wells)} Detected + {len(predicted)} Extrapolated Wells")

    template = build_template(wells, predicted, source=str(args.data_dir),
                              stage_gain=reg.get("stage_gain"))
    save_template(template, args.out)
    print(f"Template with {len(template['wells'])} Wells "
          f"(Radius ~{template['radius']:.0f}px) Written to {args.out}")


if __name__ == "__main__":
    main()
