"""Candidate scoring functions for tray-centering. Higher score = better.
"""
import cv2
import numpy as np


def laplacian(img_arr):
    """Original focus score: edge sharpness of the center patch."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    roi = img_arr[cy-75:cy+75, cx-75:cx+75]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def brightness(img_arr):
    """1. Right-half minus left-half brightness, clipped 0..10."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    band = img_arr[cy-h//4 : cy+h//4, :]
    val_l = np.mean(band[:, :cx]) / 255.0
    val_r = np.mean(band[:, cx:]) / 255.0
    return float(np.clip((val_r - val_l) * 10.0, 0, 10.0))


def edge_threshold(img_arr):
    """2. First bright column (midpoint threshold), Gaussian around center."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    band = img_arr[cy-h//4 : cy+h//4, :]
    col = band.mean(axis=0)
    col = np.convolve(col, np.ones(15)/15, mode="same")
    thr = (col.min() + col.max()) / 2
    bright = np.where(col > thr)[0]
    if bright.size == 0 or bright.size == w:         # edge not in frame
        return 0.0
    edge_x = bright[0]
    return float(10.0 * np.exp(-((edge_x - cx) / (0.2 * w))**2))


def edge_gradient(img_arr):
    """3. Strongest dark->bright column gradient, Gaussian around center."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    band = img_arr[cy-h//4 : cy+h//4, :].astype(float)
    col = np.convolve(band.mean(axis=0), np.ones(15)/15, mode="same")
    grad = np.gradient(col)
    edge_x = int(np.argmax(grad))
    if grad[edge_x] < 0.5:                           # no real edge
        return 0.0
    return float(10.0 * np.exp(-((edge_x - cx) / (0.2 * w))**2))


def hybrid(img_arr):
    """4. 5*coarse brightness hat + 5*narrow edge-position peak."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    band = img_arr[cy-h//4 : cy+h//4, :]
    val_l = band[:, :cx].mean() / 255.0
    val_r = band[:, cx:].mean() / 255.0
    coarse = np.clip(val_r - val_l, 0, 1)
    col = np.convolve(band.mean(axis=0), np.ones(15)/15, mode="same")
    thr = (col.min() + col.max()) / 2
    bright = np.where(col > thr)[0]
    if 0 < bright.size < w:
        fine = np.exp(-((bright[0] - cx) / (0.1 * w))**2)
    else:
        fine = 0.0
    return float(5.0 * coarse + 5.0 * fine)


def tiered(img_arr):
    """5. No edge: all-tray=1, all-bg=0. Edge: 2..10 by contrast."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    band = img_arr[cy-h//4 : cy+h//4, :]
    val_l = band[:, :cx].mean() / 255.0
    val_r = band[:, cx:].mean() / 255.0
    contrast = val_r - val_l
    if contrast < 0.05:
        return 1.0 if (val_l + val_r) / 2 < 0.75 else 0.0
    return float(2.0 + 8.0 * np.clip(contrast * 3.0, 0, 1))


def _otsu(band):
    """Otsu threshold of an 8-bit band."""
    t, _ = cv2.threshold(np.ascontiguousarray(band, dtype=np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def halves_otsu(img_arr):
    """7. Otsu: (frac of left half dark) * (frac of right half bright), 0..10."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    band = img_arr[cy-h//4 : cy+h//4, :]
    t = _otsu(band)
    lo, hi = band[band <= t], band[band > t]
    if lo.size == 0 or hi.size == 0 or hi.mean() - lo.mean() < 30:   # no real edge (noise)
        return 0.0
    left_dark = np.mean(band[:, :cx] <= t)
    right_bright = np.mean(band[:, cx:] > t)
    return float(10.0 * left_dark * right_bright)


def edge_otsu(img_arr):
    """8. Like edge_threshold, but Otsu threshold on the smoothed column profile."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    band = img_arr[cy-h//4 : cy+h//4, :]
    col = np.convolve(band.mean(axis=0), np.ones(15)/15, mode="same")
    t = _otsu(np.clip(col, 0, 255).reshape(1, -1))
    bright = np.where(col > t)[0]
    if bright.size == 0 or bright.size == w:
        return 0.0
    return float(10.0 * np.exp(-((bright[0] - cx) / (0.2 * w))**2))


def edge_soft(img_arr):
    """9. Like edge_gradient, but edge position = centroid of positive gradient."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    band = img_arr[cy-h//4 : cy+h//4, :].astype(float)
    col = np.convolve(band.mean(axis=0), np.ones(15)/15, mode="same")
    grad = np.clip(np.gradient(col), 0, None)
    if grad.max() < 0.5:                             # no real edge
        return 0.0
    grad[grad < 0.5 * grad.max()] = 0                # ignore noise away from the edge
    edge_x = float((grad * np.arange(w)).sum() / grad.sum())
    return float(10.0 * np.exp(-((edge_x - cx) / (0.2 * w))**2))


def brightness_median(img_arr):
    """10. Like brightness, but medians (ignores holes/outliers)."""
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    band = img_arr[cy-h//4 : cy+h//4, :]
    val_l = np.median(band[:, :cx]) / 255.0
    val_r = np.median(band[:, cx:]) / 255.0
    return float(np.clip((val_r - val_l) * 10.0, 0, 10.0))



SCORES = {f.__name__: f for f in
          (laplacian, brightness, edge_threshold, edge_gradient,
           hybrid, tiered, halves_otsu, edge_otsu, edge_soft,
           brightness_median)}
