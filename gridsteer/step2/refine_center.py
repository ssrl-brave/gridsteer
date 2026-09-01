"""Refine the center of a circular well in a single camera frame.

Given an off-axis camera image where a well is approximately centered,
find the sub-pixel offset (dx, dy) in pixels from image center to the
true well center using ring cross-correlation at a known radius.

Usage (standalone test):
    python -m gridsteer.step2.refine_center <image_url> <radius_px>

From Tcl (via fourCorners / goCirc):
    python -m gridsteer.step2.refine_center <image_url> <radius_px>
    # prints: dx_px dy_px
"""

import sys
import numpy as np
import cv2
from io import BytesIO


def ring_kernel(r, thickness=None):
    """Zero-mean annulus kernel at radius r."""
    if thickness is None:
        thickness = max(4.0, 0.1 * r)
    n = int(r + thickness + 4)
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(yy, xx)
    t = ((d > r - thickness / 2) & (d < r + thickness / 2)).astype(np.float32)
    return t - t.mean()


def refine_well_center(img, radius, search_radius=None):
    """Find the offset from image center to the well center.

    Parameters
    ----------
    img : 2-D uint8 array (grayscale camera frame)
    radius : float, well rim radius in pixels
    search_radius : int, max pixels to search from center (default: radius)

    Returns
    -------
    dx, dy : float, pixel offset (image center to well center).
             To center the well, move the content by (-dx, -dy).
    """
    from scipy.signal import fftconvolve

    if search_radius is None:
        search_radius = int(radius)

    smooth = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
    kern = ring_kernel(radius)

    resp = fftconvolve(smooth, kern[::-1, ::-1], mode="same")

    h, w = resp.shape
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - search_radius)
    y1 = min(h, cy + search_radius + 1)
    x0 = max(0, cx - search_radius)
    x1 = min(w, cx + search_radius + 1)
    crop = resp[y0:y1, x0:x1]

    # Sub-pixel via quadratic fit around the peak
    iy, ix = np.unravel_index(np.argmax(crop), crop.shape)
    peak_y = y0 + iy
    peak_x = x0 + ix

    def _subpix_y(arr, py, px):
        """Quadratic interpolation along the y axis at fixed x."""
        if py <= 0 or py >= arr.shape[0] - 1:
            return float(py)
        vm = float(arr[py - 1, px])
        v0 = float(arr[py, px])
        vp = float(arr[py + 1, px])
        denom = 2.0 * (vm - 2 * v0 + vp)
        if abs(denom) < 1e-12:
            return float(py)
        return py + (vm - vp) / denom

    def _subpix_x(arr, py, px):
        """Quadratic interpolation along the x axis at fixed y."""
        if px <= 0 or px >= arr.shape[1] - 1:
            return float(px)
        vm = float(arr[py, px - 1])
        v0 = float(arr[py, px])
        vp = float(arr[py, px + 1])
        denom = 2.0 * (vm - 2 * v0 + vp)
        if abs(denom) < 1e-12:
            return float(px)
        return px + (vm - vp) / denom

    sub_y = _subpix_y(resp, peak_y, peak_x)
    sub_x = _subpix_x(resp, peak_y, peak_x)

    dx = sub_x - cx
    dy = sub_y - cy
    return dx, dy


def img_from_url(url):
    """Fetch a grayscale image from a camera URL."""
    import requests
    from PIL import Image
    rq = requests.get(url)
    rq.raise_for_status()
    img = Image.open(BytesIO(rq.content)).convert("L")
    return np.asarray(img)


if __name__ == "__main__":
    url = sys.argv[1]
    radius = float(sys.argv[2])
    img = img_from_url(url)
    dx, dy = refine_well_center(img, radius)
    print(f"{dx:.2f} {dy:.2f}")
