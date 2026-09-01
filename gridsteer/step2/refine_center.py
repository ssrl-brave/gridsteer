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

    def _subpix(arr, idx, axis):
        """Quadratic interpolation along one axis."""
        if idx <= 0 or idx >= arr.shape[axis] - 1:
            return float(idx)
        s = [slice(None)] * arr.ndim
        s[axis] = idx - 1
        vm = float(arr[tuple(s)])
        s[axis] = idx
        v0 = float(arr[tuple(s)])
        s[axis] = idx + 1
        vp = float(arr[tuple(s)])
        denom = 2.0 * (vm - 2 * v0 + vp)
        if abs(denom) < 1e-12:
            return float(idx)
        return idx + (vm - vp) / denom

    sub_y = _subpix(resp, peak_y, 0)
    sub_x = _subpix(resp, peak_x, 1)

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
