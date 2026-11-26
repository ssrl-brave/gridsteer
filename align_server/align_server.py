import os
import sys
import cv2
import subprocess
import time
import argparse
import numpy as np
import requests
import uvicorn
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.ndimage import median_filter
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.morphology import binary_closing, disk, rectangle
from skimage.transform import hough_circle, hough_circle_peaks

# --- New Imports for Optimization ---
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound

from scores import SCORES


parser = argparse.ArgumentParser(description="Grid Alignment Server")
parser.add_argument("--host", default="0.0.0.0", help="Server host")
parser.add_argument("--port", type=int, default=8003, help="Server port")
parser.add_argument("--gridsteer-python", default=sys.executable,
                    help="python that can import gridsteer (for /optcirc/map_wells)")
parser.add_argument("--gridsteer-root", default=None,
                    help="working directory for running gridsteer (e.g. the gridsteer checkout)")

parser.add_argument("--score", choices=sorted(SCORES), default="brightness",
                    help="score function used by /bo/focus_step and /bo/focus_score")

args, _ = parser.parse_known_args()

app = FastAPI(title="Grid Aligner API")

# ==========================================
# 1. SHARED UTILITIES
# ==========================================

def calculate_focus_score(img_arr):
    """Score selected with --score (see scores.py). Higher = better."""
    return SCORES[args.score](img_arr)

#def calculate_split_score(img_pil):
#    arr = np.array(img_pil)
#    h, w = arr.shape # 400, 700
#    cy, cx = h // 2, w // 2
#    
#    patch_h, patch_w = h//10 , w // 10
#    
#    roi_L = arr[cy-patch_h:cy+patch_h, cx-patch_w:cx]
#    roi_R = arr[cy-patch_h:cy+patch_h, cx:cx+patch_w]
#    
#    # Calculate medians (robust to small holes/noise)
#    val_L = np.median(roi_L) / 255.0
#    val_R = np.median(roi_R) / 255.0
#    
#    # --- The New Math ---
#    # 1. Calculate the raw difference (Contrast)
#    # Goal: R should be much brighter than L
#    diff = val_R - val_L
#    
#    # 2. Amplify the difference. 
#    # We use a Sigmoid-like response or simple scaling to make the 
#    # peak much sharper for the Bayesian Optimizer.
#    # This will turn small improvements into big score jumps.
#    score = float(max(0, diff * 5.0))
#    #score = float(np.clip(diff * 10.0, 0, 10.0)) 
#    
#    return score, val_L, val_R

def calculate_split_score(img_arr):
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    
    # Define a small 60x60px area around the center to check the corner
    delta = 30 
    
    # Define the 4 quadrants around the absolute center
    q_top_left  = img_arr[cy-delta:cy, cx-delta:cx]
    q_bot_left  = img_arr[cy:cy+delta, cx-delta:cx]
    q_top_right = img_arr[cy-delta:cy, cx:cx+delta]
    q_bot_right = img_arr[cy:cy+delta, cx:cx+delta]
    
    # Normalize brightness (0.0 to 1.0)
    tl = np.median(q_top_left) / 255.0
    bl = np.median(q_bot_left) / 255.0
    tr = np.median(q_top_right) / 255.0
    br = np.median(q_bot_right) / 255.0

    # SCORE LOGIC: 
    # We want Left side DARK and Right side BRIGHT.
    # Specifically, the transition point is the corner.
    
    # Contrast Score: (Sum of Right) - (Sum of Left)
    contrast = (tr + br) - (tl + bl)
    
    # Perfect Corner Bonus: If all conditions are met, 
    # this creates a very sharp peak for the BO to find.
    score = float(np.clip(contrast * 5.0, 0, 10.0))
    
    # Optional: Penalize if the top and bottom on the left aren't both dark
    # This ensures we are on the vertical edge, not just a random horizontal one.
    if tl > 0.5 or bl > 0.5:
        score *= 0.5

    return score, (tl+bl)/2, (tr+br)/2


def calculate_grid_align_score(img_arr):
    """Score how well a grid/loop tip is centered at the camera crosshair.

    Expects dark sample on bright background. Base enters from the left,
    tip extends to the right. Returns 0-10 (higher = better centered).
    """
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2

    # Otsu threshold: dark sample vs bright background
    _, binary = cv2.threshold(img_arr, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = binary > 0

    dark_frac = mask.sum() / mask.size
    if dark_frac < 0.005:
        return 0.0   # empty frame — nothing visible
    if dark_frac > 0.95:
        return 0.1   # saturated / all dark

    # Tip = rightmost dark pixel in a horizontal band around center
    band_h = max(10, h // 6)
    band = mask[max(0, cy - band_h):min(h, cy + band_h), :]
    col_dark = band.any(axis=0)
    if not col_dark.any():
        return 0.1

    tip_x = float(np.max(np.where(col_dark)))

    # Vertical center of mass of the dark region
    row_dark = mask.any(axis=1)
    dark_rows = np.where(row_dark)[0]
    vert_center = float(dark_rows.mean()) if len(dark_rows) > 0 else float(cy)

    # Normalised distance from (tip_x, vert_center) to image center
    dx_norm = (tip_x - cx) / (w / 2.0)
    dy_norm = (vert_center - cy) / (h / 2.0)
    dist = np.sqrt(dx_norm ** 2 + dy_norm ** 2)

    # Smooth Gaussian score — peaks at 10 when perfectly centered
    score = 10.0 * np.exp(-(dist / 0.5) ** 2)
    return float(score)


#
#def calculate_split_score(img_pil):
#    """
#    Calculates the alignment score: 2*(1-Left) + Right
#    Left should be DARK (Sample), Right should be BRIGHT (Empty).
#    """
#    # Convert to Numpy
#    arr = np.array(img_pil)
#    h, w = arr.shape
#    
#    # Define Patches (Center Strip)
#    # Patch size: 100px tall, 50px wide
#    cy, cx = h // 2, w // 2
#    patch_h, patch_w = 100, 50
#    
#    # Left Patch (cx - 50 to cx)
#    roi_L = arr[cy-patch_h:cy+patch_h, cx-patch_w:cx]
#    # Right Patch (cx to cx + 50)
#    roi_R = arr[cy-patch_h:cy+patch_h, cx:cx+patch_w]
#    
#    # Use Median to filter out 400um holes (outliers)
#    # Normalize to 0.0 - 1.0 (assuming 8-bit image)
#    val_L = np.median(roi_L) / 255.0
#    val_R = np.median(roi_R) / 255.0
#    
#    # Objective Function
#    # Max Score (3.0) = Left is Black (0.0) + Right is White (1.0)
#    score = 2.0 * (1.0 - val_L) + val_R
#    
#    return score, val_L, val_R
#


def get_my_version():
    try:
        mtime = os.path.getmtime(__file__)
        last_mod = time.strftime("%b %d %H:%M", time.localtime(mtime))
        return f"v2.1 (Last mod: {last_mod})"
    except:
        return "unknown"

def img_from_url(url, as_grayscale=True, samp=1, med_size=1):
    try:
        rq = requests.get(url, timeout=5)
        img = Image.open(BytesIO(rq.content))
        if as_grayscale:
            img = img.convert("L")
        if samp is not None:
            new_shape = img.size[0]//samp, img.size[1]//samp
            img = img.resize(new_shape, Image.LANCZOS)
        img = np.array(img)
        if med_size > 1:
            img = median_filter(img, size=med_size)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image fetch failed: {e}")

@app.get("/version")
async def get_version():
    return {"server-version": get_my_version()}


# ==========================================
# 2. BAYESIAN OPTIMIZATION MANAGER
# ==========================================
class BOState:
    def __init__(self):
        self.optimizer = None
        # In newer versions, we instantiate the acquisition object directly
        self.acquisition_func = UpperConfidenceBound(kappa=2.5)#, xi=0.0)
        self.last_params = None
        self.initial_queue = []
        self.best_score = -1.0

    def initialize(self, h_start, v_start, h_range, v_range):
        # 'z' is used for Horizontal to match your Tcl regex
        pbounds = {
            'z': (h_start - h_range, h_start + h_range),
            'v': (v_start - v_range, v_start + v_range)
        }
        
        self.optimizer = BayesianOptimization(
            f=None, 
            pbounds=pbounds,
            verbose=2,
            random_state=1,
            acquisition_function=self.acquisition_func 
        )
        
        # 5-point probe: Center and the 4 boundaries of your symmetric box
        self.initial_queue = [
            {'z': h_start, 'v': v_start},
            {'z': h_start - h_range, 'v': v_start},
            {'z': h_start + h_range, 'v': v_start},
            {'z': h_start, 'v': v_start - v_range},
            {'z': h_start, 'v': v_start + v_range}
        ]
        self.last_params = None

    def suggest_next(self):
        # Validation check to prevent the NoneType error
        if self.optimizer is None:
            raise HTTPException(
                status_code=400, 
                detail="Optimizer not initialized. Did you call /bo/init or /bo/focus_init?"
            )

        if self.initial_queue:
            next_params = self.initial_queue.pop(0)
        else:
            next_params = self.optimizer.suggest()
        
        self.last_params = next_params
        return next_params

    def register_observation(self, score):
        if self.last_params is not None:
            self.optimizer.register(params=self.last_params, target=score)
            if score > self.best_score:
                self.best_score = score

    def focus_initialize(self, current_z, search_range=2.0):
        pbounds = {'z': (current_z - search_range, current_z + search_range)}
        
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            random_state=1,
            acquisition_function=self.acquisition_func
        )
        
        # Set the current Z as the "starting point" for iter 0
        self.last_params = {'z': current_z}
        
        # Initial probes: Check the center, then the two extremes
        self.initial_queue = [
            {'z': current_z - search_range},
            {'z': current_z + search_range}
        ]

    def grid_align_initialize(self, x_start, y_start, z_start,
                              x_range, y_range, z_range):
        """Set up 3-D BO over (sample_x, sample_y, sample_z)."""
        pbounds = {
            'x': (x_start - x_range, x_start + x_range),
            'y': (y_start - y_range, y_start + y_range),
            'z': (z_start - z_range, z_start + z_range),
        }
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            random_state=1,
            acquisition_function=self.acquisition_func,
        )
        # Start at center (we are already there after centerGoodLimits)
        self.last_params = {'x': x_start, 'y': y_start, 'z': z_start}
        # Initial probes: extremes of each axis
        self.initial_queue = [
            {'x': x_start - x_range, 'y': y_start, 'z': z_start},
            {'x': x_start + x_range, 'y': y_start, 'z': z_start},
            {'x': x_start, 'y': y_start - y_range, 'z': z_start},
            {'x': x_start, 'y': y_start + y_range, 'z': z_start},
            {'x': x_start, 'y': y_start, 'z': z_start - z_range},
            {'x': x_start, 'y': y_start, 'z': z_start + z_range},
        ]
        self.best_score = -1.0

# Instantiate Global Manager
bo_manager = BOState()

# PAYLOAD DEFINITIONS
# --- BO Pydantic Models ---
class InitBORequest(BaseModel):
    z_start: float = 0
    outdir: str = "/tmp"
    z_range: float = 1
    h_start: float = 0
    v_start: float = 0
    h_range: float = 1
    v_range: float = 1

class StepBORequest(BaseModel):
    url: str
    iter: int 

class ScoreBORequest(BaseModel):
    url: str
    samp: int = 1
    med_size: int = 1

class FocusStepRequest(BaseModel):
    url: str
    iter: int
    samp: int = 2      # Slight downsampling helps focus speed
    med_size: int = 3

class AlignGridInitRequest(BaseModel):
    x_start: float = 0.0
    y_start: float = 0.0
    z_start: float = 0.0
    x_range: float = 1.0
    y_range: float = 1.0
    z_range: float = 1.0

class AlignGridScoreRequest(BaseModel):
    url: str
    samp: int = 1
    med_size: int = 1

class AlignGridStepRequest(BaseModel):
    url: str
    face_score: float = 0.0
    samp: int = 1
    med_size: int = 1

###################
# END PAYLOAD DEF
####################


##########################
# --- BO Endpoints ---
##########################
@app.post("/bo/init")
async def init_bo(req: InitBORequest):
    bo_manager.initialize(
        h_start=req.h_start, 
        v_start=req.v_start, 
        h_range=req.h_range, 
        v_range=req.v_range
    )
    return {"status": "initialized", "h_center": req.h_start, "v_center": req.v_start}

@app.post("/bo/score")
async def score_bo(req: ScoreBORequest):
    
    img = img_from_url(req.url, samp=req.samp, med_size=req.med_size)
    current_score, val_l, val_r = calculate_split_score(img)
    
    return {
        "status": "DONE WITH SCORE",
        "score": current_score,
        "val_l": val_l,
        "val_r": val_r
    }

@app.post("/bo/step")
async def step_bo(req: StepBORequest):
    
    # If this is NOT the very first move (iter 0), we need to score the PREVIOUS move
    #    (The motor is currently sitting at last_params)
    current_score = 0.0
    val_l, val_r = 0.0, 0.0
    
    # We grab the image representing the result of the *previous* move
    img = img_from_url(req.url)
    current_score, val_l, val_r = calculate_split_score(img)
    
    # Register this result to update the model
    # (Skip for iter 0 if you haven't moved yet, but usually iter 0 counts as "Checking Start Position")
    bo_manager.register_observation(current_score)
    

    # Get Next Suggestion
    next_params = bo_manager.suggest_next()
    
    return {
        "status": "MOVE",
        "score": round(current_score, 3),
        "debug_L": round(val_l, 2),
        "debug_R": round(val_r, 2),
        "next_z": int(next_params['z']),
        "next_v": int(next_params['v'])
    }

@app.post("/bo/best_params_findGrid")
async def best_params_findGrid():
    """
    Returns the parameters that yielded the highest score so far.
    Used for the final 'homing' move.
    """
    if bo_manager.optimizer is None or not bo_manager.optimizer.res:
        raise HTTPException(status_code=400, detail="No optimization data available.")
    
    # .max returns {'target': score, 'params': {'z': value, ...}}
    best = bo_manager.optimizer.max
    
    return {
        "status": "SUCCESS",
        "score": round(best['target'], 4),
        "z": int(best['params']['z']),
        "v": int(best['params'].get('v', 0)) 
    }


@app.post("/bo/reset")
async def reset_bo():
    bo_manager.optimizer = None
    return {"status": "reset"}


####################
# AUTO-FOCUS ENDPOINTS
####################
@app.post("/bo/focus_init")
async def focus_init(req: InitBORequest):
    # Focus search range: current z +/- 1.0mm (adjust as needed)
    search_range = req.z_range
    pbounds = {'z': (req.z_start - search_range, req.z_start + search_range)}
    
    bo_manager.optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=1,
        acquisition_function=bo_manager.acquisition_func
    )
    
    # Start at the current position
    bo_manager.last_params = {'z': req.z_start}
    
    # Probes: Check the extremes
    bo_manager.initial_queue = [
        {'z': req.z_start - search_range},
        {'z': req.z_start + search_range}
    ]
    
    return {"status": "focus_initialized", "center_z": req.z_start}


@app.post("/bo/focus_step")
async def focus_step(req: FocusStepRequest):
    # 1. Capture and process image
    img = img_from_url(req.url, samp=req.samp, med_size=req.med_size)
    
    # Calculate sharpness
    current_score = calculate_focus_score(img)
    
    # 2. Register observation (attached to the 'z' move just performed)
    bo_manager.register_observation(current_score)
    
    # 3. Suggest next Z position
    # The suggest_next() logic remains the same, it just operates 
    # within the Z bounds set during focus_init
    next_params = bo_manager.suggest_next()
    
    # Note: Unlike alignment, we usually don't have a 'fixed' score for focus 
    # since every sample has different edge density. 
    # We usually run for a fixed N iterations (e.g., 10-12) then take the best.
    return {
        "status": "MOVE_Z",
        "score": round(current_score, 2),
        "next_z": int(next_params['z'])
    }

@app.post("/bo/focus_score")
async def focus_score(req: ScoreBORequest):
    
    img = img_from_url(req.url, samp=req.samp, med_size=req.med_size)
    current_score = calculate_focus_score(img)
    
    return {
        "status": "DONE WITH FOCUS SCORE",
        "score": current_score,
    }

@app.post("/bo/best_params")
async def get_best_params():
    """
    Returns the parameters that yielded the highest score so far.
    Used for the final 'homing' move.
    """
    if bo_manager.optimizer is None or not bo_manager.optimizer.res:
        raise HTTPException(status_code=400, detail="No optimization data available.")
    
    # .max returns {'target': score, 'params': {'z': value, ...}}
    best = bo_manager.optimizer.max
    
    return {
        "status": "SUCCESS",
        "score": round(best['target'], 4),
        "z": int(best['params']['z']),
    }

####################
# END of AUTO-FOCUS
#####################


# ==========================================
# 3. GRID / LOOP TIP ALIGNMENT (3-D BO)
# ==========================================

@app.post("/bo/align_grid_init")
async def align_grid_init(req: AlignGridInitRequest):
    """Initialise a 3-D Bayesian optimisation over sample_x, y, z."""
    bo_manager.grid_align_initialize(
        req.x_start, req.y_start, req.z_start,
        req.x_range, req.y_range, req.z_range,
    )
    return {
        "status": "initialized",
        "bounds": {
            "x": [req.x_start - req.x_range, req.x_start + req.x_range],
            "y": [req.y_start - req.y_range, req.y_start + req.y_range],
            "z": [req.z_start - req.z_range, req.z_start + req.z_range],
        },
    }


@app.post("/bo/align_grid_score")
async def align_grid_score(req: AlignGridScoreRequest):
    """Score a single camera frame (face or edge view)."""
    img = img_from_url(req.url, samp=req.samp, med_size=req.med_size)
    score = calculate_grid_align_score(img)
    return {"score": round(score, 4)}


@app.post("/bo/align_grid_step")
async def align_grid_step(req: AlignGridStepRequest):
    """Score edge view, combine with face_score, register, suggest next."""
    img = img_from_url(req.url, samp=req.samp, med_size=req.med_size)
    edge_score = calculate_grid_align_score(img)
    composite = req.face_score + edge_score

    bo_manager.register_observation(composite)
    next_params = bo_manager.suggest_next()

    return {
        "edge_score": round(edge_score, 4),
        "composite": round(composite, 4),
        "next_x": round(float(next_params['x']), 4),
        "next_y": round(float(next_params['y']), 4),
        "next_z": round(float(next_params['z']), 4),
    }


@app.post("/bo/align_grid_best")
async def align_grid_best():
    """Return the motor positions that yielded the highest composite score."""
    if bo_manager.optimizer is None or not bo_manager.optimizer.res:
        raise HTTPException(status_code=400,
                            detail="No optimisation data available.")
    best = bo_manager.optimizer.max
    return {
        "status": "SUCCESS",
        "score": round(best['target'], 4),
        "best_x": round(float(best['params']['x']), 4),
        "best_y": round(float(best['params']['y']), 4),
        "best_z": round(float(best['params']['z']), 4),
    }


# ==========================================
# 4. FIND FIRST CIRCLE (Hough-based search)
# ==========================================
#
# Strategy: user aligns tip face-on centered in camera.
# We step LEFT (toward base) with vertical zigzag, running Hough circle
# detection at each position. Once a circle is found, we nudge to center it.
# All state lives in memory — no files written.

from dataclasses import dataclass, field

@dataclass
class HoughParams:
    target_radius: int = 85
    radius_range: int = 20
    edge_sigma: float = 10.0
    edge_low: float = 0.15
    edge_high: float = 0.70
    hough_threshold: float = 0.15   # lower than step2 (0.2) to catch ~80% partial circles
    min_peak_distance: int = 120
    num_peaks: int = 10
    clean_edges: bool = False       # pre-blur + morphological closing to suppress noise


def count_rows(y_coords, tolerance):
    """Count distinct horizontal rows from y-coordinates."""
    if len(y_coords) == 0:
        return 0
    sorted_y = np.sort(y_coords)
    rows = 1
    last_y = sorted_y[0]
    for y in sorted_y[1:]:
        if abs(y - last_y) > tolerance:
            rows += 1
            last_y = y
    return rows


def circle_edge_coverage(edge_map, cx, cy, r, n_samples=72):
    """Fraction of circle perimeter that overlaps Canny edges.
    Real circles (well boundaries) score 0.3-0.8+; spurious detections score < 0.15."""
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    px = np.round(cx + r * np.cos(angles)).astype(int)
    py = np.round(cy + r * np.sin(angles)).astype(int)
    h, w = edge_map.shape
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    n_valid = valid.sum()
    if n_valid == 0:
        return 0.0
    # Check ±1 pixel neighborhood for edge hits (edges aren't pixel-perfect)
    hits = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            sx = np.clip(px[valid] + dx, 0, w - 1)
            sy = np.clip(py[valid] + dy, 0, h - 1)
            hits = max(hits, int(edge_map[sy, sx].sum()))
    return hits / n_valid


def detect_radius_sweep(img_gray, min_radius=30, max_radius=180, radius_step=5,
                        canny_sigma=15.0, canny_low=0.3, canny_high=0.7,
                        num_peaks=19, max_lines=2, line_tolerance=0.5,
                        debug_dir="", debug_label=""):
    """Sweep radii to auto-detect circle radius. Returns (median_radius, best_result_dict) or (None, None).
    Scores by edge coverage (fraction of perimeter with actual edges) instead of raw Hough accumulator."""
    import os

    # Normalize image to 0-1
    img = img_gray.astype(float)
    if img.max() > 1.0:
        img = img / 255.0

    edge = canny(img, sigma=canny_sigma,
                 low_threshold=canny_low, high_threshold=canny_high,
                 use_quantiles=True)

    results = []
    for test_radius in range(max_radius, min_radius - 1, -radius_step):
        r_min = max(5, test_radius - 5)
        r_max = test_radius + 5
        rads = np.arange(r_min, r_max + 1)

        hough_res = hough_circle(edge, rads)
        accum, cx, cy, radii = hough_circle_peaks(
            hough_res, rads,
            min_xdistance=int(2 * test_radius),
            min_ydistance=int(2 * test_radius),
            num_peaks=num_peaks,
            threshold=None,
        )
        if len(accum) == 0:
            continue

        num_circles = len(accum)
        avg_radius = float(np.mean(radii))
        std_radius = float(np.std(radii)) if num_circles > 1 else 0.0

        # Compute edge coverage for each detected circle
        coverages = [circle_edge_coverage(edge, float(x), float(y), float(r))
                     for x, y, r in zip(cx, cy, radii)]
        avg_coverage = float(np.mean(coverages))

        # Score: edge coverage (are circles real?) * capped count / consistency
        score = (avg_coverage * min(num_circles, 3)) / (1 + std_radius)

        tolerance = test_radius * line_tolerance
        num_rows = count_rows(cy, tolerance)
        fits = num_rows <= max_lines

        circles = [{"cx": float(x), "cy": float(y), "r": float(r),
                     "score": float(a), "edge_cov": round(cov, 2)}
                   for a, x, y, r, cov in zip(accum, cx, cy, radii, coverages)]

        results.append({
            "test_radius": test_radius,
            "num_circles": num_circles,
            "avg_radius": avg_radius,
            "std_radius": std_radius,
            "avg_edge_coverage": avg_coverage,
            "confidence_score": score,
            "num_rows": num_rows,
            "fits_lines": fits,
            "circles": circles,
        })

    if not results:
        return None, None

    # Prefer results that fit the line constraint
    fits_results = [r for r in results if r["fits_lines"]]
    pool = fits_results if fits_results else results
    best = max(pool, key=lambda r: r["confidence_score"])

    all_radii = [c["r"] for c in best["circles"]]
    median_radius = float(np.median(all_radii))

    # Debug: save annotated sweep result
    if debug_dir:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            for c in best["circles"]:
                cx_i, cy_i, r_i = int(c["cx"]), int(c["cy"]), int(c["r"])
                cov = c["edge_cov"]
                cv2.circle(vis, (cx_i, cy_i), r_i, (0, 255, 255), 2)
                cv2.circle(vis, (cx_i, cy_i), 3, (0, 255, 255), -1)
                cv2.putText(vis, f'r={r_i} cov={cov:.0%}', (cx_i - 40, cy_i - r_i - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            label = (f"radius_sweep{debug_label}: median={median_radius:.0f}px  "
                     f"n={best['num_circles']}  rows={best['num_rows']}  "
                     f"avg_cov={best['avg_edge_coverage']:.0%}")
            cv2.putText(vis, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            fname = f"findcirc_radius_sweep{debug_label}.png"
            cv2.imwrite(os.path.join(debug_dir, fname), vis)
        except Exception:
            pass

    return median_radius, best


def detect_circles_hough(img_gray, p: HoughParams):
    """Run Hough circle detection. Returns (circles_list, edge_map).
    When p.clean_edges is True: pre-blur, raise thresholds, morphological closing."""
    if p.clean_edges:
        img_pre = gaussian(img_gray, sigma=0.5, preserve_range=True)
        edge = canny(
            img_pre, sigma=15.0,
            low_threshold=0.3, high_threshold=0.7,
            use_quantiles=True,
        )
        edge = binary_closing(edge, disk(3))
        edge = binary_closing(edge, rectangle(1, 7))
        edge = binary_closing(edge, rectangle(7, 1))
    else:
        edge = canny(
            img_gray, sigma=p.edge_sigma,
            low_threshold=p.edge_low, high_threshold=p.edge_high,
            use_quantiles=True,
        )
    r_min = max(10, p.target_radius - p.radius_range)
    r_max = p.target_radius + p.radius_range
    rads = np.arange(r_min, r_max + 1)

    hspace = hough_circle(edge, rads)
    accum, cx, cy, radii = hough_circle_peaks(
        hspace, rads,
        min_xdistance=p.min_peak_distance,
        min_ydistance=p.min_peak_distance,
        num_peaks=p.num_peaks,
        threshold=p.hough_threshold,
    )
    circles = []
    for a, x, y, r in zip(accum, cx, cy, radii):
        circles.append({"cx": float(x), "cy": float(y), "r": float(r), "score": float(a)})
    return circles, edge


def annotate_circles(img_gray, circles, frame_cx, frame_cy, step_index, phase, debug_dir,
                     edge_map=None):
    """Draw detected circles + crosshair on image and save as PNG.
    If edge_map is provided, also save the Canny edge image with circles overlaid."""
    import os
    os.makedirs(debug_dir, exist_ok=True)
    # Convert grayscale to BGR for colored annotations
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Crosshair at frame center
    cv2.line(vis, (int(frame_cx) - 20, int(frame_cy)), (int(frame_cx) + 20, int(frame_cy)), (0, 255, 0), 1)
    cv2.line(vis, (int(frame_cx), int(frame_cy) - 20), (int(frame_cx), int(frame_cy) + 20), (0, 255, 0), 1)

    for i, c in enumerate(circles):
        cx, cy, r = int(c["cx"]), int(c["cy"]), int(c["r"])
        color = (0, 255, 255) if i == 0 else (255, 180, 0)
        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.circle(vis, (cx, cy), 3, color, -1)
        cv2.putText(vis, f'r={r} ({c["score"]:.2f})', (cx - 30, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    label = f"step={step_index} {phase} n={len(circles)}"
    cv2.putText(vis, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    fname = os.path.join(debug_dir, f"findcirc_{step_index:03d}_{phase}.png")
    cv2.imwrite(fname, vis)

    # Save Canny edge image with circles overlaid
    if edge_map is not None:
        edge_vis = (edge_map.astype(np.uint8) * 255)
        edge_bgr = cv2.cvtColor(edge_vis, cv2.COLOR_GRAY2BGR)
        cv2.line(edge_bgr, (int(frame_cx) - 20, int(frame_cy)), (int(frame_cx) + 20, int(frame_cy)), (0, 255, 0), 1)
        cv2.line(edge_bgr, (int(frame_cx), int(frame_cy) - 20), (int(frame_cx), int(frame_cy) + 20), (0, 255, 0), 1)
        for i, c in enumerate(circles):
            cx, cy, r = int(c["cx"]), int(c["cy"]), int(c["r"])
            color = (0, 255, 255) if i == 0 else (255, 180, 0)
            cv2.circle(edge_bgr, (cx, cy), r, color, 1)
            cv2.circle(edge_bgr, (cx, cy), 3, color, -1)
        cv2.putText(edge_bgr, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        edge_fname = os.path.join(debug_dir, f"findcirc_{step_index:03d}_{phase}_edges.png")
        cv2.imwrite(edge_fname, edge_bgr)


@dataclass
class FindCircState:
    """Holds state across the findcirc init/step loop."""
    camera_url: str = ""
    target_radius: int = 85
    hough: HoughParams = field(default_factory=HoughParams)

    step_h_px: int = 50        # horizontal step leftward (pixels)
    max_steps: int = 5

    debug_dir: str = ""        # if non-empty, write annotated PNGs here

    step_index: int = 0
    found: bool = False
    best_circle: dict = None
    img_h: int = 0
    img_w: int = 0

    centering: bool = False
    center_iters: int = 0
    max_center_iters: int = 5


findcirc_state: FindCircState = None


class FindCircInitRequest(BaseModel):
    camera_url: str
    target_radius: int = 85
    radius_range: int = 20
    step_h_px: int = 50        # horizontal search step in pixels (leftward)
    max_steps: int = 5
    debug_dir: str = ""
    clean_edges: bool = False  # pre-blur + morphological closing for noisy images


class FindCircStepRequest(BaseModel):
    pass  # camera URL already in state


class DetectRadiusRequest(BaseModel):
    camera_url: str
    min_radius: int = 30
    max_radius: int = 180
    radius_step: int = 5
    n_images: int = 3          # number of snapshots for voting
    debug_dir: str = ""


@app.post("/findcirc/detect_radius")
async def findcirc_detect_radius(req: DetectRadiusRequest):
    """Sweep radii on multiple camera images, vote on consensus radius."""
    import time

    winners = []       # (median_radius, best_dict) per image
    all_images = []

    for i in range(req.n_images):
        if i > 0:
            time.sleep(0.3)  # brief pause between grabs for variation
        img = img_from_url(req.camera_url)
        all_images.append(img)
        median_r, best = detect_radius_sweep(
            img,
            min_radius=req.min_radius,
            max_radius=req.max_radius,
            radius_step=req.radius_step,
            debug_dir=req.debug_dir,
            debug_label=f"_{i+1}" if req.n_images > 1 else "",
        )
        if median_r is not None:
            winners.append((median_r, best))

    if not winners:
        return {"status": "fail", "message": "No circles detected in any image", "radius": 0}

    # Consensus: median of winning radii across images
    radii_votes = [w[0] for w in winners]
    consensus_radius = int(round(float(np.median(radii_votes))))

    # Pick the best result closest to consensus for reporting
    best_winner = min(winners, key=lambda w: abs(w[0] - consensus_radius))
    best = best_winner[1]

    # Debug: save a summary image using the best frame
    if req.debug_dir:
        try:
            votes_str = ", ".join(f"{r:.0f}" for r in radii_votes)
            best_img = all_images[winners.index(best_winner)]
            vis = cv2.cvtColor(best_img, cv2.COLOR_GRAY2BGR)
            for c in best["circles"]:
                cx_i, cy_i, r_i = int(c["cx"]), int(c["cy"]), int(c["r"])
                cov = c["edge_cov"]
                cv2.circle(vis, (cx_i, cy_i), r_i, (0, 255, 255), 2)
                cv2.circle(vis, (cx_i, cy_i), 3, (0, 255, 255), -1)
                cv2.putText(vis, f'r={r_i} cov={cov:.0%}', (cx_i - 40, cy_i - r_i - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            label = (f"CONSENSUS r={consensus_radius}px  votes=[{votes_str}]  "
                     f"n={best['num_circles']}  cov={best['avg_edge_coverage']:.0%}")
            cv2.putText(vis, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(req.debug_dir, "findcirc_radius_sweep.png"), vis)
        except Exception:
            pass

    return {
        "status": "ok",
        "radius": consensus_radius,
        "num_circles": best["num_circles"],
        "num_rows": best["num_rows"],
        "votes": radii_votes,
        "message": (f"Detected radius {consensus_radius}px "
                    f"(votes: [{', '.join(f'{r:.0f}' for r in radii_votes)}], "
                    f"avg_cov={best['avg_edge_coverage']:.0%})"),
    }


@app.post("/findcirc/init")
async def findcirc_init(req: FindCircInitRequest):
    global findcirc_state
    hp = HoughParams(target_radius=req.target_radius, radius_range=req.radius_range,
                      clean_edges=req.clean_edges)
    findcirc_state = FindCircState(
        camera_url=req.camera_url,
        target_radius=req.target_radius,
        hough=hp,
        step_h_px=req.step_h_px,
        max_steps=req.max_steps,
        debug_dir=req.debug_dir,
    )
    # Clean old debug images
    if req.debug_dir:
        import glob as _glob
        os.makedirs(req.debug_dir, exist_ok=True)
        for old in _glob.glob(os.path.join(req.debug_dir, "findcirc_*.png")):
            if "radius_sweep" in os.path.basename(old):
                continue  # keep radius sweep images from detect_radius
            try:
                os.remove(old)
            except Exception:
                pass

    msg = "findcirc session initialized"
    if req.debug_dir:
        msg += f" (debug images → {req.debug_dir})"
    return {"status": "ok", "message": msg}


@app.post("/findcirc/step")
async def findcirc_step(req: FindCircStepRequest):
    global findcirc_state
    if findcirc_state is None:
        return {"status": "fail", "message": "No active session — call /findcirc/init first"}

    s = findcirc_state

    # Capture image (reuse align_server's img_from_url)
    img = img_from_url(s.camera_url)
    s.img_h, s.img_w = img.shape[:2]
    frame_cx = s.img_w / 2.0
    frame_cy = s.img_h / 2.0

    # Detect circles
    circles, edge_map = detect_circles_hough(img, s.hough)

    # Debug: save annotated image + edge image
    phase = "center" if s.centering else "search"
    if s.debug_dir:
        try:
            annotate_circles(img, circles, frame_cx, frame_cy, s.step_index, phase, s.debug_dir,
                             edge_map=edge_map)
        except Exception as e:
            pass  # don't let debug IO break the pipeline

    # --- CENTERING PHASE ---
    if s.centering:
        if not circles:
            s.found = True
            return {
                "status": "done",
                "move_dx_frac": 0.0, "move_dy_frac": 0.0,
                "message": f"Circle lost during centering (iter {s.center_iters}). Stopping.",
                "n_circles": 0, "step": s.step_index, "best_circle": s.best_circle,
            }

        best = min(circles, key=lambda c: (c["cx"] - frame_cx)**2 + (c["cy"] - frame_cy)**2)
        s.best_circle = best
        dx_px = best["cx"] - frame_cx
        dy_px = best["cy"] - frame_cy
        dist_px = (dx_px**2 + dy_px**2) ** 0.5
        s.center_iters += 1

        if dist_px < s.target_radius * 0.25 or s.center_iters >= s.max_center_iters:
            s.found = True
            return {
                "status": "done",
                "move_dx_frac": 0.0, "move_dy_frac": 0.0,
                "message": f"Centered on circle (offset {dist_px:.0f}px after {s.center_iters} iters)",
                "n_circles": len(circles), "step": s.step_index, "best_circle": best,
            }

        # Return image-fraction offsets for moveSample (BluIce moveSampleRelative)
        # Negate: positive dx_frac pans view LEFT, but dx_px>0 means circle is RIGHT of center
        dx_frac = -dx_px / s.img_w
        dy_frac = -dy_px / s.img_h
        s.step_index += 1
        return {
            "status": "center",
            "move_dx_frac": round(dx_frac, 6), "move_dy_frac": round(dy_frac, 6),
            "message": f"Centering iter {s.center_iters}: offset ({dx_px:.0f}, {dy_px:.0f})px",
            "n_circles": len(circles), "step": s.step_index, "best_circle": best,
        }

    # --- SEARCH PHASE ---
    if circles:
        best = min(circles, key=lambda c: (c["cx"] - frame_cx)**2 + (c["cy"] - frame_cy)**2)
        s.best_circle = best
        s.centering = True
        dx_px = best["cx"] - frame_cx
        dy_px = best["cy"] - frame_cy
        dist_px = (dx_px**2 + dy_px**2) ** 0.5

        if dist_px < s.target_radius * 0.25:
            s.found = True
            return {
                "status": "done",
                "move_dx_frac": 0.0, "move_dy_frac": 0.0,
                "message": f"Found circle already near center ({dist_px:.0f}px off)",
                "n_circles": len(circles), "step": s.step_index, "best_circle": best,
            }

        # Negate: positive dx_frac pans view LEFT, but dx_px>0 means circle is RIGHT of center
        dx_frac = -dx_px / s.img_w
        dy_frac = -dy_px / s.img_h
        s.center_iters = 1
        s.step_index += 1
        return {
            "status": "center",
            "move_dx_frac": round(dx_frac, 6), "move_dy_frac": round(dy_frac, 6),
            "message": f"Found {len(circles)} circle(s)! Centering... offset ({dx_px:.0f}, {dy_px:.0f})px",
            "n_circles": len(circles), "step": s.step_index, "best_circle": best,
        }

    # No circles — next search move
    if s.step_index >= s.max_steps:
        return {
            "status": "fail",
            "move_dx_frac": 0.0, "move_dy_frac": 0.0,
            "message": f"No circles found after {s.max_steps} steps",
            "n_circles": 0, "step": s.step_index, "best_circle": None,
        }

    # Step toward base (positive dx_frac = image pans left toward base)
    dx_frac = s.step_h_px / s.img_w if s.img_w > 0 else 0.05
    dy_frac = 0.0

    s.step_index += 1
    return {
        "status": "search",
        "move_dx_frac": round(dx_frac, 6), "move_dy_frac": round(dy_frac, 6),
        "message": f"Searching step {s.step_index}/{s.max_steps}...",
        "n_circles": 0, "step": s.step_index, "best_circle": None,
    }


@app.post("/findcirc/result")
async def findcirc_result():
    """Return final result after search is done."""
    global findcirc_state
    if findcirc_state is None:
        return {"status": "fail", "message": "No session"}
    if findcirc_state.found and findcirc_state.best_circle:
        return {
            "status": "ok",
            "circle": findcirc_state.best_circle,
            "steps": findcirc_state.step_index,
            "center_iters": findcirc_state.center_iters,
        }
    return {"status": "fail", "message": "No circle found", "steps": findcirc_state.step_index}


# ==========================================
# END of FIND FIRST CIRCLE
# ==========================================


# ==========================================
# optCirc WELL MAPPING (gridsteer step2)
# ==========================================
# Runs gridsteer.step2.map_wells here on pxgpu03 instead of on the
# beamline host, which lacks the memory for the mosaic. The scan dir
# must be visible at the same path on both machines (/data is shared).

class MapWellsRequest(BaseModel):
    data_dir: str
    outdir: str
    timeout_s: int = 900


@app.post("/optcirc/map_wells")
def optcirc_map_wells(req: MapWellsRequest):
    # Plain def: FastAPI runs it in a worker thread, so the BO endpoints
    # stay responsive while the mapping runs. Subprocess keeps the mosaic
    # memory out of the server process.
    if not os.path.isdir(req.data_dir):
        return {"status": "fail", "message": f"data_dir not found on this host: {req.data_dir}"}
    cmd = [args.gridsteer_python, "-m", "gridsteer.step2.map_wells",
           req.data_dir, "--outdir", req.outdir]
    t = time.time()
    try:
        proc = subprocess.run(cmd, cwd=args.gridsteer_root, capture_output=True,
                              text=True, timeout=req.timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "fail", "message": f"map_wells timed out after {req.timeout_s}s"}
    elapsed = round(time.time() - t, 1)
    mapping = os.path.join(req.outdir, "mapping.json")
    ok = proc.returncode == 0 and os.path.exists(mapping)
    return {
        "status": "ok" if ok else "fail",
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "mapping": mapping if ok else None,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "message": (f"mapping.json written in {elapsed}s" if ok
                    else f"map_wells failed (rc={proc.returncode}) after {elapsed}s"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
