# GridSteer

GridSteer automates crystallography sample tray alignment by determining the optimal phi angle, estimating well radius, and computing the motor positions needed to center any of the 19 wells on the tray.

## Pipeline Overview

The system runs in three sequential stages:

```
Step 1 --> Step 1.5 --> Step 2
 (phi)      (radius)    (motor positions)
```

### Step 1 — Phi Optimization (`gridsteer/step1`)

Determines the rotation angle (phi) at which the sample tray is face-on to the camera. It scans a sequence of frames, measuring the tray width in each one using a histogram-based approach (dark pixel clustering (identify tray region) -> PCA (find axis parallel to tray edge) -> perpendicular intensity profiles (estimate tray width)). It finds the frame with the minimum width, which is edge-on; adding 90° gives the face-on orientation.

**Output**: optimal `phi` angle (float)

### Step 1.5 — Well Radius Estimation (`gridsteer/step1_5`)

Estimates the radius of the wells in pixels. It performs a radial sweep using the Hough Circle Transform, validates candidates against a two-row geometric constraint, and returns the median radius of the best-scoring result. Should be run on a frame with at least two circles in each row.

**Output**: well radius in pixels (float)

### Step 2 — Well Tracking and Motor Calibration (`gridsteer/step2`)

Processes a motor scan to detect and track all 19 wells across frames, assign consistent IDs, and learn the pixel-to-motor mapping via linear regression. Once calibrated, it predicts the motor position required to center any individual well.

**Output**: JSON file with motor coordinates for each of the 19 wells

## Complete Workflow

```bash
# Step 1: Find the optimal rotation angle
python -m gridsteer.step1.optimize_phi /path/to/data --verbose

# Step 1.5: Estimate well radius from a representative frame
python -m gridsteer.step1_5.find_radius /path/to/frame.npz

# Step 2: Track wells and produce centering motor positions
python -m gridsteer.step2.main /path/to/frames --target_radius $radius --outdir ./output

# Query a specific well
python gridsteer/step2/read.py ./output 1 1
```

## Data Format

Frames are stored as NPZ files containing a grayscale image (`sample`) and motor position values (`x`, `y`, `z`, `phi`).

## Repository Structure

```
gridsteer/
├── gridsteer/
│   ├── step1/                  # Phi optimization
│   │   ├── optimize_phi.py         # Controller: iterates frames, tracks best result
│   │   ├── optimize_phi_transient.py  # Analyzer: measures tray width for a single frame
│   │   └── Old/                    # Previous approaches (line-finding, Hough)
│   ├── step1_5/                # Well radius estimation
│   │   └── find_radius.py          # Radial sweep + Hough circle detection
│   └── step2/                  # Well tracking and motor calibration
│       ├── main.py                 # Entry point and orchestration
│       ├── well_detection.py       # Hough circle and line detection
│       ├── well_tracking.py        # Well ID assignment and tracking across frames
│       ├── motor_prediction.py     # Pixel-to-motor calibration and prediction
│       ├── visualization.py        # Frame annotation and video output
│       ├── read.py                 # Query motor positions from JSON output
│       └── test_predictions.py     # Validation tool
└── pyproject.toml
```

## Further Reading

Each step has its own README with implementation details, configuration parameters, and usage examples:

- [`gridsteer/step1/README.md`](gridsteer/step1/README.md)
- [`gridsteer/step1_5/README.md`](gridsteer/step1_5/README.md)
- [`gridsteer/step2/README.md`](gridsteer/step2/README.md)
