# GridSteer

GridSteer automates crystallography sample tray alignment by determining the optimal phi angle and computing the motor positions needed to center any of the 19 wells on the tray.

## Pipeline Overview

The system runs in two sequential stages:

```
Step 1 --> Step 2
(phi)      (motor positions)
```

### Step 1 — Phi Optimization (`gridsteer/step1`)

Determines the rotation angle (phi) at which the sample tray is face-on to the camera. It scans a sequence of frames, measuring the tray width in each one using a histogram-based approach (dark pixel clustering (identify tray region) -> PCA (find axis parallel to tray edge) -> perpendicular intensity profiles (estimate tray width)). It finds the frame with the minimum width, which is edge-on; adding 90° gives the face-on orientation.

**Output**: optimal `phi` angle (float)

### Step 2 — Well Identification and Motor Calibration (`gridsteer/step2`)

Registers frames into a common tray coordinate system via SIFT + RANSAC homography, fuses them into a mosaic, and detects wells with a ring matched filter. Well radius, spacing, and rim width are all measured from the data — no manual tray geometry is needed beyond the expected well count per row. Undetected wells (from a truncated scan) are extrapolated from row geometry, and a linear ridge regression maps pixel positions to motor coordinates for centering any well.

**Output**: JSON file with motor coordinates for each of the 19 wells

## Complete Workflow

```bash
# Step 1: Find the optimal rotation angle
python -m gridsteer.step1.optimize_phi <path_to_data>

# Step 2: Identify wells and produce centering motor positions
python -m gridsteer.step2.map_wells <path_to_data> --out <output_dir>

# Check predictions against recorded frames
python -m gridsteer.step2.check_predictions \
    <output_dir>/<dataset>/well_centering_positions.json <path_to_data>
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
│   └── step2/                  # Well identification and motor calibration
│       ├── map_wells.py             # Main entry point: registration, mosaic, detection, tracking
│       ├── predict_wells.py        # Geometry helper: extrapolates wells, fits motor mapping
│       ├── check_predictions.py    # Validation: compares predictions against recorded frames
│       └── read.py                 # Query motor positions from JSON output
└── pyproject.toml
```

## Further Reading

Each step has its own README with implementation details, configuration parameters, and usage examples:

- [`gridsteer/step1/README.md`](gridsteer/step1/README.md)
- [`gridsteer/step2/README.md`](gridsteer/step2/README.md)
