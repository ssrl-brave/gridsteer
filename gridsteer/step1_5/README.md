# GridSteer Step 1.5: Circle Radius Detection

## Overview

Step 1.5 automatically detects the optimal radius for circular wells in images using Hough Circle Transform. It bridges Step 1 (phi optimization) and Step 2 (well tracking): Step 2 requires a `target_radius` parameter to accurately locate and track individual wells, and Step 1.5 finds that value.

To find the best radius, Step 1.5 performs a radius sweep — testing candidate radii from large to small (default: 200 --> 10, step 5). At each candidate radius, it runs a Hough Circle Transform and evaluates the quality of detections. Since Hough transform will find spurious circles at many radii, the sweep applies confidence scoring and row-constraint filtering to distinguish true well detections from noise.

The radius returned may not be pixel-perfect due to the step size, but Step 2 is designed to work within a tolerance so this approximation is sufficient.

> **Recommended**: Use a frame where at least 2 circles per row are clearly visible. This gives the row constraint filter enough information to reliably distinguish true well arrangements from random detections.

---

## System Architecture

Single-module design:
- **`find_radius.py`**: Loads a frame, sweeps candidate radii, scores and filters detections, and outputs the median detected radius to stdout.

---

## How It Works

### Step 1: Load Image

Loads image from NPZ file and normalizes to 0-1 range.

File: `find_radius.py:66-90` (`_load_image`)

### Step 2: Edge Detection

Applies Canny edge detection:
- **Sigma**: 15.0 (Gaussian smoothing)
- **Low threshold**: 0.3 (quantile)
- **High threshold**: 0.7 (quantile)

```python
edge = canny(img, sigma=15.0,
             low_threshold=0.3,
             high_threshold=0.7,
             use_quantiles=True)
```

File: `find_radius.py:186-189`

### Step 3: Radius Sweep

Tests radii from largest to smallest (default: 200 → 10, step 5).

The sweep starts large and works downward so that broad structures are considered before fine ones. At each candidate radius, the Hough Circle Transform is applied to the edge image. Because Hough-based detection is sensitive to edge noise, it will find many spurious circles at radii that don't correspond to actual wells — this is expected and handled by the filtering steps below.

For each test radius:
1. **Search window**: `test_radius ± 5 pixels`
2. **Hough Circle Transform**: Detect circles in edge image
3. **Extract peaks**: Find top 19 circles with highest accumulator confidence
4. **Spacing constraint**: Circles must be `2 × radius` apart (prevents overlapping detections of the same well)

File: `find_radius.py:194-207`

### Step 4: Row Counting

Counts horizontal rows by grouping circles with similar Y-coordinates:
- **Tolerance**: `test_radius × 0.5` (default)
- Circles within tolerance --> same row
- Gap > tolerance --> new row

```python
def _count_rows(y_coords, tolerance):
    sorted_y = np.sort(y_coords)
    rows = 1
    last_y = sorted_y[0]

    for y in sorted_y[1:]:
        if abs(y - last_y) > tolerance:
            rows += 1
            last_y = y

    return rows
```

File: `find_radius.py:50-64`

### Step 5: Confidence Scoring

Calculates quality score for each radius test:

```python
confidence_score = (num_circles × avg_confidence) / (1 + std_radius)
```

- More circles --> higher score
- Higher Hough accumulator --> higher score
- More radius variation --> lower score

File: `find_radius.py:209-214`

### Step 6: Row Constraint Filtering

Filters results that fit within a maximum number of rows (default: 2):

```python
fits_max_lines = (num_rows <= max_lines)
```

Wells in a plate are arranged in a staggered grid. After Step 1 aligns the tray, the visible wells appear in at most 1-2 horizontal rows in the frame. When the Hough transform runs at an incorrect radius it tends to produce spurious circles scattered across many different Y-coordinates, spanning far more than 2 rows. By requiring all detected circles to fit within `max_lines` rows, the algorithm rejects these erroneous multi-row detections and keeps only results consistent with the expected well-plate geometry.

This is why using a frame with at least 2 circles visible per row is recommended — the more circles present, the clearer the row structure and the more reliably spurious detections are rejected.

File: `find_radius.py:218`

### Step 7: Select Best Result

1. Filter to results that fit row constraint
2. Select result with highest confidence score
3. If none fit constraint, use best overall result

File: `find_radius.py:257-286` (`get_circles_on_lines`)

### Step 8: Calculate Median Radius

Extracts all radii from best result and calculates median:

```python
all_radii = [r for (cx, cy, r, acc) in best['circles']]
median_radius = np.median(all_radii)
```

The median radius is printed to stdout. Since the sweep uses a step size of 5 (default), the returned value is an approximation — but Step 2 accepts this and works within a tolerance.

File: `find_radius.py:307-308`

---

## Configuration Parameters

### Radius Search Range
- `--min-radius`: Minimum radius to test (default: 10)
- `--max-radius`: Maximum radius to test (default: 200)
- `--radius-step`: Step size (default: 5)

### Row Constraint
- `--max-lines`: Maximum rows allowed (default: 2)
- `--line-tolerance`: Y-coordinate grouping tolerance as fraction of radius (default: 0.5)

### Visualization
- `--visualize-all`: Save image for every radius tested
- `--no-visualize-best`: Skip saving best result image
- `--output-dir`: Directory for output images (default: `output_images`)

### Logging
- `--verbose` or `-v`: Enable detailed logging
- `--log-dir`: Directory for log files (default: `logs`)

---

## Usage Examples

### Basic Usage

```bash
python -m gridsteer.step1_5.find_radius <path_to_frame.npz>
```

### With Visualization

```bash
python -m gridsteer.step1_5.find_radius <path_to_frame.npz> \
    --visualize-all \
    --verbose
```

### Custom Radius Range

```bash
python -m gridsteer.step1_5.find_radius <path_to_frame.npz> \
    --min-radius <min_radius> \
    --max-radius <max_radius> \
    --radius-step <radius_step>
```
