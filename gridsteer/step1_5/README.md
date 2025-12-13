# GridSteer Step 1.5: Circle Radius Detection

## Overview

Step 1.5 automatically detects the optimal radius for circular wells in images using Hough Circle Transform. It runs before Step 2 to provide the `target_radius` parameter needed for accurate well tracking.

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

Tests radii from largest to smallest (default: 200 -> 10, step 5):

For each test radius:
1. **Search window**: `test_radius +/- 5 pixels`
2. **Hough Circle Transform**: Detect circles in edge image
3. **Extract peaks**: Find top 19 circles with highest confidence
4. **Spacing constraint**: Circles must be `2 x radius` apart

File: `find_radius.py:194-207`

### Step 4: Row Counting

Counts horizontal rows by grouping circles with similar Y-coordinates:
- **Tolerance**: `test_radius x 0.5` (default)
- Circles within tolerance -> same row
- Gap > tolerance -> new row

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

- More circles -> higher score
- Higher Hough accumulator -> higher score
- More radius variation -> lower score

File: `find_radius.py:209-214`

### Step 6: Row Constraint Filtering

Filters results that fit within maximum number of rows (default: 2):

```python
fits_max_lines = (num_rows <= max_lines)
```

This eliminates false detections that don't arrange into neat horizontal rows.

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
python -m gridsteer.step1_5.find_radius /path/to/frame.npz
```

### With Visualization

```bash
python -m gridsteer.step1_5.find_radius /path/to/frame.npz \
    --visualize-all \
    --verbose
```

### Custom Radius Range

```bash
python -m gridsteer.step1_5.find_radius /path/to/frame.npz \
    --min-radius 75 \
    --max-radius 105 \
    --radius-step 2
```