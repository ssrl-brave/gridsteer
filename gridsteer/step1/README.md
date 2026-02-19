# GridSteer Step 1: Phi Optimization

## Overview

Step 1 finds the optimal phi to orient the well plate face-on toward the camera. The face-on view is the target orientation, but it is easier to detect the edge-on view (where the tray appears at its narrowest) and then add 90° to derive the face-on phi. The algorithm processes a sequence of frames captured at different rotation angles and identifies the frame with the minimum measured tray width, which corresponds to the edge-on orientation.

## System Architecture

Two-module design:
- **`optimize_phi.py`**: Persistent controller that iterates through frames
- **`optimize_phi_transient.py`**: Non-persistent analyzer that processes individual frames

---

## How It Works

### Main Process Flow

The controller (`optimize_phi.py`) calls the analyzer (`optimize_phi_transient.py`) for each frame, tracking which frame has the minimum tray width.

### Step 1: Load Frame

Loads image and motor position from NPZ file.

File: `optimize_phi_transient.py:397-410` (`load_frame_data`)

### Step 2: Dark Point Detection

Identifies dark pixels below an adaptive threshold:
- **Threshold**: Based on percentile (default: 25th percentile)
- **Purpose**: Dark regions in a frame correspond to the tray/well plate

```python
intensity_threshold = np.percentile(img_normalized, dark_percentile)
dark_mask = img_normalized < intensity_threshold
dark_points = np.column_stack(np.where(dark_mask))
```

However, shadows and optical artifacts in the image can also appear as dark regions, so the raw set of dark points is not a clean representation of the tray alone.

File: `optimize_phi_transient.py:144-162`

### Step 3: DBSCAN Clustering

Clusters dark points to isolate the main tray region from noise, shadows, and artifacts:

```python
clustering = DBSCAN(eps=15, min_samples=50).fit(dark_points)
```

The largest cluster of dark points corresponds to the tray. Any smaller clusters are most likely shadows or imaging artifacts and are discarded. This step ensures subsequent analysis operates only on the tray's actual pixel footprint.

File: `optimize_phi_transient.py:164-193`

### Step 4: PCA

Finds the principal axis of the tray cluster. Because the tray can be oriented at any angle in the frame, PCA determines the dominant direction of the tray's pixel distribution:

```python
cov_matrix = np.cov(centered_points.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
principal_vector = eigenvectors[:, argmax(eigenvalues)]
```

**Result**: Line equation `y = m_parallel * x + b_parallel` representing the tray's main axis. This axis is used to generate perpendicular sampling lines so that width measurements are always taken across the tray rather than along it.

File: `optimize_phi_transient.py:199-220`

### Step 5: Generate Perpendicular Lines

Creates evenly-spaced lines perpendicular to the tray axis:

```python
m_perpendicular = -1.0 / m_parallel
```

Default: 5 perpendicular lines across the tray width. Sampling along these lines produces intensity histograms (profiles) that capture the cross-sectional shape of the tray, from which the width is measured.

File: `optimize_phi_transient.py:222-238`

### Step 6: Measure Width Along Each Line

For each perpendicular line:

1. **Sample intensities** along the line
2. **Apply Gaussian smoothing** (sigma=5.0)
3. **Find deepest trough** (darkest region)
4. **Detect width boundaries** using curvature analysis:
   - Calculate second derivative (curvature)
   - Stop at points where curvature exceeds threshold
   - This detects the transition from flat trough to curved edge

Since the physical tray has surface imperfections, the intensity profile is not a clean flat-bottomed trough. The transition from tray interior to tray edge is not always a sharp, smooth step, so boundary detection relies on curvature rather than a simple intensity cutoff.

File: `optimize_phi_transient.py:244-358`

### Step 7: Calculate Median Width

Calculates median width across all perpendicular lines:

```python
avg_width = np.median(widths)
```

File: `optimize_phi_transient.py:363-364`

### Step 8: Track Best Frame

The controller maintains state across frames:

```python
if width < min_width_found:
    min_width_found = width
    best_frame_info = {
        'frame_number': frame_number,
        'phi': motor_data.phi,
        'avg_width': width
    }
```

File: `optimize_phi_transient.py:447-453`

### Step 9: Output Result

After processing all frames, outputs the phi value from the best frame:

```python
corrected_phi = best_frame_info['phi'] + 90
print(f"{corrected_phi:.6f}")
```

**+90 correction**: Adds 90 so the tray's wells are facing the camera.

File: `optimize_phi.py:200-213`

---

## Configuration Parameters

### Detection Parameters
- `dark_percentile`: Percentile threshold for dark points (default: 25.0)
- `num_perpendicular_lines`: Number of width measurement lines (default: 5)

### Smoothing Parameters
- `smoothing_sigma`: Gaussian filter sigma for intensity smoothing (default: 5.0)

### Curvature Detection
- `curvature_percentile`: Percentile for adaptive curvature threshold (default: 85.0)
- `min_curvature_threshold`: Minimum curvature to avoid noise (default: 0.0001)

### Clustering
Hardcoded in `optimize_phi_transient.py`:
- `eps`: 15 pixels (DBSCAN distance parameter)
- `min_samples`: 50 points (DBSCAN minimum cluster size)

### Output
- `save_individual_frames`: Save visualization for each frame (default: True)
- `output_images_dir`: Directory for output images (default: `output_images_1`)

### Logging
- `--verbose` or `-v`: Enable detailed logging
- `--log-dir`: Directory for log files (default: `logs_1`)

---

## Usage Examples

### Basic Usage

```bash
python -m gridsteer.step1.optimize_phi /path/to/data --verbose
```

### With Custom Parameters

```bash
python -m gridsteer.step1.optimize_phi /path/to/data \
    --imgs_to_proc 50 \
    --verbose \
    --output-dir ./phi_analysis
```

### Specify Frame Range

```bash
python -m gridsteer.step1.optimize_phi /path/to/data \
    --imgs_to_proc 100
```
