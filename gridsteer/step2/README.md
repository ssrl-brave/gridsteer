# GridSteer Step 2: Well Tracking and Motor Calibration System

## Overview

This system processes image sequences to detect, track, and identify wells arranged in a staggered two-row configuration (9 wells in Row 1, 10 wells in Row 2). The system establishes well positions, tracks them across frames as the motor moves, and calibrates the relationship between pixel movements and motor movements to predict motor positions needed to center specific wells.

The system makes the following assumptions about the tray layout:
- Row 1 is on top with 9 wells (IDs 1–9)
- Row 2 is on the bottom with 10 wells (IDs 10–19)
- The two rows are staggered: each Row 1 well sits between two adjacent Row 2 wells
- Frames start on the right side of the tray, so wells are encountered right-to-left as the stage scans

## System Architecture

The codebase is organized into several modules:

- **`main.py`**: Entry point and orchestration
- **`well_detection.py`**: Circle and line detection using Hough transforms
- **`well_tracking.py`**: Well identification, tracking, and management across frames
- **`motor_prediction.py`**: Training for motor calibration and position prediction
- **`visualization.py`**: Plotting and visualization of each frame
- **`read.py`**: Utility to read motor positions from JSON output
- **`test_predictions.py`**: Validation tool to test predicted motor positions

---

## How It Works: Frame-by-Frame Processing

### Phase 1: Initial Detection

#### Step 1: Load Frame Data
For each frame, the system loads:

- **Image data**: The camera frame showing the wells
- **Motor position**: X, Y, Z coordinates and phi (rotation angle)

File: `well_tracking.py:520-536` (`load_frame_data`)

#### Step 2: Circle Detection via Hough Transform
The system detects circular wells in each frame:

1. **Edge Detection**: Apply Canny edge detection to the image
   - Uses configurable sigma, low/high thresholds
   - File: `well_detection.py:40-60` (`CircleDetector.detect_circles`)

2. **Hough Circle Transform**: Detect circles in the edge image
   - Searches for circles in a radius range (default: target_radius ± 20 pixels)
   - Returns top N peaks (default: 19, matching expected well count)
   - Each detection has: center (x, y), radius, and confidence score
   - File: `well_detection.py:50-60`

**Result**: List of detected circles with positions and radii

#### Step 3: Optional Edge Detection (Default: Disabled)

This feature was originally designed to detect the corner edge of the tray, signaling when the system should begin labeling wells. The idea: by finding a prominent non-horizontal edge line near a detected circle, the system could determine it had reached the tray boundary and could start assigning well IDs from that reference point.

In practice, this turned out to be unnecessary. Frames typically start at the edge of the tray already, and during testing the tracking logic reliably assigns the correct labels without needing an explicit edge signal. Enabling edge detection adds computational overhead without improving accuracy — so it is disabled by default. When enabled, it runs only until the corner is found and is then automatically skipped for all remaining frames.

If edge detection is enabled, the system performs the following:

1. **Background Removal** (optional, if configured):
   - Uses rembg library to remove background noise
   - File: `well_detection.py:269-317` (`ImageProcessor.remove_background`)

2. **Contour Extraction**:
   - Finds contours in edge image
   - Computes convex hull around all contours
   - Filters border points
   - File: `well_detection.py:167-243` (`ContourProcessor.extract_contour_coordinates`)

3. **Line Detection**:
   - Detects lines using Hough Line Transform
   - Filters out horizontal lines
   - Looks for non-horizontal edge lines (corresponding to the tray boundary for labeling)
   - File: `well_detection.py:69-158` (`LineDetector.detect_lines`)

4. **Edge Condition Check**:
   - Checks if any circle is near a non-horizontal line
   - This indicates the camera has reached the edge of the well plate
   - Once satisfied, edge detection is disabled for remaining frames
   - File: `well_tracking.py:883-919` (`_check_edge_condition`)

---

### Phase 2: Row Detection and Clustering

Once circles are detected, the system needs to determine which row each circle belongs to.

#### Initial Learning Phase (K-Means Clustering)

K-Means (K=2) is applied to determine whether one or two rows are currently visible in the frame. The algorithm always tries a two-cluster split, then validates whether the resulting clusters are actually far enough apart to represent distinct rows.

1. **Cluster by Y-Coordinate**:
   - Uses K-Means clustering with K=2 to separate circles into two rows
   - File: `well_detection.py:410-462` (`GeometryUtils.cluster_points_by_y`)

2. **Validate Separation**:
   - Checks if the two clusters are separated by at least `row_separation_min` pixels
   - If yes: Two distinct rows detected
   - If no: Treated as single row (insufficient separation)

3. **Assign Row IDs**:
   - Row 1 (top): Smaller Y coordinates
   - Row 2 (bottom): Larger Y coordinates

**Result**: Circles grouped into Row 1 and Row 2

File: `well_tracking.py:972-1009` (`_detect_rows`)

#### Established Phase (Closest-Row Assignment)

Once the system has established line parameters for both rows:

1. **Use Line Equations**:
   - Each row has a fitted line: `y = slope * x + intercept`
   - File: `well_tracking.py:848-849` (stored in `self.row_params`)

2. **Assign to Closest Row**:
   - For each detected circle, calculate distance to each row's line
   - Assign to the closest row (if within tolerance)
   - File: `well_tracking.py:921-970` (`_assign_to_established_rows`)

**Result**: Faster, more robust row assignment than K-Means clustering

---

### Phase 3: Well Identification (Assigning Well IDs)

Once circles are grouped by row, the system assigns specific well IDs (1-9 for Row 1, 10-19 for Row 2).

```
Row 1:   o   o   o   o   o   o   o   o   o      (Wells 1-9)
Row 2:  o   o   o   o   o   o   o   o   o   o   (Wells 10-19)
```

- Row 1 well is positioned between two Row 2 wells
- Row 2 edges extend beyond Row 1 edges

#### Identification Methods (in priority order)

The algorithm prioritizes historical data when assigning labels to wells, as consistency across frames over time is more reliable than re-deriving labels from scratch each frame. Methods 1–2 draw on information from previous frames. When those fail, the system falls back to information from the current frame alone (Methods 3–5) to fill in any remaining unlabeled detections.

**Method 1: Temporal Matching**
- Match to the well ID from the previous frame based on closest distance
- Validates against stagger constraints
- File: `well_tracking.py:1085-1133` (`_find_best_temporal_match`)

**Method 2: Spacing-Based Matching**
- Uses established well spacing (median distance between adjacent wells)
- Estimates position based on known wells from previous frames (`last_successful_row_wells`, `reference_frame_wells`, etc.)
- File: `well_tracking.py:419-466` (`determine_well_id_from_spacing`)

**Method 3: Stagger Relationship**
- Uses already-identified wells from the current frame in the other row to infer IDs via the stagger pattern
- For Row 1 well at column C: Should be between Row 2 wells at columns C and C+1
- For Row 2 well at column C: Should be offset by half-spacing from Row 1
- File: `well_tracking.py:270-340` (`WellIdentifier.identify_well_using_stagger`)

**Method 4: Spatial Consistency**
- Uses already-identified wells in current frame to estimate ID
- File: `well_tracking.py:1258-1298` (`_determine_id_from_spatial_layout`)

**Method 5: Sequential Fallback**
- Assigns next available ID in the row (used only for initial frames)
- File: `well_tracking.py:1431-1443` (`_assign_initial_id`)

#### Stagger Validation

After assignment, the system validates that well positions are consistent with the stagger pattern:
- File: `well_tracking.py:1011-1046` (`_validate_stagger_consistency`)
- Checks that Row 1 wells are correctly positioned between Row 2 wells

---

### Phase 4: Line Fitting and Tracking

Once wells are identified in each row, the system fits a line to each row. Together with the well spacing calculated in Phase 5, these line equations allow the system to predict the positions of wells that are not currently visible in the frame — for example, wells that have scrolled off-screen or are temporarily obscured.

#### Fitting Lines to Rows

For each row with >= 2 detected wells:

1. **RANSAC Line Fitting**:
   - Fits line: `y = slope * x + intercept`
   - File: `well_detection.py:351-380` (`GeometryUtils.fit_line_ransac`)

2. **Validate Horizontality**:
   - Checks that line is approximately horizontal (within `max_line_angle_degrees`)
   - Rejects steep lines (not valid for horizontal well rows)
   - File: `well_detection.py:403-407` (`is_line_horizontal`)

3. **Update Row Parameters**:
   - Stores (slope, intercept) for each row
   - File: `well_tracking.py:1459-1501` (in `_process_two_rows`)

#### Line Updates

**Per-Frame Updates**:
- Lines are re-fitted every frame based on current detections
- This allows the system to track row positions as the camera/stage moves

**Handling Single Detection**:
- If only 1 well detected in a row: Update intercept, keep slope
- File: `well_tracking.py:1486-1497`

**Handling Missing Rows**:
- If a row has no detections: Keep previous line parameters
- File: `well_tracking.py:1499-1501`

#### Inter-Row Spacing and Extrapolation

**Measuring Inter-Row Distance**:
- When both rows are visible, calculates vertical distance between row lines
- Uses exponential moving average to smooth measurements
- File: `well_tracking.py:1509-1541`

**Row Extrapolation**:
- If Row 1 visible but Row 2 missing: Extrapolate Row 2 position using inter-row spacing
- If Row 2 visible but Row 1 missing: Extrapolate Row 1 position
- Enables tracking even when one row moves out of view
- File: `well_tracking.py:1544-1577`

---

### Phase 5: Spacing Calculation

The system calculates the horizontal spacing between adjacent wells in each row.

**Calculation Method**:
1. For each row, sort wells by x-coordinate
2. Calculate distances between consecutive wells
3. Uses median spacing so it's more robust to outliers
4. File: `well_tracking.py:1478-1482`

**Established Spacing**:
- Once reliable spacing is determined, it becomes the "established spacing"
- Used for predicting positions of undetected wells
- File: `well_tracking.py:1061-1073` (in `_update_successful_frame_tracking`)

---

### Phase 6: Prediction of Missing Wells

If some wells are not detected but spacing and row lines are established:

**Prediction Algorithm**:
1. Determine which wells are missing (1-19 minus detected wells)
2. For each missing well:
   - Find closest detected well in same row
   - Calculate column offset from that well
   - Estimate x-position: `x = anchor_x - offset * spacing`
   - Estimate y-position: `y = slope * x + intercept`
3. Store predicted positions with lower confidence

File: `well_tracking.py:1591-1646` (`_generate_predictions_for_missing_wells`)

---

### Phase 7: Motor Calibration

The system learns the relationship between pixel movements and motor movements.

#### Data Collection

**Observation Generation**:
- For each frame with detected wells, store:
  - Motor position (X, Y, Z, phi)
  - Pixel positions of each detected well
  - Frame number
- File: `motor_prediction.py:213-259` (`add_observation`)

#### Training Pair Generation

**Averaged Movement (Default)**:
- For each pair of frames:
  1. Find wells visible in both frames (common wells)
  2. Calculate pixel shift for each common well
  3. Filter outliers using IQR method
  4. Average pixel shifts across all common wells
  5. Calculate motor shift (difference in motor positions)
  6. Add pair for training: (average_pixel_shift -> motor_shift)
- File: `motor_prediction.py:261-315` (`_generate_training_pairs_averaged`)

#### Outlier Detection

**IQR Method**:
- Uses: Q1 - 1.5 x IQR to Q3 + 1.5 x IQR
- Filters out abnormal pixel shifts before averaging (in case a well gets mis-labeled)
- File: `motor_prediction.py:93-138` (`_filter_outliers`)

**Score Drop Detection**:
- Monitors calibration model performance (R² score)
- Rejects observations that significantly degrade model quality
- File: `motor_prediction.py:140-181` (`_is_score_drop_significant`)

#### Model Training

Once sufficient training samples collected (default: >= 10 pairs):

**Model Options**:

I tested 3 different models for learning the mapping, Linear Ridge Regression worked the best during my testing. The other models are disabled, but the logic is still in the code. 

1. **Linear Ridge Regression** (default):
   - Maps (Δpixel_x, Δpixel_y) -> (Δmotor_x, Δmotor_y, Δmotor_z)
   - Three separate models for X, Y, Z
   - Phi remains constant

2. **Polynomial Ridge Regression**:
   - Adds quadratic terms: x², xy, y²

3. **Spline Interpolation (RBF)**:
   - Uses thin-plate spline kernel

File: `motor_prediction.py:379-437` (`_train_models`)

#### Model Persistence

Once calibration is complete, the trained model is saved as a pickle file (`calibration_model.pkl`) in the output directory:

```
<output_dir>/calibration_model.pkl
```

This allows the calibration to be reloaded and used downstream — for example, to fine-tune the centering without re-running the full tracking pipeline.

File: `well_tracking.py:772-777`, `motor_prediction.py:579-584` (`save_model`)

---

### Phase 8: Well Centering Predictions - Creating Output JSON File

Predict motor positions needed to center each well in the frame.

#### Reference Frame Selection

**When Established**:
- The first frame where both rows have fitted line parameters (requiring at least `min_circles_per_row` detections per row, default: 2) is designated the reference frame
- This frame's motor position is recorded as the origin for all subsequent motor predictions
- File: `well_tracking.py:88-111` (`WellCenterTracker.update`)

> **Important**: If no frame with at least 2 detected circles in each row is encountered, the reference frame is never established and the system cannot generate motor centering predictions.

**Pixel Offset Calculation**:
For each well detected in reference frame:
- `offset_x = frame_center_x - well_x`
- `offset_y = frame_center_y - well_y`
- Store these offsets for each well ID

#### Missing Well Offset Estimation

For wells never detected in reference frame:
- Use row line equation and established spacing
- Estimate well position based on detected neighbors
- Calculate offset to center
- File: `well_tracking.py:113-153` (`_estimate_missing_well_offsets`)

#### Motor Position Prediction

For each well (1-19):
1. Retrieve pixel offset to center
2. Use calibrated model to predict motor shift:
   - `motor_shift = model.predict(pixel_offset)`
3. Add shift to reference motor position:
   - `target_motor = reference_motor + motor_shift`
4. Store predicted motor position in JSON file

File: `well_tracking.py:155-259` (`WellCenterTracker.save_to_json`)

---

## Configuration Parameters

### Video Output

If `save_video=True`:
- Creates annotated video showing:
  - Detected circles
  - Well IDs and row assignments
  - Fitted row lines
  - Predicted well positions
  - Motor position and calibration status

### Frame Images

The `save_individual_frames` variable in `main.py` controls debug frame output. If set to `True`:
- Saves each annotated frame as a PNG image
- Visualizations include detected circles, assigned well IDs, fitted row lines, predicted well positions, and motor/calibration status
- Useful for understanding what the system sees at each step and diagnosing tracking issues

### Circle Detection
- `target_radius`: Expected well radius in pixels (default: 85)
- `radius_range`: Search range around target (default: 20)
- `min_x_distance`, `min_y_distance`: Minimum spacing between detected circles
- `hough_num_peaks`: Maximum circles to detect (default: 19)
- `hough_threshold`: Detection confidence threshold

### Row Detection
- `row_separation_min`: Minimum vertical spacing to consider 2 rows (default: 150)
- `row_y_tolerance`: Maximum deviation from row line (default: 40)
- `max_line_angle_degrees`: Maximum slope for horizontal lines (default: 10°)

### Well Tracking
- `association_distance_threshold`: Maximum distance for temporal matching (default: 50px)
- `min_circles_per_row`: Minimum circles to fit row line (default: 2)

### Motor Calibration
- `calibration_method`: "linear", "polynomial", or "spline" (default: linear)
- `calibration_min_samples`: Minimum training pairs needed (default: 10)
- `calibration_max_samples`: Maximum pairs to keep (default: 100)
- `calibration_use_average_movement`: Use averaged vs. individual well movements
- `calibration_outlier_detection`: "iqr" or "none" (default: iqr)

### Edge Detection
- `enable_edge_detection`: Enable edge-based row detection (default: False)
- `edge_distance_multiplier`: Circle-to-line distance threshold (default: 2.0)

---

## Usage Examples

### Basic Usage

```bash
python -m gridsteer.step2.main <path_to_data>
```

### With Custom Parameters

```bash
python -m gridsteer.step1.optimize_phi <path_to_data> \
    --imgs_to_proc <num_images> \
    --verbose \
    --output-dir <output_dir>
```

### Specify Frame Range and Radius

```bash
python -m gridsteer.step1.optimize_phi <path_to_data> \
    --imgs_to_proc <num_images> \
    --target_radius <radius>
```
