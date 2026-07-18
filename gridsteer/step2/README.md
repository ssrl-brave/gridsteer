# GridSteer Step 2: Tray Well Identification and Motor Centering

## Overview

These scripts process a directory of `.npz` tray-scan frames to register,
detect, and label the wells of a tray, then compute the absolute motor
coordinates that center each well in the camera frame. The pipeline
requires no manual tray geometry beyond the expected well count per row —
well radius, spacing, and rim width are all measured from the data.

The system makes the following assumptions about the tray layout:
- The tray has two rows of wells, with row 1 holding 9 wells and row 2
  holding 10 wells (configurable via `EXPECTED_WELLS_PER_ROW`)
- Column 1 (`C1`) is the rightmost well in each row
- The scan starts at the `C1` end and travels left, so a truncated scan
  is always missing wells from the left end of a row

## System Architecture

The codebase is organized into three scripts:

- **`map_wells.py`**: Main entry point — registers frames, fuses
  them into a mosaic, detects wells, and orchestrates tracking and
  motor calibration
- **`predict_wells.py`**: Geometry helper imported by `map_wells.py`
  — extrapolates undetected wells and fits the motor-to-pixel mapping
- **`check_predictions.py`**: Standalone validation tool that sanity-checks
  `map_wells.py`'s output against the recorded frames

---

## How It Works: `map_wells.py`

### Phase 1: Registration

Every frame is registered into a common tray coordinate system (frame 0's).

1. **Pairwise Matching**: SIFT features + RANSAC homography between each
   consecutive pair of frames
   - File: `map_wells.py:87-115` (`match_pairs`)

2. **Trajectory Validation**: The recorded stage trajectory (x, y, z) is
   fit to pixel motion via RANSAC and used to check each image-derived
   homography
   - Agreeing pairs pass through unchanged
   - Weak pairs (defocus, too few inliers) are bridged with the
     stage-motion prediction, refined with ECC
   - Confident-but-contradicting pairs (aliasing on the repeating well
     lattice) are rejected and rebuilt the same way
   - File: `map_wells.py:118-216` (`fit_stage_map`, `ecc_refine`,
     `apply_trajectory`)

3. **Chaining**: Pairwise transforms are composed into per-frame
   transforms mapping frame *t* to frame 0
   - File: `map_wells.py:219-235` (`register_frames`)

**Result**: `Hs`, one homography per frame, plus a report of any
bridged/rejected pairs

---

### Phase 2: Mosaic Fusion

1. **Canvas Sizing**: Computes a global transform per frame so every
   warped frame lands on a common, non-negative canvas; guards against a
   runaway canvas size from a bad homography by checking available memory
   - File: `map_wells.py:246-268` (`mosaic_transforms`)

2. **Temporal Median**: Warps every frame onto the canvas and takes the
   per-pixel median across frames (NaN-aware, so unobserved canvas
   regions do not pollute the result)
   - File: `map_wells.py:271-287` (`build_mosaic`)

**Result**: A single fused mosaic image plus a per-pixel observation count

---

### Phase 3: Well Detection

Wells are detected once on the fused mosaic, rather than by per-frame
circle detection. The primary detector is a ring matched filter; trays
whose wells image as filled dark domes instead of rimmed circles fall
back to a disk matched filter on the local smoothness map (see below).

1. **Radius Measurement**: A 1-D sweep over candidate radii finds the one
   with the strongest ring-template NCC response
   - File: `map_wells.py:349-367` (`measure_radius`)

2. **Rim Width Measurement**: The FWHM of the dark rim trough in the
   radial intensity profile around the best ring response
   - File: `map_wells.py:369-385` (`measure_rim_width`)

3. **Matched Filtering**: A zero-mean annulus template at the measured
   radius/rim-width is correlated (NCC) against the mosaic; degenerate
   regions (flat/unobserved) are zeroed out
   - File: `map_wells.py:308-347` (`ring_template`, `_ncc`)

4. **Peak Selection**: Local maxima above a fraction of the strongest peak
   are accepted outright; weaker peaks require a nearly complete, high-contrast
   dark rim and must clear a null-hypothesis noise ceiling
   - File: `map_wells.py:421-500` (`detect_wells`)

**Result**: A list of detected wells with mosaic-coordinate centers, radii,
and confidence scores

#### Fallback: Smooth-Dark-Disk Matched Filter

Some trays image with each well filled by a dark dome (a sample) rather
than a bright interior ringed by a dark rim. The ring filter then has no
annulus to match, and because the dome's shading gradient overlaps the
substrate's intensity range, no global threshold separates wells from
background either. What does separate them is **texture**: the domes are
locally smooth, the substrate is heavily speckled, and the smooth
off-tray background is bright where wells are dark.

When the ring detector returns nothing (or with `--detector disk`), the
fallback:

1. Computes a local-standard-deviation texture map over a small window
   (the speckle grain scale, independent of well size)
   - File: `map_wells.py:515-521` (`texture_map`)

2. Runs the same 1-D radius sweep, but with a zero-mean filled-disk
   template correlated against the inverted (smoothness) texture map;
   all wells share the single measured radius
   - File: `map_wells.py:503-512` (`disk_template`)

3. Accepts response peaks with a rank test against measured background
   rather than fixed thresholds. Three statistics describe a dark dome
   well — a smooth interior of the measured radius, a rough (speckled)
   exterior ring, and an interior darker than its surround — and the
   same statistics are measured at ~500 random control locations away
   from the candidates. No single axis separates wells from background
   (off-tray regions are smooth inside, speckle is rough outside, and
   shadow bands out-contrast a well), but no background location matches
   a well on *all* axes at once. So each location is scored by its worst
   per-axis exceedance over the controls, that score is calibrated
   against the controls' own score distribution (Westfall–Young-style
   min-p), and a candidate is kept when at most a fraction `alpha`
   (default 0.01) of background looks as jointly well-like as it does.
   The significance level is the only acceptance knob. This replaces
   both fixed statistic thresholds and the ring path's MAD-based null
   gate, which is unusable here because the smoothness map's
   large-scale structure inflates the response MAD
   - File: `map_wells.py:524-634` (`control_locations`,
     `detect_wells_disk`)

The `--detector` CLI flag selects `ring`, `disk`, or `auto` (ring with
disk fallback, the default).

#### Labeling

- Wells are grouped into rows by y-coordinate, then labeled `R{row}C{col}`
  within each row, columns numbered right-to-left (`C1` is rightmost)
- File: `map_wells.py:637-653` (`label_wells`)

---

### Phase 4: Missing-Well Prediction

The scan often stops partway across the tray, so each row is completed by
extrapolating from the detections that were made.

1. **Per-Row Spacing**: Detected wells in a row are sorted by x-coordinate;
   the average spacing vector between the first and last detection is
   computed
   - File: `predict_wells.py:47-88` (`predict_missing_wells`)

2. **Leftward Extension**: Starting from the leftmost detection, positions
   are stepped further left by the spacing vector until the row reaches
   its expected well count (row 1: 9, row 2: 10)

**Result**: `PredictedWell` entries for positions the scan never reached,
marked distinct from real detections

---

### Phase 5: Tracking and Pose

1. **Per-Frame Projection**: Each well's fixed tray-coordinate position is
   projected into every frame through the inverse of that frame's
   homography, with visibility classified as full/partial/out based on
   whether the projected well fits inside the frame
   - File: `map_wells.py:660-687` (`local_scale`, `project_tracks`)

2. **Planar Pose**: Each frame's homography is decomposed into rotation,
   isotropic scale, translation, and a perspective magnitude used as a
   tilt indicator
   - File: `map_wells.py:690-710` (`planar_pose`)

---

### Phase 6: Motor Calibration and Centering

1. **Motor-to-Pixel Mapping**: A linear ridge regression is trained from
   pixel shift (dx, dy) to motor shift (dx, dy, dz), using every frame
   pair (not just consecutive ones) so that a constant per-axis step
   doesn't become indistinguishable from the intercept
   - File: `predict_wells.py:130-178` (`fit_motor_pixel_map`)

2. **Centering Positions**: For each well (detected and predicted), the
   pixel offset from its reference-frame position to the frame center is
   passed through the fitted mapping to get a motor shift, which is added
   to the reference frame's recorded motor position
   - File: `predict_wells.py:181-237` (`motor_centering_positions`)

**Result**: `well_centering_positions.json`, the absolute motor
coordinates that center each well in the frame

---

## How It Works: `check_predictions.py`

Sanity-checks `map_wells.py`'s output by comparing each well's
predicted motor position against the frames that were actually recorded.

1. **Closest-Frame Search**: For each well's predicted motor position,
   finds the recorded frame whose stage coordinates (x, y, z) are
   nearest by euclidean distance
   - File: `check_predictions.py:48-62` (`closest_frames`)

2. **Montage Output**: Renders one panel per well showing its closest
   frame with a center crosshair and the motor-space distance — a well
   predicted correctly should appear centered, assuming a frame was
   actually taken near that position
   - File: `check_predictions.py:65-95` (`save_montage`)

---

## Usage Examples

### Identify and Track Wells

```bash
python -m gridsteer.step2.map_wells data/mapper15 --out output_tracks
```

Outputs (in `<out>/<dataset-name>/`): `mosaic_labeled.png`,
`frames_overlay.png`, `tracks.csv`, `pose.csv`, `wells.json` (detected
and predicted wells, flagged by an `observed` field), and
`well_centering_positions.json`.

### Check Predictions Against Recorded Frames

```bash
python -m gridsteer.step2.check_predictions \
    output_tracks/mapper15/well_centering_positions.json data/mapper15
```

Outputs (alongside the input JSON by default): `closest_frames.png`.

---

## Dependencies

`numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `matplotlib`.
