# GridSteer Validation Suite

Automated checks that run on every push and pull request
(`.github/workflows/ci.yml`). They catch a change to Step 1 or Step 2 that
moves the result on a scan that used to work, while tolerating small
numeric drift.

```
tests/
├── conftest.py                    fixtures, --shard, --results-csv, --outputs-dir, --no-keep-outputs
├── test_metrics.py                unit tests of the comparison logic
├── test_results.py                unit tests of the results CSV
├── test_step1_validation.py       Step 1 vs ground_truth/step1 (slow)
├── test_step2_validation.py       Step 2 vs ground_truth/step2 (slow)
├── validation/
│   ├── runners.py                 runs the two CLIs, parses their outputs
│   ├── metrics.py                 tolerances, compare_step1 / compare_step2
│   ├── results.py                 results CSV, one row per dataset and step
│   └── make_ground_truth.py       generates the reference files
└── ground_truth/
    ├── datasets.json              manifest: datasets, orientation, overrides
    ├── step1/<dataset>.json       one reference per Step 1 scan
    └── step2/<dataset>.json       one reference per Step 2 scan
```

## Running Locally

```bash
pip install -e ".[test]"

pytest -m "not validation"          # unit tests only
pytest -m validation -s             # every scan
pytest tests/test_step2_validation.py -s -k mapper15   # one scan
pytest -m validation --shard 2/4    # what CI shard 2 runs
pytest -m validation -s --results-csv validation-results.csv
```

`-s` prints every check for each dataset. `--results-csv PATH` writes the
results table below. Every run's working directory is kept (Step 1's
`output_images/` and `logs/`; Step 2's `out/` with `wells.json`,
`well_centering_positions.json`, `mosaic_labeled.png`, `frames_overlay.png`
and the other debug images). Its path is printed at the end of the session
and recorded in the CSV. `--outputs-dir PATH` chooses where the runs go;
`--no-keep-outputs` deletes them afterwards.

The validation tests run the real entry points as subprocesses
(`python -m gridsteer.step1.optimize_phi`, `python -m gridsteer.step2.map_wells`)
because those are what the beamline calls.

## Results CSV

One row per dataset and step, sorted by step then dataset:

| column | meaning |
|---|---|
| `step` | `Step 1` or `Step 2` |
| `dataset` | manifest name of the scan |
| `shard` | CI shard index; its outputs are in the `validation-outputs-shard-<shard>` artifact (empty locally) |
| `orientation` | the `--orientation` Step 2 ran with (empty for Step 1) |
| `n_frames` | frames in the scan |
| `result` | `PASS` or `FAIL` |
| `actual` | headline numbers from this run: Step 1 phi; Step 2 exit code, well counts, residual, bridged+rejected pairs, layout scale |
| `expected` | the same numbers from the reference |
| `failed_checks` | `name (value vs tolerance); ...`, empty on a pass |
| `reference_file` | the reference the run was compared with |
| `output_dir` | the run's working directory, empty with `--no-keep-outputs` |

In CI each shard writes its own `results.csv`; the `results` job merges
them into `validation-results.csv`, uploads it and shows the table on the
job summary.

## Reference Data

Each `ground_truth/step*/<dataset>.json` is the pipeline's own output on
that scan, reviewed and committed. It is a behavioural baseline, not
absolute truth.

Step 1 reference:

| field | meaning |
|---|---|
| `phi` | the face-on phi the CLI printed |

Step 2 reference:

| field | meaning |
|---|---|
| `orientation` | the `--orientation` the scan runs with, from the manifest |
| `returncode` | 0, or the CLI's exit code for a known failure |
| `registration` | `calibrated`, `residual_px`, `n_bridged`, `n_rejected`, `stage_gain` (px/mm) |
| `layout` | `scale`, `rotation_deg`, `well_radius_px`, `n_observed`, `spacing_rejected` (only `scale` is checked) |
| `wells{label}` | `observed`, `score`, `mosaic_px`, `ref_px` (frame-0 pixels), `motor` (x, y, z) |

Positions are compared in frame-0 pixel coordinates (`ref_px`), not mosaic
coordinates, because the mosaic origin moves whenever the registration
changes slightly.

## Metrics and Tolerances

Numbers live in `validation/metrics.py` (`DEFAULT_TOLERANCES`) and can be
overridden per dataset in the manifest. Every check is reported; the test
fails if any check fails.

### Step 1

| check | passes when | why |
|---|---|---|
| `cli_exit_and_phi` | exit code 0 and a finite phi printed | `NaN` means no frame yielded a width |
| `phi_deg` | \|phi − ref\| ≤ 2.5° (mod 360) | scans step phi by ~1° and the width minimum is shallow, so a benign change may move the argmin a frame or two |

Only the printed phi is pinned, because that is all the beamline consumes.

### Step 2

| check | passes when | why |
|---|---|---|
| `cli_exit`, `all_labels_present` | exit 0 and `wells.json` has every reference label | the full layout must always be produced |
| `stage_calibrated` | the stage-to-pixel fit still succeeds | without it the pipeline falls back to the image-only sweep |
| `residual_px` | ≤ max(2 px, 2 × reference) | calibration accuracy is 1–2 px; aliasing errors are hundreds |
| `trajectory_interventions` | bridged + rejected pairs ≤ reference + 10 % of frames | more bridging means weaker registration |
| `layout_scale_rel` | within 5 % of the reference scale | the stage prior fixes scale to <1 % |
| `position_observed_px` | every observed well within 0.2 r (~20 px) | snapped centres repeat to a few px; a one-column slide is ~2.5 r |
| `position_predicted_px` | every predicted well within 0.2 r + 1 % of its distance from the frame centre | lattice extrapolation error grows with the lever arm |
| `observed_loss` | ≤ max(2, 20 % of reference observed) wells no longer observed | a borderline rim flipping to lattice-placed is not a regression |
| `observed_gain` | ≤ max(2, 20 % of reference observed) wells newly observed | guards against false positives; a real improvement beyond that needs a regenerated reference |
| `motor_map_fitted` | `well_centering_positions.json` is still produced | the file the beamline reads |
| `motor_position_mm` | each well's (x, y, z) within the pixel tolerance divided by the stage gain | the same tolerance in motor units (~0.1 mm) |

## Regenerating or Adding References

```bash
python -m tests.validation.make_ground_truth                 # everything
python -m tests.validation.make_ground_truth --step 2 --only mapper15
```

Review `git diff tests/ground_truth` before committing: every changed
number is a behaviour change you are accepting.

To add a scan: copy its frames under `data/Step2TestData/<name>/` (or
`data/Step1TestData/`), add an entry to `datasets.json`, generate its
reference, check the mosaic, and commit both.

## CI Layout

* `unit`: the metric and results-table tests, no data needed.
* `validate`: four shards, each running a balanced subset of the scans
  (`--shard i/4`) and uploading its `results.csv` and every run's outputs
  (`validation-outputs-shard-<i>`). Add shards in `ci.yml` as the dataset
  list grows.
* `results`: merges the shard CSVs into `validation-results.csv` and is
  green only when the unit tests and every shard passed. Point branch
  protection at it.
