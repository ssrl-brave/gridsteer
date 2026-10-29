"""Fast unit tests for the validation metrics.

They prove that the tolerance checks fail when they should (a wrong
frame, a column slide, a lost calibration) and pass on benign noise, so
the validation suite's verdicts can be trusted. No datasets are needed.
"""

import copy

import pytest

from tests.validation import metrics
from tests.validation.metrics import Check, compare_step1, compare_step2


# --------------------------------------------------------------------------
# Fixtures: small synthetic reference runs
# --------------------------------------------------------------------------

def _step1_ref():
    return dict(phi=104.348, returncode=0, n_frames=11)


def _step1_run_from(ref):
    return dict(returncode=0, stdout="", stderr="", phi=ref["phi"],
                n_frames=ref["n_frames"])


LABELS = [f"R1C{c}" for c in range(1, 10)] + [f"R2C{c}" for c in range(1, 11)]
PITCH = 254.7       # px between columns in the template
RADIUS = 101.0
GAIN = 203.25       # px / mm


def _step2_ref(n_observed=4):
    wells = {}
    for lab in LABELS:
        row, col = int(lab[1]), int(lab.split("C")[1])
        x = 600.0 - (col - 1) * PITCH + (127.3 if row == 2 else 0.0)
        y = 420.0 if row == 1 else 170.0
        observed = lab in ["R1C1", "R1C2", "R2C2", "R2C3", "R1C3", "R2C4",
                           "R1C4", "R2C5", "R1C5", "R2C6"][:n_observed]
        wells[lab] = dict(observed=observed, score=80.0 if observed else None,
                          mosaic_px=[x + 50, y + 30], r=RADIUS,
                          ref_px=[x, y],
                          motor=[0.1, -0.2, 1.0 + (x - 360) / GAIN])
    return dict(returncode=0, n_frames=24, frame_shape=[480, 720],
                registration=dict(calibrated=True, residual_px=0.66, n_bridged=0,
                                  n_rejected=0, stage_gain=GAIN),
                layout=dict(scale=0.995, rotation_deg=1.5, well_radius_px=RADIUS,
                            orientation="prior", expected_scale=1.0,
                            spacing_rejected=[], n_observed=n_observed),
                motor_map=dict(n_pairs=276, r2={"x": 1.0, "y": 1.0, "z": 1.0}),
                wells=wells)


def _step2_run_from(ref):
    return dict(returncode=0, stdout="", stderr="", n_frames=ref["n_frames"],
                frame_shape=ref["frame_shape"],
                registration=copy.deepcopy(ref["registration"]),
                layout=copy.deepcopy(ref["layout"]),
                motor_map=copy.deepcopy(ref["motor_map"]),
                wells=copy.deepcopy(ref["wells"]), out_dir=None)


def _names(checks, passed):
    return sorted(c.name for c in checks if c.passed is passed)


# --------------------------------------------------------------------------
# Generic
# --------------------------------------------------------------------------

def test_merged_tolerances_rejects_unknown_key():
    with pytest.raises(KeyError):
        metrics.merged_tolerances({"not_a_tolerance": 1})


def test_angle_diff_wraps():
    assert metrics.angle_diff_deg(359.0, 1.0) == pytest.approx(2.0)
    assert metrics.angle_diff_deg(104.3, 104.3 + 360) == pytest.approx(0.0)


def test_format_report_marks_failures():
    text = metrics.format_report([Check("a", True, 1, 2), Check("b", False, 3, 2, "x")])
    assert "PASS a" in text and "FAIL b" in text


# --------------------------------------------------------------------------
# Step 1
# --------------------------------------------------------------------------

def test_step1_identical_run_passes():
    ref = _step1_ref()
    checks = compare_step1(_step1_run_from(ref), ref)
    assert not metrics.failures(checks), metrics.format_report(checks)


def test_step1_one_frame_shift_is_tolerated():
    ref = _step1_ref()
    run = _step1_run_from(ref)
    run["phi"] = ref["phi"] + 1.0      # argmin moved one frame over
    assert not metrics.failures(compare_step1(run, ref))


def test_step1_wrong_frame_fails_phi():
    ref = _step1_ref()
    run = _step1_run_from(ref)
    run["phi"] = ref["phi"] - 5.0
    assert "phi_deg" in _names(compare_step1(run, ref), False)


def test_step1_phi_wraparound_is_handled():
    ref = _step1_ref()
    ref["phi"] = 359.5
    run = _step1_run_from(ref)
    run["phi"] = 0.5
    assert "phi_deg" in _names(compare_step1(run, ref), True)


def test_step1_cli_failure_short_circuits():
    ref = _step1_ref()
    run = _step1_run_from(ref)
    run.update(returncode=1, phi=None)
    checks = compare_step1(run, ref)
    assert [c.name for c in checks] == ["cli_exit_and_phi"] and not checks[0].passed


def test_step1_nan_output_fails():
    ref = _step1_ref()
    run = _step1_run_from(ref)
    run["phi"] = None                  # the CLI printed NaN
    assert not compare_step1(run, ref)[0].passed


# --------------------------------------------------------------------------
# Step 2
# --------------------------------------------------------------------------

def test_step2_identical_run_passes():
    ref = _step2_ref()
    checks = compare_step2(_step2_run_from(ref), ref)
    assert not metrics.failures(checks), metrics.format_report(checks)


def test_step2_small_jitter_passes():
    ref = _step2_ref()
    run = _step2_run_from(ref)
    for w in run["wells"].values():
        w["ref_px"] = [w["ref_px"][0] + 6.0, w["ref_px"][1] - 4.0]
        w["motor"] = [w["motor"][0] + 0.01, w["motor"][1], w["motor"][2] - 0.02]
    run["registration"]["residual_px"] = 1.1
    run["layout"]["scale"] = 1.004
    run["layout"]["rotation_deg"] = 0.6
    assert not metrics.failures(compare_step2(run, ref)), \
        metrics.format_report(compare_step2(run, ref))


def test_step2_column_slide_fails_positions_and_motor():
    ref = _step2_ref()
    run = _step2_run_from(ref)
    for w in run["wells"].values():
        w["ref_px"][0] += PITCH             # every well relabelled one column over
        w["motor"][2] += PITCH / GAIN
    failed = _names(compare_step2(run, ref), False)
    assert {"position_observed_px", "position_predicted_px",
            "motor_position_mm"} <= set(failed)


def test_step2_extrapolated_wells_get_lever_arm_slack():
    ref = _step2_ref()
    run = _step2_run_from(ref)
    # 0.5% scale drift: far extrapolations move a lot, observed wells barely.
    for w in run["wells"].values():
        dx = w["ref_px"][0] - 360.0
        w["ref_px"][0] = 360.0 + dx * 1.005
    checks = compare_step2(run, ref)
    assert not metrics.failures(checks), metrics.format_report(checks)


def test_step2_two_lost_observed_wells_pass_three_fail():
    ref = _step2_ref(n_observed=4)
    run = _step2_run_from(ref)
    run["wells"]["R2C3"]["observed"] = False
    run["wells"]["R2C2"]["observed"] = False      # 2/4 kept: two misses are allowed
    assert "observed_loss" in _names(compare_step2(run, ref), True)
    run["wells"]["R1C2"]["observed"] = False      # 1/4 kept: a real loss
    assert "observed_loss" in _names(compare_step2(run, ref), False)


def test_step2_recall_scales_with_many_observed_wells():
    ref = _step2_ref(n_observed=10)
    run = _step2_run_from(ref)
    run["wells"]["R2C3"]["observed"] = False
    run["wells"]["R2C4"]["observed"] = False      # 8/10 = 0.8: still fine
    assert "observed_loss" in _names(compare_step2(run, ref), True)
    run["wells"]["R2C5"]["observed"] = False      # 7/10: fails
    assert "observed_loss" in _names(compare_step2(run, ref), False)


def test_step2_recall_tiny_set_loses_everything_and_fails():
    ref = _step2_ref(n_observed=3)
    run = _step2_run_from(ref)
    for lab in ("R1C1", "R1C2", "R2C2"):
        run["wells"][lab]["observed"] = False
    assert "observed_loss" in _names(compare_step2(run, ref), False)


def test_step2_a_few_newly_observed_wells_are_fine_many_fail():
    ref = _step2_ref(n_observed=4)
    run = _step2_run_from(ref)
    for lab in ("R1C3", "R2C4"):                 # +2: allowed
        run["wells"][lab]["observed"] = True
        run["wells"][lab]["score"] = 50.0
    assert not metrics.failures(compare_step2(run, ref))
    run["wells"]["R1C4"]["observed"] = True      # +3 on a 4-well reference: fails
    run["wells"]["R1C4"]["score"] = 50.0
    assert "observed_gain" in _names(compare_step2(run, ref), False)
    assert "observed_loss" in _names(compare_step2(run, ref), True)


def test_step2_gain_scales_with_many_observed_wells():
    ref = _step2_ref(n_observed=10)
    run = _step2_run_from(ref)
    for lab in ("R1C6", "R2C7"):                 # +2 of 10: fine
        run["wells"][lab]["observed"] = True
        run["wells"][lab]["score"] = 50.0
    assert "observed_gain" in _names(compare_step2(run, ref), True)
    run["wells"]["R1C7"]["observed"] = True      # +3 > 20% of 10: fails
    run["wells"]["R1C7"]["score"] = 50.0
    assert "observed_gain" in _names(compare_step2(run, ref), False)


def test_step2_lost_calibration_fails():
    ref = _step2_ref()
    run = _step2_run_from(ref)
    run["registration"]["calibrated"] = False
    run["registration"]["residual_px"] = None
    assert "stage_calibrated" in _names(compare_step2(run, ref), False)


def test_step2_registration_degradation_fails():
    ref = _step2_ref()
    run = _step2_run_from(ref)
    run["registration"]["residual_px"] = 3.5
    run["registration"]["n_bridged"] = 3          # > 10% of 24 frames
    failed = _names(compare_step2(run, ref), False)
    assert {"residual_px", "trajectory_interventions"} <= set(failed)
    run["registration"]["n_bridged"] = 2          # <= 2.4: fine
    assert "trajectory_interventions" in _names(compare_step2(run, ref), True)


def test_step2_layout_scale_drift_fails():
    ref = _step2_ref()
    run = _step2_run_from(ref)
    run["layout"]["scale"] = ref["layout"]["scale"] * 1.04
    assert "layout_scale_rel" in _names(compare_step2(run, ref), True)
    run["layout"]["scale"] = ref["layout"]["scale"] * 1.08
    assert "layout_scale_rel" in _names(compare_step2(run, ref), False)
    names = {c.name for c in compare_step2(run, ref)}
    assert not {"layout_rotation_deg", "well_radius_rel", "human_truth_mm"} & names


def test_step2_missing_motor_map_fails():
    ref = _step2_ref()
    run = _step2_run_from(ref)
    for w in run["wells"].values():
        w["motor"] = None
    assert "motor_map_fitted" in _names(compare_step2(run, ref), False)


def test_step2_reference_without_stage_gain_fails_motor_check():
    ref = _step2_ref()
    ref["registration"]["stage_gain"] = None
    assert "motor_position_mm" in _names(compare_step2(_step2_run_from(ref), ref), False)


def test_step2_cli_failure_short_circuits():
    ref = _step2_ref()
    run = _step2_run_from(ref)
    run.update(returncode=1, wells=None, stderr="No Wells Detected")
    checks = compare_step2(run, ref)
    assert [c.name for c in checks] == ["cli_exit"] and not checks[0].passed


def test_step2_reference_failure_never_blocks():
    ref = _step2_ref()
    ref["returncode"] = 1
    run = _step2_run_from(ref)
    assert not metrics.failures(compare_step2(run, ref))
    run.update(returncode=1, wells=None)
    assert not metrics.failures(compare_step2(run, ref))


def test_step2_per_dataset_override():
    ref = _step2_ref()
    run = _step2_run_from(ref)
    run["layout"]["scale"] = ref["layout"]["scale"] * 1.08
    assert "layout_scale_rel" in _names(compare_step2(run, ref), False)
    assert "layout_scale_rel" in _names(
        compare_step2(run, ref, tol={"scale_rel": 0.1}), True)
