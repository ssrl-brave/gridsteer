"""Tolerance-based comparison of a GridSteer run against its reference.

Every comparison returns a list of :class:`Check` objects rather than
raising, so one pipeline run can be scored on several independent
criteria and every check shows up in the test output.

Tolerances live in ``DEFAULT_TOLERANCES`` and can be overridden per dataset
from the manifest (``tests/ground_truth/datasets.json``). See
``tests/README.md`` for the rationale behind each number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------

DEFAULT_TOLERANCES = {
    # ---- Step 1 --------------------------------------------------------
    # Max |phi - phi_ref| in degrees (mod 360). Scans step phi by ~1 deg
    # and the width minimum is shallow, so allow a two-frame shift.
    "phi_deg": 2.5,

    # ---- Step 2 --------------------------------------------------------
    # Registration residual (px) <= max(abs, rel * reference).
    "residual_px_abs": 2.0,
    "residual_px_rel": 2.0,
    # Extra bridged/rejected frame pairs allowed, as a fraction of n_frames.
    "interventions_frac": 0.1,
    # Matched layout scale, relative to the reference.
    "scale_rel": 0.05,
    # Well position (reference-frame px): within pos_r * well radius.
    # Predicted (lattice-placed) wells also get pos_lever_frac of their
    # distance from the frame centre. A one-column slide is ~2.5 r.
    "pos_r": 0.2,
    "pos_lever_frac": 0.01,
    # Reference-observed wells lost / new wells gained, each allowed up to
    # max(2, frac * n_ref_observed).
    "observed_loss": 0.2,
    "observed_gain": 0.2,
    # Motor positions reuse the pixel tolerance divided by the reference's
    # stage gain (px/mm); there is no separate knob.
}


@dataclass
class Check:
    name: str
    passed: bool
    value: float | None = None
    tolerance: float | None = None
    detail: str = ""

    def as_row(self) -> str:
        v = "-" if self.value is None else f"{self.value:.4g}"
        t = "-" if self.tolerance is None else f"{self.tolerance:.4g}"
        mark = "PASS" if self.passed else "FAIL"
        return f"{mark:4s} {self.name:<26s} value={v:<10s} tol={t:<10s} {self.detail}"


def merged_tolerances(overrides: dict | None) -> dict:
    tol = dict(DEFAULT_TOLERANCES)
    if overrides:
        unknown = set(overrides) - set(DEFAULT_TOLERANCES)
        if unknown:
            raise KeyError(f"unknown tolerance override(s): {sorted(unknown)}")
        tol.update(overrides)
    return tol


def failures(checks: list[Check]) -> list[Check]:
    return [c for c in checks if not c.passed]


def format_report(checks: list[Check]) -> str:
    return "\n".join(c.as_row() for c in checks)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(float(d))


# --------------------------------------------------------------------------
# Step 1
# --------------------------------------------------------------------------

def compare_step1(run: dict, ref: dict, tol: dict | None = None) -> list[Check]:
    """Score a Step 1 run (see runners.run_step1) against its reference.

    Two checks: the CLI finished and printed a finite phi, and that phi
    is within ``phi_deg`` of the reference (angles compared modulo 360).
    """
    tol = merged_tolerances(tol)
    checks: list[Check] = []

    ok = run["returncode"] == 0 and run["phi"] is not None
    checks.append(Check("cli_exit_and_phi", ok,
                        detail=f"returncode={run['returncode']}, phi={run['phi']}"
                        + ("" if ok else f"; stderr: {run['stderr'][-400:]}")))
    if not ok:
        return checks

    d = angle_diff_deg(run["phi"], ref["phi"])
    checks.append(Check("phi_deg", d <= tol["phi_deg"], d, tol["phi_deg"],
                        f"phi={run['phi']:.3f} ref={ref['phi']:.3f}"))
    return checks


# --------------------------------------------------------------------------
# Step 2
# --------------------------------------------------------------------------

def _pos_tolerance_px(ref_well: dict, radius: float, frame_center, tol: dict) -> float:
    lim = tol["pos_r"] * radius
    if not ref_well["observed"] and ref_well.get("ref_px") is not None \
            and frame_center is not None:
        lim += tol["pos_lever_frac"] * float(np.hypot(
            ref_well["ref_px"][0] - frame_center[0],
            ref_well["ref_px"][1] - frame_center[1]))
    return lim


def _allowed(frac: float, n: int) -> int:
    """max(2, frac * n), rounded down (with a guard against 0.2 * 10 = 1.999...)."""
    return max(2, int(np.floor(frac * n + 1e-9)))


def compare_step2(run: dict, ref: dict, tol: dict | None = None) -> list[Check]:
    """Score a Step 2 run (see runners.run_step2) against its reference."""
    tol = merged_tolerances(tol)
    checks: list[Check] = []

    ref_ok = ref.get("returncode", 0) == 0
    if not ref_ok:
        # The reference itself records a failure (e.g. "No Wells
        # Detected"). Any success is an improvement; a failure must at
        # least be the same kind.
        same = run["returncode"] != 0
        checks.append(Check("expected_failure", True,
                            detail=f"reference fails (rc={ref['returncode']}); run "
                                   f"rc={run['returncode']} "
                                   + ("(still failing)" if same else "(now succeeds!)")))
        return checks

    ok = run["returncode"] == 0 and run["wells"] is not None
    checks.append(Check("cli_exit", ok,
                        detail=f"returncode={run['returncode']}"
                        + ("" if ok else f"; tail: {(run['stderr'] or run['stdout'])[-600:]}")))
    if not ok:
        return checks

    # ---- completeness --------------------------------------------------
    missing = sorted(set(ref["wells"]) - set(run["wells"]))
    checks.append(Check("all_labels_present", not missing, len(missing), 0,
                        f"missing labels: {missing}" if missing else
                        f"{len(run['wells'])} wells"))

    # ---- registration ---------------------------------------------------
    rr, rf = run["registration"], ref["registration"]
    if rf["calibrated"]:
        checks.append(Check("stage_calibrated", bool(rr["calibrated"]),
                            detail="stage-to-pixel calibration must not be lost"))
        if rr["residual_px"] is not None and rf["residual_px"] is not None:
            lim = max(tol["residual_px_abs"], tol["residual_px_rel"] * rf["residual_px"])
            checks.append(Check("residual_px", rr["residual_px"] <= lim,
                                rr["residual_px"], lim, f"ref={rf['residual_px']}"))
    n_int = rr["n_bridged"] + rr["n_rejected"]
    n_ref = rf["n_bridged"] + rf["n_rejected"]
    lim = n_ref + tol["interventions_frac"] * run["n_frames"]
    checks.append(Check("trajectory_interventions", n_int <= lim, n_int, lim,
                        f"bridged+rejected pairs; ref={n_ref}"))

    # ---- layout ---------------------------------------------------------
    lr, lf = run["layout"], ref["layout"]
    if lr and lf and lf.get("scale") is not None and lr.get("scale") is not None:
        rel = abs(lr["scale"] / lf["scale"] - 1)
        checks.append(Check("layout_scale_rel", rel <= tol["scale_rel"], rel,
                            tol["scale_rel"], f"scale={lr['scale']} ref={lf['scale']}"))
    radius = float(np.median([w["r"] for w in ref["wells"].values()]))

    # ---- per-well positions (reference-frame pixels) --------------------
    fc = None
    if ref.get("frame_shape"):
        fc = (ref["frame_shape"][1] / 2.0, ref["frame_shape"][0] / 2.0)
    worst = {True: (0.0, None, 0.0, 0.0), False: (0.0, None, 0.0, 0.0)}
    pos_fail = {True: [], False: []}
    for lab, wf in ref["wells"].items():
        wr = run["wells"].get(lab)
        if wr is None or wf.get("ref_px") is None or wr.get("ref_px") is None:
            continue
        d = float(np.hypot(wr["ref_px"][0] - wf["ref_px"][0],
                           wr["ref_px"][1] - wf["ref_px"][1]))
        lim = _pos_tolerance_px(wf, radius, fc, tol)
        obs = bool(wf["observed"])
        if d > lim:
            pos_fail[obs].append(f"{lab}:{d:.0f}px>{lim:.0f}")
        if worst[obs][1] is None or d / lim > worst[obs][0]:
            worst[obs] = (d / lim, lab, d, lim)
    for obs, name in ((True, "position_observed_px"), (False, "position_predicted_px")):
        w = worst[obs]
        checks.append(Check(name, not pos_fail[obs], w[2], w[3],
                            f"worst {w[1]} ({w[0]:.2f}x tol); fails: {pos_fail[obs]}"))

    # ---- detection set: lost and gained wells ---------------------------
    ref_obs = {l for l, w in ref["wells"].items() if w["observed"]}
    run_obs = {l for l, w in run["wells"].items() if w["observed"]}
    lost, gained = sorted(ref_obs - run_obs), sorted(run_obs - ref_obs)
    for name, labels in (("observed_loss", lost), ("observed_gain", gained)):
        allowed = _allowed(tol[name], len(ref_obs))
        checks.append(Check(name, len(labels) <= allowed, len(labels), allowed,
                            f"{len(ref_obs)} observed in the reference; "
                            f"{'lost' if name == 'observed_loss' else 'gained'}={labels}"))

    # ---- motor positions ------------------------------------------------
    gain = (ref.get("registration") or {}).get("stage_gain")
    have_motor = [l for l, w in ref["wells"].items()
                  if w.get("motor") is not None and run["wells"].get(l, {}).get("motor")]
    if ref.get("motor_map") is not None:
        checks.append(Check("motor_map_fitted", bool(have_motor),
                            detail="well_centering_positions.json must still be produced"))
    if have_motor and not gain:
        checks.append(Check("motor_position_mm", False,
                            detail="reference has no stage_gain; regenerate it"))
    elif have_motor:
        worst = (0.0, None, 0.0, 0.0)
        mfail = []
        for lab in have_motor:
            wf, wr = ref["wells"][lab], run["wells"][lab]
            d = float(np.linalg.norm(np.array(wr["motor"]) - np.array(wf["motor"])))
            lim = _pos_tolerance_px(wf, radius, fc, tol) / gain
            if d > lim:
                mfail.append(f"{lab}:{d:.3f}mm>{lim:.3f}")
            if worst[1] is None or d / lim > worst[0]:
                worst = (d / lim, lab, d, lim)
        checks.append(Check("motor_position_mm", not mfail, worst[2], worst[3],
                            f"worst {worst[1]} ({worst[0]:.2f}x tol); fails: {mfail}"))

    return checks
