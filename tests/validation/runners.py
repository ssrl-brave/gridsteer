"""Run the GridSteer Step 1 and Step 2 command-line entry points and
collect their results into plain dictionaries.

The validation suite deliberately exercises the *CLIs* (``python -m
gridsteer.step1.optimize_phi`` and ``python -m gridsteer.step2.map_wells``)
rather than importing internals, because those are the interfaces the
beamline Tcl operations call. Everything the tests compare is read back
from what the CLIs emit: Step 1's phi from stdout, Step 2's results from
the files it writes plus a few scalar diagnostics (registration residual,
layout scale/rotation) parsed from stdout.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _numeric_key(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def list_frames(data_dir: Path) -> list[Path]:
    return sorted(Path(data_dir).glob("*.npz"), key=_numeric_key)


def _run(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout)


# --------------------------------------------------------------------------
# Step 1
# --------------------------------------------------------------------------

def run_step1(data_dir: Path, work_dir: Path, timeout: float = 1800.0,
              python: str = sys.executable) -> dict:
    """Run ``optimize_phi`` end to end on one directory of frames.

    Returns a dict with:
      returncode, stdout, stderr      raw CLI outcome
      phi                             float parsed from the CLI's last stdout
                                      line, or None when it printed NaN/failed
      n_frames                        number of .npz frames in the directory
      work_dir                        where output_images/ and logs/ were written
    """
    data_dir = Path(data_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-m", "gridsteer.step1.optimize_phi", str(data_dir),
           "--output-dir", str(work_dir / "output_images"),
           "--log-dir", str(work_dir / "logs")]
    proc = _run(cmd, REPO_ROOT, timeout)

    phi = None
    if proc.returncode == 0:
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        if lines:
            try:
                val = float(lines[-1])
                phi = None if np.isnan(val) else val
            except ValueError:
                phi = None

    return dict(returncode=proc.returncode, stdout=proc.stdout,
                stderr=proc.stderr, phi=phi,
                n_frames=len(list_frames(data_dir)), work_dir=str(work_dir))


# --------------------------------------------------------------------------
# Step 2
# --------------------------------------------------------------------------

_RE_RESIDUAL = re.compile(r"Calibration Residual:\s*([-\d.]+)px")
_RE_LAYOUT = re.compile(r"Matched Layout: Scale=([-\d.]+), Rotation=([-+\d.]+) deg, "
                        r"Orientation=(\w+), Well Radius ~([-\d.]+)px")
_RE_BRIDGED = re.compile(r"Bridged Pairs \(Frame Indices\): \[([^\]]*)\]")
_RE_REJECTED = re.compile(r"Rejected Pairs \(Frame Indices\): \[([^\]]*)\]")
_RE_SPACING = re.compile(r"Spacing Check\): (.+)")
_RE_EXPECTED_SCALE = re.compile(r"Expected Scale ([-\d.]+)")


def _count_list(m) -> int:
    if not m:
        return 0
    body = m.group(1).strip()
    return len([x for x in body.split(",") if x.strip()]) if body else 0


def run_step2(data_dir: Path, work_dir: Path, orientation: str = "bottom-right",
              timeout: float = 3600.0, python: str = sys.executable) -> dict:
    """Run ``map_wells`` on one dataset and collect its outputs.

    Returns a dict with raw CLI outcome plus, when the run succeeded:
      registration   {calibrated, residual_px, n_bridged, n_rejected}
      layout         {scale, rotation_deg, well_radius_px, orientation,
                      expected_scale, spacing_rejected: [labels]}
      wells          {label: {observed, score, mosaic_px: [cx, cy], r,
                              ref_px: [x, y] | None, motor: [x, y, z] | None}}
      motor_map      {n_pairs, r2} | None
      frame_shape    [h, w]
    plus ``work_dir`` (the directory holding ``out/`` with map_wells' files).
    """
    data_dir = Path(data_dir)
    work_dir = Path(work_dir)
    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-m", "gridsteer.step2.map_wells", str(data_dir),
           "--outdir", str(out_dir), "--orientation", orientation]
    proc = _run(cmd, REPO_ROOT, timeout)

    result = dict(returncode=proc.returncode, stdout=proc.stdout,
                  stderr=proc.stderr, orientation=orientation,
                  n_frames=len(list_frames(data_dir)), work_dir=str(work_dir),
                  registration=None, layout=None, wells=None, motor_map=None,
                  frame_shape=None)
    result.update(parse_step2_stdout(proc.stdout))
    if proc.returncode != 0:
        return result

    wells_path = out_dir / "wells.json"
    if wells_path.exists():
        result["wells"] = load_step2_wells(out_dir)
    pos_path = out_dir / "well_centering_positions.json"
    if pos_path.exists():
        with open(pos_path) as f:
            centering = json.load(f)
        cal = centering.get("calibration", {})
        result["motor_map"] = dict(n_pairs=cal.get("n_training_pairs"),
                                   r2=cal.get("r2_scores"))
        fc = centering.get("frame_center")
        if fc:
            result["frame_shape"] = [int(round(2 * fc[1])), int(round(2 * fc[0]))]
    return result


def parse_step2_stdout(stdout: str) -> dict:
    """Scalar diagnostics map_wells prints but does not write to a file."""
    reg = dict(calibrated="Calibration Residual" in stdout,
               residual_px=None, n_bridged=_count_list(_RE_BRIDGED.search(stdout)),
               n_rejected=_count_list(_RE_REJECTED.search(stdout)))
    m = _RE_RESIDUAL.search(stdout)
    if m:
        reg["residual_px"] = float(m.group(1))

    layout = None
    m = _RE_LAYOUT.search(stdout)
    if m:
        layout = dict(scale=float(m.group(1)), rotation_deg=float(m.group(2)),
                      orientation=m.group(3).lower(),
                      well_radius_px=float(m.group(4)))
        m = _RE_EXPECTED_SCALE.search(stdout)
        layout["expected_scale"] = float(m.group(1)) if m else None
        m = _RE_SPACING.search(stdout)
        layout["spacing_rejected"] = (
            re.findall(r"(R\d+C\d+) \(", m.group(1)) if m else [])
    return dict(registration=reg, layout=layout)


def load_step2_wells(out_dir: Path) -> dict:
    """Merge wells.json and well_centering_positions.json per label."""
    out_dir = Path(out_dir)
    with open(out_dir / "wells.json") as f:
        wells_list = json.load(f)
    wells = {}
    for w in wells_list:
        wells[w["label"]] = dict(
            observed=bool(w["observed"]),
            score=float(w["score"]) if w.get("score") is not None else None,
            mosaic_px=[float(w["cx"]), float(w["cy"])],
            r=float(w["r"]), ref_px=None, motor=None)
    pos_path = out_dir / "well_centering_positions.json"
    if pos_path.exists():
        with open(pos_path) as f:
            centering = json.load(f)
        for entry in centering["well_centering_positions"].values():
            lab = entry["label"]
            if lab not in wells:
                continue
            p = entry["pixel_position_in_reference_frame"]
            m = entry["motor_position"]
            wells[lab]["ref_px"] = [float(p["x"]), float(p["y"])]
            wells[lab]["motor"] = [float(m["x"]), float(m["y"]), float(m["z"])]
    return wells

