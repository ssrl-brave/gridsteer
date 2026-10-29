#!/usr/bin/env python3
"""(Re)generate the validation reference data under tests/ground_truth/.

The references are *the current code's own output*, reviewed by a person
and committed. They are not absolute truth; they pin down behaviour that
is known to be good so that a later change which moves it is noticed and
has to be justified.

Usage (from the repository root, with the package installed):

    # everything listed in tests/ground_truth/datasets.json
    python -m tests.validation.make_ground_truth

    # only some datasets, e.g. after adding new scans to the manifest
    python -m tests.validation.make_ground_truth --only mapper15 scan.31

    # Step 1 or Step 2 only
    python -m tests.validation.make_ground_truth --step 2

Review ``git diff tests/ground_truth`` before committing a regeneration:
every changed number is a behaviour change you are accepting.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import runners

HERE = Path(__file__).resolve().parent
GT_DIR = HERE.parent / "ground_truth"
MANIFEST = GT_DIR / "datasets.json"
REPO_ROOT = runners.REPO_ROOT


def load_manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def step1_reference(entry: dict, work: Path) -> dict:
    data_dir = REPO_ROOT / entry["data_dir"]
    run = runners.run_step1(data_dir, work)
    if run["returncode"] != 0 or run["phi"] is None:
        raise RuntimeError(f"Step 1 failed on {entry['name']}:\n{run['stderr']}")
    return dict(
        dataset=entry["name"], data_dir=entry["data_dir"], step=1,
        n_frames=run["n_frames"], returncode=run["returncode"],
        phi=_r(run["phi"], 6))


def step2_reference(entry: dict, work: Path) -> dict:
    data_dir = REPO_ROOT / entry["data_dir"]
    orientation = entry.get("orientation", "bottom-right")
    run = runners.run_step2(data_dir, work, orientation=orientation)
    ref = dict(dataset=entry["name"], data_dir=entry["data_dir"], step=2,
               orientation=orientation, n_frames=run["n_frames"],
               returncode=run["returncode"], frame_shape=run["frame_shape"],
               registration=None, layout=None, motor_map=None, wells=None)
    if run["returncode"] != 0:
        ref["failure_tail"] = (run["stderr"] or run["stdout"])[-500:]
        return ref

    reg = dict(run["registration"])
    gain = None
    # The stage gain is not printed; derive it from the template's gain
    # and the scale prior the run reported.
    try:
        from gridsteer.step2.well_template import DEFAULT_TEMPLATE_PATH, load_template
        tgain = load_template(DEFAULT_TEMPLATE_PATH).get("stage_gain")
        if tgain and run["layout"] and run["layout"].get("expected_scale"):
            gain = run["layout"]["expected_scale"] * tgain
    except Exception:  # pragma: no cover - template missing
        gain = None
    reg["stage_gain"] = _r(gain, 2)
    reg["residual_px"] = _r(reg["residual_px"], 2)
    ref["registration"] = reg

    if run["layout"]:
        lay = dict(run["layout"])
        lay["scale"] = _r(lay.get("scale"), 4)
        lay["rotation_deg"] = _r(lay.get("rotation_deg"), 2)
        lay["expected_scale"] = _r(lay.get("expected_scale"), 4)
        lay["n_observed"] = sum(w["observed"] for w in run["wells"].values())
        ref["layout"] = lay
    ref["motor_map"] = run["motor_map"]
    ref["wells"] = {
        lab: dict(observed=w["observed"], score=_r(w["score"], 2),
                  mosaic_px=[_r(w["mosaic_px"][0], 1), _r(w["mosaic_px"][1], 1)],
                  r=_r(w["r"], 1),
                  ref_px=None if w["ref_px"] is None else [_r(w["ref_px"][0], 2), _r(w["ref_px"][1], 2)],
                  motor=None if w["motor"] is None else [_r(v, 5) for v in w["motor"]])
        for lab, w in sorted(run["wells"].items(),
                             key=lambda kv: (int(kv[0][1]), int(kv[0].split("C")[1])))}
    return ref


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, choices=(1, 2), default=None,
                    help="regenerate only this step (default: both)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="dataset names to regenerate (default: all)")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="where to run (default: a temporary directory)")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="gridsteer-gt-"))
    print(f"Working directory: {work_root}")

    jobs = []
    for step, key in ((1, "step1"), (2, "step2")):
        if args.step and args.step != step:
            continue
        for entry in manifest[key]:
            if args.only and entry["name"] not in args.only:
                continue
            jobs.append((step, entry))
    if not jobs:
        sys.exit("nothing selected")

    for step, entry in jobs:
        print(f"[step {step}] {entry['name']} ...", end=" ", flush=True)
        work = work_root / f"step{step}" / entry["name"]
        ref = step1_reference(entry, work) if step == 1 else step2_reference(entry, work)
        out = GT_DIR / f"step{step}" / f"{entry['name']}.json"
        write_json(out, ref)
        if step == 1:
            print(f"phi={ref['phi']} -> {out.relative_to(REPO_ROOT)}")
        elif ref["returncode"] != 0:
            print(f"FAILS (rc={ref['returncode']}) -> {out.relative_to(REPO_ROOT)}")
        else:
            print(f"{ref['layout']['n_observed'] if ref['layout'] else '?'} observed, "
                  f"residual={ref['registration']['residual_px']}px -> "
                  f"{out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
