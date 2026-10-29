"""The validation results table: one CSV row per dataset and step.

Each validation test records a row (``conftest.report``); ``pytest
--results-csv PATH`` writes the rows as CSV at the end of the session.
CI runs the datasets in several shards and merges their CSVs into one::

    python -m tests.validation.results merge shard-*/results.csv \
        -o validation-results.csv [--markdown summary.md]

This module deliberately imports nothing beyond the standard library so
the merge can run on a bare Python without the package installed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

COLUMNS = [
    "step",             # "Step 1" or "Step 2"
    "dataset",          # manifest name, e.g. mapper15
    "shard",            # CI shard index (artifact validation-outputs-shard-<shard>); empty locally
    "orientation",      # Step 2 --orientation; empty for Step 1
    "n_frames",         # frames in the dataset
    "result",           # PASS or FAIL
    "actual",           # what this run produced
    "expected",         # what the reference holds
    "failed_checks",    # "name (value vs tolerance); ..." or empty
    "reference_file",   # tests/ground_truth/step<N>/<dataset>.json
    "output_dir",       # the run's working directory (empty with --no-keep-outputs)
]


# --------------------------------------------------------------------------
# Describing a run and its reference
# --------------------------------------------------------------------------

def _one_line(text: str, limit: int = 160) -> str:
    """Last non-empty line of a CLI's output, trimmed."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    tail = lines[-1] if lines else ""
    return tail if len(tail) <= limit else tail[:limit - 3] + "..."


def describe_step1(run: dict, ref: dict, tol: dict) -> tuple[str, str]:
    if run["returncode"] != 0 or run["phi"] is None:
        actual = (f"rc={run['returncode']}, phi={run['phi']}; "
                  + _one_line(run["stderr"] or run["stdout"]))
    else:
        actual = f"phi={run['phi']:.3f} deg"
    expected = f"phi={ref['phi']:.3f} deg (+/- {tol['phi_deg']:g})"
    return actual, expected


def _step2_outcome(d: dict) -> str:
    """Headline numbers of a successful Step 2 run or reference."""
    reg = d.get("registration") or {}
    lay = d.get("layout") or {}
    wells = d.get("wells") or {}
    n_obs = sum(1 for w in wells.values() if w.get("observed"))
    parts = [f"rc={d['returncode']}", f"wells={len(wells)} ({n_obs} observed)"]
    if reg.get("residual_px") is not None:
        parts.append(f"residual={reg['residual_px']:.2f} px")
    parts.append(f"bridged+rejected={reg.get('n_bridged', 0) + reg.get('n_rejected', 0)}")
    if lay.get("scale") is not None:
        parts.append(f"scale={lay['scale']:.3f}")
    return "; ".join(parts)


def describe_step2(run: dict, ref: dict) -> tuple[str, str]:
    if ref.get("returncode", 0) != 0:
        expected = f"rc={ref['returncode']} (known failure)"
    else:
        expected = _step2_outcome(ref)
    if run["returncode"] != 0 or run.get("wells") is None:
        actual = f"rc={run['returncode']}; " + _one_line(run["stderr"] or run["stdout"])
    else:
        actual = _step2_outcome(run)
    return actual, expected


def failed_checks(checks) -> str:
    out = []
    for c in checks:
        if c.passed:
            continue
        if c.value is not None and c.tolerance is not None:
            out.append(f"{c.name} ({c.value:.4g} vs {c.tolerance:.4g})")
        else:
            out.append(c.name)
    return "; ".join(out)


def make_row(step: int, entry: dict, run: dict, ref: dict, checks, tol: dict,
             reference_file: str, output_dir: str = "", shard: str = "") -> dict:
    if step == 1:
        actual, expected = describe_step1(run, ref, tol)
    else:
        actual, expected = describe_step2(run, ref)
    return dict(
        step=f"Step {step}",
        dataset=entry["name"],
        shard=shard,
        orientation=entry.get("orientation", "") if step == 2 else "",
        n_frames=run["n_frames"],
        result="PASS" if all(c.passed for c in checks) else "FAIL",
        actual=actual,
        expected=expected,
        failed_checks=failed_checks(checks),
        reference_file=reference_file,
        output_dir=output_dir,
    )


# --------------------------------------------------------------------------
# CSV and Markdown
# --------------------------------------------------------------------------

def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r["step"], r["dataset"]))


def write_csv(rows: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(sort_rows(rows))


def read_csv(path: Path | str) -> list[dict]:
    with open(path, newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def to_markdown(rows: list[dict]) -> str:
    cols = ["step", "dataset", "shard", "result", "actual", "expected", "failed_checks"]
    esc = lambda s: str(s).replace("|", "\\|")  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sort_rows(rows):
        lines.append("| " + " | ".join(esc(r.get(c, "")) for c in cols) + " |")
    n_fail = sum(r["result"] != "PASS" for r in rows)
    head = (f"**{len(rows)} dataset runs, {n_fail} failed**" if rows
            else "**No results were recorded**")
    return head + "\n\n" + "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="concatenate per-shard CSVs into one")
    m.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    m.add_argument("-o", "--output", required=True, type=Path)
    m.add_argument("--markdown", type=Path, default=None,
                   help="also append a Markdown table to this file")
    args = ap.parse_args(argv)

    paths = sorted({p for pat in args.inputs for p in glob.glob(pat)} | {
        p for p in args.inputs if Path(p).is_file()})
    rows = [r for p in paths for r in read_csv(p)]
    write_csv(rows, args.output)
    print(f"{len(rows)} rows from {len(paths)} file(s) -> {args.output}")
    if args.markdown:
        with open(args.markdown, "a") as f:
            f.write("## GridSteer Validation Results Summary\n\n" + to_markdown(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
