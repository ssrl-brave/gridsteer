"""Shared fixtures for the GridSteer test-suite.

Validation tests (``tests/test_step*_validation.py``) run the real CLIs on
the datasets under ``data/`` and compare against ``tests/ground_truth/``.
They are marked ``validation``; run only the unit tests with
``-m "not validation"``.

Options:
  --shard I/N        run only the I-th of N dataset shards (balanced by
                     frame count); CI uses this to spread the datasets over
                     several jobs
  --results-csv P    write one CSV row per dataset and step to P
                     (see tests/validation/results.py for the columns)
  --outputs-dir P    put every run's working directory under P (default: a
                     fresh temporary directory); CI uses this to upload the
                     outputs as a build artifact
  --no-keep-outputs  delete every run's working directory at the end of the
                     session. By default they are kept (debug frames, mosaics,
                     JSON files, logs), the path is printed at the end of the
                     session and recorded in the CSV's output_dir column
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.validation import metrics, results, runners

TESTS_DIR = Path(__file__).resolve().parent
GT_DIR = TESTS_DIR / "ground_truth"
REPO_ROOT = TESTS_DIR.parent


def pytest_addoption(parser):
    parser.addoption("--shard", default=None,
                     help="I/N: run only shard I (1-based) of N for validation datasets")
    parser.addoption("--results-csv", default=None,
                     help="write one CSV row per dataset and step to this path")
    parser.addoption("--outputs-dir", default=None,
                     help="directory for the runs' outputs (default: a temporary one)")
    parser.addoption("--keep-outputs", action="store_true", dest="keep_outputs",
                     default=True, help="keep pipeline outputs (the default)")
    parser.addoption("--no-keep-outputs", action="store_false", dest="keep_outputs",
                     help="delete pipeline outputs at the end of the session")


def pytest_configure(config):
    config.addinivalue_line("markers", "validation: slow end-to-end run on a dataset")
    config._validation_rows = []
    config._validation_work_root = None


# --------------------------------------------------------------------------
# Manifest / reference discovery
# --------------------------------------------------------------------------

def load_manifest() -> dict:
    with open(GT_DIR / "datasets.json") as f:
        return json.load(f)


def reference_path(step: int, name: str) -> Path:
    return GT_DIR / f"step{step}" / f"{name}.json"


def manifest_entries(step: int) -> list[dict]:
    """Datasets of one step that have a committed reference file."""
    return [entry for entry in load_manifest()[f"step{step}"]
            if reference_path(step, entry["name"]).exists()]


def dataset_params(step: int):
    return [pytest.param(e, id=e["name"]) for e in manifest_entries(step)]


# --------------------------------------------------------------------------
# Sharding: deterministic, balanced by frame count
# --------------------------------------------------------------------------

def shard_assignment(entries: list[dict], n_shards: int) -> dict[str, int]:
    """Greedy longest-processing-time bin packing on the manifest's frame
    counts (falling back to counting the .npz files)."""
    load = [0] * n_shards
    assign = {}
    weights = [(e.get("n_frames") or len(runners.list_frames(REPO_ROOT / e["data_dir"])),
                e["name"]) for e in entries]
    for n, name in sorted(weights, reverse=True):
        i = load.index(min(load))
        assign[name] = i
        load[i] += n
    return assign


def pytest_collection_modifyitems(config, items):
    shard = config.getoption("--shard")
    if not shard:
        return
    idx, n = (int(x) for x in shard.split("/"))
    assign = shard_assignment(manifest_entries(2) + manifest_entries(1), n)
    keep, skip = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        entry = callspec.params.get("entry") if callspec else None
        if entry is None:
            keep.append(item)           # unit tests run in every shard
        elif assign.get(entry["name"]) == idx - 1:
            keep.append(item)
        else:
            skip.append(item)
    items[:] = keep
    config.hook.pytest_deselected(items=skip)


# --------------------------------------------------------------------------
# Running the pipelines
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def work_root(request):
    outputs_dir = request.config.getoption("--outputs-dir")
    if outputs_dir:
        root = Path(outputs_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="gridsteer-validation-"))
    request.config._validation_work_root = root
    yield root
    if not request.config.getoption("keep_outputs"):
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def step1_run(entry, work_root):
    return runners.run_step1(REPO_ROOT / entry["data_dir"],
                             work_root / "step1" / entry["name"])


@pytest.fixture
def step2_run(entry, work_root):
    return runners.run_step2(REPO_ROOT / entry["data_dir"],
                             work_root / "step2" / entry["name"],
                             orientation=entry.get("orientation", "bottom-right"))


@pytest.fixture
def reference(entry, request):
    """The committed reference for this dataset; each validation module
    names its step in a module-level STEP constant."""
    with open(reference_path(request.module.STEP, entry["name"])) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Recording and reporting
# --------------------------------------------------------------------------

def report(request, step: int, entry: dict, run: dict, ref: dict,
           checks: list[metrics.Check]) -> None:
    """Record the dataset's CSV row, print its checks, and fail the test
    if any check missed its tolerance."""
    config = request.config
    config._validation_rows.append(results.make_row(
        step, entry, run, ref, checks,
        tol=metrics.merged_tolerances(entry.get("tolerances")),
        reference_file=str(reference_path(step, entry["name"]).relative_to(REPO_ROOT)),
        output_dir=run["work_dir"] if config.getoption("keep_outputs") else "",
        shard=(config.getoption("--shard") or "").split("/")[0]))
    print(f"\n[{entry['name']}] Step {step} checks\n{metrics.format_report(checks)}")
    failed = metrics.failures(checks)
    assert not failed, (f"{len(failed)} check(s) failed for {entry['name']}:\n"
                        + metrics.format_report(failed))


def pytest_sessionfinish(session, exitstatus):
    path = session.config.getoption("--results-csv")
    if path:
        results.write_csv(session.config._validation_rows, path)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    path = config.getoption("--results-csv")
    if path:
        terminalreporter.write_line(
            f"validation results: {len(config._validation_rows)} row(s) -> {path}")
    if config.getoption("keep_outputs") and config._validation_work_root:
        terminalreporter.write_line(
            f"pipeline outputs kept under {config._validation_work_root}")
