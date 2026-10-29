"""Unit tests for the validation results table (tests/validation/results.py)."""

from tests.validation import results
from tests.validation.metrics import Check, DEFAULT_TOLERANCES


def _entry(name="scan", **kw):
    return dict(name=name, data_dir=f"data/{name}", **kw)


def test_step1_row_pass():
    run = dict(returncode=0, stdout="104.9\n", stderr="", phi=104.9, n_frames=11,
               work_dir="/tmp/w")
    ref = dict(phi=104.348)
    row = results.make_row(1, _entry("phi.5"), run, ref,
                           [Check("phi_deg", True, 0.55, 2.5)], DEFAULT_TOLERANCES,
                           "tests/ground_truth/step1/phi.5.json", output_dir="/tmp/w")
    assert row["step"] == "Step 1" and row["result"] == "PASS"
    assert row["actual"] == "phi=104.900 deg"
    assert row["expected"] == "phi=104.348 deg (+/- 2.5)"
    assert row["failed_checks"] == "" and row["orientation"] == ""
    assert row["output_dir"] == "/tmp/w"
    assert list(row) == results.COLUMNS


def test_step1_row_cli_failure_quotes_last_line():
    run = dict(returncode=1, stdout="", stderr="Traceback\nValueError: no frames\n",
               phi=None, n_frames=0, work_dir="")
    row = results.make_row(1, _entry(), run, dict(phi=1.0),
                           [Check("cli_exit_and_phi", False)], DEFAULT_TOLERANCES, "r")
    assert row["result"] == "FAIL"
    assert row["actual"] == "rc=1, phi=None; ValueError: no frames"
    assert row["failed_checks"] == "cli_exit_and_phi"


def _step2(returncode=0, n_wells=3, n_obs=2, residual=0.3, scale=1.01):
    wells = {f"R1C{i}": dict(observed=i <= n_obs) for i in range(1, n_wells + 1)}
    return dict(returncode=returncode, stdout="", stderr="", n_frames=24, work_dir="",
                registration=dict(calibrated=True, residual_px=residual, n_bridged=1,
                                  n_rejected=0), layout=dict(scale=scale), wells=wells)


def test_step2_row_describes_both_sides_and_lists_failures():
    run, ref = _step2(residual=3.5), _step2()
    checks = [Check("cli_exit", True), Check("residual_px", False, 3.5, 2.0, "ref=0.3"),
              Check("stage_calibrated", False)]
    row = results.make_row(2, _entry("m", orientation="upper-right"), run, ref, checks,
                           DEFAULT_TOLERANCES, "r")
    assert row["orientation"] == "upper-right" and row["result"] == "FAIL"
    assert row["actual"] == "rc=0; wells=3 (2 observed); residual=3.50 px; bridged+rejected=1; scale=1.010"
    assert row["expected"] == "rc=0; wells=3 (2 observed); residual=0.30 px; bridged+rejected=1; scale=1.010"
    assert row["failed_checks"] == "residual_px (3.5 vs 2); stage_calibrated"


def test_step2_known_failure_reference():
    ref = dict(returncode=2, wells=None, registration=None, layout=None)
    run = dict(returncode=2, stdout="No Wells Detected\n", stderr="", wells=None,
               n_frames=6, work_dir="", registration=None, layout=None)
    row = results.make_row(2, _entry(), run, ref, [Check("expected_failure", True)],
                           DEFAULT_TOLERANCES, "r")
    assert row["result"] == "PASS"
    assert row["expected"] == "rc=2 (known failure)"
    assert row["actual"] == "rc=2; No Wells Detected"


def test_csv_round_trip_merge_and_markdown(tmp_path):
    a = dict.fromkeys(results.COLUMNS, "")
    a.update(step="Step 2", dataset="b", result="PASS", actual="x|y", n_frames=3)
    b = dict(a, dataset="a", result="FAIL", failed_checks="residual_px (3 vs 2)")
    c = dict(a, step="Step 1", dataset="phi")
    results.write_csv([a], tmp_path / "s1" / "results.csv")
    results.write_csv([b, c], tmp_path / "s2" / "results.csv")

    md = tmp_path / "summary.md"
    rc = results.main(["merge", str(tmp_path / "*" / "results.csv"),
                       "-o", str(tmp_path / "all.csv"), "--markdown", str(md)])
    assert rc == 0
    rows = results.read_csv(tmp_path / "all.csv")
    assert [(r["step"], r["dataset"]) for r in rows] == [
        ("Step 1", "phi"), ("Step 2", "a"), ("Step 2", "b")]
    assert rows[2]["actual"] == "x|y" and rows[2]["n_frames"] == "3"

    text = md.read_text()
    assert "3 dataset runs, 1 failed" in text
    assert "| Step 2 | b |  | PASS | x\\|y |" in text


def test_merge_with_no_inputs_writes_header_only(tmp_path):
    results.main(["merge", str(tmp_path / "nothing-*.csv"), "-o", str(tmp_path / "o.csv")])
    assert (tmp_path / "o.csv").read_text().strip() == ",".join(results.COLUMNS)


def test_make_row_records_shard():
    run = dict(returncode=0, phi=1.0, n_frames=3, stdout="", stderr="", work_dir="w")
    row = results.make_row(1, _entry("phi.5"), run, dict(phi=1.0),
                           [Check("phi_deg", True, 0.0, 2.5)], DEFAULT_TOLERANCES,
                           "r", shard="3")
    assert row["shard"] == "3"
    assert "| shard |" in results.to_markdown([row]).splitlines()[2]
