"""CLI performs validation before writing any comparison artifacts."""

import json
from pathlib import Path

import pytest

from assumption_ops.business_cli import main

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/business_pilot/manifest.json"


def test_validation_has_no_output_artifacts(tmp_path, capsys):
    output = tmp_path / "absent"
    main(["--manifest", str(MANIFEST), "--validate-only", "--output-dir", str(output)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["synthetic"] is True
    assert [s["inventory_units"] for s in summary["snapshots"]] == [8, 9, 8]
    assert [s["open_order_units"] for s in summary["snapshots"]] == [10, 8, 7]
    assert not output.exists()


def test_comparison_exports_all_methods_and_keeps_last_snapshot_separate(tmp_path, capsys):
    main(["--manifest", str(MANIFEST), "--output-dir", str(tmp_path)])
    summary = json.loads(capsys.readouterr().out)
    report = json.loads((tmp_path / "business_report.json").read_text())
    assert summary["summary"]["lp_milp_proven_equal_snapshots"] == 3
    assert report["summary"]["final_method_evaluations"]["milp"]["metrics"]["requested_units"] == 7
    assert len((tmp_path / "business_comparison.csv").read_text().splitlines()) == 13
    opening = {
        m["method"]: m["evaluation"]["costs"]["total"]
        for m in report["snapshots"][0]["comparison"]["results"]
    }
    assert opening["milp"] == opening["verified_lp"] == 2044
    assert opening["priority_cost_greedy"] == 4028
    assert report["external_dispatch"] is False


def test_invalid_manifest_cannot_create_results(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version":1}')
    output = tmp_path / "absent"
    with pytest.raises(ValueError):
        main(["--manifest", str(path), "--output-dir", str(output)])
    assert not output.exists()
