"""Verify reusable CLI output boundaries and per-case accounting."""

import json

from assumption_ops import Order, Policy, Supply
from assumption_ops.replay import ReplayCase
from assumption_ops.replay_cli import main
from assumption_ops.replay_io import load_replay_case, save_replay_case


def test_cli_exports_reloadable_inputs_and_separate_cases(tmp_path, capsys):
    case = ReplayCase(
        name="../../outside",
        supplies=(Supply("s", "A", 1, 0, 2),),
        orders=(Order("o", "A", 3, 1),),
        policy=Policy(),
        batches=(),
        synthetic=True,
    )
    source = tmp_path / "source.json"
    save_replay_case(case, source)
    original = source.read_bytes()
    destination = tmp_path / "output"
    main(["--case", str(source), "--case", str(source), "--output-dir", str(destination)])
    printed = json.loads(capsys.readouterr().out)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert printed["external_dispatch"] is False
    assert len(manifest["cases"]) == 2
    assert "total_cost" not in manifest
    assert source.read_bytes() == original
    for entry in manifest["cases"]:
        assert "/" not in entry["input"]
        assert load_replay_case(destination / entry["input"]) == case
        report = json.loads((destination / entry["report"]).read_text())
        assert report["summary"]["final_optimized"]["metrics"]["requested_units"] == 3
        assert (destination / entry["snapshots_csv"]).is_file()


def test_cli_failed_input_does_not_create_success_manifest(tmp_path):
    import pytest

    source = tmp_path / "invalid.json"
    source.write_text('{"schema_version": 99}')
    destination = tmp_path / "output"
    with pytest.raises(ValueError):
        main(["--case", str(source), "--output-dir", str(destination)])
    assert not destination.exists()
