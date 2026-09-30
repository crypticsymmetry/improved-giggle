"""Exercise pilot validation and report export without touching source files."""

import json
from pathlib import Path

from assumption_ops.pilot_cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "pilot"


def test_validate_only_does_not_solve_or_write(tmp_path, capsys, monkeypatch):
    import assumption_ops.calibration as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must not solve")

    monkeypatch.setattr(module, "optimize", forbidden)
    destination = tmp_path / "output"
    main(
        [
            "--manifest",
            str(EXAMPLE / "pilot.json"),
            "--validate-only",
            "--output-dir",
            str(destination),
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["splits"]["calibration"]["episodes"] == 2
    assert summary["splits"]["holdout"]["episodes"] == 2
    assert not destination.exists()


def test_pilot_cli_exports_separate_selection_and_holdout_reports(tmp_path, capsys):
    before = {path.name: path.read_bytes() for path in EXAMPLE.iterdir()}
    destination = tmp_path / "output"
    main(
        [
            "--manifest",
            str(EXAMPLE / "pilot.json"),
            "--config",
            str(EXAMPLE / "calibration_config.json"),
            "--output-dir",
            str(destination),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    report = json.loads((destination / "calibration_report.json").read_text())
    assert printed["selected_candidate"] == report["selected_candidate"]
    assert report["external_dispatch"] is False
    assert len(report["holdout"]["episodes"]) == 2
    for row in report["calibration"]["rankings"]:
        assert all(
            result["episode_id"].startswith("calibration-") for result in row["episode_results"]
        )
    assert (destination / "calibration_rankings.csv").is_file()
    assert {path.name: path.read_bytes() for path in EXAMPLE.iterdir()} == before
