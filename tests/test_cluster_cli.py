"""CLI uses verified inputs and never exports failed benchmark results."""

import json

import pytest

from assumption_ops.cluster_cli import main


def test_half_input_pair_fails_before_download(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("must not download")

    monkeypatch.setattr("assumption_ops.cluster_cli.download_cluster", unexpected)
    with pytest.raises(SystemExit):
        main(["--machine-events", "one.gz", "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_invalid_benchmark_does_not_create_output(tmp_path, monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("bad trace")

    monkeypatch.setattr("assumption_ops.cluster_cli.run_cluster_benchmark", invalid)
    with pytest.raises(ValueError, match="bad trace"):
        main(
            [
                "--machine-events",
                "m.gz",
                "--task-events",
                "t.gz",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert not (tmp_path / "out").exists()


def test_downloaded_paths_pass_to_benchmark_and_reports_export(tmp_path, capsys, monkeypatch):
    machine, task = tmp_path / "m.gz", tmp_path / "t.gz"
    calls = []
    monkeypatch.setattr(
        "assumption_ops.cluster_cli.download_cluster",
        lambda directory: {"machines": machine, "tasks": task},
    )
    report = {
        "controlled_model": True,
        "summary": {"snapshots": 1},
        "limitations": ["controlled test"],
        "snapshots": [],
    }

    def run(m, t, **kwargs):
        calls.append((m, t, kwargs))
        return report

    monkeypatch.setattr("assumption_ops.cluster_cli.run_cluster_benchmark", run)
    main(["--tasks", "12", "--machines", "2", "--output-dir", str(tmp_path / "out")])
    assert calls[0][:2] == (machine, task)
    assert calls[0][2]["max_tasks"] == 12
    assert calls[0][2]["max_machines"] == 2
    saved = json.loads((tmp_path / "out/cluster_report.json").read_text())
    assert saved == report
    assert (tmp_path / "out/cluster_comparison.csv").is_file()
    assert json.loads(capsys.readouterr().out)["controlled_model"] is True


@pytest.mark.parametrize(
    "flags",
    [
        ["--holdout", "--machines", "2"],
        ["--holdout", "--cutoffs-us", "900000000"],
        ["--holdout", "--migration-penalty", "1"],
        ["--machines-grid", "1", "2"],
    ],
)
def test_mode_conflicts_fail_before_download(flags, monkeypatch):
    monkeypatch.setattr(
        "assumption_ops.cluster_cli.download_cluster",
        lambda *a: pytest.fail("conflicting modes must not download"),
    )
    with pytest.raises(SystemExit):
        main(flags)


def test_holdout_mode_passes_fixed_settings_and_exports_distinct_files(
    tmp_path, capsys, monkeypatch
):
    calls = []
    report = {"controlled_model": True, "summary": {}, "limitations": [], "cases": []}

    def run(m, t, **kwargs):
        calls.append((m, t, kwargs))
        return report

    monkeypatch.setattr("assumption_ops.cluster_cli.run_cluster_holdout", run)
    monkeypatch.setattr(
        "assumption_ops.cluster_cli.run_cluster_benchmark",
        lambda *a, **k: pytest.fail("must use holdout"),
    )
    main(
        [
            "--holdout",
            "--machine-events",
            "m.gz",
            "--task-events",
            "t.gz",
            "--machines-grid",
            "1",
            "4",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert calls[0][2] == {"machine_counts": (1, 4), "max_tasks": 64, "time_limit": 5}
    assert json.loads((tmp_path / "cluster_holdout_report.json").read_text()) == report
    assert (tmp_path / "cluster_holdout_comparison.csv").exists()
    assert not (tmp_path / "cluster_report.json").exists()
    assert json.loads(capsys.readouterr().out)["controlled_model"] is True


def test_failed_holdout_does_not_create_output(tmp_path, monkeypatch):
    def fail(*a, **k):
        raise ValueError("invalid protocol")

    monkeypatch.setattr("assumption_ops.cluster_cli.run_cluster_holdout", fail)
    with pytest.raises(ValueError, match="invalid protocol"):
        main(
            [
                "--holdout",
                "--machine-events",
                "m.gz",
                "--task-events",
                "t.gz",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert not (tmp_path / "out").exists()
