"""Unfilled templates must not become valid evidence or overwrite exports."""

import json
from pathlib import Path

import pytest

from assumption_ops.business_cli import main
from assumption_ops.business_io import load_business_pilot
from assumption_ops.business_template import create_business_template


def test_unfilled_template_is_not_an_evaluable_dataset(tmp_path):
    manifest = create_business_template(tmp_path / "new")
    data = json.loads(manifest.read_text())
    assert data["synthetic"] is None
    assert set(json.loads((manifest.parent / "policy.json").read_text()).values()) == {None}
    with pytest.raises(ValueError):
        load_business_pilot(manifest)
    output = tmp_path / "results"
    with pytest.raises(ValueError):
        main(["--manifest", str(manifest), "--output-dir", str(output)])
    assert not output.exists()


def test_template_can_be_filled_with_explicit_fixture_records(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/business_pilot"
    manifest = create_business_template(tmp_path / "new")
    for path in source.iterdir():
        (manifest.parent / path.name).write_bytes(path.read_bytes())
    pilot = load_business_pilot(manifest)
    assert pilot.synthetic is True
    assert len(pilot.stock_snapshots) == 3


@pytest.mark.parametrize("existing", ["empty", "filled", "file", "symlink"])
def test_existing_destination_is_never_overwritten(tmp_path, existing):
    target = tmp_path / "exports"
    if existing == "file":
        target.write_text("keep")
    elif existing == "symlink":
        actual = tmp_path / "actual"
        actual.mkdir()
        (actual / "keep").write_text("keep")
        target.symlink_to(actual, target_is_directory=True)
    else:
        target.mkdir()
        if existing == "filled":
            (target / "keep").write_text("keep")
    before = sorted(
        (str(p.relative_to(tmp_path)), p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()
    )
    with pytest.raises(FileExistsError):
        create_business_template(target)
    after = sorted(
        (str(p.relative_to(tmp_path)), p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()
    )
    assert after == before


def test_cli_initialization_does_not_invoke_solver(tmp_path, capsys, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Initialization must not read a dataset or solve")

    monkeypatch.setattr("assumption_ops.business_cli.load_business_pilot", unexpected)
    monkeypatch.setattr("assumption_ops.business_cli.compare_business_pilot", unexpected)
    main(["--init-dir", str(tmp_path / "new")])
    result = json.loads(capsys.readouterr().out)
    assert result["ready_for_evaluation"] is False
    assert Path(result["manifest"]).is_file()


def test_initialization_and_evaluation_flags_cannot_mix(tmp_path):
    for extra in (["--manifest", "other.json"], ["--validate-only"], ["--output-dir", "results"]):
        with pytest.raises(SystemExit):
            main(["--init-dir", str(tmp_path / "new"), *extra])
        assert not (tmp_path / "new").exists()
