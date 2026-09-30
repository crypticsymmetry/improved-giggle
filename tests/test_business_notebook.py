"""Exercise the uploaded-data path that is intentionally idle in Run All."""

import ast
import json
from pathlib import Path

import pytest

from assumption_ops import load_business_pilot, compile_business_pilot, compare_business_pilot

ROOT = Path(__file__).resolve().parents[1]


def uploaded_function():
    notebook = json.loads((ROOT / "notebooks/business_pilot.ipynb").read_text())
    source = "".join(notebook["cells"][-1]["source"])
    definition = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    namespace = dict(
        load_business_pilot=load_business_pilot,
        compile_business_pilot=compile_business_pilot,
        compare_business_pilot=compare_business_pilot,
    )
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), "uploaded-data-cell", "exec"),
        namespace,
    )
    return namespace["evaluate_uploaded_bundle"]


def test_uploaded_section_rejects_synthetic_demo():
    with pytest.raises(ValueError, match="synthetic"):
        uploaded_function()(ROOT / "examples/business_pilot/manifest.json")


def test_uploaded_section_uses_inputs_without_fixture_score_assertions(tmp_path):
    # A fixture relabeled for API-branch testing is not evidence of real data.
    for source in (ROOT / "examples/business_pilot").iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    manifest = tmp_path / "manifest.json"
    data = json.loads(manifest.read_text())
    data["synthetic"] = False
    manifest.write_text(json.dumps(data))
    snapshots, report = uploaded_function()(manifest)
    assert len(snapshots) == 3
    assert report["synthetic"] is False
    assert report["external_dispatch"] is False
    assert report["summary"]["lp_milp_proven_equal_snapshots"] == 3
