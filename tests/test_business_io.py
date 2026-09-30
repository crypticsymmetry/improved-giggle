import csv
import json

import pytest

from assumption_ops.business_io import load_business_pilot, write_business_csv


def _bundle(tmp_path):
    files = {
        "catalog.csv": "sku,unit_cost\nA,2\nB,3\n",
        "stock.csv": (
            "snapshot_id,observed_at,sku,quantity,source\n"
            "s0,2026-01-01T00:00:00Z,A,8,erp\n"
            "s0,2026-01-01T00:00:00Z,B,0,erp\n"
        ),
        "movements.csv": "movement_id,event_time,observed_at,sku,kind,quantity,source\n",
        "orders.csv": (
            "order_id,event_time,observed_at,sku,quantity,due_date,priority,source\n"
            "o1,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,A,2,2026-01-02,1,erp\n"
        ),
        "policy.json": "{}",
    }
    for filename, text in files.items():
        (tmp_path / filename).write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "name": "fixture",
        "group_id": "warehouse",
        "synthetic": True,
        "catalog_path": "catalog.csv",
        "stock_snapshots_path": "stock.csv",
        "movements_path": "movements.csv",
        "open_orders_path": "orders.csv",
        "policy_path": "policy.json",
    }
    path = tmp_path / "business.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_load_csv_bundle_bom_grouping_and_read_only(tmp_path):
    path, _ = _bundle(tmp_path)
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("\ufeffsku,unit_cost\nA,02\nB,3\n", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    pilot = load_business_pilot(path)
    assert pilot.name == "fixture"
    assert pilot.catalog[0].unit_cost == 2
    assert pilot.stock_snapshots[0].quantities == {"A": 8, "B": 0}
    assert pilot.order_observations[0].quantity == 2
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("quantity", ["1.5", "1e2", "-1", "+2", "true", "١", ""])
def test_integer_quantity_has_source_line(tmp_path, quantity):
    path, _ = _bundle(tmp_path)
    stock = tmp_path / "stock.csv"
    stock.write_text(
        f"snapshot_id,observed_at,sku,quantity,source\ns0,2026-01-01T00:00:00Z,A,{quantity},erp\n"
    )
    with pytest.raises(ValueError, match="stock.csv: line 2.*quantity"):
        load_business_pilot(path)


@pytest.mark.parametrize("header", ["sku,sku", "sku", "sku,unit_cost,unknown", ""])
def test_exact_unique_headers(tmp_path, header):
    path, _ = _bundle(tmp_path)
    (tmp_path / "catalog.csv").write_text(header + "\n")
    with pytest.raises(ValueError, match="catalog.csv: line 1"):
        load_business_pilot(path)


@pytest.mark.parametrize(
    "row",
    [
        "s0,2026-01-01T00:00:00Z,A,3,erp",
        "s0,2026-01-02T00:00:00Z,B,3,erp",
        "s0,2026-01-01T00:00:00Z,B,3,different",
    ],
)
def test_snapshot_duplicate_sku_or_inconsistent_metadata(tmp_path, row):
    path, _ = _bundle(tmp_path)
    stock = tmp_path / "stock.csv"
    stock.write_text(
        stock.read_text().splitlines()[0] + "\ns0,2026-01-01T00:00:00Z,A,3,erp\n" + row + "\n"
    )
    with pytest.raises(ValueError, match="stock.csv: line 3"):
        load_business_pilot(path)


@pytest.mark.parametrize("filename", ["catalog.csv", "orders.csv"])
def test_duplicate_records(tmp_path, filename):
    path, _ = _bundle(tmp_path)
    source = tmp_path / filename
    source.write_text(source.read_text() + source.read_text().splitlines()[1] + "\n")
    with pytest.raises(ValueError, match=f"{filename}: line.*Duplicate"):
        load_business_pilot(path)


@pytest.mark.parametrize("row", ["", "A", "A,2,extra", '"unclosed,2'])
def test_bad_row_shape_and_csv_syntax(tmp_path, row):
    path, _ = _bundle(tmp_path)
    (tmp_path / "catalog.csv").write_text("sku,unit_cost\n" + row + "\n")
    with pytest.raises(ValueError, match="catalog.csv: line 2"):
        load_business_pilot(path)


@pytest.mark.parametrize(
    "change",
    [{"schema_version": True}, {"schema_version": 2}, {"unknown": 1}, {"synthetic": "false"}],
)
def test_manifest_validation(tmp_path, change):
    path, manifest = _bundle(tmp_path)
    path.write_text(json.dumps({**manifest, **change}))
    with pytest.raises(ValueError, match="business.json: line 1"):
        load_business_pilot(path)


@pytest.mark.parametrize(
    "value", ["../catalog.csv", "/etc/passwd", "C:\\catalog.csv", "folder\\catalog.csv"]
)
def test_confined_paths(tmp_path, value):
    path, manifest = _bundle(tmp_path)
    path.write_text(json.dumps({**manifest, "catalog_path": value}))
    with pytest.raises(ValueError, match="business.json: line 1"):
        load_business_pilot(path)


def test_symlink_escape(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    path, manifest = _bundle(directory)
    outside = tmp_path / "outside.csv"
    outside.write_text("sku,unit_cost\nA,1\n")
    (directory / "link.csv").symlink_to(outside)
    path.write_text(json.dumps({**manifest, "catalog_path": "link.csv"}))
    with pytest.raises(ValueError, match="symlink"):
        load_business_pilot(path)


def test_domain_validation_before_return(tmp_path):
    path, _ = _bundle(tmp_path)
    stock = tmp_path / "stock.csv"
    stock.write_text(stock.read_text().replace("2026-01-01T00:00:00Z", "invalid"))
    with pytest.raises(ValueError):
        load_business_pilot(path)


def test_existing_order_created_before_opening_snapshot(tmp_path):
    path, _ = _bundle(tmp_path)
    orders = tmp_path / "orders.csv"
    orders.write_text(
        orders.read_text().replace("o1,2026-01-01T00:00:00Z", "o1,2025-12-15T00:00:00Z")
    )
    pilot = load_business_pilot(path)
    assert pilot.order_observations[0].event_time == "2025-12-15T00:00:00Z"


def test_duplicate_movement_id_reports_source_line(tmp_path):
    path, _ = _bundle(tmp_path)
    movement = "m1,2026-01-02T00:00:00Z,2026-01-02T00:00:00Z,A,receipt,2,erp\n"
    target = tmp_path / "movements.csv"
    target.write_text(target.read_text() + movement + movement)
    with pytest.raises(ValueError, match="movements.csv: line 3.*Duplicate"):
        load_business_pilot(path)


def test_json_duplicate_manifest_key_rejected(tmp_path):
    path, _ = _bundle(tmp_path)
    path.write_text(
        path.read_text().replace(
            '{"schema_version": 1', '{"schema_version": 1, "schema_version": 1'
        )
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_business_pilot(path)


def test_csv_export_metrics_and_formula_escaping(tmp_path):
    report = {
        "snapshots": [
            {
                "snapshot_id": "=unsafe",
                "cutoff": "2026-01-01T00:00:00Z",
                "comparison": {
                    "results": [
                        {
                            "method": "+formula",
                            "status": "optimal",
                            "evaluation": {
                                "metrics": {
                                    "requested_units": 9,
                                    "allocated_units": 7,
                                    "on_time_units": 6,
                                    "late_units": 1,
                                    "unfilled_units": 2,
                                },
                                "costs": {"total": 600},
                            },
                        }
                    ]
                },
            }
        ]
    }
    target = tmp_path / "report.csv"
    write_business_csv(report, target)
    with target.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["snapshot_id"] == "'=unsafe"
    assert rows[0]["method"] == "'+formula"
    assert rows[0]["filled_units"] == "7"
    assert rows[0]["weighted_score"] == "600"


def test_empty_report_exports_header(tmp_path):
    target = tmp_path / "report.csv"
    write_business_csv({"snapshots": []}, target)
    assert target.read_text().startswith("snapshot_id,cutoff,method,status,")
    assert len(target.read_text().splitlines()) == 1
