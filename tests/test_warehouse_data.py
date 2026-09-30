from hashlib import md5, sha256
from io import BytesIO
from types import SimpleNamespace
import sys

import pytest

import assumption_ops.warehouse_data as warehouse
from assumption_ops.warehouse_data import (
    WarehouseData,
    WarehouseRecord,
    download_warehouse,
    read_warehouse,
    warehouse_scenario,
)


def sample():
    return WarehouseData(
        {"A": 4, "B": 2},
        (WarehouseRecord(0, "A", 3), WarehouseRecord(2, "A", 5)),
        (WarehouseRecord(1, "A", 3), WarehouseRecord(1, "A", 2), WarehouseRecord(2, "B", 1)),
        {"test": True},
    )


def test_projection_aggregates_and_scaling_is_explicit_without_changing_demand():
    data = sample()
    baseline = warehouse_scenario(data)
    stressed = warehouse_scenario(data, stock_fraction=0.5)
    assert sum(s.quantity for s in baseline.supplies) == 14
    assert sum(s.quantity for s in stressed.supplies) == 6
    assert baseline.orders == stressed.orders
    assert sum(o.quantity for o in baseline.orders) == 6
    assert any(s.sku == "A" and s.available_day == 0 and s.quantity == 7 for s in baseline.supplies)
    assert any(o.sku == "A" and o.due_day == 1 and o.quantity == 5 for o in baseline.orders)
    assert baseline.policy.allow_late is True
    assert baseline.policy.disruption_penalty == 0
    assert all(s.unit_cost == 0 for s in baseline.supplies)
    assert all(o.priority == 1 for o in baseline.orders)
    assert data.initial_inventory == {"A": 4, "B": 2}


def test_projection_subset_and_empty_subset():
    selected = warehouse_scenario(sample(), sku_ids=("A",))
    assert {s.sku for s in selected.supplies} == {"A"}
    assert {o.sku for o in selected.orders} == {"A"}
    assert warehouse_scenario(sample(), sku_ids=()).supplies == ()


@pytest.mark.parametrize("fraction", [True, -0.1, 1.1, float("nan"), float("inf"), "1"])
def test_invalid_supply_fraction(fraction):
    with pytest.raises(ValueError, match="stock_fraction"):
        warehouse_scenario(sample(), stock_fraction=fraction)


@pytest.mark.parametrize("labels", [("missing",), ("A", "A"), (" ",), ["A"]])
def test_invalid_sku_selection(labels):
    with pytest.raises(ValueError, match="sku_ids"):
        warehouse_scenario(sample(), sku_ids=labels)


def test_data_detaches_provenance_and_revalidates_mutated_inventory():
    provenance = {"nested": {"flag": True}}
    data = WarehouseData({"A": 1}, (), (), provenance)
    provenance["nested"]["flag"] = False
    assert data.provenance["nested"]["flag"] is True
    data.initial_inventory["A"] = True
    with pytest.raises(ValueError, match="quantities"):
        warehouse_scenario(data)


def frozen_shape_tables():
    q = {key: [] for key in ("T", "Q", "item_SKU", "item_BATCH", "item_MONTH", "item_WEEK")}
    d = {key: [] for key in q}
    for index in range(368):
        for key, value in {
            "T": 0,
            "Q": 36610 if index == 0 else 0,
            "item_SKU": "1",
            "item_BATCH": "1_1_1",
            "item_MONTH": "1_1",
            "item_WEEK": "1_1_25",
        }.items():
            q[key].append(value)
    for index in range(1257):
        for key, value in {
            "T": 1,
            "Q": 1,
            "item_SKU": "1_1_1",
            "item_BATCH": "1_1",
            "item_MONTH": "1_1_25",
            "item_WEEK": "25686" if index == 0 else "0",
        }.items():
            d[key].append(value)
    h = {
        key: []
        for key in (
            "T",
            "Z",
            "SIDE",
            "item_SKU",
            "h_LAST_SKU",
            "item_BATCH",
            "h_LAST_BATCH",
            "item_MONTH",
            "h_LAST_MONTH",
            "item_WEEK",
            "h_LAST_WEEK",
        )
    }
    for index in range(780):
        values = {
            "T": 0.0,
            "Z": f"z{index // 2 + 1}",
            "SIDE": "dx" if index % 2 == 0 else "sx",
            "item_SKU": "1" if index == 0 else "NaN",
            "h_LAST_SKU": 10709.0 if index == 0 else 0.0,
            "item_BATCH": "NaN",
            "h_LAST_BATCH": 0,
            "item_MONTH": "NaN",
            "h_LAST_MONTH": 0,
            "item_WEEK": "NaN",
            "h_LAST_WEEK": 0,
        }
        for key, value in values.items():
            h[key].append(value)
    return {"Q": q, "D": d, "h_LAST": h, "ITEM_SKU": {"SKU": [str(i) for i in range(1, 102)]}}


def test_full_shape_normalization_and_documented_inferred_mapping():
    data = warehouse._normalize_tables(frozen_shape_tables())
    assert data.initial_inventory == {"1": 10709}
    assert sum(record.quantity for record in data.demand) == 25686
    assert data.demand[0] == WarehouseRecord(1, "1", 25686)
    assert data.provenance["inferred_demand_mapping"] is True
    assert data.provenance["author_confirmation"] is False
    assert data.provenance["demand_mapping"]["quantity"] == "int(D.item_WEEK)"


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t["D"]["item_SKU"].__setitem__(0, "2_1_1"),
        lambda t: t["D"]["item_WEEK"].__setitem__(0, "1.2"),
        lambda t: t["D"]["T"].__setitem__(0, 0),
        lambda t: t["Q"]["T"].__setitem__(0, 131),
        lambda t: t["Q"]["Q"].__setitem__(0, 1),
        lambda t: t["h_LAST"]["item_SKU"].__setitem__(0, "NaN"),
        lambda t: t["Q"]["item_SKU"].__setitem__(0, "missing"),
        lambda t: t["D"]["Q"].pop(),
    ],
)
def test_parse_invariants_fail_closed(change):
    tables = frozen_shape_tables()
    change(tables)
    with pytest.raises(ValueError):
        warehouse._normalize_tables(tables)


def test_read_requires_explicit_mapping_consent_before_optional_import(monkeypatch):
    monkeypatch.setattr(warehouse, "_verify", lambda path: None)
    with pytest.raises(ValueError, match="allow_inferred_demand=True"):
        read_warehouse("not-read.accdb")


def test_optional_reader_can_be_mocked_without_installing_dependency(monkeypatch):
    tables = frozen_shape_tables()
    monkeypatch.setattr(warehouse, "_verify", lambda path: None)

    class Parser:
        catalog = tables

        def __init__(self, path):
            pass

        def parse_table(self, name):
            return tables[name]

    monkeypatch.setitem(sys.modules, "access_parser", SimpleNamespace(AccessParser=Parser))
    assert (
        read_warehouse("not-read.accdb", allow_inferred_demand=True).provenance["totals"]["demand"]
        == 25686
    )


def pin_payload(monkeypatch, payload):
    monkeypatch.setattr(warehouse, "WAREHOUSE_SHA256", sha256(payload).hexdigest())
    monkeypatch.setattr(warehouse, "WAREHOUSE_MD5", md5(payload, usedforsecurity=False).hexdigest())


def test_downloader_pinned_url_timeout_atomic_cache_and_verification(tmp_path, monkeypatch):
    payload = b"verified test database"
    pin_payload(monkeypatch, payload)
    calls = []

    def response(url, timeout):
        calls.append((url, timeout))
        return BytesIO(payload)

    monkeypatch.setattr(warehouse, "urlopen", response)
    target = tmp_path / "cache" / "warehouse.accdb"
    assert download_warehouse(target) == target.resolve()
    assert target.read_bytes() == payload
    assert calls == [(warehouse.WAREHOUSE_URL, 30)]
    assert download_warehouse(target) == target.resolve()
    assert len(calls) == 1
    target.write_bytes(b"bad cache")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_warehouse(target)
    assert target.read_bytes() == b"bad cache"


@pytest.mark.parametrize("oversized", [False, True])
def test_download_failure_does_not_publish_or_leave_temporary_files(
    tmp_path, monkeypatch, oversized
):
    pin_payload(monkeypatch, b"correct")
    if oversized:
        monkeypatch.setattr(warehouse, "MAX_DOWNLOAD_BYTES", 4)
    monkeypatch.setattr(warehouse, "urlopen", lambda *args, **kwargs: BytesIO(b"incorrect"))
    with pytest.raises(ValueError):
        download_warehouse(tmp_path / "warehouse.accdb")
    assert list(tmp_path.iterdir()) == []


def test_known_fixture_hash_is_pinned():
    assert (
        warehouse.WAREHOUSE_SHA256
        == "4db26e1552f495f840e5c61e5d80fc7ed51bd28e1af6e10bd04b606d164f4548"
    )
    assert warehouse.WAREHOUSE_MD5 == "3844b06c3cb5326084c6dab1bdc284f0"
