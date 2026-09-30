from dataclasses import replace
import json

import pytest

from assumption_ops.events import EvidenceUpdate
from assumption_ops.pilot import (
    PilotDataset,
    PilotEpisode,
    dataset_summary,
    load_pilot_dataset,
    save_pilot_dataset,
)
from assumption_ops.replay import ReplayBatch
from assumption_ops.replay_fixtures import synthetic_cases


def dataset():
    calibration = synthetic_cases(order_count=4, seed=17)[0]
    holdout = synthetic_cases(order_count=4, seed=23)[0]
    return PilotDataset(
        "test-pilot",
        (
            PilotEpisode("cal-1", "site-1", "calibration", calibration),
            PilotEpisode("hold-1", "site-2", "holdout", holdout),
        ),
    )


def test_pilot_roundtrip_and_summary_counts(tmp_path):
    original = dataset()
    path = save_pilot_dataset(original, tmp_path / "pilot")
    assert load_pilot_dataset(path) == original
    assert path == (tmp_path / "pilot" / "pilot.json").resolve()
    summary = dataset_summary(original)
    assert summary["episodes"] == 2
    for split in ("calibration", "holdout"):
        values = summary["splits"][split]
        assert values["episodes"] == 1
        assert values["groups"] == 1
        assert values["synthetic_episodes"] == 1
        assert values["orders"] == 4
        assert values["event_batches"] == 3
        assert values["updated_fields"] == 9
        assert values["distinct_update_sources"] == 3


def test_episodes_detach_nested_caller_policy_and_values():
    case = synthetic_cases(order_count=4, seed=17)[0]
    episode = PilotEpisode("a", "g", "calibration", case)
    case.policy.substitutions["A"] = ("changed",)
    assert episode.case.policy.substitutions["A"] == ("A-alt",)


def test_duplicate_episode_ids_and_groups_cannot_cross_splits():
    original = dataset()
    calibration, holdout = original.episodes
    with pytest.raises(ValueError, match="Duplicate episode_id"):
        PilotDataset("invalid", (calibration, replace(holdout, episode_id=calibration.episode_id)))
    with pytest.raises(ValueError, match="Group leakage"):
        PilotDataset("invalid", (calibration, replace(holdout, group_id=calibration.group_id)))


@pytest.mark.parametrize(
    "changes",
    [
        {"unfilled_penalty": 1},
        {"allow_late": False},
        {"substitutions": {}},
    ],
)
def test_policy_changes_cannot_disguise_duplicate_holdout_facts(changes):
    calibration = dataset().episodes[0]
    altered = replace(calibration.case, policy=replace(calibration.case.policy, **changes))
    holdout = PilotEpisode("different-id", "different-group", "holdout", altered)
    with pytest.raises(ValueError, match="Duplicate business trace"):
        PilotDataset("leak", (calibration, holdout))


def test_same_group_may_have_multiple_episodes_in_same_split():
    original = dataset()
    another = PilotEpisode(
        "cal-2", "site-1", "calibration", synthetic_cases(order_count=4, seed=31)[0]
    )
    result = PilotDataset("valid", original.episodes + (another,))
    assert dataset_summary(result)["splits"]["calibration"]["groups"] == 1


@pytest.mark.parametrize("episodes", [(), ("bad",)])
def test_missing_splits_or_invalid_types(episodes):
    with pytest.raises(ValueError):
        PilotDataset("invalid", episodes)
    with pytest.raises(ValueError, match="at least one"):
        PilotDataset("invalid", (dataset().episodes[0],))


@pytest.mark.parametrize("split", ["train", "test", "Calibration", "", True])
def test_episode_requires_exact_split(split):
    with pytest.raises(ValueError, match="split"):
        PilotEpisode("a", "g", split, synthetic_cases(order_count=4, seed=17)[0])


def test_renaming_provenance_cannot_hide_cross_split_business_trace():
    original = dataset().episodes[0]
    case = original.case
    batches = tuple(
        ReplayBatch(
            f"renamed-{index}",
            tuple(replace(update, source="renamed source") for update in batch.updates),
        )
        for index, batch in enumerate(case.batches)
    )
    renamed = replace(
        case,
        name="different-name",
        synthetic=False,
        seed=None,
        batches=batches,
        supplies=tuple(reversed(case.supplies)),
        orders=tuple(reversed(case.orders)),
    )
    holdout = PilotEpisode("hold-renamed", "other-site", "holdout", renamed)
    with pytest.raises(ValueError, match="Duplicate business trace"):
        PilotDataset("leak", (original, holdout))


def test_reordered_atomic_update_fields_do_not_hide_leakage():
    original = dataset().episodes[0]
    batches = tuple(
        replace(batch, updates=tuple(reversed(batch.updates))) for batch in original.case.batches
    )
    holdout = PilotEpisode("hold", "other-site", "holdout", replace(original.case, batches=batches))
    with pytest.raises(ValueError, match="Duplicate business trace"):
        PilotDataset("leak", (original, holdout))


def test_substitution_membership_order_and_duplicates_do_not_hide_leakage():
    case = synthetic_cases(order_count=4, seed=17)[0]
    case.policy.substitutions["A"] = ("B", "C")
    calibration = PilotEpisode("cal", "site-a", "calibration", case)
    case.policy.substitutions["A"] = ("C", "B", "B")
    holdout = PilotEpisode("hold", "site-b", "holdout", case)
    with pytest.raises(ValueError, match="Duplicate business trace"):
        PilotDataset("leak", (calibration, holdout))


def test_current_nested_case_values_are_revalidated_before_json_coercion():
    case = synthetic_cases(order_count=4, seed=17)[0]
    case.policy.substitutions[1] = ("B",)
    with pytest.raises(ValueError, match="substitution SKU"):
        PilotEpisode("cal", "site-a", "calibration", case)


def test_dataset_constructor_and_summary_reject_mutated_episode_values():
    original = dataset()
    original.episodes[0].case.policy.substitutions[1] = ("B",)
    with pytest.raises(ValueError, match="substitution SKU"):
        PilotDataset(original.name, original.episodes)
    with pytest.raises(ValueError, match="substitution SKU"):
        dataset_summary(original)


def test_policy_config_events_are_disallowed():
    case = synthetic_cases(order_count=4, seed=17)[0]
    policy_batch = ReplayBatch(
        "policy", (EvidenceUpdate("policy", "config", {"allow_late": True}, "review"),)
    )
    case = replace(case, batches=case.batches + (policy_batch,))
    with pytest.raises(ValueError, match="policy config"):
        PilotEpisode("policy-episode", "site", "calibration", case)


@pytest.mark.parametrize(
    "bad_path",
    ["../outside.json", "/absolute.json", "C:\\outside.json", "cases/../../outside.json"],
)
def test_manifest_case_paths_reject_escape(tmp_path, bad_path):
    manifest = save_pilot_dataset(dataset(), tmp_path / "pilot")
    data = json.loads(manifest.read_text())
    data["episodes"][0]["case_path"] = bad_path
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pilot.json: line.*(relative|traversal)"):
        load_pilot_dataset(manifest)


def test_manifest_symlink_escape_rejected(tmp_path):
    manifest = save_pilot_dataset(dataset(), tmp_path / "pilot")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    link = manifest.parent / "escape.json"
    link.symlink_to(outside)
    data = json.loads(manifest.read_text())
    data["episodes"][0]["case_path"] = "escape.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="symlink"):
        load_pilot_dataset(manifest)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(schema_version=2),
        lambda d: d.update(extra=1),
        lambda d: d.update(episodes=[]),
        lambda d: d["episodes"][0].update(extra=1),
        lambda d: d["episodes"][0].pop("group_id"),
        lambda d: d["episodes"][1].update(split="calibration"),
    ],
)
def test_strict_manifest_schema(tmp_path, change):
    manifest = save_pilot_dataset(dataset(), tmp_path / "pilot")
    data = json.loads(manifest.read_text())
    change(data)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pilot.json: line"):
        load_pilot_dataset(manifest)


@pytest.mark.parametrize("text", ['{"name":"a","name":"b"}', '{"episodes":NaN}', "{\n broken\n}"])
def test_manifest_duplicate_keys_nonfinite_and_parse_errors(tmp_path, text):
    path = tmp_path / "pilot.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="pilot.json: line"):
        load_pilot_dataset(path)
