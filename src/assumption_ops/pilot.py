"""Grouped pilot datasets with strict local manifests and trace leakage checks.

The synthetic flag is caller-provided provenance, not verified source truth.
This module validates and summarizes inputs; it never solves or dispatches work.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
from typing import Any

from .intake import _error, _read_json
from .replay import ReplayCase
from .replay_io import _parse_case, load_replay_case, replay_case_to_dict, save_replay_case


def _name(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _trace_fingerprint(case: ReplayCase) -> str:
    """Hash business facts and chronology, excluding cosmetic provenance.

    Initial record order and independent fields within an atomic batch do not
    change its business meaning. Batch chronology remains significant.
    """
    # Policy variants are experimental settings, not independent observations;
    # changing them must not disguise the same facts as held-out data.
    payload = {
        "initial": {
            "supplies": sorted((asdict(record) for record in case.supplies), key=lambda r: r["id"]),
            "orders": sorted((asdict(record) for record in case.orders), key=lambda r: r["id"]),
        },
        "batches": [
            sorted(
                (
                    {"entity": update.entity, "field": update.field, "value": update.value}
                    for update in batch.updates
                ),
                key=lambda u: (u["entity"], u["field"]),
            )
            for batch in case.batches
        ],
    }
    canonical = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PilotEpisode:
    episode_id: str
    group_id: str
    split: str
    case: ReplayCase

    def __post_init__(self) -> None:
        _name(self.episode_id, "episode_id")
        _name(self.group_id, "group_id")
        if self.split not in ("calibration", "holdout"):
            raise ValueError("split must be exactly calibration or holdout")
        if not isinstance(self.case, ReplayCase):
            raise ValueError("case must be a ReplayCase")
        # Reconstruct before JSON conversion: the core validates current nested
        # values without allowing JSON to silently stringify invalid object keys.
        copied_case = replace(deepcopy(self.case))
        if any(
            update.entity == "policy" and update.field == "config"
            for batch in copied_case.batches
            for update in batch.updates
        ):
            raise ValueError("Pilot cases cannot contain policy config updates")
        # Detach mutable nested policy/update dictionaries from the caller.
        object.__setattr__(self, "case", _parse_case(replay_case_to_dict(copied_case)))


@dataclass(frozen=True)
class PilotDataset:
    name: str
    episodes: tuple[PilotEpisode, ...]

    def __post_init__(self) -> None:
        _name(self.name, "dataset name")
        if not isinstance(self.episodes, tuple) or any(
            not isinstance(episode, PilotEpisode) for episode in self.episodes
        ):
            raise ValueError("episodes must be a tuple of PilotEpisode records")
        episodes = tuple(
            PilotEpisode(episode.episode_id, episode.group_id, episode.split, episode.case)
            for episode in self.episodes
        )
        if {episode.split for episode in episodes} != {"calibration", "holdout"}:
            raise ValueError("A pilot needs at least one calibration and one holdout episode")
        ids: set[str] = set()
        groups: dict[str, str] = {}
        traces: dict[str, str] = {}
        for episode in episodes:
            if episode.episode_id in ids:
                raise ValueError(f"Duplicate episode_id: {episode.episode_id}")
            ids.add(episode.episode_id)
            if episode.group_id in groups and groups[episode.group_id] != episode.split:
                raise ValueError(f"Group leakage across splits: {episode.group_id}")
            groups[episode.group_id] = episode.split
            fingerprint = _trace_fingerprint(episode.case)
            if fingerprint in traces and traces[fingerprint] != episode.split:
                raise ValueError("Duplicate business trace across calibration and holdout splits")
            traces[fingerprint] = episode.split
        object.__setattr__(self, "episodes", episodes)


def _keys(value: Any, required: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} requires exactly {sorted(required)}")


def _case_path(directory: Path, value: Any) -> Path:
    _name(value, "case_path")
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).is_absolute() or "\\" in value:
        raise ValueError("case_path must be a relative local POSIX path")
    if ".." in relative.parts:
        raise ValueError("case_path cannot contain parent traversal")
    resolved = (directory / relative).resolve(strict=True)
    if not resolved.is_relative_to(directory):
        raise ValueError("case_path escapes the manifest directory through a symlink")
    if not resolved.is_file():
        raise ValueError("case_path must identify a file")
    return resolved


def load_pilot_dataset(path: str | Path) -> PilotDataset:
    """Load a strict version-1 manifest and cases confined to its directory."""
    source = Path(path)
    data = _read_json(source)
    try:
        _keys(data, {"schema_version", "name", "episodes"}, "pilot manifest")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("schema_version must be integer 1")
        if not isinstance(data["episodes"], list):
            raise ValueError("episodes must be an array")
        directory = source.resolve().parent
        episodes = []
        for record in data["episodes"]:
            _keys(record, {"episode_id", "group_id", "split", "case_path"}, "episode")
            case = load_replay_case(_case_path(directory, record["case_path"]))
            episodes.append(
                PilotEpisode(record["episode_id"], record["group_id"], record["split"], case)
            )
        return PilotDataset(data["name"], tuple(episodes))
    except (ValueError, TypeError, OSError) as exc:
        raise _error(source, 1, str(exc)) from exc


def dataset_summary(dataset: PilotDataset) -> dict[str, Any]:
    """Summarize workload/provenance counts without estimating business outcomes."""
    dataset = PilotDataset(dataset.name, dataset.episodes)
    result: dict[str, Any] = {"name": dataset.name, "episodes": len(dataset.episodes), "splits": {}}
    for split in ("calibration", "holdout"):
        episodes = [episode for episode in dataset.episodes if episode.split == split]
        sources = {
            update.source
            for episode in episodes
            for batch in episode.case.batches
            for update in batch.updates
        }
        result["splits"][split] = {
            "episodes": len(episodes),
            "groups": len({episode.group_id for episode in episodes}),
            "synthetic_episodes": sum(episode.case.synthetic for episode in episodes),
            "caller_labeled_nonsynthetic_episodes": sum(
                not episode.case.synthetic for episode in episodes
            ),
            "distinct_update_sources": len(sources),
            "supplies": sum(len(episode.case.supplies) for episode in episodes),
            "orders": sum(len(episode.case.orders) for episode in episodes),
            "initial_demand_units": sum(
                order.quantity for episode in episodes for order in episode.case.orders
            ),
            "event_batches": sum(len(episode.case.batches) for episode in episodes),
            "updated_fields": sum(
                len(batch.updates) for episode in episodes for batch in episode.case.batches
            ),
        }
    result["provenance_note"] = (
        "Synthetic flags and source labels are caller claims, not verified truth."
    )
    return result


def save_pilot_dataset(dataset: PilotDataset, directory: str | Path) -> Path:
    """Validate first, then export numbered cases and a portable local manifest."""
    checked = PilotDataset(dataset.name, dataset.episodes)
    # Validate nested case schemas before creating any directory or file.
    for episode in checked.episodes:
        _parse_case(replay_case_to_dict(episode.case))
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for index, episode in enumerate(checked.episodes):
        filename = f"case_{index:04d}.json"
        save_replay_case(episode.case, target / filename)
        records.append(
            {
                "episode_id": episode.episode_id,
                "group_id": episode.group_id,
                "split": episode.split,
                "case_path": filename,
            }
        )
    manifest = target / "pilot.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "name": checked.name, "episodes": records}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest.resolve()
