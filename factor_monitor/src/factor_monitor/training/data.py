"""Trajectory-grouped records, strict collation, and family-uniform sampling."""

from __future__ import annotations

import bisect
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..io import iter_jsonl
from ..prompt_contract import (
    factor_suffix_choice_contract,
    factor_suffix_choice_token_ids,
    render_factor_suffix_choice_text,
    tokenize_factor_suffix_choice_prompt,
)
from ..schema import LEVELS, FactorSpec, Taxonomy
from .config import FactorMonitorArchitecture, PHASE1_FAMILY_RELATIONS
from .encoding import (
    CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID,
    ENDPOINT_CONTRACT_ID,
    ENDPOINT_ROLE_FIELDS,
    FACTOR_SUFFIX_CHOICE_CONTRACT_ID,
    FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,
    FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID,
    TokenizedInput,
    factor_suffix_relay_insertion_count,
    factor_suffix_relay_token_id,
    render_endpoint_inputs,
    render_evidence,
    render_factor,
    rendered_for_tokenizer,
    tokenize_rendered,
)
from .model import RELATION_TO_ID, ROUTE_TO_ID

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except ModuleNotFoundError:
    torch = None
    Dataset = object  # type: ignore[assignment,misc]
    Sampler = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class FactorQueryRecord:
    query_id: str
    trajectory_id: str
    factor_instance_id: str
    factor_level: str
    family: str
    factor_spec: FactorSpec
    label: int | None
    leakage_group_id: str
    negative_type: str | None
    loss_weight: float = 1.0
    factor_scope: str | None = None
    intervention: str = "matched"
    source_query_id: str | None = None
    donor_query_id: str | None = None
    contrast_group_id: str | None = None


@dataclass(frozen=True)
class TrajectoryGroup:
    trajectory_id: str
    task_context: str
    model_cot: str
    family: str
    leakage_group_id: str
    queries: tuple[FactorQueryRecord, ...]


@dataclass(frozen=True)
class FactorInstanceContrastUnit:
    """One sampling unit containing one trajectory or one complete contrast pair."""

    unit_id: str
    family: str
    groups: tuple[TrajectoryGroup, ...]
    factor_instance_count: int
    contrast_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class LongCotMilTrainingUnit:
    """One source Factor-instance unit and all of its lossless CoT windows."""

    unit_id: str
    family: str
    source_groups: tuple[TrajectoryGroup, ...]
    window_groups: tuple[TrajectoryGroup, ...]
    factor_instance_count: int


@dataclass(frozen=True)
class LongCotMilScheduledIndex:
    """One sampler index, optionally repeated only for synchronized DDP shape."""

    index: int
    is_sync_padding: bool = False


@dataclass(frozen=True)
class LongCotMilScheduledUnit:
    """A materialized MIL unit carrying its zero-loss synchronization role."""

    unit: LongCotMilTrainingUnit | FactorInstanceContrastUnit
    is_sync_padding: bool = False


_EVIDENCE_WINDOW_MARKERS = ("::cot-window=", "::context-window=")


def _evidence_window_parts(identifier: str) -> tuple[str, str, int] | None:
    matches = [marker for marker in _EVIDENCE_WINDOW_MARKERS if marker in identifier]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"ambiguous evidence-window identifier: {identifier}")
    marker = matches[0]
    source, value = identifier.rsplit(marker, 1)
    if not source or not value.isdigit():
        raise ValueError(f"invalid evidence-window identifier: {identifier}")
    return source, marker.removeprefix("::").removesuffix("="), int(value)


def _cot_window_index(query_id: str) -> int | None:
    """Return the historical or field-aware evidence-window index."""

    parts = _evidence_window_parts(query_id)
    return None if parts is None else parts[2]


class LongCotMilMiningState:
    """Label-independent cached scoring and rotating exploration for long bags."""

    def __init__(
        self,
        *,
        exact_max_windows: int,
        active_windows: int,
        exploitation_windows: int,
        exploration_windows: int,
        initial_selection: str = "cached_rank_then_position",
    ) -> None:
        if exact_max_windows != 4 or active_windows != 4:
            raise ValueError("the frozen MIL screen requires four exact/active windows")
        if exploitation_windows != 2 or exploration_windows != 2:
            raise ValueError("the frozen MIL screen requires two exploit and two explore windows")
        if initial_selection not in {
            "cached_rank_then_position",
            "position_stratified_no_model",
        }:
            raise ValueError(f"unsupported MIL initial selection: {initial_selection!r}")
        self.exact_max_windows = exact_max_windows
        self.active_windows = active_windows
        self.exploitation_windows = exploitation_windows
        self.exploration_windows = exploration_windows
        self.initial_selection = initial_selection
        self.logits: dict[tuple[str, int], float] = {}
        self.visits: Counter[tuple[str, int]] = Counter()
        self.refresh_count = 0

    @staticmethod
    def _source_trajectory_id(group: TrajectoryGroup) -> str:
        parts = _evidence_window_parts(group.trajectory_id)
        if parts is None:
            raise ValueError("MIL window group lacks an evidence-window trajectory suffix")
        return parts[0]

    @staticmethod
    def _window_index(group: TrajectoryGroup) -> int:
        values = {_cot_window_index(query.query_id) for query in group.queries}
        if None in values or len(values) != 1:
            raise ValueError("one MIL window group must contain one shared window index")
        return int(next(iter(values)))

    def update_rows(self, rows: Iterable[Mapping[str, Any]], *, full_refresh: bool) -> None:
        seen = 0
        for row in rows:
            source = row.get("source_query_id")
            query_id = row.get("query_id")
            logit = row.get("logit")
            if not isinstance(source, str) or not isinstance(query_id, str):
                raise ValueError("MIL cache rows require query and source-query IDs")
            index = _cot_window_index(query_id)
            if index is None:
                continue
            if isinstance(logit, bool) or not isinstance(logit, (int, float)):
                raise ValueError("MIL cache rows require numeric logits")
            self.logits[(source, index)] = float(logit)
            seen += 1
        if full_refresh:
            if seen <= 0:
                raise ValueError("full MIL cache refresh received no window logits")
            self.refresh_count += 1

    def update_batch(
        self,
        source_query_ids: Sequence[Sequence[str | None]],
        window_indices: Any,
        logits: Any,
        query_mask: Any,
    ) -> None:
        if torch is None:
            raise RuntimeError("MIL cache updates require torch")
        detached = logits.detach().float().cpu()
        indices = window_indices.detach().cpu()
        valid = query_mask.detach().bool().cpu()
        for row in range(detached.shape[0]):
            for column in range(detached.shape[1]):
                if not bool(valid[row, column]):
                    continue
                index = int(indices[row, column].item())
                source = source_query_ids[row][column]
                if index < 0:
                    continue
                if not isinstance(source, str) or not source:
                    raise ValueError("active MIL windows require source-query IDs")
                self.logits[(source, index)] = float(detached[row, column].item())

    def select_groups(
        self,
        unit: LongCotMilTrainingUnit,
        *,
        record_visit: bool = True,
    ) -> tuple[TrajectoryGroup, ...]:
        groups = tuple(
            sorted(unit.window_groups, key=self._window_index)
        )
        if len(groups) <= self.exact_max_windows:
            selected = groups
        else:
            source_trajectory = self._source_trajectory_id(groups[0])
            if any(self._source_trajectory_id(group) != source_trajectory for group in groups):
                raise ValueError("one MIL unit may not cross source trajectories")

            def cached_score(group: TrajectoryGroup) -> float:
                index = self._window_index(group)
                values = [
                    self.logits.get((query.source_query_id or query.query_id, index))
                    for query in group.queries
                ]
                present = [value for value in values if value is not None]
                return max(present) if present else float("-inf")

            has_cached_score = any(
                cached_score(group) != float("-inf") for group in groups
            )
            if (
                self.initial_selection == "position_stratified_no_model"
                and not has_cached_score
            ):
                final_index = len(groups) - 1
                positions = tuple((step * final_index + 1) // 3 for step in range(4))
                if len(set(positions)) != self.active_windows:
                    raise ValueError("position-stratified MIL selection was not unique")
                selected = tuple(groups[position] for position in positions)
            else:
                ranked = sorted(
                    groups,
                    key=lambda group: (-cached_score(group), self._window_index(group)),
                )
                exploit = ranked[: self.exploitation_windows]
                exploit_indices = {self._window_index(group) for group in exploit}
                positional = []
                left, right = 0, len(groups) - 1
                while left <= right:
                    positional.append(groups[left])
                    if left != right:
                        positional.append(groups[right])
                    left += 1
                    right -= 1
                remaining = [
                    group
                    for group in positional
                    if self._window_index(group) not in exploit_indices
                ]
                remaining.sort(
                    key=lambda group: (
                        self.visits[(source_trajectory, self._window_index(group))],
                        positional.index(group),
                    )
                )
                explore = remaining[: self.exploration_windows]
                selected = tuple(
                    sorted((*exploit, *explore), key=self._window_index)
                )
                if len(selected) != self.active_windows:
                    raise ValueError("approximate MIL selection did not produce four windows")
        source_trajectory = self._source_trajectory_id(selected[0])
        if record_visit:
            for group in selected:
                self.visits[(source_trajectory, self._window_index(group))] += 1
        return tuple(selected)

    def audit(self) -> dict[str, Any]:
        return {
            "policy": "cached_top2_plus_rotating_least_visited_position_stratified",
            "initial_selection": self.initial_selection,
            "exact_max_windows": self.exact_max_windows,
            "active_windows": self.active_windows,
            "exploitation_windows": self.exploitation_windows,
            "exploration_windows": self.exploration_windows,
            "cached_query_windows": len(self.logits),
            "visited_trajectory_windows": len(self.visits),
            "full_refreshes": self.refresh_count,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "audit": self.audit(),
            "logits": [
                {
                    "source_query_id": source,
                    "window_index": index,
                    "logit": logit,
                }
                for (source, index), logit in sorted(self.logits.items())
            ],
            "visits": [
                {
                    "source_trajectory_id": trajectory,
                    "window_index": index,
                    "count": count,
                }
                for (trajectory, index), count in sorted(self.visits.items())
            ],
        }

    def restore(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") not in {1, 2}:
            raise ValueError("unsupported MIL mining-cache snapshot")
        audit = value.get("audit", {})
        if value.get("schema_version") == 2:
            cached_initial_selection = (
                audit.get("initial_selection") if isinstance(audit, Mapping) else None
            )
            if cached_initial_selection != self.initial_selection:
                raise ValueError("MIL mining-cache initial-selection policy differs")
        logits: dict[tuple[str, int], float] = {}
        for row in value.get("logits", []):
            source = row.get("source_query_id")
            index = row.get("window_index")
            logit = row.get("logit")
            if (
                not isinstance(source, str)
                or isinstance(index, bool)
                or not isinstance(index, int)
                or isinstance(logit, bool)
                or not isinstance(logit, (int, float))
            ):
                raise ValueError("invalid MIL mining-cache logit row")
            logits[(source, index)] = float(logit)
        if (
            not logits
            and self.initial_selection != "position_stratified_no_model"
        ):
            raise ValueError("MIL mining-cache snapshot contains no logits")
        visits: Counter[tuple[str, int]] = Counter()
        for row in value.get("visits", []):
            trajectory = row.get("source_trajectory_id")
            index = row.get("window_index")
            count = row.get("count")
            if (
                not isinstance(trajectory, str)
                or isinstance(index, bool)
                or not isinstance(index, int)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("invalid MIL mining-cache visit row")
            visits[(trajectory, index)] = count
        refreshes = audit.get("full_refreshes", 0) if isinstance(audit, Mapping) else 0
        if isinstance(refreshes, bool) or not isinstance(refreshes, int) or refreshes < 0:
            raise ValueError("invalid MIL mining-cache refresh count")
        self.logits = logits
        self.visits = visits
        self.refresh_count = refreshes


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _factor_instance_id(row: Mapping[str, Any]) -> str:
    direct = row.get("factor_instance_id")
    if isinstance(direct, str) and direct.strip():
        return direct
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("factor_instance_id")
        if isinstance(nested, str) and nested.strip():
            return nested
    raise ValueError(
        f"query {row.get('query_id')} lacks explicit factor_instance_id; "
        "the loader will not infer L1/L3 grouping"
    )


def _negative_subtype(row: Mapping[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    for field in ("negative_type", "factor_bucket", "training_subtype"):
        value = metadata.get(field)
        if isinstance(value, str) and value.strip():
            return value
    causal = metadata.get("causal")
    if isinstance(causal, Mapping):
        role = causal.get("triplet_role")
        if isinstance(role, str) and role.strip():
            return role
    return None


def _factor_scope(row: Mapping[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("factor_scope")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"query {row.get('query_id')}.metadata.factor_scope must be a non-empty string"
        )
    return value.strip()


def _loss_weight(row: Mapping[str, Any]) -> float:
    """Load one positive finite record weight without hiding field disagreement."""

    parsed: dict[str, float] = {}
    for field in ("loss_weight", "record_loss_weight"):
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"query {row.get('query_id')}.{field} must be numeric")
        numeric = float(value)
        if numeric <= 0 or numeric == float("inf") or numeric != numeric:
            raise ValueError(
                f"query {row.get('query_id')}.{field} must be positive and finite"
            )
        parsed[field] = numeric
    if len(set(parsed.values())) > 1:
        raise ValueError(
            f"query {row.get('query_id')} loss_weight fields disagree: {parsed}"
        )
    return next(iter(parsed.values()), 1.0)


def _canonical_contrast_group_id(row: Mapping[str, Any]) -> str | None:
    """Read only source-owned Factor contrast identifiers.

    ``metadata.contrast_group_id`` is intentionally ignored: Deception uses
    that field for scenario grouping rather than positive/negative Factor
    contrasts.
    """

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    values: list[str] = []
    for field in ("factor_contrast", "contrast_pair"):
        nested = metadata.get(field)
        if not isinstance(nested, Mapping):
            continue
        value = nested.get("contrast_group_id")
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"query {row.get('query_id')}.metadata.{field}.contrast_group_id "
                "must be a non-empty string"
            )
        values.append(value.strip())
    if len(set(values)) > 1:
        raise ValueError(
            f"query {row.get('query_id')} has conflicting canonical contrast IDs"
        )
    return values[0] if values else None


def _lossless_factor_spec(value: Any, name: str) -> FactorSpec:
    """Validate canonical Factor fields without normalizing their source text."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    fields = {"factor", "relation", "applied_criterion", "not_applied_criterion"}
    if set(value) != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields-set(value))}, "
            f"unexpected={sorted(set(value)-fields)}"
        )
    parsed: dict[str, str] = {}
    for field in fields:
        item = value[field]
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}.{field} must be a non-empty string")
        parsed[field] = item
    return FactorSpec(
        factor=parsed["factor"],
        relation=parsed["relation"],
        applied_criterion=parsed["applied_criterion"],
        not_applied_criterion=parsed["not_applied_criterion"],
    )


def load_trajectory_groups(
    trajectory_file: Path,
    query_file: Path,
    *,
    architecture: FactorMonitorArchitecture | None = None,
    taxonomy: Taxonomy | None = None,
    allow_unlabelled: bool = False,
) -> list[TrajectoryGroup]:
    """Join one frozen query split to its trajectory store and validate Q/I identity.

    Training callers validate against a frozen architecture. Pure data operations
    may instead validate against the repository taxonomy so that materialization
    is not coupled to a particular training backend's family whitelist.
    """

    if (architecture is None) == (taxonomy is None):
        raise ValueError("provide exactly one of architecture or taxonomy")

    trajectories: dict[str, tuple[str, str]] = {}
    for row in iter_jsonl(trajectory_file):
        trajectory_id = _text(row.get("trajectory_id"), "trajectory.trajectory_id")
        if trajectory_id in trajectories:
            raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
        trajectories[trajectory_id] = (
            _text(row.get("task_context"), f"trajectory {trajectory_id}.task_context"),
            _text(row.get("model_cot"), f"trajectory {trajectory_id}.model_cot"),
        )

    grouped: dict[str, list[FactorQueryRecord]] = defaultdict(list)
    seen_queries: set[str] = set()
    for row in iter_jsonl(query_file):
        query_id = _text(row.get("query_id"), "query.query_id")
        if query_id in seen_queries:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen_queries.add(query_id)
        trajectory_id = _text(row.get("trajectory_id"), f"query {query_id}.trajectory_id")
        if trajectory_id not in trajectories:
            raise ValueError(f"query {query_id} references missing trajectory {trajectory_id}")
        family = _text(row.get("family"), f"query {query_id}.family")
        level = _text(row.get("factor_level"), f"query {query_id}.factor_level")
        allowed_levels = architecture.factor_levels if architecture is not None else LEVELS
        if level not in allowed_levels:
            raise ValueError(f"query {query_id} has invalid factor level {level!r}")
        label = row.get("label_id")
        if label is None and allow_unlabelled:
            parsed_label = None
        elif isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"query {query_id}.label_id must be integer 0 or 1")
        else:
            parsed_label = int(label)
        factor = _lossless_factor_spec(
            row.get("factor_spec"), f"query {query_id}.factor_spec"
        )
        if architecture is not None:
            if family not in architecture.phase1_families:
                raise ValueError(f"query {query_id} uses non-Phase-1 family {family!r}")
            if factor.relation not in architecture.relations:
                raise ValueError(
                    f"query {query_id} has invalid relation {factor.relation!r}"
                )
            expected_relation = PHASE1_FAMILY_RELATIONS[family]
            relation_owner = f"Phase-1 family {family!r}"
        else:
            assert taxonomy is not None
            cell = _text(row.get("cell"), f"query {query_id}.cell")
            expected_relation = taxonomy.expected_relation(cell, family)
            relation_owner = f"taxonomy family {cell!r}/{family!r}"
        if factor.relation != expected_relation:
            raise ValueError(
                f"query {query_id} relation {factor.relation!r} disagrees with "
                f"{relation_owner}: expected {expected_relation!r}"
            )
        grouped[trajectory_id].append(
            FactorQueryRecord(
                query_id=query_id,
                trajectory_id=trajectory_id,
                factor_instance_id=(
                    query_id if allow_unlabelled and label is None else _factor_instance_id(row)
                ),
                factor_level=level,
                family=family,
                factor_spec=factor,
                label=parsed_label,
                leakage_group_id=_text(
                    row.get("leakage_group_id"), f"query {query_id}.leakage_group_id"
                ),
                negative_type=_negative_subtype(row),
                loss_weight=_loss_weight(row),
                factor_scope=_factor_scope(row),
                contrast_group_id=_canonical_contrast_group_id(row),
            )
        )

    groups: list[TrajectoryGroup] = []
    for trajectory_id in sorted(grouped):
        queries = sorted(grouped[trajectory_id], key=lambda item: item.query_id)
        families = {query.family for query in queries}
        leakage_groups = {query.leakage_group_id for query in queries}
        if len(families) != 1:
            raise ValueError(
                f"trajectory {trajectory_id} crosses families; family-uniform sampling is undefined"
            )
        if len(leakage_groups) != 1:
            raise ValueError(f"trajectory {trajectory_id} crosses leakage groups")
        levels_by_instance: dict[str, set[str]] = defaultdict(set)
        labels_by_instance: dict[str, set[int | None]] = defaultdict(set)
        scopes_by_instance: dict[str, set[str | None]] = defaultdict(set)
        for query in queries:
            if query.factor_level in levels_by_instance[query.factor_instance_id]:
                raise ValueError(
                    f"trajectory {trajectory_id} instance {query.factor_instance_id} "
                    f"duplicates level {query.factor_level}"
                )
            levels_by_instance[query.factor_instance_id].add(query.factor_level)
            labels_by_instance[query.factor_instance_id].add(query.label)
            scopes_by_instance[query.factor_instance_id].add(query.factor_scope)
        disagreeing = [key for key, labels in labels_by_instance.items() if len(labels) != 1]
        if disagreeing:
            raise ValueError(
                f"trajectory {trajectory_id} has L1/L3 label disagreement for {disagreeing}"
            )
        scope_disagreement = [
            key for key, scopes in scopes_by_instance.items() if len(scopes) != 1
        ]
        if scope_disagreement:
            raise ValueError(
                f"trajectory {trajectory_id} has L1/L3 factor_scope disagreement for "
                f"{scope_disagreement}"
            )
        task_context, model_cot = trajectories[trajectory_id]
        groups.append(
            TrajectoryGroup(
                trajectory_id=trajectory_id,
                task_context=task_context,
                model_cot=model_cot,
                family=next(iter(families)),
                leakage_group_id=next(iter(leakage_groups)),
                queries=tuple(queries),
            )
        )
    if not groups:
        raise ValueError(f"query split is empty: {query_file}")
    return groups


def build_factor_instance_contrast_units(
    groups: Sequence[TrajectoryGroup],
    *,
    co_schedule_complete_contrasts: bool = True,
) -> tuple[list[FactorInstanceContrastUnit], dict[str, Any]]:
    """Build trajectory units and audit canonical Factor-pair metadata.

    A canonical group is complete only when it contains exactly two Factor
    instances with opposite labels. Incomplete groups left by the no-truncation
    partition are deliberately downgraded to singleton trajectory units. Training
    may co-schedule complete pairs, while validation can retain one source
    trajectory per unit without discarding the canonical-pair audit.
    """

    if not groups:
        raise ValueError("contrast-unit construction requires trajectories")
    trajectory_index = {
        group.trajectory_id: index for index, group in enumerate(groups)
    }
    if len(trajectory_index) != len(groups):
        raise ValueError("trajectory IDs must be unique before contrast grouping")
    members: dict[str, set[tuple[str, str, int | None, str]]] = defaultdict(set)
    for group in groups:
        for query in group.queries:
            if query.contrast_group_id is not None:
                members[query.contrast_group_id].add(
                    (
                        query.trajectory_id,
                        query.factor_instance_id,
                        query.label,
                        query.family,
                    )
                )

    complete: dict[str, set[str]] = {}
    incomplete: dict[str, str] = {}
    for contrast_id, records in sorted(members.items()):
        factor_instances = {(trajectory, instance) for trajectory, instance, _, _ in records}
        labels = {label for _, _, label, _ in records}
        families = {family for _, _, _, family in records}
        if len(families) != 1:
            raise ValueError(f"contrast group {contrast_id} crosses families")
        if len(factor_instances) == 2 and labels == {0, 1}:
            complete[contrast_id] = {trajectory for trajectory, _, _, _ in records}
        else:
            incomplete[contrast_id] = (
                f"factor_instances={len(factor_instances)},labels={sorted(labels, key=str)}"
            )

    parents = list(range(len(groups)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    if co_schedule_complete_contrasts:
        for trajectories in complete.values():
            ordered = sorted(trajectories)
            for trajectory in ordered[1:]:
                union(trajectory_index[ordered[0]], trajectory_index[trajectory])

    components: dict[int, list[TrajectoryGroup]] = defaultdict(list)
    for index, group in enumerate(groups):
        components[root(index)].append(group)

    units: list[FactorInstanceContrastUnit] = []
    for component in sorted(
        components.values(), key=lambda selected: min(group.trajectory_id for group in selected)
    ):
        ordered_groups = tuple(sorted(component, key=lambda group: group.trajectory_id))
        if len(ordered_groups) > 2:
            raise ValueError(
                "canonical contrast component exceeds the frozen two-trajectory unit limit: "
                f"{[group.trajectory_id for group in ordered_groups]}"
            )
        families = {group.family for group in ordered_groups}
        if len(families) != 1:
            raise ValueError("one contrast training unit may not cross families")
        component_trajectories = {group.trajectory_id for group in ordered_groups}
        contrast_ids = tuple(
            contrast_id
            for contrast_id, trajectories in sorted(complete.items())
            if trajectories <= component_trajectories
            and any(
                query.contrast_group_id == contrast_id
                for group in ordered_groups
                for query in group.queries
            )
        )
        factor_instances = {
            (group.trajectory_id, query.factor_instance_id)
            for group in ordered_groups
            for query in group.queries
        }
        units.append(
            FactorInstanceContrastUnit(
                unit_id="contrast-unit:" + "+".join(
                    group.trajectory_id for group in ordered_groups
                ),
                family=next(iter(families)),
                groups=ordered_groups,
                factor_instance_count=len(factor_instances),
                contrast_group_ids=contrast_ids,
            )
        )

    audit = {
        "policy": (
            "canonical_complete_pair_else_singleton"
            if co_schedule_complete_contrasts
            else "trajectory_singleton_with_canonical_contrast_audit"
        ),
        "co_schedule_complete_contrasts": co_schedule_complete_contrasts,
        "canonical_metadata_paths": [
            "metadata.factor_contrast.contrast_group_id",
            "metadata.contrast_pair.contrast_group_id",
        ],
        "ignored_metadata_paths": ["metadata.contrast_group_id"],
        "trajectories": len(groups),
        "factor_instances": sum(unit.factor_instance_count for unit in units),
        "training_units": len(units),
        "unit_trajectory_counts": dict(
            sorted(Counter(len(unit.groups) for unit in units).items())
        ),
        "complete_contrast_groups": len(complete),
        "incomplete_contrast_groups": [
            {"contrast_group_id": key, "reason": value}
            for key, value in sorted(incomplete.items())
        ],
        "by_family": {
            family: {
                "training_units": sum(unit.family == family for unit in units),
                "trajectories": sum(
                    len(unit.groups) for unit in units if unit.family == family
                ),
                "factor_instances": sum(
                    unit.factor_instance_count for unit in units if unit.family == family
                ),
            }
            for family in sorted({unit.family for unit in units})
        },
    }
    return units, audit


def build_long_cot_mil_units(
    source_units: Sequence[FactorInstanceContrastUnit],
    window_groups: Sequence[TrajectoryGroup],
) -> list[LongCotMilTrainingUnit]:
    """Attach lossless materialized windows to singleton long source units."""

    by_source: dict[str, list[TrajectoryGroup]] = defaultdict(list)
    for group in window_groups:
        parts = _evidence_window_parts(group.trajectory_id)
        if parts is None:
            raise ValueError("long-CoT MIL received a non-window trajectory")
        by_source[parts[0]].append(group)
    units: list[LongCotMilTrainingUnit] = []
    for source in source_units:
        if len(source.groups) != 1:
            raise ValueError(
                "the frozen long-CoT MIL screen requires singleton Factor-instance units"
            )
        group = source.groups[0]
        windows = tuple(
            sorted(
                by_source.pop(group.trajectory_id, ()),
                key=LongCotMilMiningState._window_index,
            )
        )
        if not windows:
            raise ValueError(f"long source unit has no windows: {group.trajectory_id}")
        source_queries = {query.query_id for query in group.queries}
        for window in windows:
            if {query.source_query_id for query in window.queries} != source_queries:
                raise ValueError("MIL window query membership differs from its source unit")
        units.append(
            LongCotMilTrainingUnit(
                unit_id="long-mil:" + group.trajectory_id,
                family=source.family,
                source_groups=source.groups,
                window_groups=windows,
                factor_instance_count=source.factor_instance_count,
            )
        )
    if by_source:
        raise ValueError(f"orphan MIL window trajectories: {sorted(by_source)[:3]}")
    return units


class TrajectoryGroupDataset(Dataset):
    def __init__(self, groups: Sequence[TrajectoryGroup]) -> None:
        if not groups:
            raise ValueError("TrajectoryGroupDataset requires at least one group")
        self.groups = tuple(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> TrajectoryGroup:
        return self.groups[index]


class FactorInstanceContrastDataset(Dataset):
    def __init__(self, units: Sequence[FactorInstanceContrastUnit]) -> None:
        if not units:
            raise ValueError("FactorInstanceContrastDataset requires training units")
        self.units = tuple(units)

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> FactorInstanceContrastUnit:
        return self.units[index]


class LongCotMilDataset(Dataset):
    """Combined long-MIL and direct short-replay training units."""

    def __init__(
        self,
        long_units: Sequence[LongCotMilTrainingUnit],
        short_units: Sequence[FactorInstanceContrastUnit],
    ) -> None:
        if not long_units:
            raise ValueError("LongCotMilDataset requires long training units")
        if not short_units:
            raise ValueError("LongCotMilDataset requires short replay units")
        self.long_units = tuple(long_units)
        self.short_units = tuple(short_units)
        self.units = (*self.long_units, *self.short_units)
        self.long_indices = tuple(range(len(self.long_units)))
        self.short_indices = tuple(range(len(self.long_units), len(self.units)))

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(
        self, index: int | LongCotMilScheduledIndex
    ) -> LongCotMilTrainingUnit | FactorInstanceContrastUnit | LongCotMilScheduledUnit:
        if isinstance(index, LongCotMilScheduledIndex):
            if index.index < 0 or index.index >= len(self.units):
                raise IndexError(f"MIL scheduled index is out of range: {index.index}")
            return LongCotMilScheduledUnit(
                unit=self.units[index.index],
                is_sync_padding=index.is_sync_padding,
            )
        return self.units[index]


def evidence_token_lengths(tokenizer: Any, groups: Sequence[TrajectoryGroup]) -> list[int]:
    return [
        len(tokenize_rendered(tokenizer, render_evidence(
            task_context=group.task_context, model_cot=group.model_cot
        )).input_ids)
        for group in groups
    ]


def factor_token_lengths(
    tokenizer: Any, groups: Sequence[TrajectoryGroup]
) -> list[tuple[str, int]]:
    return [
        (
            query.query_id,
            len(tokenize_rendered(tokenizer, render_factor(query.factor_spec)).input_ids),
        )
        for group in groups
        for query in group.queries
    ]


def endpoint_token_lengths(
    tokenizer: Any,
    groups: Sequence[TrajectoryGroup],
    *,
    contract_id: str = ENDPOINT_CONTRACT_ID,
) -> list[tuple[str, int]]:
    """Return exact M0 joint-serialization lengths for every Factor query."""

    results: list[tuple[str, int]] = []
    pending: list[tuple[str, tuple[Any, ...]]] = []

    def flush() -> None:
        if not pending:
            return
        flattened = [rendered for _, branches in pending for rendered in branches]
        lengths = _rendered_token_lengths(tokenizer, flattened)
        offset = 0
        for query_id, branches in pending:
            branch_count = len(branches)
            results.append((query_id, max(lengths[offset : offset + branch_count])))
            offset += branch_count
        pending.clear()

    for group in groups:
        for query in group.queries:
            pending.append(
                (
                    query.query_id,
                    render_endpoint_inputs(
                        query.factor_spec,
                        task_context=group.task_context,
                        model_cot=group.model_cot,
                        contract_id=contract_id,
                        omit_factor=query.intervention == "no_factor",
                    ),
                )
            )
            if len(pending) == 64:
                flush()
    flush()
    return results


def causal_choice_token_lengths(
    tokenizer: Any,
    groups: Sequence[TrajectoryGroup],
    *,
    chat_template_kwargs: Mapping[str, Any],
) -> list[tuple[str, int]]:
    """Return exact chat prompt plus one answer-token lengths for every query."""

    instruction = factor_suffix_choice_contract()["instruction"]
    return [
        (
            query.query_id,
            len(
                tokenize_factor_suffix_choice_prompt(
                    tokenizer,
                    render_factor_suffix_choice_text(group, query, instruction),
                    chat_template_kwargs,
                )
            )
            + 1,
        )
        for group in groups
        for query in group.queries
    ]


def _rendered_token_lengths(tokenizer: Any, rendered: Sequence[Any]) -> list[int]:
    """Batch exact fast-tokenizer lengths, with strict renderer fallback for tests."""

    if not rendered:
        return []
    # Keep the batched fast path on the same tokenizer-facing text contract as
    # tokenize_rendered(), including the Llama boundary-safe structural spaces.
    prepared = [rendered_for_tokenizer(tokenizer, item) for item in rendered]
    if any(
        item.contract_id == FACTOR_SUFFIX_RELAY_ENDPOINT_CONTRACT_ID
        for item in prepared
    ):
        if not (
            getattr(tokenizer, "is_fast", False)
            and hasattr(tokenizer, "backend_tokenizer")
        ):
            return [len(tokenize_rendered(tokenizer, item).input_ids) for item in rendered]
        factor_suffix_relay_token_id(tokenizer)
        encoded = tokenizer(
            [item.text for item in prepared],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        ids_batch = encoded.get("input_ids")
        offsets_batch = encoded.get("offset_mapping")
        attention_batch = encoded.get("attention_mask")
        if not all(
            isinstance(values, list) and len(values) == len(prepared)
            for values in (ids_batch, offsets_batch, attention_batch)
        ):
            raise ValueError("batched relay tokenizer returned invalid collections")
        results: list[int] = []
        for item, ids, raw_offsets, attention in zip(
            prepared, ids_batch, offsets_batch, attention_batch, strict=True
        ):
            offsets = [(int(start), int(end)) for start, end in raw_offsets]
            attention_values = [int(value) for value in attention]
            if len(ids) != len(offsets) or len(ids) != len(attention_values):
                raise ValueError("batched relay tokenizer lengths differ")
            results.append(
                len(ids)
                + factor_suffix_relay_insertion_count(
                    item, offsets, attention_values
                )
            )
        return results
    if getattr(tokenizer, "is_fast", False) and hasattr(tokenizer, "backend_tokenizer"):
        encoded = tokenizer(
            [item.text for item in prepared],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_offsets_mapping=False,
        )
        values = encoded.get("input_ids")
        if not isinstance(values, list) or len(values) != len(prepared):
            raise ValueError("batched tokenizer returned an invalid input_ids collection")
        return [len(value) for value in values]
    return [len(tokenize_rendered(tokenizer, item).input_ids) for item in rendered]


def factor_intervention_groups(
    groups: Sequence[TrajectoryGroup],
    tokenizer: Any,
    *,
    variants: Sequence[str],
    seed: int,
    mixed_labels_only: bool,
) -> tuple[list[TrajectoryGroup], dict[str, Any]]:
    """Build deterministic in-memory matched/ablation/permutation variants."""

    requested = tuple(variants)
    supported = {"matched", "shuffled", "constant", "no_factor"}
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("factor intervention variants must be unique and non-empty")
    unknown = set(requested) - supported
    if unknown:
        raise ValueError(f"unknown factor intervention variants: {sorted(unknown)}")
    selected = [
        group
        for group in groups
        if not mixed_labels_only or {query.label for query in group.queries} == {0, 1}
    ]
    if not selected:
        raise ValueError("factor intervention selection contains no trajectories")
    by_stratum: dict[tuple[str, str, str], list[FactorQueryRecord]] = defaultdict(list)
    for group in selected:
        for query in group.queries:
            by_stratum[
                (query.family, query.factor_level, query.factor_spec.relation)
            ].append(query)
    constant_donors: dict[tuple[str, str, str], FactorQueryRecord] = {}
    shuffled_donors: dict[str, FactorQueryRecord] = {}
    # A formal permutation must change the trajectory donor.  A tiny source
    # stratum can make that mathematically impossible (the overlength
    # partition may contain one trajectory only).  Keep the complete variant
    # grid for coverage/audit purposes, but mark those identity rows so the
    # reviewer-facing permutation metric can exclude them instead of silently
    # treating an unchanged Factor as a valid shuffle.
    invalid_shuffled: set[str] = set()
    invalid_strata: list[dict[str, Any]] = []
    randomizer = random.Random(seed)
    for stratum, queries in sorted(by_stratum.items()):
        lengths = {
            query.query_id: len(
                tokenize_rendered(tokenizer, render_factor(query.factor_spec)).input_ids
            )
            for query in queries
        }
        ordered = sorted(queries, key=lambda query: (lengths[query.query_id], query.query_id))
        constant_donors[stratum] = ordered[(len(ordered) - 1) // 2]
        # Preserve the exact Factor token length during permutation.  The
        # long-CoT evaluator reuses matched window boundaries for every
        # intervention; allowing a longer donor Factor would otherwise make
        # an otherwise-valid 32k window exceed the endpoint budget.
        by_factor_length: dict[int, list[FactorQueryRecord]] = defaultdict(list)
        for query in queries:
            by_factor_length[lengths[query.query_id]].append(query)
        for factor_length, length_queries in sorted(by_factor_length.items()):
            recipients = sorted(length_queries, key=lambda query: query.query_id)
            donor_order = list(recipients)
            matched_derangement = None
            for _ in range(1000):
                randomizer.shuffle(donor_order)
                if all(
                    recipient.trajectory_id != donor.trajectory_id
                    for recipient, donor in zip(recipients, donor_order, strict=True)
                ):
                    matched_derangement = tuple(donor_order)
                    break
            if matched_derangement is None:
                for recipient in recipients:
                    shuffled_donors[recipient.query_id] = recipient
                    invalid_shuffled.add(recipient.query_id)
                invalid_strata.append(
                    {
                        "family": stratum[0],
                        "factor_level": stratum[1],
                        "relation": stratum[2],
                        "factor_token_length": factor_length,
                        "query_count": len(recipients),
                        "trajectory_count": len(
                            {query.trajectory_id for query in recipients}
                        ),
                        "reason": (
                            "insufficient_distinct_trajectories_for_"
                            "length_preserving_derangement"
                        ),
                    }
                )
                continue
            for recipient, donor in zip(recipients, matched_derangement, strict=True):
                shuffled_donors[recipient.query_id] = donor

    transformed: list[TrajectoryGroup] = []
    manifest_rows: list[dict[str, Any]] = []
    for group in selected:
        records: list[FactorQueryRecord] = []
        for query in group.queries:
            stratum = (query.family, query.factor_level, query.factor_spec.relation)
            for variant in requested:
                donor = query
                if variant == "shuffled":
                    donor = shuffled_donors[query.query_id]
                elif variant == "constant":
                    donor = constant_donors[stratum]
                diagnostic_id = f"{query.query_id}::factor={variant}"
                records.append(
                    FactorQueryRecord(
                        query_id=diagnostic_id,
                        trajectory_id=query.trajectory_id,
                        factor_instance_id=f"{query.factor_instance_id}::factor={variant}",
                        factor_level=query.factor_level,
                        family=query.family,
                        factor_spec=donor.factor_spec,
                        label=query.label,
                        leakage_group_id=query.leakage_group_id,
                        negative_type=query.negative_type,
                        factor_scope=query.factor_scope,
                        intervention=variant,
                        source_query_id=query.query_id,
                        donor_query_id=donor.query_id,
                    )
                )
                manifest_rows.append(
                    {
                        "query_id": diagnostic_id,
                        "source_query_id": query.query_id,
                        "donor_query_id": donor.query_id,
                        "trajectory_id": query.trajectory_id,
                        "donor_trajectory_id": donor.trajectory_id,
                        "family": query.family,
                        "factor_level": query.factor_level,
                        "relation": query.factor_spec.relation,
                        "label_id": query.label,
                        "intervention": variant,
                        "shuffle_valid": not (
                            variant == "shuffled" and query.query_id in invalid_shuffled
                        ),
                    }
                )
        transformed.append(
            TrajectoryGroup(
                trajectory_id=group.trajectory_id,
                task_context=group.task_context,
                model_cot=group.model_cot,
                family=group.family,
                leakage_group_id=group.leakage_group_id,
                queries=tuple(records),
            )
        )
    return transformed, {
        "schema_version": 1,
        "seed": seed,
        "mixed_labels_only": mixed_labels_only,
        "shuffle_stratum": [
            "family",
            "factor_level",
            "relation",
            "factor_token_length",
        ],
        "variants": list(requested),
        "trajectories": len(transformed),
        "source_queries": sum(len(group.queries) for group in selected),
        "diagnostic_queries": len(manifest_rows),
        "invalid_shuffled_queries": len(invalid_shuffled),
        "invalid_shuffled_strata": invalid_strata,
        "rows": manifest_rows,
    }


def _length_statistics(values: Sequence[int], thresholds: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("length statistics require at least one value")
    ordered = sorted(int(value) for value in values)

    def nearest_rank(percentile: int) -> int:
        index = max(0, ((percentile * len(ordered) + 99) // 100) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "p50": nearest_rank(50),
        "p90": nearest_rank(90),
        "p95": nearest_rank(95),
        "p99": nearest_rank(99),
        "max": ordered[-1],
        "above": {
            str(threshold): {
                "count": sum(value > threshold for value in ordered),
                "ratio": sum(value > threshold for value in ordered) / len(ordered),
            }
            for threshold in thresholds
        },
    }


def _token_length_rows(
    tokenizer: Any,
    groups_by_split: Mapping[str, Sequence[TrajectoryGroup]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render each trajectory and Factor exactly once for length-based decisions."""

    trajectory_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for split, groups in groups_by_split.items():
        for group in groups:
            encoded_evidence = tokenize_rendered(
                tokenizer,
                render_evidence(task_context=group.task_context, model_cot=group.model_cot),
            )
            trajectory_length = {
                "context_tokens": sum(encoded_evidence.span_masks["context"]),
                "cot_tokens": sum(encoded_evidence.span_masks["cot"]),
                "evidence_tokens": len(encoded_evidence.input_ids),
            }
            trajectory_rows.append(
                {
                    "split": split,
                    "family": group.family,
                    "trajectory_id": group.trajectory_id,
                    **trajectory_length,
                }
            )
            for query in group.queries:
                encoded_factor = tokenize_rendered(tokenizer, render_factor(query.factor_spec))
                query_rows.append(
                    {
                        "split": split,
                        "family": query.family,
                        "label": "applied" if query.label == 1 else "not_applied",
                        "factor_level": query.factor_level,
                        "negative_subtype": (
                            "applied" if query.label == 1 else query.negative_type or "unspecified_negative"
                        ),
                        "query_id": query.query_id,
                        "trajectory_id": query.trajectory_id,
                        "factor_instance_id": query.factor_instance_id,
                        **trajectory_length,
                        "factor_tokens": len(encoded_factor.input_ids),
                    }
                )
    return trajectory_rows, query_rows


def token_limit_factor_instance_partition(
    tokenizer: Any,
    groups_by_split: Mapping[str, Sequence[TrajectoryGroup]],
    *,
    max_evidence_tokens: int,
    max_factor_tokens: int,
) -> dict[str, Any]:
    """Select whole Factor instances under frozen no-truncation token limits."""

    if max_evidence_tokens <= 0 or max_factor_tokens <= 0:
        raise ValueError("token limits must be positive")
    _, query_rows = _token_length_rows(tokenizer, groups_by_split)
    instances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        instances[row["factor_instance_id"]].append(row)

    accepted_ids = {
        factor_instance_id
        for factor_instance_id, rows in instances.items()
        if max(int(row["evidence_tokens"]) for row in rows) <= max_evidence_tokens
        and max(int(row["factor_tokens"]) for row in rows) <= max_factor_tokens
    }
    overlength_ids = set(instances) - accepted_ids
    accepted_rows = [
        row for row in query_rows if row["factor_instance_id"] in accepted_ids
    ]
    overlength_rows = [
        row for row in query_rows if row["factor_instance_id"] in overlength_ids
    ]

    def counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(rows),
            "factor_instances": len({row["factor_instance_id"] for row in rows}),
            "trajectories": len({row["trajectory_id"] for row in rows}),
            "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
            "by_family": dict(sorted(Counter(row["family"] for row in rows).items())),
            "by_label": dict(sorted(Counter(row["label"] for row in rows).items())),
            "by_factor_level": dict(
                sorted(Counter(row["factor_level"] for row in rows).items())
            ),
        }

    return {
        "schema_version": 1,
        "policy": "drop_complete_factor_instance_no_truncation",
        "max_evidence_tokens": max_evidence_tokens,
        "max_factor_tokens": max_factor_tokens,
        "accepted_factor_instance_ids": sorted(accepted_ids),
        "overlength_factor_instance_ids": sorted(overlength_ids),
        "accepted": counts(accepted_rows),
        "overlength": counts(overlength_rows),
    }


def token_length_audit(
    tokenizer: Any,
    groups_by_split: Mapping[str, Sequence[TrajectoryGroup]],
    *,
    evidence_thresholds: Sequence[int],
    factor_limits: Sequence[int],
) -> dict[str, Any]:
    """Audit rendered lengths and candidate fail-closed filters without truncation."""

    evidence_thresholds = tuple(sorted(set(int(value) for value in evidence_thresholds)))
    factor_limits = tuple(sorted(set(int(value) for value in factor_limits)))
    if not evidence_thresholds or any(value <= 0 for value in evidence_thresholds):
        raise ValueError("evidence_thresholds must contain positive integers")
    if not factor_limits or any(value <= 0 for value in factor_limits):
        raise ValueError("factor_limits must contain positive integers")

    trajectory_rows, query_rows = _token_length_rows(tokenizer, groups_by_split)

    metrics = ("context_tokens", "cot_tokens", "evidence_tokens", "factor_tokens")

    def grouped_statistics(
        rows: Sequence[dict[str, Any]], dimensions: tuple[str, ...], selected_metrics: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(str(row[dimension]) for dimension in dimensions)].append(row)
        result = []
        for key in sorted(grouped):
            selected = grouped[key]
            result.append(
                {
                    **dict(zip(dimensions, key, strict=True)),
                    "rows": len(selected),
                    "trajectories": len({row["trajectory_id"] for row in selected}),
                    **{
                        metric: _length_statistics(
                            [int(row[metric]) for row in selected],
                            evidence_thresholds if metric != "factor_tokens" else factor_limits,
                        )
                        for metric in selected_metrics
                    },
                }
            )
        return result

    instances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        instances[row["factor_instance_id"]].append(row)
    filter_scenarios = []
    for evidence_limit in evidence_thresholds:
        for factor_limit in factor_limits:
            kept_instances = {
                factor_instance_id
                for factor_instance_id, rows in instances.items()
                if max(int(row["evidence_tokens"]) for row in rows) <= evidence_limit
                and max(int(row["factor_tokens"]) for row in rows) <= factor_limit
            }
            kept = [row for row in query_rows if row["factor_instance_id"] in kept_instances]
            filter_scenarios.append(
                {
                    "max_evidence_tokens": evidence_limit,
                    "max_factor_tokens": factor_limit,
                    "policy": "drop_complete_factor_instance_no_truncation",
                    "remaining_factor_instances": len(kept_instances),
                    "dropped_factor_instances": len(instances) - len(kept_instances),
                    "remaining_queries": len(kept),
                    "dropped_queries": len(query_rows) - len(kept),
                    "remaining_by_family": dict(sorted(Counter(row["family"] for row in kept).items())),
                    "remaining_by_split": dict(sorted(Counter(row["split"] for row in kept).items())),
                    "remaining_by_label": dict(sorted(Counter(row["label"] for row in kept).items())),
                    "remaining_by_factor_level": dict(
                        sorted(Counter(row["factor_level"] for row in kept).items())
                    ),
                    "remaining_by_negative_subtype": dict(
                        sorted(Counter(row["negative_subtype"] for row in kept).items())
                    ),
                    "remaining_by_family_split_label": [
                        {
                            **dict(zip(("family", "split", "label"), key, strict=True)),
                            "queries": count,
                        }
                        for key, count in sorted(
                            Counter(
                                (row["family"], row["split"], row["label"])
                                for row in kept
                            ).items()
                        )
                    ],
                }
            )

    return {
        "schema_version": 1,
        "unit_policy": {
            "trajectory_lengths": "one rendered evidence input per trajectory",
            "label_conditioned_lengths": "evidence repeated per Factor query for label slicing",
            "percentiles": "nearest_rank",
            "threshold_comparison": "strictly_greater_than",
            "filtering": "candidate scenarios only; no artifact rows are removed",
        },
        "evidence_thresholds": list(evidence_thresholds),
        "factor_limits": list(factor_limits),
        "totals": {
            "trajectories": len(trajectory_rows),
            "queries": len(query_rows),
            "factor_instances": len(instances),
        },
        "trajectory_lengths_by_split_family": grouped_statistics(
            trajectory_rows,
            ("split", "family"),
            ("context_tokens", "cot_tokens", "evidence_tokens"),
        ),
        "query_lengths_by_split_family_label": grouped_statistics(
            query_rows,
            ("split", "family", "label"),
            metrics,
        ),
        "query_lengths_by_split_family_label_level": grouped_statistics(
            query_rows,
            ("split", "family", "label", "factor_level"),
            metrics,
        ),
        "negative_subtypes": dict(
            sorted(Counter(row["negative_subtype"] for row in query_rows).items())
        ),
        "filter_scenarios": filter_scenarios,
    }


def endpoint_token_length_audit(
    tokenizer: Any,
    groups_by_split: Mapping[str, Sequence[TrajectoryGroup]],
    *,
    sequence_thresholds: Sequence[int],
    contract_id: str = ENDPOINT_CONTRACT_ID,
) -> dict[str, Any]:
    """Audit exact M0 joint Factor/Relation/Context/CoT sequence lengths."""

    thresholds = tuple(sorted(set(int(value) for value in sequence_thresholds)))
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("sequence_thresholds must contain positive integers")
    rows: list[dict[str, Any]] = []
    for split, groups in groups_by_split.items():
        pending: list[tuple[TrajectoryGroup, FactorQueryRecord, tuple[Any, ...]]] = []

        def flush() -> None:
            if not pending:
                return
            flattened = [rendered for _, _, branches in pending for rendered in branches]
            lengths = _rendered_token_lengths(tokenizer, flattened)
            offset = 0
            for group, query, branches in pending:
                branch_count = len(branches)
                branch_lengths = lengths[offset : offset + branch_count]
                offset += branch_count
                rows.append(
                    {
                        "split": split,
                        "family": query.family,
                        "factor_level": query.factor_level,
                        "label": "applied" if query.label == 1 else "not_applied",
                        "query_id": query.query_id,
                        "trajectory_id": query.trajectory_id,
                        "tokens": max(branch_lengths),
                        "branch_tokens": branch_lengths,
                    }
                )
            pending.clear()

        for group in groups:
            for query in group.queries:
                pending.append(
                    (
                        group,
                        query,
                        render_endpoint_inputs(
                            query.factor_spec,
                            task_context=group.task_context,
                            model_cot=group.model_cot,
                            contract_id=contract_id,
                            omit_factor=query.intervention == "no_factor",
                        ),
                    )
                )
                if len(pending) == 64:
                    flush()
        flush()
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["family"])].append(int(row["tokens"]))
    report = {
        "schema_version": 1,
        "encoding": contract_id,
        "truncation": "none",
        "thresholds": list(thresholds),
        "totals": {
            "queries": len(rows),
            "trajectories": len({row["trajectory_id"] for row in rows}),
            "lengths": _length_statistics(
                [int(row["tokens"]) for row in rows], thresholds
            ),
        },
        "by_split_family": [
            {
                "split": split,
                "family": family,
                "lengths": _length_statistics(values, thresholds),
            }
            for (split, family), values in sorted(grouped.items())
        ],
        "candidate_limits": [
            {
                "max_sequence_length": threshold,
                "remaining_queries": sum(row["tokens"] <= threshold for row in rows),
                "dropped_queries": sum(row["tokens"] > threshold for row in rows),
            }
            for threshold in thresholds
        ],
    }
    if contract_id == CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID:
        applied_lengths = [int(row["branch_tokens"][0]) for row in rows]
        not_applied_lengths = [int(row["branch_tokens"][1]) for row in rows]
        report["totals"]["branch_sequences"] = len(rows) * 2
        report["totals"]["branch_lengths"] = {
            "applied": _length_statistics(applied_lengths, thresholds),
            "not_applied": _length_statistics(not_applied_lengths, thresholds),
        }
    return report


def causal_choice_token_length_audit(
    tokenizer: Any,
    groups_by_split: Mapping[str, Sequence[TrajectoryGroup]],
    *,
    sequence_thresholds: Sequence[int],
    chat_template_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit exact chat-formatted prompt plus one A/B answer token."""

    thresholds = tuple(sorted(set(int(value) for value in sequence_thresholds)))
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("sequence_thresholds must contain positive integers")
    rows: list[dict[str, Any]] = []
    for split, groups in groups_by_split.items():
        by_id = dict(
            causal_choice_token_lengths(
                tokenizer,
                groups,
                chat_template_kwargs=chat_template_kwargs,
            )
        )
        for group in groups:
            for query in group.queries:
                rows.append(
                    {
                        "split": split,
                        "family": query.family,
                        "factor_level": query.factor_level,
                        "label": "applied" if query.label == 1 else "not_applied",
                        "query_id": query.query_id,
                        "trajectory_id": query.trajectory_id,
                        "tokens": by_id[query.query_id],
                    }
                )
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["family"])].append(int(row["tokens"]))
    return {
        "schema_version": 1,
        "encoding": FACTOR_SUFFIX_CHOICE_CONTRACT_ID,
        "endpoint_encoding": FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID,
        "chat_template": "tokenizer_native",
        "chat_template_kwargs": dict(chat_template_kwargs),
        "answer_tokens": 1,
        "length_semantics": "prompt_plus_answer_token",
        "truncation": "none",
        "thresholds": list(thresholds),
        "totals": {
            "queries": len(rows),
            "trajectories": len({row["trajectory_id"] for row in rows}),
            "lengths": _length_statistics(
                [int(row["tokens"]) for row in rows], thresholds
            ),
        },
        "by_split_family": [
            {
                "split": split,
                "family": family,
                "lengths": _length_statistics(values, thresholds),
            }
            for (split, family), values in sorted(grouped.items())
        ],
        "candidate_limits": [
            {
                "max_sequence_length": threshold,
                "remaining_queries": sum(row["tokens"] <= threshold for row in rows),
                "dropped_queries": sum(row["tokens"] > threshold for row in rows),
            }
            for threshold in thresholds
        ],
    }


class TrajectoryCollator:
    """Dynamically right-pad a microbatch while preserving one evidence row per trajectory."""

    def __init__(
        self,
        tokenizer: Any,
        architecture: FactorMonitorArchitecture,
        *,
        max_evidence_tokens: int,
        max_factor_tokens: int,
        max_queries_per_trajectory: int | None,
        padding_side: str,
        pad_queries_to: int | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("TrajectoryCollator requires torch")
        if padding_side != "right":
            raise ValueError("Phase 1 collator currently requires explicit right padding")
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_id, int) or pad_id < 0:
            raise ValueError("tokenizer must define a nonnegative pad_token_id")
        if max_evidence_tokens <= 0 or max_factor_tokens <= 0:
            raise ValueError("token limits must be positive")
        if pad_queries_to is not None and pad_queries_to <= 0:
            raise ValueError("pad_queries_to must be positive when provided")
        self.tokenizer = tokenizer
        self.architecture = architecture
        self.max_evidence_tokens = max_evidence_tokens
        self.max_factor_tokens = max_factor_tokens
        self.max_queries_per_trajectory = max_queries_per_trajectory
        self.pad_queries_to = pad_queries_to
        self.pad_id = pad_id
        self.family_to_id = {
            family: index for index, family in enumerate(architecture.phase1_families)
        }
        self.level_to_id = {
            level: index for index, level in enumerate(architecture.factor_levels)
        }

    def __call__(self, groups: Sequence[TrajectoryGroup]) -> dict[str, Any]:
        if not groups:
            raise ValueError("cannot collate an empty microbatch")
        evidence = [
            tokenize_rendered(
                self.tokenizer,
                render_evidence(task_context=group.task_context, model_cot=group.model_cot),
            )
            for group in groups
        ]
        factor_tokens = [
            [tokenize_rendered(self.tokenizer, render_factor(query.factor_spec)) for query in group.queries]
            for group in groups
        ]
        evidence_over = [
            group.trajectory_id
            for group, encoded in zip(groups, evidence, strict=True)
            if len(encoded.input_ids) > self.max_evidence_tokens
        ]
        if evidence_over:
            raise ValueError(
                f"{len(evidence_over)} trajectories exceed max_evidence_tokens and must be "
                f"excluded to Phase 2 before collation: {evidence_over[:3]}"
            )
        factor_over = [
            query.query_id
            for group, encoded_group in zip(groups, factor_tokens, strict=True)
            for query, encoded in zip(group.queries, encoded_group, strict=True)
            if len(encoded.input_ids) > self.max_factor_tokens
        ]
        if factor_over:
            raise ValueError(
                f"{len(factor_over)} Factors exceed max_factor_tokens: {factor_over[:3]}"
            )
        if self.max_queries_per_trajectory is not None:
            excess = [group.trajectory_id for group in groups if len(group.queries) > self.max_queries_per_trajectory]
            if excess:
                raise ValueError(f"trajectories exceed max_queries_per_trajectory: {excess[:3]}")
        if self.pad_queries_to is not None:
            excess = [group.trajectory_id for group in groups if len(group.queries) > self.pad_queries_to]
            if excess:
                raise ValueError(f"trajectories exceed pad_queries_to: {excess[:3]}")

        batch = len(groups)
        max_evidence = max(len(item.input_ids) for item in evidence)
        max_records = self.pad_queries_to or max(len(group.queries) for group in groups)
        max_factor = max(
            len(item.input_ids) for encoded_group in factor_tokens for item in encoded_group
        )
        evidence_ids = torch.full((batch, max_evidence), self.pad_id, dtype=torch.long)
        evidence_attention = torch.zeros((batch, max_evidence), dtype=torch.bool)
        context_mask = torch.zeros((batch, max_evidence), dtype=torch.bool)
        cot_mask = torch.zeros((batch, max_evidence), dtype=torch.bool)
        factor_ids = torch.full((batch, max_records, max_factor), self.pad_id, dtype=torch.long)
        factor_attention = torch.zeros((batch, max_records, max_factor), dtype=torch.bool)
        factor_mask = torch.zeros((batch, max_records, max_factor), dtype=torch.bool)
        applied_mask = torch.zeros((batch, max_records, max_factor), dtype=torch.bool)
        not_applied_mask = torch.zeros((batch, max_records, max_factor), dtype=torch.bool)
        query_mask = torch.zeros((batch, max_records), dtype=torch.bool)
        labels = torch.zeros((batch, max_records), dtype=torch.float32)
        loss_weights = torch.zeros((batch, max_records), dtype=torch.float32)
        instance_indices = torch.full((batch, max_records), -1, dtype=torch.long)
        levels = torch.full((batch, max_records), -1, dtype=torch.long)
        families = torch.full((batch, max_records), -1, dtype=torch.long)
        relations = torch.zeros((batch, max_records), dtype=torch.long)
        routes = torch.zeros((batch, max_records), dtype=torch.long)
        query_ids: list[list[str | None]] = [[None] * max_records for _ in range(batch)]

        for batch_index, (group, evidence_item, factor_group) in enumerate(
            zip(groups, evidence, factor_tokens, strict=True)
        ):
            evidence_length = len(evidence_item.input_ids)
            evidence_ids[batch_index, :evidence_length] = torch.tensor(evidence_item.input_ids)
            evidence_attention[batch_index, :evidence_length] = torch.tensor(
                evidence_item.attention_mask, dtype=torch.bool
            )
            context_mask[batch_index, :evidence_length] = torch.tensor(
                evidence_item.span_masks["context"], dtype=torch.bool
            )
            cot_mask[batch_index, :evidence_length] = torch.tensor(
                evidence_item.span_masks["cot"], dtype=torch.bool
            )
            instance_map: dict[str, int] = {}
            for record_index, (query, factor_item) in enumerate(
                zip(group.queries, factor_group, strict=True)
            ):
                expected_relation = PHASE1_FAMILY_RELATIONS[query.family]
                if query.factor_spec.relation != expected_relation:
                    raise ValueError(
                        f"query {query.query_id} relation disagrees with its Phase-1 family"
                    )
                factor_length = len(factor_item.input_ids)
                factor_ids[batch_index, record_index, :factor_length] = torch.tensor(
                    factor_item.input_ids
                )
                factor_attention[batch_index, record_index, :factor_length] = torch.tensor(
                    factor_item.attention_mask, dtype=torch.bool
                )
                factor_mask[batch_index, record_index, :factor_length] = torch.tensor(
                    factor_item.span_masks["factor"], dtype=torch.bool
                )
                applied_mask[batch_index, record_index, :factor_length] = torch.tensor(
                    factor_item.span_masks["applied"], dtype=torch.bool
                )
                not_applied_mask[batch_index, record_index, :factor_length] = torch.tensor(
                    factor_item.span_masks["not_applied"], dtype=torch.bool
                )
                if query.factor_instance_id not in instance_map:
                    instance_map[query.factor_instance_id] = len(instance_map)
                query_mask[batch_index, record_index] = True
                labels[batch_index, record_index] = (
                    -100.0 if query.label is None else float(query.label)
                )
                loss_weights[batch_index, record_index] = query.loss_weight
                instance_indices[batch_index, record_index] = instance_map[query.factor_instance_id]
                levels[batch_index, record_index] = self.level_to_id[query.factor_level]
                families[batch_index, record_index] = self.family_to_id[query.family]
                relations[batch_index, record_index] = RELATION_TO_ID[query.factor_spec.relation]
                routes[batch_index, record_index] = ROUTE_TO_ID[
                    self.architecture.family_routes[query.family]
                ]
                query_ids[batch_index][record_index] = query.query_id

        return {
            "evidence_input_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention,
            "context_span_mask": context_mask,
            "cot_span_mask": cot_mask,
            "factor_input_ids": factor_ids,
            "factor_attention_mask": factor_attention,
            "factor_span_mask": factor_mask,
            "applied_span_mask": applied_mask,
            "not_applied_span_mask": not_applied_mask,
            "query_mask": query_mask,
            "relation": relations,
            "evidence_route": routes,
            "labels": labels,
            "loss_weight": loss_weights,
            "factor_instance_index": instance_indices,
            "factor_level": levels,
            "family_id": families,
            "trajectory_id": [group.trajectory_id for group in groups],
            "leakage_group_id": [group.leakage_group_id for group in groups],
            "query_id": query_ids,
        }


class EndpointTrajectoryCollator:
    """Collate the shared joint Factor/Evidence serialization for M0 or M1."""

    def __init__(
        self,
        tokenizer: Any,
        architecture: FactorMonitorArchitecture,
        *,
        max_sequence_length: int,
        max_queries_per_trajectory: int | None,
        padding_side: str,
        pad_queries_to: int | None = None,
        output_mode: str = "endpoint",
        endpoint_contract_id: str = ENDPOINT_CONTRACT_ID,
        include_endpoint_span_masks: bool = False,
        choice_contract_id: str | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("EndpointTrajectoryCollator requires torch")
        if padding_side != "right":
            raise ValueError("joint-input collator requires explicit right padding")
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_id, int) or pad_id < 0:
            raise ValueError("tokenizer must define a nonnegative pad_token_id")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if pad_queries_to is not None and pad_queries_to <= 0:
            raise ValueError("pad_queries_to must be positive when provided")
        if output_mode not in {"endpoint", "factor_joint"}:
            raise ValueError("output_mode must be endpoint or factor_joint")
        if (
            endpoint_contract_id == CRITERION_DIFFERENCE_ENDPOINT_CONTRACT_ID
            and output_mode != "endpoint"
        ):
            raise ValueError("criterion-difference contract requires endpoint output mode")
        if include_endpoint_span_masks and (
            output_mode != "endpoint" or endpoint_contract_id != ENDPOINT_CONTRACT_ID
        ):
            raise ValueError(
                "endpoint span masks require the original single-endpoint contract"
            )
        if choice_contract_id is not None:
            if choice_contract_id != FACTOR_SUFFIX_CHOICE_CONTRACT_ID:
                raise ValueError(f"unsupported choice contract: {choice_contract_id!r}")
            if output_mode != "endpoint":
                raise ValueError("causal choice prompts require endpoint output mode")
            if endpoint_contract_id != FACTOR_SUFFIX_ENDPOINT_CONTRACT_ID:
                raise ValueError("causal choice prompts require the Factor-Suffix contract")
            if include_endpoint_span_masks:
                raise ValueError("causal choice prompts do not expose endpoint span masks")
            if chat_template_kwargs is None:
                raise ValueError("causal choice prompts require chat_template_kwargs")
        self.tokenizer = tokenizer
        self.architecture = architecture
        self.max_sequence_length = max_sequence_length
        self.max_queries_per_trajectory = max_queries_per_trajectory
        self.pad_queries_to = pad_queries_to
        self.output_mode = output_mode
        self.endpoint_contract_id = endpoint_contract_id
        self.include_endpoint_span_masks = include_endpoint_span_masks
        self.choice_contract_id = choice_contract_id
        self.chat_template_kwargs = (
            dict(chat_template_kwargs) if chat_template_kwargs is not None else None
        )
        self.choice_token_ids = (
            factor_suffix_choice_token_ids(tokenizer)
            if choice_contract_id is not None
            else None
        )
        self.pad_id = pad_id
        self.family_to_id = {
            family: index for index, family in enumerate(architecture.phase1_families)
        }
        self.level_to_id = {
            level: index for index, level in enumerate(architecture.factor_levels)
        }

    def __call__(self, groups: Sequence[TrajectoryGroup]) -> dict[str, Any]:
        if not groups:
            raise ValueError("cannot collate an empty microbatch")
        if self.choice_contract_id is None:
            encoded = [
                [
                    tuple(
                        tokenize_rendered(self.tokenizer, rendered)
                        for rendered in render_endpoint_inputs(
                            query.factor_spec,
                            task_context=group.task_context,
                            model_cot=group.model_cot,
                            contract_id=self.endpoint_contract_id,
                            omit_factor=query.intervention == "no_factor",
                        )
                    )
                    for query in group.queries
                ]
                for group in groups
            ]
        else:
            instruction = factor_suffix_choice_contract()["instruction"]

            def encode_choice(group: TrajectoryGroup, query: FactorQueryRecord) -> TokenizedInput:
                input_ids = tuple(
                    tokenize_factor_suffix_choice_prompt(
                        self.tokenizer,
                        render_factor_suffix_choice_text(group, query, instruction),
                        self.chat_template_kwargs or {},
                    )
                )
                return TokenizedInput(
                    contract_id=self.choice_contract_id,
                    input_ids=input_ids,
                    attention_mask=(1,) * len(input_ids),
                    span_masks={},
                )

            encoded = [
                [
                    (encode_choice(group, query),)
                    for query in group.queries
                ]
                for group in groups
            ]
        overlength = [
            query.query_id
            for group, encoded_group in zip(groups, encoded, strict=True)
            for query, branches in zip(group.queries, encoded_group, strict=True)
            if any(
                len(item.input_ids)
                + (1 if self.choice_contract_id is not None else 0)
                > self.max_sequence_length
                for item in branches
            )
        ]
        if overlength:
            raise ValueError(
                f"{len(overlength)} joint inputs exceed max_sequence_length; "
                f"full-CoT joint runs never truncate: {overlength[:3]}"
            )
        if self.max_queries_per_trajectory is not None:
            excess = [
                group.trajectory_id
                for group in groups
                if len(group.queries) > self.max_queries_per_trajectory
            ]
            if excess:
                raise ValueError(
                    f"trajectories exceed max_queries_per_trajectory: {excess[:3]}"
                )
        max_records = self.pad_queries_to or max(len(group.queries) for group in groups)
        excess_padding = [
            group.trajectory_id for group in groups if len(group.queries) > max_records
        ]
        if excess_padding:
            raise ValueError(f"trajectories exceed pad_queries_to: {excess_padding[:3]}")
        branch_counts = {
            len(branches) for encoded_group in encoded for branches in encoded_group
        }
        if len(branch_counts) != 1:
            raise RuntimeError("endpoint contract produced inconsistent branch counts")
        branch_count = branch_counts.pop()
        if branch_count not in {1, 2}:
            raise RuntimeError("endpoint contract must produce one or two branches")
        max_tokens = max(
            len(item.input_ids)
            for encoded_group in encoded
            for branches in encoded_group
            for item in branches
        )
        batch = len(groups)
        input_ids = torch.full(
            (batch, max_records, branch_count, max_tokens),
            self.pad_id,
            dtype=torch.long,
        )
        attention = torch.zeros(
            (batch, max_records, branch_count, max_tokens), dtype=torch.bool
        )
        factor_mask = torch.zeros_like(attention)
        relation_mask = torch.zeros_like(attention)
        applied_mask = torch.zeros_like(attention)
        not_applied_mask = torch.zeros_like(attention)
        context_mask = torch.zeros_like(attention)
        cot_mask = torch.zeros_like(attention)
        query_mask = torch.zeros((batch, max_records), dtype=torch.bool)
        labels = torch.zeros((batch, max_records), dtype=torch.float32)
        loss_weights = torch.zeros((batch, max_records), dtype=torch.float32)
        instances = torch.full((batch, max_records), -1, dtype=torch.long)
        levels = torch.full((batch, max_records), -1, dtype=torch.long)
        families = torch.full((batch, max_records), -1, dtype=torch.long)
        relations = torch.zeros((batch, max_records), dtype=torch.long)
        routes = torch.zeros((batch, max_records), dtype=torch.long)
        query_ids: list[list[str | None]] = [[None] * max_records for _ in range(batch)]
        for batch_index, (group, encoded_group) in enumerate(
            zip(groups, encoded, strict=True)
        ):
            instance_map: dict[str, int] = {}
            for record_index, (query, branches) in enumerate(
                zip(group.queries, encoded_group, strict=True)
            ):
                if query.factor_instance_id not in instance_map:
                    instance_map[query.factor_instance_id] = len(instance_map)
                for branch_index, item in enumerate(branches):
                    length = len(item.input_ids)
                    input_ids[batch_index, record_index, branch_index, :length] = torch.tensor(
                        item.input_ids, dtype=torch.long
                    )
                    attention[
                        batch_index, record_index, branch_index, :length
                    ] = torch.tensor(
                        item.attention_mask, dtype=torch.bool
                    )
                    for mask_name, target in (
                        ("factor", factor_mask),
                        ("relation", relation_mask),
                        ("applied", applied_mask),
                        ("not_applied", not_applied_mask),
                        ("context", context_mask),
                        ("cot", cot_mask),
                    ):
                        if mask_name in item.span_masks:
                            target[
                                batch_index, record_index, branch_index, :length
                            ] = torch.tensor(
                                item.span_masks[mask_name], dtype=torch.bool
                            )
                query_mask[batch_index, record_index] = True
                labels[batch_index, record_index] = (
                    -100.0 if query.label is None else float(query.label)
                )
                loss_weights[batch_index, record_index] = query.loss_weight
                instances[batch_index, record_index] = instance_map[
                    query.factor_instance_id
                ]
                levels[batch_index, record_index] = self.level_to_id[query.factor_level]
                families[batch_index, record_index] = self.family_to_id[query.family]
                expected_relation = PHASE1_FAMILY_RELATIONS[query.family]
                if query.factor_spec.relation != expected_relation:
                    raise ValueError(
                        f"query {query.query_id} relation disagrees with its Phase-1 family"
                    )
                relations[batch_index, record_index] = RELATION_TO_ID[
                    query.factor_spec.relation
                ]
                routes[batch_index, record_index] = ROUTE_TO_ID[
                    self.architecture.family_routes[query.family]
                ]
                query_ids[batch_index][record_index] = query.query_id
        role_ids = torch.full_like(input_ids, -1)
        for role_id, role_mask in enumerate(
            (
                factor_mask,
                relation_mask,
                applied_mask,
                not_applied_mask,
                context_mask,
                cot_mask,
            )
        ):
            role_ids.masked_fill_(role_mask, role_id)
        if len(ENDPOINT_ROLE_FIELDS) != 6:
            raise RuntimeError("endpoint role mapping must contain exactly six fields")
        if branch_count == 1:
            input_ids = input_ids.squeeze(2)
            attention = attention.squeeze(2)
            role_ids = role_ids.squeeze(2)
            factor_mask = factor_mask.squeeze(2)
            relation_mask = relation_mask.squeeze(2)
            applied_mask = applied_mask.squeeze(2)
            not_applied_mask = not_applied_mask.squeeze(2)
            context_mask = context_mask.squeeze(2)
            cot_mask = cot_mask.squeeze(2)
        if self.output_mode == "endpoint":
            inputs = {
                "endpoint_input_ids": input_ids,
                "endpoint_attention_mask": attention,
                "endpoint_role_ids": role_ids,
            }
            if self.include_endpoint_span_masks:
                inputs.update(
                    {
                        "factor_span_mask": factor_mask,
                        "relation_span_mask": relation_mask,
                        "applied_span_mask": applied_mask,
                        "not_applied_span_mask": not_applied_mask,
                        "context_span_mask": context_mask,
                        "cot_span_mask": cot_mask,
                    }
                )
        else:
            inputs = {
                "joint_input_ids": input_ids,
                "joint_attention_mask": attention,
                "factor_span_mask": factor_mask,
                "applied_span_mask": applied_mask,
                "not_applied_span_mask": not_applied_mask,
                "context_span_mask": context_mask,
                "cot_span_mask": cot_mask,
                "relation": relations,
            }
        return {
            **inputs,
            "query_mask": query_mask,
            "labels": labels,
            "loss_weight": loss_weights,
            "factor_instance_index": instances,
            "factor_level": levels,
            "family_id": families,
            "evidence_route": routes,
            "trajectory_id": [group.trajectory_id for group in groups],
            "leakage_group_id": [group.leakage_group_id for group in groups],
            "query_id": query_ids,
        }


class FactorInstanceContrastCollator:
    """Flatten co-scheduled units while retaining the selected-unit identity."""

    def __init__(self, trajectory_collator: Any) -> None:
        if torch is None:
            raise RuntimeError("FactorInstanceContrastCollator requires torch")
        self.trajectory_collator = trajectory_collator

    def __call__(
        self, units: Sequence[FactorInstanceContrastUnit]
    ) -> dict[str, Any]:
        if not units:
            raise ValueError("cannot collate an empty contrast-unit microbatch")
        groups: list[TrajectoryGroup] = []
        unit_by_trajectory_row: list[int] = []
        for unit_index, unit in enumerate(units):
            groups.extend(unit.groups)
            unit_by_trajectory_row.extend([unit_index] * len(unit.groups))
        batch = self.trajectory_collator(groups)
        training_unit_index = torch.full_like(batch["factor_instance_index"], -1)
        for row, unit_index in enumerate(unit_by_trajectory_row):
            training_unit_index[row, batch["query_mask"][row].bool()] = unit_index
        batch["training_unit_index"] = training_unit_index
        batch["training_unit_id"] = [unit.unit_id for unit in units]
        return batch


class LongCotMilCollator:
    """Materialize active long windows and preserve their source hierarchy."""

    def __init__(
        self,
        trajectory_collator: Any,
        *,
        mining_state: LongCotMilMiningState | None,
        teacher_logits: Mapping[tuple[str, int], float] | None = None,
        source_query_indices: Mapping[str, int] | None = None,
        full_long_bags: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("LongCotMilCollator requires torch")
        if full_long_bags and mining_state is not None:
            raise ValueError("full validation bags must not use the training miner")
        self.trajectory_collator = trajectory_collator
        self.mining_state = mining_state
        self.teacher_logits = dict(teacher_logits) if teacher_logits is not None else None
        self.source_query_indices = dict(source_query_indices or {})
        self.full_long_bags = full_long_bags

    @staticmethod
    def _source_metadata(
        unit: LongCotMilTrainingUnit | FactorInstanceContrastUnit,
    ) -> dict[str, FactorQueryRecord]:
        groups = unit.source_groups if isinstance(unit, LongCotMilTrainingUnit) else unit.groups
        result: dict[str, FactorQueryRecord] = {}
        for group in groups:
            for query in group.queries:
                if query.query_id in result:
                    raise ValueError(f"duplicate source query in MIL unit: {query.query_id}")
                result[query.query_id] = query
        return result

    def __call__(
        self,
        units: Sequence[
            LongCotMilTrainingUnit
            | FactorInstanceContrastUnit
            | LongCotMilScheduledUnit
        ],
    ) -> dict[str, Any]:
        if not units:
            raise ValueError("cannot collate an empty MIL-unit microbatch")
        scheduled_units = tuple(
            unit
            if isinstance(unit, LongCotMilScheduledUnit)
            else LongCotMilScheduledUnit(unit=unit)
            for unit in units
        )
        materialized: list[TrajectoryGroup] = []
        group_unit_indices: list[int] = []
        source_records: list[dict[str, FactorQueryRecord]] = []
        long_flags: list[bool] = []
        for unit_index, scheduled in enumerate(scheduled_units):
            unit = scheduled.unit
            source_records.append(self._source_metadata(unit))
            is_long = isinstance(unit, LongCotMilTrainingUnit)
            long_flags.append(is_long)
            if is_long:
                if self.full_long_bags:
                    groups = unit.window_groups
                else:
                    if self.mining_state is None:
                        raise ValueError("training long bags require a mining state")
                    groups = self.mining_state.select_groups(
                        unit,
                        record_visit=not scheduled.is_sync_padding,
                    )
            else:
                groups = unit.groups
            materialized.extend(groups)
            group_unit_indices.extend([unit_index] * len(groups))

        batch = self.trajectory_collator(materialized)
        shape = batch["query_mask"].shape
        mil_bag_index = torch.full(shape, -1, dtype=torch.long)
        mil_factor_instance_index = torch.full(shape, -1, dtype=torch.long)
        mil_source_query_index = torch.full(shape, -1, dtype=torch.long)
        mil_window_index = torch.full(shape, -1, dtype=torch.long)
        mil_is_long = torch.full(shape, -1, dtype=torch.long)
        mil_teacher_logit = (
            torch.full(shape, float("nan"), dtype=torch.float32)
            if self.teacher_logits is not None
            else None
        )
        training_unit_index = torch.full(shape, -1, dtype=torch.long)
        source_query_ids: list[list[str | None]] = [
            [None] * shape[1] for _ in range(shape[0])
        ]
        bag_ids: dict[tuple[int, str], int] = {}
        factor_ids: dict[tuple[int, str], int] = {}
        for row, (group, unit_index) in enumerate(
            zip(materialized, group_unit_indices, strict=True)
        ):
            originals = source_records[unit_index]
            for column, query in enumerate(group.queries):
                if not bool(batch["query_mask"][row, column]):
                    raise ValueError("MIL collator query order disagrees with query_mask")
                source_id = query.source_query_id or query.query_id
                if source_id not in originals:
                    raise ValueError(f"MIL window lacks source metadata: {source_id}")
                original = originals[source_id]
                if (
                    query.label != original.label
                    or query.factor_level != original.factor_level
                    or query.family != original.family
                    or query.loss_weight != original.loss_weight
                ):
                    raise ValueError("MIL window changed source label/hierarchy metadata")
                bag_key = (unit_index, source_id)
                factor_key = (unit_index, original.factor_instance_id)
                bag_ids.setdefault(bag_key, len(bag_ids))
                factor_ids.setdefault(factor_key, len(factor_ids))
                mil_bag_index[row, column] = bag_ids[bag_key]
                mil_factor_instance_index[row, column] = factor_ids[factor_key]
                training_unit_index[row, column] = unit_index
                mil_is_long[row, column] = int(long_flags[unit_index])
                source_query_ids[row][column] = source_id
                if self.source_query_indices:
                    if source_id not in self.source_query_indices:
                        raise ValueError(f"missing global source-query index: {source_id}")
                    mil_source_query_index[row, column] = self.source_query_indices[source_id]
                else:
                    mil_source_query_index[row, column] = bag_ids[bag_key]
                window_index = _cot_window_index(query.query_id)
                if window_index is not None:
                    mil_window_index[row, column] = window_index
                if self.teacher_logits is not None and long_flags[unit_index]:
                    if window_index is None:
                        raise ValueError("long Teacher query lacks a window index")
                    teacher_key = (source_id, window_index)
                    if teacher_key not in self.teacher_logits:
                        raise ValueError(
                            f"missing Teacher logit for source/window {teacher_key}"
                        )
                    assert mil_teacher_logit is not None
                    mil_teacher_logit[row, column] = self.teacher_logits[teacher_key]
        batch.update(
            {
                "mil_bag_index": mil_bag_index,
                "mil_factor_instance_index": mil_factor_instance_index,
                "mil_source_query_index": mil_source_query_index,
                "mil_window_index": mil_window_index,
                "mil_is_long": mil_is_long,
                "training_unit_index": training_unit_index,
                "source_query_id": source_query_ids,
                "training_unit_id": [
                    scheduled.unit.unit_id for scheduled in scheduled_units
                ],
                "training_unit_weight": torch.tensor(
                    [
                        0.0 if scheduled.is_sync_padding else 1.0
                        for scheduled in scheduled_units
                    ],
                    dtype=torch.float32,
                ),
            }
        )
        if mil_teacher_logit is not None:
            batch["mil_teacher_logit"] = mil_teacher_logit
        return batch


class FactorInstanceUniformUnitSampler(Sampler):
    """Family-priority draws whose within-family anchors are Factor instances."""

    def __init__(
        self,
        units: Sequence[FactorInstanceContrastUnit],
        *,
        num_trajectory_slots: int,
        seed: int,
        family_weights: Mapping[str, float] | None = None,
    ) -> None:
        if num_trajectory_slots <= 0:
            raise ValueError("num_trajectory_slots must be positive")
        self.units = tuple(units)
        self.num_trajectory_slots = num_trajectory_slots
        self.seed = seed
        self.epoch = 0
        anchors: dict[str, list[int]] = defaultdict(list)
        for index, unit in enumerate(units):
            if unit.factor_instance_count <= 0:
                raise ValueError("every training unit must contain a Factor instance")
            anchors[unit.family].extend([index] * unit.factor_instance_count)
        if not anchors:
            raise ValueError("sampler received no training units")
        self.anchors = {key: tuple(values) for key, values in sorted(anchors.items())}
        self.families = tuple(self.anchors)
        self.family_weights = {
            family: float((family_weights or {}).get(family, 1.0))
            for family in self.families
        }
        if any(weight <= 0 for weight in self.family_weights.values()):
            raise ValueError("family sampling weights must be positive")
        unknown = set(family_weights or {}) - set(self.families)
        if unknown:
            raise ValueError(
                f"family sampling weights contain unknown families: {sorted(unknown)}"
            )

    def __iter__(self) -> Iterator[int]:
        randomizer = random.Random(self.seed + self.epoch)
        families = list(self.families)
        weights = [self.family_weights[family] for family in families]
        uniform = len(set(weights)) == 1
        trajectory_slots = 0
        offset = 0
        while trajectory_slots < self.num_trajectory_slots:
            if uniform and offset % len(families) == 0:
                randomizer.shuffle(families)
            family = (
                families[offset % len(families)]
                if uniform
                else randomizer.choices(families, weights=weights, k=1)[0]
            )
            index = randomizer.choice(self.anchors[family])
            yield index
            trajectory_slots += len(self.units[index].groups)
            offset += 1

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def audit(self) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        factor_instances = 0
        trajectory_slots = 0
        for index in self:
            unit = self.units[index]
            counts[unit.family] += 1
            factor_instances += unit.factor_instance_count
            trajectory_slots += len(unit.groups)
        return {
            "unit_draws": sum(counts.values()),
            "trajectory_slots": trajectory_slots,
            "factor_instances": factor_instances,
            "family_draws": dict(sorted(counts.items())),
            "family_weights": dict(sorted(self.family_weights.items())),
            "epoch": self.epoch,
        }

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a nonnegative integer")
        self.epoch = epoch


class FactorInstanceContrastLengthBucketBatchSampler(Sampler):
    """Pack whole 1--2 trajectory units into synchronized two-slot rank batches."""

    def __init__(
        self,
        units: Sequence[FactorInstanceContrastUnit],
        trajectory_lengths: Mapping[str, int],
        *,
        bucket_boundaries: Sequence[int],
        batch_size: int,
        num_trajectory_slots: int,
        seed: int,
        drop_last: bool,
        synchronized_replicas: int = 1,
        family_weights: Mapping[str, float] | None = None,
    ) -> None:
        if batch_size != 2:
            raise ValueError(
                "factor_instance_contrast currently requires exactly two trajectory "
                "slots per rank"
            )
        if synchronized_replicas <= 0:
            raise ValueError("synchronized_replicas must be positive")
        boundaries = tuple(int(value) for value in bucket_boundaries)
        if not boundaries or tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError("bucket boundaries must be strictly increasing")
        unit_lengths: list[int] = []
        for unit in units:
            missing = [
                group.trajectory_id
                for group in unit.groups
                if group.trajectory_id not in trajectory_lengths
            ]
            if missing:
                raise ValueError(f"training units lack trajectory lengths: {missing[:3]}")
            maximum = max(trajectory_lengths[group.trajectory_id] for group in unit.groups)
            if maximum <= 0 or maximum > boundaries[-1]:
                raise ValueError("all accepted unit lengths must fit the final bucket")
            unit_lengths.append(int(maximum))
        self.units = tuple(units)
        self.unit_lengths = tuple(unit_lengths)
        self.boundaries = boundaries
        # One dataset item can occupy one or two trajectory slots. Exposing a
        # numeric ``batch_size`` makes Accelerate discard one-item/two-slot
        # batches, so advertise variable item count and record capacity
        # separately.
        self.batch_size = None
        self.trajectory_slots_per_replica = batch_size
        self.drop_last = drop_last
        self.synchronized_replicas = synchronized_replicas
        self.base = FactorInstanceUniformUnitSampler(
            units,
            num_trajectory_slots=num_trajectory_slots,
            seed=seed,
            family_weights=family_weights,
        )

    def _batches(self) -> list[list[int]]:
        buckets: list[list[int]] = [[] for _ in self.boundaries]
        for index in self.base:
            bucket = bisect.bisect_left(self.boundaries, self.unit_lengths[index])
            buckets[bucket].append(index)
        result: list[list[int]] = []
        for bucket in buckets:
            two_slot = [index for index in bucket if len(self.units[index].groups) == 2]
            one_slot = [index for index in bucket if len(self.units[index].groups) == 1]
            rank_batches = [[index] for index in two_slot]
            for start in range(0, len(one_slot) - 1, 2):
                rank_batches.append(one_slot[start : start + 2])
            if len(one_slot) % 2 and not self.drop_last:
                rank_batches.append([one_slot[-1], one_slot[-1]])
            if self.drop_last:
                complete = len(rank_batches) - len(rank_batches) % self.synchronized_replicas
                rank_batches = rank_batches[:complete]
            elif rank_batches:
                original = tuple(tuple(batch) for batch in rank_batches)
                offset = 0
                while len(rank_batches) % self.synchronized_replicas:
                    rank_batches.append(list(original[offset % len(original)]))
                    offset += 1
            result.extend(rank_batches)
        return result

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)


class LongCotMilTwoStreamBatchSampler(Sampler):
    """Emit synchronized long/replay microsteps for a two-rank global batch four."""

    def __init__(
        self,
        dataset: LongCotMilDataset,
        *,
        exposures: int,
        seed: int,
        synchronized_replicas: int,
        replay_only: bool,
        expected_long_units: int,
        sampled_long_units_per_exposure: int,
    ) -> None:
        if synchronized_replicas != 2:
            raise ValueError("the frozen MIL screen requires exactly two synchronized ranks")
        if exposures <= 0:
            raise ValueError("MIL exposures must be positive")
        if len(dataset.long_indices) != expected_long_units:
            raise ValueError(
                "frozen MIL train split requires "
                f"{expected_long_units} long units, found {len(dataset.long_indices)}"
            )
        if sampled_long_units_per_exposure < expected_long_units:
            raise ValueError("MIL sampled-long count may not drop source long units")
        if sampled_long_units_per_exposure % synchronized_replicas:
            raise ValueError("MIL sampled-long count must divide across replicas")
        self.dataset = dataset
        self.exposures = exposures
        self.seed = seed
        self.synchronized_replicas = synchronized_replicas
        self.replay_only = replay_only
        self.sampled_long_units_per_exposure = sampled_long_units_per_exposure
        self.batch_size = None
        self.trajectory_slots_per_replica = 1
        self.drop_last = False
        by_family: dict[str, list[int]] = defaultdict(list)
        for index in dataset.short_indices:
            by_family[dataset.units[index].family].append(index)
        if len(by_family) != 9:
            raise ValueError(
                f"short replay requires all nine families, found {sorted(by_family)}"
            )
        self.short_by_family = {
            family: tuple(indices) for family, indices in sorted(by_family.items())
        }

    @property
    def optimizer_steps_per_exposure(self) -> int:
        return self.sampled_long_units_per_exposure // self.synchronized_replicas

    def _long_pairs(self) -> list[tuple[int, int]]:
        ordered = list(self.dataset.long_indices)
        random.Random(self.seed).shuffle(ordered)
        source = tuple(ordered)
        while len(ordered) < self.sampled_long_units_per_exposure:
            ordered.append(source[(len(ordered) - len(source)) % len(source)])
        return [tuple(ordered[start : start + 2]) for start in range(0, len(ordered), 2)]

    def _short_pairs(self, *, exposure: int, count: int) -> list[tuple[int, int]]:
        randomizer = random.Random(self.seed + 10_000 + exposure)
        families = list(self.short_by_family)
        draws: list[int] = []
        for offset in range(count * 2):
            if offset % len(families) == 0:
                randomizer.shuffle(families)
            family = families[offset % len(families)]
            draws.append(randomizer.choice(self.short_by_family[family]))
        return [tuple(draws[start : start + 2]) for start in range(0, len(draws), 2)]

    def _batches(self) -> list[list[int]]:
        long_pairs = self._long_pairs()
        result: list[list[int]] = []
        for exposure in range(self.exposures):
            short_pair_count = len(long_pairs) * (2 if self.replay_only else 1)
            short_pairs = self._short_pairs(exposure=exposure, count=short_pair_count)
            for step, long_pair in enumerate(long_pairs):
                first_pair = short_pairs[2 * step] if self.replay_only else long_pair
                second_pair = short_pairs[2 * step + 1] if self.replay_only else short_pairs[step]
                # Accelerate shards consecutive batch-sampler entries across ranks.
                result.extend(([first_pair[0]], [first_pair[1]]))
                result.extend(([second_pair[0]], [second_pair[1]]))
        return result

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return self.exposures * self.optimizer_steps_per_exposure * 4

    def set_epoch(self, epoch: int) -> None:
        # The complete exposure schedule is pre-registered and must not be
        # reshuffled by Trainer's epoch bookkeeping.
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a nonnegative integer")


class LongCotMilFullEpochBatchSampler(Sampler):
    """Cover every MIL unit once per epoch with zero-loss DDP synchronization pads."""

    def __init__(
        self,
        dataset: LongCotMilDataset,
        unit_lengths: Sequence[int],
        unit_costs: Sequence[int],
        *,
        bucket_boundaries: Sequence[int],
        seed: int,
        synchronized_replicas: int,
        gradient_accumulation_steps: int,
        expected_long_units: int,
        expected_short_units: int,
    ) -> None:
        if synchronized_replicas != 2:
            raise ValueError("full-epoch MIL training requires exactly two ranks")
        if gradient_accumulation_steps != 2:
            raise ValueError("full-epoch MIL training requires gradient accumulation two")
        if len(dataset.long_indices) != expected_long_units:
            raise ValueError(
                f"expected {expected_long_units} long MIL units, "
                f"found {len(dataset.long_indices)}"
            )
        if len(dataset.short_indices) != expected_short_units:
            raise ValueError(
                f"expected {expected_short_units} short MIL units, "
                f"found {len(dataset.short_indices)}"
            )
        if len(unit_lengths) != len(dataset) or len(unit_costs) != len(dataset):
            raise ValueError("full-epoch MIL schedule requires one length and cost per unit")
        boundaries = tuple(int(value) for value in bucket_boundaries)
        if not boundaries or tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError("bucket boundaries must be strictly increasing")
        if any(length <= 0 or length > boundaries[-1] for length in unit_lengths):
            raise ValueError("all full-epoch MIL unit lengths must fit the final bucket")
        if any(cost <= 0 for cost in unit_costs):
            raise ValueError("all full-epoch MIL unit costs must be positive")
        self.dataset = dataset
        self.unit_lengths = tuple(int(value) for value in unit_lengths)
        self.unit_costs = tuple(int(value) for value in unit_costs)
        self.boundaries = boundaries
        self.seed = seed
        self.synchronized_replicas = synchronized_replicas
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.expected_long_units = expected_long_units
        self.expected_short_units = expected_short_units
        self.batch_size = None
        self.trajectory_slots_per_replica = 1
        self.drop_last = False
        self.epoch = 0
        self._slots_per_optimizer_step = (
            synchronized_replicas * gradient_accumulation_steps
        )
        real_units = expected_long_units + expected_short_units
        self.optimizer_steps_per_epoch = (
            real_units + self._slots_per_optimizer_step - 1
        ) // self._slots_per_optimizer_step
        self.sync_padding_units_per_epoch = (
            self.optimizer_steps_per_epoch * self._slots_per_optimizer_step
            - real_units
        )
        if expected_long_units > self.optimizer_steps_per_epoch:
            raise ValueError("full-epoch MIL cannot spread at most one long unit per step")

    def _bucket(self, index: int) -> int:
        return bisect.bisect_left(self.boundaries, self.unit_lengths[index])

    def _schedule(self, epoch: int) -> list[LongCotMilScheduledIndex]:
        randomizer = random.Random(self.seed + epoch)
        long_order = list(self.dataset.long_indices)
        randomizer.shuffle(long_order)
        short_by_bucket: dict[int, list[int]] = defaultdict(list)
        for index in self.dataset.short_indices:
            short_by_bucket[self._bucket(index)].append(index)
        for indices in short_by_bucket.values():
            randomizer.shuffle(indices)

        step_sizes = [self._slots_per_optimizer_step] * self.optimizer_steps_per_epoch
        for offset in range(self.sync_padding_units_per_epoch):
            step_sizes[-1 - offset] -= 1
        long_steps = {
            ((2 * offset + 1) * self.optimizer_steps_per_epoch)
            // (2 * self.expected_long_units)
            for offset in range(self.expected_long_units)
        }
        if len(long_steps) != self.expected_long_units:
            raise RuntimeError("full-epoch MIL long-step allocation is not unique")

        def take_short(preferred_bucket: int) -> int:
            available = [bucket for bucket, values in short_by_bucket.items() if values]
            if not available:
                raise RuntimeError("full-epoch MIL exhausted short units early")
            bucket = min(
                available,
                key=lambda value: (
                    abs(value - preferred_bucket),
                    -len(short_by_bucket[value]),
                    value,
                ),
            )
            return short_by_bucket[bucket].pop()

        blocks: list[list[LongCotMilScheduledIndex]] = []
        long_offset = 0
        for step, real_size in enumerate(step_sizes):
            block: list[int] = []
            if step in long_steps:
                long_index = long_order[long_offset]
                long_offset += 1
                block.append(long_index)
                preferred_bucket = self._bucket(long_index)
            else:
                available = [
                    bucket for bucket, values in short_by_bucket.items() if values
                ]
                if not available:
                    raise RuntimeError("full-epoch MIL has no short bucket for a block")
                preferred_bucket = max(
                    available,
                    key=lambda value: (len(short_by_bucket[value]), -value),
                )
            while len(block) < real_size:
                block.append(take_short(preferred_bucket))
            scheduled = [LongCotMilScheduledIndex(index) for index in block]
            while len(scheduled) < self._slots_per_optimizer_step:
                padding_source = min(
                    block,
                    key=lambda index: (self.unit_costs[index], index),
                )
                scheduled.append(
                    LongCotMilScheduledIndex(
                        padding_source,
                        is_sync_padding=True,
                    )
                )
            scheduled.sort(
                key=lambda item: (
                    self._bucket(item.index),
                    self.unit_costs[item.index],
                    item.index,
                    item.is_sync_padding,
                )
            )
            blocks.append(scheduled)

        if long_offset != len(long_order):
            raise RuntimeError("full-epoch MIL did not schedule every long unit")
        if any(short_by_bucket.values()):
            raise RuntimeError("full-epoch MIL did not schedule every short unit")
        return [item for block in blocks for item in block]

    def audit(self, *, epoch: int) -> dict[str, Any]:
        schedule = self._schedule(epoch)
        real = [item.index for item in schedule if not item.is_sync_padding]
        padding = [item for item in schedule if item.is_sync_padding]
        if len(real) != len(set(real)):
            raise RuntimeError("full-epoch MIL schedule repeats a real unit")
        expected = set(range(len(self.dataset)))
        if set(real) != expected:
            raise RuntimeError("full-epoch MIL schedule does not exactly cover the dataset")
        blocks = [
            schedule[start : start + self._slots_per_optimizer_step]
            for start in range(0, len(schedule), self._slots_per_optimizer_step)
        ]
        long_blocks = [
            index
            for index, block in enumerate(blocks)
            if any(
                not item.is_sync_padding and item.index in self.dataset.long_indices
                for item in block
            )
        ]
        payload = "\n".join(
            f"{item.index}:{int(item.is_sync_padding)}" for item in schedule
        ).encode("utf-8")
        gaps = [right - left for left, right in zip(long_blocks, long_blocks[1:])]
        return {
            "schedule": "full_epoch_joint",
            "epoch": epoch,
            "schedule_sha256": hashlib.sha256(payload).hexdigest(),
            "real_units": len(real),
            "long_units": len(self.dataset.long_indices),
            "short_units": len(self.dataset.short_indices),
            "sync_padding_units": len(padding),
            "optimizer_steps": self.optimizer_steps_per_epoch,
            "long_optimizer_steps": len(long_blocks),
            "maximum_long_step_gap": max(gaps) if gaps else 0,
        }

    def __iter__(self) -> Iterator[list[LongCotMilScheduledIndex]]:
        for item in self._schedule(self.epoch):
            yield [item]

    def __len__(self) -> int:
        return self.optimizer_steps_per_epoch * self._slots_per_optimizer_step

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a nonnegative integer")
        self.epoch = epoch


class FamilyUniformTrajectorySampler(Sampler):
    """Deterministic family-priority then within-family trajectory draws."""

    def __init__(
        self,
        groups: Sequence[TrajectoryGroup],
        *,
        num_samples: int,
        seed: int,
        family_weights: Mapping[str, float] | None = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0
        self.groups = tuple(groups)
        by_family: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(groups):
            by_family[group.family].append(index)
        if not by_family:
            raise ValueError("sampler received no trajectories")
        self.by_family = {key: tuple(values) for key, values in sorted(by_family.items())}
        self.families = tuple(self.by_family)
        self.family_weights = {
            family: float((family_weights or {}).get(family, 1.0))
            for family in self.families
        }
        if any(weight <= 0 for weight in self.family_weights.values()):
            raise ValueError("family sampling weights must be positive")
        unknown = set(family_weights or {}) - set(self.families)
        if unknown:
            raise ValueError(
                f"family sampling weights contain unknown families: {sorted(unknown)}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def audit(self) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        for index in self:
            counts[self.groups[index].family] += 1
        return {
            "draws": self.num_samples,
            "family_draws": dict(sorted(counts.items())),
            "family_weights": dict(sorted(self.family_weights.items())),
            "epoch": self.epoch,
        }

    def __iter__(self) -> Iterator[int]:
        randomizer = random.Random(self.seed + self.epoch)
        families = list(self.families)
        weights = [self.family_weights[family] for family in families]
        uniform = len(set(weights)) == 1
        for offset in range(self.num_samples):
            if uniform and offset % len(families) == 0:
                randomizer.shuffle(families)
            family = (
                families[offset % len(families)]
                if uniform
                else randomizer.choices(families, weights=weights, k=1)[0]
            )
            yield randomizer.choice(self.by_family[family])

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a nonnegative integer")
        self.epoch = epoch


class FamilyUniformLengthBucketBatchSampler(Sampler):
    """Bucket family-uniform draws and align bucket lengths across DDP ranks."""

    def __init__(
        self,
        groups: Sequence[TrajectoryGroup],
        lengths: Sequence[int],
        *,
        bucket_boundaries: Sequence[int],
        batch_size: int,
        num_samples: int,
        seed: int,
        drop_last: bool,
        synchronized_replicas: int = 1,
        family_weights: Mapping[str, float] | None = None,
    ) -> None:
        if len(groups) != len(lengths):
            raise ValueError("one evidence length is required per trajectory")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if synchronized_replicas <= 0:
            raise ValueError("synchronized_replicas must be positive")
        boundaries = tuple(int(value) for value in bucket_boundaries)
        if not boundaries or tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError("bucket boundaries must be strictly increasing")
        if any(length <= 0 or length > boundaries[-1] for length in lengths):
            raise ValueError("all accepted lengths must fall inside the final bucket")
        self.base = FamilyUniformTrajectorySampler(
            groups,
            num_samples=num_samples,
            seed=seed,
            family_weights=family_weights,
        )
        self.lengths = tuple(int(value) for value in lengths)
        self.boundaries = boundaries
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.synchronized_replicas = synchronized_replicas

    @property
    def synchronized_batch_size(self) -> int:
        return self.batch_size * self.synchronized_replicas

    def _yield_rank_batches(self, indices: Sequence[int]) -> Iterator[list[int]]:
        if len(indices) != self.synchronized_batch_size:
            raise RuntimeError("synchronized batch has an invalid number of indices")
        for start in range(0, len(indices), self.batch_size):
            yield list(indices[start : start + self.batch_size])

    def __iter__(self) -> Iterator[list[int]]:
        queues: list[list[int]] = [[] for _ in self.boundaries]
        histories: list[list[int]] = [[] for _ in self.boundaries]
        for index in self.base:
            bucket = bisect.bisect_left(self.boundaries, self.lengths[index])
            queues[bucket].append(index)
            histories[bucket].append(index)
            if len(queues[bucket]) == self.synchronized_batch_size:
                yield from self._yield_rank_batches(queues[bucket])
                queues[bucket] = []
        if not self.drop_last:
            for queue, history in zip(queues, histories, strict=True):
                if queue:
                    # Accelerate shards consecutive batches across ranks. Pad
                    # deterministically from the same length bucket so every
                    # synchronized step remains full and bucket-aligned.
                    source_index = 0
                    while len(queue) < self.synchronized_batch_size:
                        queue.append(history[source_index % len(history)])
                        source_index += 1
                    yield from self._yield_rank_batches(queue)

    def __len__(self) -> int:
        counts = [0] * len(self.boundaries)
        for index in self.base:
            counts[bisect.bisect_left(self.boundaries, self.lengths[index])] += 1
        if self.drop_last:
            synchronized = sum(count // self.synchronized_batch_size for count in counts)
        else:
            synchronized = sum(
                (count + self.synchronized_batch_size - 1) // self.synchronized_batch_size
                for count in counts
            )
        return synchronized * self.synchronized_replicas

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)


class SynchronizedLengthBucketBatchSampler(Sampler):
    """Exact validation coverage with optional per-unit cost balancing."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        bucket_boundaries: Sequence[int],
        batch_size: int,
        synchronized_replicas: int,
        costs: Sequence[int] | None = None,
    ) -> None:
        if not lengths:
            raise ValueError("validation lengths must not be empty")
        if batch_size <= 0 or synchronized_replicas <= 0:
            raise ValueError("validation batch size and replicas must be positive")
        boundaries = tuple(int(value) for value in bucket_boundaries)
        if not boundaries or tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError("bucket boundaries must be strictly increasing")
        if any(length <= 0 or length > boundaries[-1] for length in lengths):
            raise ValueError("all validation lengths must fall inside the final bucket")
        if costs is not None and len(costs) != len(lengths):
            raise ValueError("one validation cost is required per validation length")
        if costs is not None and any(cost <= 0 for cost in costs):
            raise ValueError("validation costs must be positive")
        self.lengths = tuple(int(value) for value in lengths)
        self.costs = (
            tuple(int(value) for value in costs)
            if costs is not None
            else self.lengths
        )
        self.balance_by_cost = costs is not None
        self.boundaries = boundaries
        self.batch_size = batch_size
        self.synchronized_replicas = synchronized_replicas
        self.drop_last = False

    @property
    def synchronized_batch_size(self) -> int:
        return self.batch_size * self.synchronized_replicas

    def _bucket_indices(self) -> list[list[int]]:
        buckets: list[list[int]] = [[] for _ in self.boundaries]
        for index, length in enumerate(self.lengths):
            buckets[bisect.bisect_left(self.boundaries, length)].append(index)
        for bucket in buckets:
            bucket.sort(
                key=lambda index: (
                    self.costs[index] if self.balance_by_cost else self.lengths[index],
                    self.lengths[index],
                    index,
                )
            )
        return buckets

    def _cost_balanced_rank_batches(self, indices: Sequence[int]) -> list[list[int]]:
        """Greedily balance one complete synchronized step across replicas."""

        rank_batches: list[list[int]] = [
            [] for _ in range(self.synchronized_replicas)
        ]
        rank_costs = [0] * self.synchronized_replicas
        for index in sorted(indices, key=lambda value: (-self.costs[value], value)):
            eligible = [
                rank
                for rank, batch in enumerate(rank_batches)
                if len(batch) < self.batch_size
            ]
            rank = min(eligible, key=lambda value: (rank_costs[value], value))
            rank_batches[rank].append(index)
            rank_costs[rank] += self.costs[index]
        return rank_batches

    def __iter__(self) -> Iterator[list[int]]:
        complete: list[list[int]] = []
        remainders: list[int] = []
        for bucket in self._bucket_indices():
            full_end = (
                len(bucket) // self.synchronized_batch_size
            ) * self.synchronized_batch_size
            complete.extend(
                list(bucket[start : start + self.synchronized_batch_size])
                for start in range(0, full_end, self.synchronized_batch_size)
            )
            remainders.extend(bucket[full_end:])
        remainders.sort(
            key=lambda index: (
                self.costs[index] if self.balance_by_cost else self.lengths[index],
                self.lengths[index],
                index,
            )
        )
        complete.extend(
            list(remainders[start : start + self.synchronized_batch_size])
            for start in range(0, len(remainders), self.synchronized_batch_size)
        )
        for synchronized in complete:
            unpadded_size = len(synchronized)
            if len(synchronized) < self.synchronized_batch_size:
                padding_source = tuple(synchronized)
                source_index = 0
                while len(synchronized) < self.synchronized_batch_size:
                    synchronized.append(
                        padding_source[source_index % len(padding_source)]
                    )
                    source_index += 1
            if self.balance_by_cost and unpadded_size == self.synchronized_batch_size:
                yield from self._cost_balanced_rank_batches(synchronized)
                continue
            for rank_start in range(0, len(synchronized), self.batch_size):
                yield synchronized[rank_start : rank_start + self.batch_size]

    def __len__(self) -> int:
        synchronized = (
            len(self.lengths) + self.synchronized_batch_size - 1
        ) // self.synchronized_batch_size
        return synchronized * self.synchronized_replicas
