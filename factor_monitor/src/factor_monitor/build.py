"""Config-driven canonical train/eval dataset builder."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .config import (
    ConfigurationError,
    file_sha256,
    load_config,
    repository_state,
    reject_unknown,
    require_keys,
    require_mapping,
    resolve_path,
    stable_hash,
)
from .io import read_jsonl, write_json, write_jsonl
from .prompt_contract import CONTRACT_ID, contract_hash
from .schema import CanonicalRecord, LEVELS, LABEL_NAMES, Taxonomy, validate_record_collection
from .training.encoding import (
    DEFAULT_FIELD_BOUNDARY_POLICY,
    FIELD_BOUNDARY_POLICIES,
    set_field_boundary_policy,
)


BUILD_KINDS = {"train_dataset": "train", "eval_dataset": "eval"}
COMPOSITE_KIND = "train_composite_dataset"
LENGTH_PARTITION_KIND = "train_length_partition"
EVAL_LENGTH_PARTITION_KIND = "eval_length_partition"
LENGTH_PARTITION_KINDS = {
    LENGTH_PARTITION_KIND: "train",
    EVAL_LENGTH_PARTITION_KIND: "eval",
}

MANAGED_OUTPUT_NAMES = {
    "trajectories.jsonl",
    "pool_queries.jsonl",
    "train_queries.jsonl",
    "validation_queries.jsonl",
    "holdout_queries.jsonl",
    "eval_queries.jsonl",
    "effective_config.json",
    "length_audit.json",
    "summary.json",
}


@dataclass(frozen=True)
class _SelectionUnit:
    unit_id: str
    exclusive_id: str
    leakage_group_id: str
    rows: tuple[tuple[CanonicalRecord, dict[str, Any]], ...]
    representative: CanonicalRecord
    attributes: dict[str, str]
    resource_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FactorInstance:
    factor_instance_id: str
    trajectory_id: str
    factor_scope: str
    label: int
    negative_type: str | None
    target_method_ids: tuple[str, ...]
    rows: tuple[tuple[CanonicalRecord, dict[str, Any]], ...]
    representative: CanonicalRecord


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ConfigurationError(f"{name} must be a positive finite number")
    return number


def _source_config(value: Any, index: int, *, config_dir: Path) -> dict[str, Any]:
    source = require_mapping(value, f"sources[{index}]")
    required = {"mixture_id", "path", "include_levels", "sampling_weight", "loss_weight"}
    require_keys(source, required, f"sources[{index}]")
    reject_unknown(
        source,
        required
        | {"record_loss_weight_field", "factor_instance_id_field", "filters"},
        f"sources[{index}]",
    )
    mixture_id = source["mixture_id"]
    if not isinstance(mixture_id, str) or not mixture_id.strip():
        raise ConfigurationError(f"sources[{index}].mixture_id must be non-empty")
    levels = source["include_levels"]
    if not isinstance(levels, list) or not levels or any(level not in LEVELS for level in levels):
        raise ConfigurationError(
            f"sources[{index}].include_levels must be a non-empty subset of {list(LEVELS)}"
        )
    if len(levels) != len(set(levels)):
        raise ConfigurationError(f"sources[{index}].include_levels contains duplicates")
    weight_field = source.get("record_loss_weight_field")
    if weight_field is not None and (
        not isinstance(weight_field, str) or not weight_field.strip()
    ):
        raise ConfigurationError(
            f"sources[{index}].record_loss_weight_field must be null or a non-empty string"
        )
    factor_instance_id_field = source.get("factor_instance_id_field")
    if factor_instance_id_field is not None and (
        not isinstance(factor_instance_id_field, str)
        or (
            factor_instance_id_field != "trajectory_id"
            and not factor_instance_id_field.startswith("metadata.")
        )
    ):
        raise ConfigurationError(
            f"sources[{index}].factor_instance_id_field must be null, trajectory_id, "
            "or a dotted metadata field"
        )
    filters_value = source.get("filters", {})
    filters = require_mapping(filters_value, f"sources[{index}].filters")
    if any(not isinstance(field, str) or not field.strip() for field in filters):
        raise ConfigurationError(
            f"sources[{index}].filters keys must be non-empty dotted fields"
        )
    normalized = {
        "mixture_id": mixture_id.strip(),
        "path": resolve_path(source["path"], relative_to=config_dir),
        "include_levels": list(levels),
        "sampling_weight": _positive_number(
            source["sampling_weight"], f"sources[{index}].sampling_weight"
        ),
        "loss_weight": _positive_number(source["loss_weight"], f"sources[{index}].loss_weight"),
        "record_loss_weight_field": weight_field,
        "filters": dict(filters),
    }
    if factor_instance_id_field is not None:
        normalized["factor_instance_id_field"] = factor_instance_id_field
    return normalized


def _split_config(value: Any) -> dict[str, Any]:
    split = require_mapping(value, "split")
    strategy = split.get("strategy")
    if strategy == "unsplit_pool_v1":
        reject_unknown(split, {"strategy"}, "split")
        return {"strategy": strategy}
    if strategy == "selection_partition_v1":
        required = {"strategy", "group_field"}
        require_keys(split, required, "split")
        reject_unknown(split, required, "split")
        if split["group_field"] != "leakage_group_id":
            raise ConfigurationError(
                "selection_partition_v1 group_field must be leakage_group_id"
            )
        return dict(split)
    if strategy == "metadata_partition_v1":
        required = {
            "strategy",
            "field",
            "train_value",
            "validation_value",
            "holdout_value",
        }
        require_keys(split, required, "split")
        reject_unknown(split, required, "split")
        if not isinstance(split["field"], str) or not split["field"].startswith("metadata."):
            raise ConfigurationError(
                "metadata_partition_v1 field must be a dotted path below metadata"
            )
        values = [split["train_value"], split["validation_value"], split["holdout_value"]]
        if any(not isinstance(value, str) or not value for value in values):
            raise ConfigurationError("source partition values must be non-empty strings")
        if len(set(values)) != 3:
            raise ConfigurationError("source partition values must be distinct")
        return dict(split)
    required = {
        "strategy",
        "group_field",
        "seed",
        "train_fraction",
        "validation_fraction",
    }
    require_keys(split, required, "split")
    reject_unknown(split, required, "split")
    if split["strategy"] != "deterministic_group_hash_v1":
        raise ConfigurationError(
            "split.strategy must be deterministic_group_hash_v1, "
            "metadata_partition_v1, selection_partition_v1, or unsplit_pool_v1"
        )
    if split["group_field"] != "leakage_group_id":
        raise ConfigurationError("split.group_field must be leakage_group_id")
    if isinstance(split["seed"], bool) or not isinstance(split["seed"], int):
        raise ConfigurationError("split.seed must be an integer")
    train = float(split["train_fraction"])
    validation = float(split["validation_fraction"])
    if train <= 0 or validation <= 0 or not math.isclose(train + validation, 1.0):
        raise ConfigurationError("train_fraction and validation_fraction must be positive and sum to 1")
    return {
        "strategy": split["strategy"],
        "group_field": split["group_field"],
        "seed": split["seed"],
        "train_fraction": train,
        "validation_fraction": validation,
    }


def _selection_config(
    value: Any,
    *,
    config_dir: Path | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    selection = require_mapping(value, "selection")
    strategy = selection.get("strategy")
    exclusion_files_value = selection.get("exclude_overlap_query_files", [])
    if not isinstance(exclusion_files_value, list) or any(
        not isinstance(path, str) or not path.strip() for path in exclusion_files_value
    ):
        raise ConfigurationError(
            "selection.exclude_overlap_query_files must be a list of non-empty paths"
        )
    if len(exclusion_files_value) != len(set(exclusion_files_value)):
        raise ConfigurationError("selection.exclude_overlap_query_files contains duplicates")
    if exclusion_files_value and config_dir is None:
        raise ConfigurationError("selection exclusion paths require config_dir")
    exclusion_files = [
        str(resolve_path(path, relative_to=config_dir))
        for path in exclusion_files_value
    ]
    if strategy == "partitioned_group_quota_v1":
        required = {
            "strategy",
            "unit_mode",
            "unit_field",
            "seed",
            "max_attempts",
            "required_levels",
            "passthrough_splits",
            "report_fields",
            "coverage",
            "strata",
        }
        require_keys(selection, required, "selection")
        reject_unknown(
            selection,
            required
            | {
                "preferred_unit_files",
                "exclude_overlap_query_files",
                "resource_capacities",
                "minimum_distinct_resources",
            },
            "selection",
        )
        if selection["unit_mode"] not in {
            "records",
            "sycophancy_pair",
            "sycophancy_target_contrast",
            "sycophancy_mixed_contrast",
            "factor_contrast",
            "factor_mixed_contrast",
            "factor_pointwise_mix",
            "mem2act_pair",
        }:
            raise ConfigurationError(
                "selection.unit_mode must be records, sycophancy_pair, "
                "sycophancy_target_contrast, sycophancy_mixed_contrast, "
                "factor_contrast, factor_mixed_contrast, factor_pointwise_mix, "
                "or mem2act_pair"
            )
        unit_field = selection["unit_field"]
        if not isinstance(unit_field, str) or not unit_field.strip():
            raise ConfigurationError("selection.unit_field must be non-empty text")
        if isinstance(selection["seed"], bool) or not isinstance(selection["seed"], int):
            raise ConfigurationError("selection.seed must be an integer")
        max_attempts = selection["max_attempts"]
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ConfigurationError("selection.max_attempts must be a positive integer")
        levels = selection["required_levels"]
        if (
            not isinstance(levels, list)
            or not levels
            or len(levels) != len(set(levels))
            or any(level not in LEVELS for level in levels)
        ):
            raise ConfigurationError(
                f"selection.required_levels must be a unique non-empty subset of {list(LEVELS)}"
            )
        passthrough = selection["passthrough_splits"]
        if (
            not isinstance(passthrough, list)
            or len(passthrough) != len(set(passthrough))
            or any(value not in {"holdout"} for value in passthrough)
        ):
            raise ConfigurationError(
                "selection.passthrough_splits must be a unique list containing only holdout"
            )
        report_fields = selection["report_fields"]
        if (
            not isinstance(report_fields, list)
            or len(report_fields) != len(set(report_fields))
            or any(not isinstance(field, str) or not field for field in report_fields)
        ):
            raise ConfigurationError(
                "selection.report_fields must be a unique list of non-empty dotted fields"
            )
        coverage_value = selection["coverage"]
        if not isinstance(coverage_value, list):
            raise ConfigurationError("selection.coverage must be a list")
        coverage: list[dict[str, Any]] = []
        seen_coverage: set[tuple[str, str]] = set()
        for index, value in enumerate(coverage_value):
            where = f"selection.coverage[{index}]"
            row = require_mapping(value, where)
            allowed_coverage = {"split", "field", "values", "minimum_distinct"}
            require_keys(row, {"split", "field"}, where)
            reject_unknown(row, allowed_coverage, where)
            if row["split"] not in {"train", "validation"}:
                raise ConfigurationError(f"{where}.split must be train or validation")
            if not isinstance(row["field"], str) or not row["field"]:
                raise ConfigurationError(f"{where}.field must be non-empty text")
            values = row.get("values", [])
            if (
                not isinstance(values, list)
                or len({str(item) for item in values}) != len(values)
            ):
                raise ConfigurationError(f"{where}.values must be a unique list")
            minimum_distinct = row.get("minimum_distinct")
            if minimum_distinct is not None and (
                isinstance(minimum_distinct, bool)
                or not isinstance(minimum_distinct, int)
                or minimum_distinct <= 0
            ):
                raise ConfigurationError(
                    f"{where}.minimum_distinct must be a positive integer"
                )
            if not values and minimum_distinct is None:
                raise ConfigurationError(
                    f"{where} must define values and/or minimum_distinct"
                )
            key = (row["split"], row["field"])
            if key in seen_coverage:
                raise ConfigurationError(f"duplicate selection coverage target: {key}")
            seen_coverage.add(key)
            normalized_coverage = {
                "split": row["split"],
                "field": row["field"],
                "values": list(values),
            }
            if minimum_distinct is not None:
                normalized_coverage["minimum_distinct"] = minimum_distinct
            coverage.append(normalized_coverage)
        preferred_value = selection.get("preferred_unit_files", [])
        if not isinstance(preferred_value, list):
            raise ConfigurationError("selection.preferred_unit_files must be a list")
        preferred_unit_files: list[dict[str, Any]] = []
        for index, value in enumerate(preferred_value):
            where = f"selection.preferred_unit_files[{index}]"
            row = require_mapping(value, where)
            required_preference = {"split", "path", "field"}
            require_keys(row, required_preference, where)
            reject_unknown(row, required_preference | {"require_all"}, where)
            if row["split"] not in {"train", "validation"}:
                raise ConfigurationError(f"{where}.split must be train or validation")
            if not isinstance(row["field"], str) or not row["field"]:
                raise ConfigurationError(f"{where}.field must be non-empty text")
            if config_dir is None:
                raise ConfigurationError("selection preference paths require config_dir")
            require_all = row.get("require_all", False)
            if not isinstance(require_all, bool):
                raise ConfigurationError(f"{where}.require_all must be boolean")
            preferred_unit_files.append(
                {
                    "split": row["split"],
                    "path": str(resolve_path(row["path"], relative_to=config_dir)),
                    "field": row["field"],
                    "require_all": require_all,
                }
            )
        strata_value = selection["strata"]
        if not isinstance(strata_value, list) or not strata_value:
            raise ConfigurationError("selection.strata must be a non-empty list")
        strata: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, value in enumerate(strata_value):
            where = f"selection.strata[{index}]"
            stratum = require_mapping(value, where)
            required_stratum = {"id", "split", "filters", "count"}
            require_keys(stratum, required_stratum, where)
            reject_unknown(stratum, required_stratum | {"priority"}, where)
            stratum_id = stratum["id"]
            if not isinstance(stratum_id, str) or not stratum_id.strip():
                raise ConfigurationError(f"{where}.id must be non-empty")
            if stratum_id in seen_ids:
                raise ConfigurationError(f"duplicate selection stratum id: {stratum_id}")
            seen_ids.add(stratum_id)
            if stratum["split"] not in {"train", "validation"}:
                raise ConfigurationError(f"{where}.split must be train or validation")
            filters = require_mapping(stratum["filters"], f"{where}.filters")
            if not filters or any(not isinstance(field, str) or not field for field in filters):
                raise ConfigurationError(f"{where}.filters must have non-empty dotted fields")
            count = stratum["count"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ConfigurationError(f"{where}.count must be positive")
            priority = stratum.get("priority", 0)
            if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
                raise ConfigurationError(f"{where}.priority must be a non-negative integer")
            normalized_stratum = {
                "id": stratum_id.strip(),
                "split": stratum["split"],
                "filters": dict(filters),
                "count": count,
            }
            if "priority" in stratum:
                normalized_stratum["priority"] = priority
            strata.append(normalized_stratum)
        if {row["split"] for row in strata} != {"train", "validation"}:
            raise ConfigurationError(
                "partitioned_group_quota_v1 requires train and validation strata"
            )
        resource_capacities_value = selection.get("resource_capacities", {})
        minimum_distinct_value = selection.get("minimum_distinct_resources", {})
        if selection["unit_mode"] == "factor_mixed_contrast":
            if not resource_capacities_value or not minimum_distinct_value:
                raise ConfigurationError(
                    "factor_mixed_contrast requires resource_capacities and "
                    "minimum_distinct_resources"
                )
        elif resource_capacities_value or minimum_distinct_value:
            raise ConfigurationError(
                "resource policies are supported only by factor_mixed_contrast"
            )

        def normalize_resource_policy(
            raw: Any,
            *,
            name: str,
            allow_zero: bool,
        ) -> dict[str, dict[str, int]]:
            if not raw:
                return {}
            mapping = require_mapping(raw, f"selection.{name}")
            if set(mapping) != {"train", "validation"}:
                raise ConfigurationError(
                    f"selection.{name} must define train and validation"
                )
            normalized: dict[str, dict[str, int]] = {}
            for split_name in ("train", "validation"):
                resources = require_mapping(
                    mapping[split_name], f"selection.{name}.{split_name}"
                )
                if not resources:
                    raise ConfigurationError(
                        f"selection.{name}.{split_name} must not be empty"
                    )
                normalized_resources: dict[str, int] = {}
                for resource_type, limit in resources.items():
                    if not isinstance(resource_type, str) or not resource_type.strip():
                        raise ConfigurationError(
                            f"selection.{name}.{split_name} resource names must be non-empty"
                        )
                    minimum = 0 if allow_zero else 1
                    if (
                        isinstance(limit, bool)
                        or not isinstance(limit, int)
                        or limit < minimum
                    ):
                        qualifier = "non-negative" if allow_zero else "positive"
                        raise ConfigurationError(
                            f"selection.{name}.{split_name}.{resource_type} must be a "
                            f"{qualifier} integer"
                        )
                    normalized_resources[resource_type.strip()] = limit
                normalized[split_name] = normalized_resources
            return normalized

        resource_capacities = normalize_resource_policy(
            resource_capacities_value,
            name="resource_capacities",
            allow_zero=False,
        )
        minimum_distinct_resources = normalize_resource_policy(
            minimum_distinct_value,
            name="minimum_distinct_resources",
            allow_zero=True,
        )
        if resource_capacities and any(
            not set(minimum_distinct_resources[split_name]).issubset(
                resource_capacities[split_name]
            )
            for split_name in ("train", "validation")
        ):
            raise ConfigurationError(
                "minimum_distinct_resources may only reference resource types declared "
                "in resource_capacities for the same split"
            )
        normalized_selection = {
            "strategy": strategy,
            "unit_mode": selection["unit_mode"],
            "unit_field": unit_field,
            "seed": selection["seed"],
            "max_attempts": max_attempts,
            "required_levels": list(levels),
            "passthrough_splits": list(passthrough),
            "report_fields": list(report_fields),
            "coverage": coverage,
            "preferred_unit_files": preferred_unit_files,
            "exclude_overlap_query_files": exclusion_files,
            "strata": strata,
        }
        if resource_capacities:
            normalized_selection["resource_capacities"] = resource_capacities
            normalized_selection["minimum_distinct_resources"] = (
                minimum_distinct_resources
            )
        return normalized_selection
    if strategy == "deterministic_group_quota_v1":
        required = {"strategy", "unit_field", "seed", "required_levels", "strata"}
        require_keys(selection, required, "selection")
        reject_unknown(
            selection,
            required | {"exclude_overlap_query_files", "unit_mode", "report_fields"},
            "selection",
        )
        unit_field = selection["unit_field"]
        if not isinstance(unit_field, str) or not unit_field.strip():
            raise ConfigurationError("selection.unit_field must be a non-empty dotted path")
        if isinstance(selection["seed"], bool) or not isinstance(selection["seed"], int):
            raise ConfigurationError("selection.seed must be an integer")
        levels = selection["required_levels"]
        if (
            not isinstance(levels, list)
            or not levels
            or len(levels) != len(set(levels))
            or any(level not in LEVELS for level in levels)
        ):
            raise ConfigurationError(
                f"selection.required_levels must be a unique non-empty subset of {list(LEVELS)}"
            )
        strata_value = selection["strata"]
        if not isinstance(strata_value, list) or not strata_value:
            raise ConfigurationError("selection.strata must be a non-empty list")
        strata = []
        seen_ids = set()
        for index, value in enumerate(strata_value):
            stratum = require_mapping(value, f"selection.strata[{index}]")
            require_keys(stratum, {"id", "filters", "count"}, f"selection.strata[{index}]")
            reject_unknown(stratum, {"id", "filters", "count"}, f"selection.strata[{index}]")
            stratum_id = stratum["id"]
            if not isinstance(stratum_id, str) or not stratum_id.strip():
                raise ConfigurationError(f"selection.strata[{index}].id must be non-empty")
            if stratum_id in seen_ids:
                raise ConfigurationError(f"duplicate selection stratum id: {stratum_id}")
            seen_ids.add(stratum_id)
            filters = require_mapping(stratum["filters"], f"selection.strata[{index}].filters")
            if not filters or any(not isinstance(field, str) or not field for field in filters):
                raise ConfigurationError(
                    f"selection.strata[{index}].filters must have non-empty dotted fields"
                )
            count = stratum["count"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ConfigurationError(f"selection.strata[{index}].count must be positive")
            strata.append({"id": stratum_id, "filters": dict(filters), "count": count})
        unit_mode = selection.get("unit_mode", "records")
        if unit_mode not in {"records", "factor_pointwise_mix"}:
            raise ConfigurationError(
                "deterministic_group_quota_v1 unit_mode must be records or "
                "factor_pointwise_mix"
            )
        report_fields = selection.get("report_fields", [])
        if (
            not isinstance(report_fields, list)
            or len(report_fields) != len(set(report_fields))
            or any(not isinstance(field, str) or not field for field in report_fields)
        ):
            raise ConfigurationError("selection.report_fields must be unique dotted fields")
        return {
            "strategy": strategy,
            "unit_mode": unit_mode,
            "unit_field": unit_field,
            "seed": selection["seed"],
            "required_levels": list(levels),
            "exclude_overlap_query_files": exclusion_files,
            "strata": strata,
            "report_fields": list(report_fields),
        }

    required = {"strategy", "field", "unit_field", "seed", "caps"}
    require_keys(selection, required, "selection")
    reject_unknown(selection, required, "selection")
    if strategy != "metadata_stratified_cap_v1":
        raise ConfigurationError(
            "selection.strategy must be metadata_stratified_cap_v1 or "
            "deterministic_group_quota_v1"
        )
    field = selection["field"]
    if not isinstance(field, str) or not field.startswith("metadata."):
        raise ConfigurationError("selection.field must be a dotted path below metadata")
    if selection["unit_field"] != "trajectory_id":
        raise ConfigurationError("selection.unit_field must be trajectory_id")
    seed = selection["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("selection.seed must be an integer")
    caps = require_mapping(selection["caps"], "selection.caps")
    if not caps:
        raise ConfigurationError("selection.caps must not be empty")
    normalized_caps: dict[str, int] = {}
    for key, cap in caps.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigurationError("selection.caps keys must be non-empty strings")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ConfigurationError(
                f"selection.caps[{key!r}] must be a positive integer"
            )
        normalized_caps[key.strip()] = cap
    return {
        "strategy": selection["strategy"],
        "field": field,
        "unit_field": selection["unit_field"],
        "seed": seed,
        "caps": dict(sorted(normalized_caps.items())),
    }


def _validate_unit_rows(
    unit_id: str,
    rows: list[tuple[CanonicalRecord, dict[str, Any]]],
    required_levels: set[str],
) -> None:
    levels = [record.factor_level for record, _ in rows]
    if len(levels) != len(required_levels) or set(levels) != required_levels:
        raise ConfigurationError(
            f"selection unit {unit_id!r} must contain exactly {sorted(required_levels)}; "
            f"observed={levels}"
        )
    invariants = {
        "label": {record.label for record, _ in rows},
        "trajectory_id": {record.trajectory_id for record, _ in rows},
        "task_id": {record.task_id for record, _ in rows},
        "leakage_group_id": {record.leakage_group_id for record, _ in rows},
    }
    drifted = {name: sorted(values) for name, values in invariants.items() if len(values) != 1}
    if drifted:
        raise ConfigurationError(
            f"selection unit {unit_id!r} violates paired-record invariants: {drifted}"
        )


def _record_selection_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit]:
    grouped: dict[str, list[tuple[CanonicalRecord, dict[str, Any]]]] = defaultdict(list)
    for candidate in candidates:
        value = _path_value(candidate[0].to_dict(), selection["unit_field"])
        if not isinstance(value, str) or not value:
            raise ConfigurationError(
                f"selection unit field {selection['unit_field']} must resolve to non-empty text"
            )
        grouped[value].append(candidate)
    required_levels = set(selection["required_levels"])
    units: list[_SelectionUnit] = []
    for unit_id, rows in grouped.items():
        _validate_unit_rows(unit_id, rows, required_levels)
        representative = min((record for record, _ in rows), key=lambda record: record.factor_level)
        attributes = {
            "task_prefix": representative.task_id.split("__", 1)[0],
        }
        constraint_count = representative.metadata.get("constraint_count")
        if isinstance(constraint_count, int) and not isinstance(constraint_count, bool):
            if constraint_count <= 0:
                raise ConfigurationError(
                    f"selection unit {unit_id!r} has invalid constraint_count={constraint_count}"
                )
            attributes["constraint_cardinality"] = (
                "single" if constraint_count == 1 else "multiple"
            )
        units.append(
            _SelectionUnit(
                unit_id=unit_id,
                exclusive_id=unit_id,
                leakage_group_id=representative.leakage_group_id,
                rows=tuple(rows),
                representative=representative,
                attributes=attributes,
            )
        )
    return units


def _sycophancy_selection_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit]:
    grouped: dict[
        tuple[str, str], dict[str, dict[str, tuple[CanonicalRecord, dict[str, Any]]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for candidate in candidates:
        record, mixture = candidate
        if isinstance(record.metadata.get("target_contrast"), dict):
            continue
        mapping = record.to_dict()
        triplet_id = _path_value(mapping, selection["unit_field"])
        role = _path_value(mapping, "metadata.causal.triplet_role")
        key = (mixture["mixture_id"], str(triplet_id))
        if record.factor_level in grouped[key][str(role)]:
            raise ConfigurationError(
                f"duplicate sycophancy role/level in triplet {key}: {role}/{record.factor_level}"
            )
        grouped[key][str(role)][record.factor_level] = candidate

    required_levels = set(selection["required_levels"])
    role_to_arm = {
        "control_x0_y0": "control",
        "hard_negative_x1_y0": "hard",
    }
    units: list[_SelectionUnit] = []
    for (mixture_id, triplet_id), roles in grouped.items():
        required_roles = {"applied_x1_y1", *role_to_arm}
        if set(roles) != required_roles:
            raise ConfigurationError(
                f"sycophancy triplet {(mixture_id, triplet_id)} roles drifted: {sorted(roles)}"
            )
        for role, level_rows in roles.items():
            if set(level_rows) != required_levels:
                raise ConfigurationError(
                    f"sycophancy triplet {(mixture_id, triplet_id)} role {role} levels "
                    f"drifted: {sorted(level_rows)}"
                )
        all_records = [candidate[0] for level_rows in roles.values() for candidate in level_rows.values()]
        invariant_fields = {
            "leakage_group_id": {record.leakage_group_id for record in all_records},
            "source_partition": {
                _path_value(record.to_dict(), "metadata.source_partition")
                for record in all_records
            },
            "cue_family": {
                _path_value(record.to_dict(), "metadata.intervention.cue_family")
                for record in all_records
            },
            "target_correctness": {
                _path_value(record.to_dict(), "metadata.intervention.target_correctness")
                for record in all_records
            },
            "cue_template": {
                _path_value(record.to_dict(), "metadata.intervention.cue_template_id")
                for record in all_records
            },
        }
        drifted = {
            field: sorted(values) for field, values in invariant_fields.items() if len(values) != 1
        }
        if drifted:
            raise ConfigurationError(
                f"sycophancy triplet {(mixture_id, triplet_id)} invariants drifted: {drifted}"
            )
        representative = roles["applied_x1_y1"][sorted(required_levels)[0]][0]
        exclusive_id = f"{mixture_id}:{triplet_id}"
        for negative_role, arm in role_to_arm.items():
            rows = [
                roles[role][level]
                for level in sorted(required_levels)
                for role in ("applied_x1_y1", negative_role)
            ]
            units.append(
                _SelectionUnit(
                    unit_id=f"{exclusive_id}:{arm}",
                    exclusive_id=exclusive_id,
                    leakage_group_id=representative.leakage_group_id,
                    rows=tuple(rows),
                    representative=representative,
                    attributes={"negative_arm": arm},
                )
            )
    return units


def _sycophancy_target_contrast_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit]:
    grouped: dict[str, list[tuple[CanonicalRecord, dict[str, Any]]]] = defaultdict(list)
    for candidate in candidates:
        record, _ = candidate
        contrast = record.metadata.get("target_contrast")
        if not isinstance(contrast, dict):
            continue
        group_id = _path_value(record.to_dict(), selection["unit_field"])
        if not isinstance(group_id, str) or not group_id:
            raise ConfigurationError("sycophancy target contrast lacks a group identity")
        grouped[group_id].append(candidate)

    units: list[_SelectionUnit] = []
    for group_id, rows in grouped.items():
        records = [record for record, _ in rows]
        if len(records) != 2:
            raise ConfigurationError(
                f"sycophancy target contrast {group_id!r} must contain two L1 records"
            )
        roles = {
            str(record.metadata["target_contrast"].get("role")) for record in records
        }
        invariants = {
            "levels": {record.factor_level for record in records},
            "labels": {record.label for record in records},
            "trajectory_id": {record.trajectory_id for record in records},
            "task_id": {record.task_id for record in records},
            "leakage_group_id": {record.leakage_group_id for record in records},
            "task_context": {record.task_context for record in records},
            "model_cot": {record.model_cot for record in records},
        }
        expected = {
            "levels": {"L1"},
            "labels": {0, 1},
        }
        drifted = {
            name: len(values)
            for name, values in invariants.items()
            if name not in expected and len(values) != 1
        }
        if invariants["levels"] != expected["levels"] or invariants["labels"] != expected["labels"]:
            drifted.update(
                {
                    "levels": sorted(invariants["levels"]),
                    "labels": sorted(invariants["labels"]),
                }
            )
        if roles != {"actual_target_applied", "alternate_target_not_applied"}:
            drifted["roles"] = sorted(roles)
        factor_inputs = {
            stable_hash(asdict(record.factor_spec)) for record in records
        }
        if len(factor_inputs) != 2:
            drifted["factor_inputs"] = "not_distinct"
        if drifted:
            raise ConfigurationError(
                f"sycophancy target contrast {group_id!r} violates invariants: {drifted}"
            )
        representative = next(record for record in records if record.label == 1)
        units.append(
            _SelectionUnit(
                unit_id=group_id,
                exclusive_id=group_id,
                leakage_group_id=representative.leakage_group_id,
                rows=tuple(rows),
                representative=representative,
                attributes={"contrast_type": "alternate_target"},
            )
        )
    if not units:
        raise ConfigurationError("no sycophancy target-contrast units were found")
    return units


def _sycophancy_mixed_contrast_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit]:
    """Build orthogonal target-swap and causal L1 pairs from one TE pool."""

    grouped: dict[
        tuple[str, str], list[tuple[CanonicalRecord, dict[str, Any]]]
    ] = defaultdict(list)
    for candidate in candidates:
        record, mixture = candidate
        if record.factor_level != "L1":
            continue
        group_id = _path_value(record.to_dict(), "metadata.causal.triplet_group_id")
        grouped[(mixture["mixture_id"], str(group_id))].append(candidate)

    units: list[_SelectionUnit] = []
    negative_use: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (mixture_id, triplet_group_id), rows in sorted(grouped.items()):
        actual_rows = [
            candidate
            for candidate in rows
            if _path_value(candidate[0].to_dict(), "metadata.target_role") == "actual"
        ]
        alternate_by_trajectory = {
            candidate[0].trajectory_id: candidate
            for candidate in rows
            if _path_value(candidate[0].to_dict(), "metadata.target_role") == "alternate"
        }
        positives = sorted(
            (
                candidate
                for candidate in actual_rows
                if _path_value(candidate[0].to_dict(), "metadata.causal.training_role")
                == "applied"
            ),
            key=lambda candidate: stable_hash(
                {
                    "contract": "sycophancy-l1-mixed-contrast-v1",
                    "seed": selection["seed"],
                    "role": "positive",
                    "sample_id": candidate[0].sample_id,
                }
            ),
        )
        negatives_by_axis = {
            axis: sorted(
                (
                    candidate
                    for candidate in actual_rows
                    if _path_value(
                        candidate[0].to_dict(), "metadata.causal.training_role"
                    )
                    == axis
                ),
                key=lambda candidate: stable_hash(
                    {
                        "contract": "sycophancy-l1-mixed-contrast-v1",
                        "seed": selection["seed"],
                        "role": axis,
                        "sample_id": candidate[0].sample_id,
                    }
                ),
            )
            for axis in ("hard", "simple")
        }

        def add_unit(
            *,
            positive: tuple[CanonicalRecord, dict[str, Any]],
            negative: tuple[CanonicalRecord, dict[str, Any]],
            axis: str,
        ) -> None:
            positive_record, positive_mixture = positive
            negative_record, negative_mixture = negative
            if positive_mixture["mixture_id"] != negative_mixture["mixture_id"]:
                raise ConfigurationError("sycophancy contrast pair crosses mixtures")
            if positive_record.label != 1 or negative_record.label != 0:
                raise ConfigurationError("sycophancy contrast labels must be applied/not-applied")
            positive_factor = stable_hash(asdict(positive_record.factor_spec))
            negative_factor = stable_hash(asdict(negative_record.factor_spec))
            if axis == "factor":
                invariants = {
                    positive_record.trajectory_id == negative_record.trajectory_id,
                    positive_record.task_context == negative_record.task_context,
                    positive_record.model_cot == negative_record.model_cot,
                    positive_record.task_id == negative_record.task_id,
                }
                if invariants != {True} or positive_factor == negative_factor:
                    raise ConfigurationError(
                        "factor-axis sycophancy pair must keep Evidence fixed and change Factor"
                    )
                negative_type = "alternate_target_negative"
            else:
                if (
                    positive_factor != negative_factor
                    or positive_record.trajectory_id == negative_record.trajectory_id
                    or _path_value(
                        positive_record.to_dict(), "metadata.causal.triplet_group_id"
                    )
                    != _path_value(
                        negative_record.to_dict(), "metadata.causal.triplet_group_id"
                    )
                ):
                    raise ConfigurationError(
                        "causal-axis sycophancy pair must keep Factor/group fixed and change Evidence"
                    )
                negative_type = f"{axis}_causal_negative"
                reuse_key = (axis, negative_record.sample_id)
                if negative_record.sample_id in negative_use[(mixture_id, axis)]:
                    raise ConfigurationError(
                        f"sycophancy {axis} negative reused within mixture: {reuse_key}"
                    )
                negative_use[(mixture_id, axis)].add(negative_record.sample_id)
            contrast_group_id = "syc_l1_" + stable_hash(
                {
                    "contract": "sycophancy-l1-mixed-contrast-v1",
                    "axis": axis,
                    "positive": positive_record.sample_id,
                    "negative": negative_record.sample_id,
                }
            )[:24]
            decorated: list[tuple[CanonicalRecord, dict[str, Any]]] = []
            for role, record, mixture in (
                ("positive_applied", positive_record, positive_mixture),
                ("negative_not_applied", negative_record, negative_mixture),
            ):
                pair_metadata = {
                    "contract_id": "sycophancy-l1-mixed-contrast-v1",
                    "contrast_group_id": contrast_group_id,
                    "contrast_axis": axis,
                    "role": role,
                    "negative_type": negative_type,
                    "positive_trajectory_id": positive_record.trajectory_id,
                    "negative_trajectory_id": negative_record.trajectory_id,
                }
                decorated.append(
                    (
                        replace(
                            record,
                            metadata={**record.metadata, "contrast_pair": pair_metadata},
                        ),
                        mixture,
                    )
                )
            units.append(
                _SelectionUnit(
                    unit_id=contrast_group_id,
                    exclusive_id=positive_record.sample_id,
                    leakage_group_id=positive_record.leakage_group_id,
                    rows=tuple(decorated),
                    representative=positive_record,
                    attributes={
                        "contrast_axis": axis,
                        "negative_type": negative_type,
                    },
                )
            )

        for positive in positives:
            alternate = alternate_by_trajectory.get(positive[0].trajectory_id)
            if alternate is None:
                raise ConfigurationError(
                    f"applied trajectory lacks alternate target: {positive[0].trajectory_id}"
                )
            add_unit(positive=positive, negative=alternate, axis="factor")
        for axis, negatives in negatives_by_axis.items():
            for positive, negative in zip(positives, negatives):
                add_unit(positive=positive, negative=negative, axis=axis)
    if not units:
        raise ConfigurationError("no mixed-axis sycophancy L1 units were found")
    return units


_COMPOSITE_NEGATIVE_PRIORITY = {
    "alternative_path_negative": 0,
    "partial_path_negative": 1,
    "fully_absent_negative": 2,
}


def _prepaired_factor_contrast_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit] | None:
    """Validate and preserve source-owned Factor contrast groups when present."""

    decorated = [
        isinstance(record.metadata.get("factor_contrast"), dict)
        for record, _ in candidates
    ]
    if not any(decorated):
        return None
    if not all(decorated):
        raise ConfigurationError(
            "factor contrast input cannot mix source-owned paired and unpaired records"
        )

    grouped: dict[str, list[tuple[CanonicalRecord, dict[str, Any]]]] = defaultdict(list)
    for candidate in candidates:
        record, _ = candidate
        contrast = _path_value(record.to_dict(), "metadata.factor_contrast")
        group_id = contrast.get("contrast_group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ConfigurationError(
                f"source-owned factor contrast record {record.sample_id} lacks contrast_group_id"
            )
        grouped[group_id].append(candidate)

    required_levels = set(selection["required_levels"])
    expected_roles = {
        "positive_applied": 1,
        "negative_not_applied": 0,
    }
    units: list[_SelectionUnit] = []
    for group_id, rows in sorted(grouped.items()):
        roles: dict[str, list[tuple[CanonicalRecord, dict[str, Any]]]] = defaultdict(list)
        common_values: dict[str, set[Any]] = defaultdict(set)
        for record, mixture in rows:
            contrast = _path_value(record.to_dict(), "metadata.factor_contrast")
            role = contrast.get("role")
            if role not in expected_roles:
                raise ConfigurationError(
                    f"source-owned factor contrast {group_id} has invalid role {role!r}"
                )
            if record.label != expected_roles[role]:
                raise ConfigurationError(
                    f"source-owned factor contrast {group_id}/{role} has label {record.label}"
                )
            if _path_value(record.to_dict(), "metadata.training_eligible") is not True:
                raise ConfigurationError(
                    f"source-owned factor contrast {group_id} is not training eligible"
                )
            roles[role].append((record, mixture))
            common_values["trajectory_id"].add(record.trajectory_id)
            common_values["task_id"].add(record.task_id)
            common_values["leakage_group_id"].add(record.leakage_group_id)
            common_values["task_context_hash"].add(stable_hash(record.task_context))
            common_values["model_cot_hash"].add(stable_hash(record.model_cot))
            for field in (
                "contract_id",
                "contrast_group_id",
                "factor_scope",
                "selection_tier",
                "pair_rank",
                "negative_type",
            ):
                common_values[field].add(contrast.get(field))

        if set(roles) != set(expected_roles):
            raise ConfigurationError(
                f"source-owned factor contrast {group_id} must contain both roles"
            )
        drifted = sorted(field for field, values in common_values.items() if len(values) != 1)
        if drifted:
            raise ConfigurationError(
                f"source-owned factor contrast {group_id} drifts in fields {drifted}"
            )
        scope = next(iter(common_values["factor_scope"]))
        if scope not in {"atomic", "composite", "dependency_atomic", "task_complete"}:
            raise ConfigurationError(
                f"source-owned factor contrast {group_id} has invalid scope {scope!r}"
            )

        factor_ids: dict[str, str] = {}
        for role, role_rows in roles.items():
            contrast = _path_value(role_rows[0][0].to_dict(), "metadata.factor_contrast")
            factor_id = contrast.get("factor_instance_id")
            if not isinstance(factor_id, str) or not factor_id:
                raise ConfigurationError(
                    f"source-owned factor contrast {group_id}/{role} lacks factor_instance_id"
                )
            _validate_unit_rows(factor_id, role_rows, required_levels)
            if any(
                _path_value(record.to_dict(), selection["unit_field"]) != factor_id
                or _path_value(record.to_dict(), "metadata.factor_contrast.factor_instance_id")
                != factor_id
                for record, _ in role_rows
            ):
                raise ConfigurationError(
                    f"source-owned factor contrast {group_id}/{role} factor identity drifted"
                )
            factor_ids[role] = factor_id
        for role, paired_role in (
            ("positive_applied", "negative_not_applied"),
            ("negative_not_applied", "positive_applied"),
        ):
            if any(
                _path_value(
                    record.to_dict(),
                    "metadata.factor_contrast.paired_factor_instance_id",
                )
                != factor_ids[paired_role]
                for record, _ in roles[role]
            ):
                raise ConfigurationError(
                    f"source-owned factor contrast {group_id}/{role} paired identity drifted"
                )

        representative = min(
            (record for record, _ in roles["positive_applied"]),
            key=lambda record: record.factor_level,
        )
        units.append(
            _SelectionUnit(
                unit_id=group_id,
                exclusive_id=group_id,
                leakage_group_id=representative.leakage_group_id,
                rows=tuple(rows),
                representative=representative,
                attributes={
                    "factor_scope": str(scope),
                    "selection_tier": str(next(iter(common_values["selection_tier"]))),
                    "negative_type": str(next(iter(common_values["negative_type"]))),
                },
            )
        )
    return units


def _validated_factor_instances(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_FactorInstance]:
    """Collect complete, training-eligible L1/L3 Factor instances once."""

    grouped: dict[
        tuple[str, str, int, str],
        list[tuple[CanonicalRecord, dict[str, Any]]],
    ] = defaultdict(list)
    for candidate in candidates:
        record, _ = candidate
        scope = _path_value(record.to_dict(), "metadata.factor_scope")
        factor_id = _path_value(record.to_dict(), selection["unit_field"])
        if scope not in {"atomic", "composite", "dependency_atomic", "task_complete"}:
            continue
        if not isinstance(factor_id, str) or not factor_id:
            raise ConfigurationError("factor contrast record lacks a factor identity")
        grouped[(record.trajectory_id, str(scope), record.label, factor_id)].append(candidate)

    required_levels = set(selection["required_levels"])
    instances: list[_FactorInstance] = []
    for (trajectory_id, scope, label, factor_id), rows in grouped.items():
        _validate_unit_rows(factor_id, rows, required_levels)
        representative = min((record for record, _ in rows), key=lambda record: record.factor_level)
        if _path_value(representative.to_dict(), "metadata.training_eligible") is not True:
            continue
        negative_type = _path_value(representative.to_dict(), "metadata.negative_type")
        if label == 0:
            if scope == "atomic" and negative_type != "atomic_absent":
                raise ConfigurationError(
                    f"atomic contrast negative {factor_id} must use atomic_absent"
                )
            if scope == "composite" and negative_type not in _COMPOSITE_NEGATIVE_PRIORITY:
                raise ConfigurationError(
                    f"composite contrast negative {factor_id} has invalid negative_type"
                )
        target_method_ids_value = representative.metadata.get("target_method_ids")
        if target_method_ids_value is None:
            target_method_ids: tuple[str, ...] = ()
        elif (
            isinstance(target_method_ids_value, list)
            and target_method_ids_value
            and all(isinstance(value, str) and value for value in target_method_ids_value)
        ):
            target_method_ids = tuple(target_method_ids_value)
        else:
            raise ConfigurationError(
                f"factor instance {factor_id} has invalid metadata.target_method_ids"
            )
        instances.append(
            _FactorInstance(
                factor_instance_id=factor_id,
                trajectory_id=trajectory_id,
                factor_scope=scope,
                label=label,
                negative_type=str(negative_type) if negative_type is not None else None,
                target_method_ids=target_method_ids,
                rows=tuple(rows),
                representative=representative,
            )
        )
    return instances


def _factor_semantic_id(instance: _FactorInstance) -> str:
    if not instance.target_method_ids:
        raise ConfigurationError(
            f"factor instance {instance.factor_instance_id} lacks target_method_ids"
        )
    _, mixture = instance.rows[0]
    return "factor_semantic_" + stable_hash(
        {
            "contract": "factor_semantic_identity_v1",
            "mixture_id": mixture["mixture_id"],
            "leakage_group_id": instance.representative.leakage_group_id,
            "task_id": instance.representative.task_id,
            "factor_scope": instance.factor_scope,
            "target_method_ids": list(instance.target_method_ids),
        }
    )[:24]


def _factor_contrast_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit]:
    """Build deterministic same-trajectory, same-scope Factor contrast pairs."""

    prepaired = _prepaired_factor_contrast_units(candidates, selection)
    if prepaired is not None:
        return prepaired

    by_trajectory: dict[
        str,
        dict[str, dict[int, list[tuple[str, tuple[tuple[CanonicalRecord, dict[str, Any]], ...]]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for instance in _validated_factor_instances(candidates, selection):
        by_trajectory[instance.trajectory_id][instance.factor_scope][instance.label].append(
            (instance.factor_instance_id, instance.rows)
        )

    units: list[_SelectionUnit] = []
    for trajectory_id, scopes in sorted(by_trajectory.items()):
        for scope in ("atomic", "composite"):
            positives = scopes[scope][1]
            negatives = scopes[scope][0]
            if not positives or not negatives:
                continue
            positives.sort(
                key=lambda item: stable_hash(
                    {
                        "contract": "factor_contrast_pair_v1",
                        "seed": selection["seed"],
                        "trajectory_id": trajectory_id,
                        "scope": scope,
                        "role": "positive",
                        "factor_id": item[0],
                    }
                )
            )
            negatives.sort(
                key=lambda item: (
                    _COMPOSITE_NEGATIVE_PRIORITY.get(
                        _path_value(item[1][0][0].to_dict(), "metadata.negative_type"), 0
                    ),
                    stable_hash(
                        {
                            "contract": "factor_contrast_pair_v1",
                            "seed": selection["seed"],
                            "trajectory_id": trajectory_id,
                            "scope": scope,
                            "role": "negative",
                            "factor_id": item[0],
                        }
                    ),
                )
            )
            for pair_rank, ((positive_id, positive_rows), (negative_id, negative_rows)) in enumerate(
                zip(positives, negatives)
            ):
                if pair_rank > 1 or (scope == "composite" and pair_rank > 0):
                    break
                tier = "primary" if pair_rank == 0 else "secondary"
                negative_type = str(
                    _path_value(negative_rows[0][0].to_dict(), "metadata.negative_type")
                )
                contrast_group_id = "factor_contrast_" + stable_hash(
                    {
                        "contract": "factor_contrast_pair_v1",
                        "trajectory_id": trajectory_id,
                        "scope": scope,
                        "positive_factor_instance_id": positive_id,
                        "negative_factor_instance_id": negative_id,
                    }
                )[:24]
                paired_rows: list[tuple[CanonicalRecord, dict[str, Any]]] = []
                for role, own_id, paired_id, rows in (
                    ("positive_applied", positive_id, negative_id, positive_rows),
                    ("negative_not_applied", negative_id, positive_id, negative_rows),
                ):
                    for record, mixture in rows:
                        contrast = {
                            "contract_id": "factor_contrast_pair_v1",
                            "contrast_group_id": contrast_group_id,
                            "role": role,
                            "factor_scope": scope,
                            "selection_tier": tier,
                            "pair_rank": pair_rank,
                            "factor_instance_id": own_id,
                            "paired_factor_instance_id": paired_id,
                            "negative_type": negative_type,
                        }
                        paired_rows.append(
                            (
                                replace(
                                    record,
                                    metadata={**record.metadata, "factor_contrast": contrast},
                                ),
                                mixture,
                            )
                        )
                representative = min(
                    (record for record, _ in positive_rows),
                    key=lambda record: record.factor_level,
                )
                partition = str(
                    _path_value(representative.to_dict(), "metadata.source_partition")
                )
                if tier == "primary":
                    exclusive_id = (
                        representative.leakage_group_id
                        if partition == "train"
                        else trajectory_id
                    )
                else:
                    exclusive_id = f"secondary:{trajectory_id}"
                units.append(
                    _SelectionUnit(
                        unit_id=contrast_group_id,
                        exclusive_id=exclusive_id,
                        leakage_group_id=representative.leakage_group_id,
                        rows=tuple(paired_rows),
                        representative=representative,
                        attributes={
                            "factor_scope": scope,
                            "selection_tier": tier,
                            "negative_type": negative_type,
                        },
                    )
                )
    if not units:
        raise ConfigurationError("no same-trajectory Factor contrast units were found")
    return units


def _factor_instance_specs(instance: _FactorInstance) -> dict[str, dict[str, Any]]:
    return {
        record.factor_level: asdict(record.factor_spec)
        for record, _ in instance.rows
    }


def _factor_mixed_contrast_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit]:
    """Build Factor-axis and Evidence-axis HARP contrast pairs."""

    instances = _validated_factor_instances(candidates, selection)
    if not instances:
        raise ConfigurationError("no complete Factor instances were found")
    for instance in instances:
        _factor_semantic_id(instance)

    by_trajectory: dict[
        tuple[str, str], dict[int, list[_FactorInstance]]
    ] = defaultdict(lambda: defaultdict(list))
    by_semantic: dict[str, dict[int, list[_FactorInstance]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for instance in instances:
        by_trajectory[(instance.trajectory_id, instance.factor_scope)][instance.label].append(
            instance
        )
        by_semantic[_factor_semantic_id(instance)][instance.label].append(instance)

    def stable_instances(values: list[_FactorInstance]) -> list[_FactorInstance]:
        return sorted(
            values,
            key=lambda instance: stable_hash(
                {
                    "contract": "factor_mixed_contrast_v1",
                    "seed": selection["seed"],
                    "factor_instance_id": instance.factor_instance_id,
                }
            ),
        )

    def build_unit(
        *,
        axis: str,
        positive: _FactorInstance,
        negative: _FactorInstance,
    ) -> _SelectionUnit:
        if positive.label != 1 or negative.label != 0:
            raise AssertionError("mixed Factor contrast roles have invalid labels")
        if positive.factor_scope != negative.factor_scope:
            raise ConfigurationError("mixed Factor contrast crosses Factor scopes")
        scope = positive.factor_scope
        if scope == "atomic" and negative.negative_type != "atomic_absent":
            raise ConfigurationError("Atomic mixed contrast requires atomic_absent negative")
        if scope == "composite" and negative.negative_type not in _COMPOSITE_NEGATIVE_PRIORITY:
            raise ConfigurationError("Composite mixed contrast has invalid negative type")

        positive_semantic_id = _factor_semantic_id(positive)
        negative_semantic_id = _factor_semantic_id(negative)
        if axis == "factor":
            invariant_values = {
                "trajectory_id": {positive.trajectory_id, negative.trajectory_id},
                "task_id": {
                    positive.representative.task_id,
                    negative.representative.task_id,
                },
                "leakage_group_id": {
                    positive.representative.leakage_group_id,
                    negative.representative.leakage_group_id,
                },
                "task_context": {
                    positive.representative.task_context,
                    negative.representative.task_context,
                },
                "model_cot": {
                    positive.representative.model_cot,
                    negative.representative.model_cot,
                },
            }
            drifted = sorted(name for name, values in invariant_values.items() if len(values) != 1)
            if drifted:
                raise ConfigurationError(f"Factor-axis pair drifts in {drifted}")
            if positive_semantic_id == negative_semantic_id:
                raise ConfigurationError("Factor-axis pair must change semantic Factor identity")
            if scope == "composite" and negative.negative_type != "alternative_path_negative":
                raise ConfigurationError(
                    "Factor-axis Composite negative must be an alternative method"
                )
            exclusive_id = f"factor:{positive.trajectory_id}:{scope}"
        elif axis == "evidence":
            invariant_values = {
                "task_id": {
                    positive.representative.task_id,
                    negative.representative.task_id,
                },
                "leakage_group_id": {
                    positive.representative.leakage_group_id,
                    negative.representative.leakage_group_id,
                },
                "task_context": {
                    positive.representative.task_context,
                    negative.representative.task_context,
                },
                "source_partition": {
                    _path_value(positive.representative.to_dict(), "metadata.source_partition"),
                    _path_value(negative.representative.to_dict(), "metadata.source_partition"),
                },
            }
            drifted = sorted(name for name, values in invariant_values.items() if len(values) != 1)
            if drifted:
                raise ConfigurationError(f"Evidence-axis pair drifts in {drifted}")
            if positive_semantic_id != negative_semantic_id:
                raise ConfigurationError("Evidence-axis pair must preserve semantic Factor identity")
            if positive.trajectory_id == negative.trajectory_id:
                raise ConfigurationError("Evidence-axis pair must use different trajectories")
            if positive.representative.model_cot == negative.representative.model_cot:
                raise ConfigurationError("Evidence-axis pair must change model CoT")
            if _factor_instance_specs(positive) != _factor_instance_specs(negative):
                raise ConfigurationError("Evidence-axis pair drifts in exact L1/L3 Factor specs")
            exclusive_id = f"evidence:{positive_semantic_id}"
        else:
            raise AssertionError(f"unsupported mixed Factor contrast axis: {axis}")

        negative_type = str(negative.negative_type)
        contrast_group_id = "factor_mixed_" + stable_hash(
            {
                "contract": "factor_mixed_contrast_v1",
                "axis": axis,
                "positive_factor_instance_id": positive.factor_instance_id,
                "negative_factor_instance_id": negative.factor_instance_id,
            }
        )[:24]
        paired_rows: list[tuple[CanonicalRecord, dict[str, Any]]] = []
        for role, own, paired, own_semantic_id, paired_semantic_id in (
            (
                "positive_applied",
                positive,
                negative,
                positive_semantic_id,
                negative_semantic_id,
            ),
            (
                "negative_not_applied",
                negative,
                positive,
                negative_semantic_id,
                positive_semantic_id,
            ),
        ):
            for record, mixture in own.rows:
                contrast = {
                    "contract_id": "factor_mixed_contrast_v1",
                    "contrast_group_id": contrast_group_id,
                    "contrast_axis": axis,
                    "role": role,
                    "factor_scope": scope,
                    "factor_semantic_id": own_semantic_id,
                    "paired_factor_semantic_id": paired_semantic_id,
                    "factor_instance_id": own.factor_instance_id,
                    "paired_factor_instance_id": paired.factor_instance_id,
                    "positive_trajectory_id": positive.trajectory_id,
                    "negative_trajectory_id": negative.trajectory_id,
                    "negative_type": negative_type,
                }
                paired_rows.append(
                    (
                        replace(
                            record,
                            metadata={**record.metadata, "factor_contrast": contrast},
                        ),
                        mixture,
                    )
                )

        resource_ids = tuple(
            sorted(
                {
                    f"trajectory:{positive.trajectory_id}",
                    f"trajectory:{negative.trajectory_id}",
                    f"factor_instance:{positive.factor_instance_id}",
                    f"factor_instance:{negative.factor_instance_id}",
                }
            )
        )
        return _SelectionUnit(
            unit_id=contrast_group_id,
            exclusive_id=exclusive_id,
            leakage_group_id=positive.representative.leakage_group_id,
            rows=tuple(paired_rows),
            representative=positive.representative,
            attributes={
                "contrast_axis": axis,
                "factor_scope": scope,
                "negative_type": negative_type,
            },
            resource_ids=resource_ids,
        )

    units: list[_SelectionUnit] = []
    for (_, scope), labels in sorted(by_trajectory.items()):
        for positive in stable_instances(labels[1]):
            for negative in stable_instances(labels[0]):
                if scope == "composite" and negative.negative_type != "alternative_path_negative":
                    continue
                units.append(build_unit(axis="factor", positive=positive, negative=negative))
    for _, labels in sorted(by_semantic.items()):
        for positive in stable_instances(labels[1]):
            for negative in stable_instances(labels[0]):
                if positive.trajectory_id == negative.trajectory_id:
                    continue
                units.append(build_unit(axis="evidence", positive=positive, negative=negative))
    if not units:
        raise ConfigurationError("no mixed-axis Factor contrast units were found")
    return units


def _mem2act_selection_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit]:
    by_trajectory: dict[str, list[tuple[CanonicalRecord, dict[str, Any]]]] = defaultdict(list)
    for candidate in candidates:
        by_trajectory[candidate[0].trajectory_id].append(candidate)
    required_levels = set(selection["required_levels"])
    for trajectory_id, rows in by_trajectory.items():
        _validate_unit_rows(trajectory_id, rows, required_levels)

    units: list[_SelectionUnit] = []
    used_hard: set[str] = set()
    for trajectory_id, rows in by_trajectory.items():
        representative = min((record for record, _ in rows), key=lambda record: record.factor_level)
        subtype = _path_value(representative.to_dict(), "metadata.training_subtype")
        if subtype == "positive":
            units.append(
                _SelectionUnit(
                    unit_id=f"positive:{trajectory_id}",
                    exclusive_id=f"positive:{trajectory_id}",
                    leakage_group_id=representative.leakage_group_id,
                    rows=tuple(rows),
                    representative=representative,
                    attributes={"role": "positive"},
                )
            )
            continue
        if subtype != "simple_negative":
            continue
        pairing = _path_value(representative.to_dict(), "metadata.control_pairing")
        if pairing == "same_difficulty_fallback":
            continue
        hard_id = _path_value(
            representative.to_dict(), "metadata.paired_hard_negative_trajectory_id"
        )
        if hard_id not in by_trajectory:
            raise ConfigurationError(
                f"memory simple negative {trajectory_id} references missing hard negative {hard_id}"
            )
        hard_rows = by_trajectory[str(hard_id)]
        hard_rep = min((record for record, _ in hard_rows), key=lambda record: record.factor_level)
        if _path_value(hard_rep.to_dict(), "metadata.training_subtype") != "hard_negative":
            raise ConfigurationError(
                f"memory pair {trajectory_id}/{hard_id} does not reference a hard negative"
            )
        if representative.leakage_group_id != hard_rep.leakage_group_id:
            raise ConfigurationError(
                f"memory pair {trajectory_id}/{hard_id} crosses leakage groups"
            )
        if str(hard_id) in used_hard:
            raise ConfigurationError(f"memory hard negative {hard_id} has multiple controls")
        used_hard.add(str(hard_id))
        units.append(
            _SelectionUnit(
                unit_id=f"negative_pair:{trajectory_id}:{hard_id}",
                exclusive_id=f"negative_pair:{trajectory_id}:{hard_id}",
                leakage_group_id=representative.leakage_group_id,
                rows=tuple([*rows, *hard_rows]),
                representative=representative,
                attributes={"role": "negative_pair"},
            )
        )
    return units


def _factor_pointwise_mix_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit]:
    """Validate a source-owned mixture of Factor-axis pairs and pointwise rows."""

    paired = [
        candidate
        for candidate in candidates
        if _path_value(candidate[0].to_dict(), "metadata.composition_role")
        == "factor_axis"
    ]
    pointwise = [
        candidate
        for candidate in candidates
        if _path_value(candidate[0].to_dict(), "metadata.composition_role")
        == "pointwise"
    ]
    if len(paired) + len(pointwise) != len(candidates) or not paired or not pointwise:
        raise ConfigurationError(
            "factor_pointwise_mix requires only, and at least one of each, "
            "factor_axis and pointwise records"
        )
    paired_units = _prepaired_factor_contrast_units(paired, selection)
    if paired_units is None:
        raise ConfigurationError("factor-axis rows lack source-owned contrast metadata")
    result: list[_SelectionUnit] = []
    for unit in paired_units:
        negative_rows = [record for record, _ in unit.rows if record.label == 0]
        if not negative_rows:
            raise ConfigurationError(f"Factor pair {unit.unit_id} lacks negative row")
        negative_subtypes = {
            str(_path_value(record.to_dict(), "metadata.training_subtype"))
            for record in negative_rows
        }
        difficulties = {
            str(_path_value(record.to_dict(), "metadata.mem2act_difficulty"))
            for record, _ in unit.rows
        }
        if len(negative_subtypes) != 1 or len(difficulties) != 1:
            raise ConfigurationError(
                f"Factor pair {unit.unit_id} drifts in subtype or difficulty"
            )
        result.append(
            _SelectionUnit(
                unit_id=unit.unit_id,
                exclusive_id=unit.exclusive_id,
                leakage_group_id=unit.leakage_group_id,
                rows=unit.rows,
                representative=unit.representative,
                attributes={
                    **unit.attributes,
                    "composition_role": "factor_axis",
                    "training_subtype": next(iter(negative_subtypes)),
                    "position_relation": str(
                        (
                            negative_rows[0].metadata.get("factor_contrast") or {}
                        ).get("position_relation", "unspecified")
                    ),
                    "mem2act_difficulty": next(iter(difficulties)),
                },
                resource_ids=unit.resource_ids,
            )
        )
    pointwise_units = _record_selection_units(pointwise, selection)
    for unit in pointwise_units:
        subtype = str(
            _path_value(unit.representative.to_dict(), "metadata.training_subtype")
        )
        if subtype not in {"positive", "hard_negative", "simple_negative"}:
            raise ConfigurationError(
                f"pointwise unit {unit.unit_id} has invalid subtype {subtype!r}"
            )
        result.append(
            _SelectionUnit(
                unit_id=unit.unit_id,
                exclusive_id=unit.exclusive_id,
                leakage_group_id=unit.leakage_group_id,
                rows=unit.rows,
                representative=unit.representative,
                attributes={
                    **unit.attributes,
                    "composition_role": "pointwise",
                    "training_subtype": subtype,
                    "factor_scope": str(
                        unit.representative.metadata.get("factor_scope", "unspecified")
                    ),
                    "negative_mechanism": str(
                        unit.representative.metadata.get("negative_mechanism", "none")
                    ),
                    "mem2act_difficulty": str(
                        _path_value(
                            unit.representative.to_dict(),
                            "metadata.mem2act_difficulty",
                        )
                    ),
                },
                resource_ids=unit.resource_ids,
            )
        )
    return result


def _selection_units(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
) -> list[_SelectionUnit]:
    if selection["unit_mode"] == "records":
        return _record_selection_units(candidates, selection)
    if selection["unit_mode"] == "sycophancy_pair":
        return _sycophancy_selection_units(candidates, selection)
    if selection["unit_mode"] == "sycophancy_target_contrast":
        return _sycophancy_target_contrast_units(candidates, selection)
    if selection["unit_mode"] == "sycophancy_mixed_contrast":
        return _sycophancy_mixed_contrast_units(candidates, selection)
    if selection["unit_mode"] == "factor_contrast":
        return _factor_contrast_units(candidates, selection)
    if selection["unit_mode"] == "factor_mixed_contrast":
        return _factor_mixed_contrast_units(candidates, selection)
    if selection["unit_mode"] == "factor_pointwise_mix":
        return _factor_pointwise_mix_units(candidates, selection)
    return _mem2act_selection_units(candidates, selection)


def _unit_path_value(unit: _SelectionUnit, field: str) -> Any:
    if field.startswith("unit."):
        key = field.removeprefix("unit.")
        if key not in unit.attributes:
            raise ConfigurationError(f"selection unit field not found: {field}")
        return unit.attributes[key]
    if field.startswith("metadata."):
        return _path_value(
            unit.representative.metadata,
            field.removeprefix("metadata."),
        )
    top_level = {
        "sample_id": unit.representative.sample_id,
        "trajectory_id": unit.representative.trajectory_id,
        "task_id": unit.representative.task_id,
        "leakage_group_id": unit.representative.leakage_group_id,
        "cell": unit.representative.cell,
        "family": unit.representative.family,
        "source": unit.representative.source,
        "factor_level": unit.representative.factor_level,
        "label": unit.representative.label,
        "annotation_status": unit.representative.annotation_status,
    }
    if field not in top_level:
        raise ConfigurationError(f"selection record field not found: {field}")
    return top_level[field]


def _matching_stratum(
    unit: _SelectionUnit,
    strata: list[dict[str, Any]],
) -> str | None:
    matched = [
        row["id"]
        for row in strata
        if all(_unit_path_value(unit, field) == expected for field, expected in row["filters"].items())
    ]
    if len(matched) > 1:
        raise ConfigurationError(
            f"selection unit {unit.unit_id!r} matches overlapping strata: {matched}"
        )
    return matched[0] if matched else None


def _resource_type(resource_id: str) -> str:
    resource_type, separator, identity = resource_id.partition(":")
    if not separator or not resource_type or not identity:
        raise ConfigurationError(f"invalid selection resource id: {resource_id!r}")
    return resource_type


def _choose_quota_units(
    units: list[_SelectionUnit],
    strata: list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    split_name: str,
    attempt: int,
    preferred_ids: set[str],
) -> tuple[list[_SelectionUnit], dict[str, int], dict[str, int]] | None:
    units_by_stratum: dict[str, list[_SelectionUnit]] = {row["id"]: [] for row in strata}
    for unit in units:
        matched = _matching_stratum(unit, strata)
        if matched is not None:
            units_by_stratum[matched].append(unit)
    available = {key: len(value) for key, value in units_by_stratum.items()}
    mixed_axis_priority = {"hard": 0, "simple": 1, "factor": 2}
    ordered_strata = sorted(
        strata,
        key=lambda row: (
            row.get("priority", 0),
            mixed_axis_priority.get(
                row["filters"].get("unit.contrast_axis"), 0
            )
            if selection["unit_mode"] == "sycophancy_mixed_contrast"
            else 0,
            available[row["id"]] / row["count"],
            row["id"],
        ),
    )
    chosen: list[_SelectionUnit] = []
    used_exclusive: set[str] = set()
    resource_capacities = selection.get("resource_capacities", {}).get(split_name, {})
    resource_counts: Counter[str] = Counter()
    selected_counts: dict[str, int] = {}
    for stratum in ordered_strata:
        stratum_id = stratum["id"]
        eligible = [
            unit
            for unit in units_by_stratum[stratum_id]
            if unit.exclusive_id not in used_exclusive
        ]
        picked: list[_SelectionUnit] = []
        picked_exclusive: set[str] = set()
        if not resource_capacities:
            ordered = sorted(
                eligible,
                key=lambda unit: (
                    unit.unit_id not in preferred_ids,
                    stable_hash(
                        {
                            "strategy": selection["strategy"],
                            "seed": selection["seed"],
                            "attempt": attempt,
                            "split": split_name,
                            "stratum": stratum_id,
                            "unit": unit.unit_id,
                        }
                    ),
                ),
            )
            for unit in ordered:
                if unit.exclusive_id in picked_exclusive:
                    continue
                picked.append(unit)
                picked_exclusive.add(unit.exclusive_id)
                if len(picked) == stratum["count"]:
                    break
        else:
            remaining = list(eligible)
            while len(picked) < stratum["count"]:
                feasible = []
                for unit in remaining:
                    if unit.exclusive_id in picked_exclusive:
                        continue
                    for resource_id in unit.resource_ids:
                        resource_type = _resource_type(resource_id)
                        if resource_type not in resource_capacities:
                            raise ConfigurationError(
                                f"selection unit {unit.unit_id} uses undeclared resource type "
                                f"{resource_type!r}"
                            )
                        if resource_counts[resource_id] >= resource_capacities[resource_type]:
                            break
                    else:
                        feasible.append(unit)
                if not feasible:
                    break
                unit = min(
                    feasible,
                    key=lambda candidate: (
                        candidate.unit_id not in preferred_ids,
                        -sum(
                            resource_counts[resource_id] == 0
                            for resource_id in candidate.resource_ids
                        ),
                        sum(resource_counts[resource_id] for resource_id in candidate.resource_ids),
                        stable_hash(
                            {
                                "strategy": selection["strategy"],
                                "seed": selection["seed"],
                                "attempt": attempt,
                                "split": split_name,
                                "stratum": stratum_id,
                                "unit": candidate.unit_id,
                            }
                        ),
                    ),
                )
                picked.append(unit)
                picked_exclusive.add(unit.exclusive_id)
                for resource_id in unit.resource_ids:
                    resource_counts[resource_id] += 1
                remaining.remove(unit)
        if len(picked) < stratum["count"]:
            return None
        chosen.extend(picked)
        used_exclusive.update(unit.exclusive_id for unit in picked)
        selected_counts[stratum_id] = len(picked)
    for resource_type, minimum in selection.get("minimum_distinct_resources", {}).get(
        split_name, {}
    ).items():
        observed = sum(
            count > 0
            for resource_id, count in resource_counts.items()
            if _resource_type(resource_id) == resource_type
        )
        if observed < minimum:
            return None
    return chosen, available, selected_counts


def _metadata_split_name(record: CanonicalRecord, split: dict[str, Any]) -> str:
    partition_map = {
        split["train_value"]: "train",
        split["validation_value"]: "validation",
        split["holdout_value"]: "holdout",
    }
    field = split["field"]
    if not field.startswith("metadata."):
        raise ConfigurationError("metadata partition field must be below metadata")
    partition = _path_value(record.metadata, field.removeprefix("metadata."))
    if partition not in partition_map:
        raise ConfigurationError(
            f"record {record.sample_id} has unsupported metadata partition {partition!r}"
        )
    return partition_map[partition]


def _coverage_report(
    units_by_split: dict[str, list[_SelectionUnit]],
    selection: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    report = []
    passed = True
    for target in selection["coverage"]:
        observed = {
            str(_unit_path_value(unit, target["field"]))
            for unit in units_by_split[target["split"]]
        }
        required = {str(value) for value in target["values"]}
        missing = sorted(required - observed)
        minimum_distinct = target.get("minimum_distinct")
        distinct_shortfall = (
            max(0, int(minimum_distinct) - len(observed))
            if minimum_distinct is not None
            else 0
        )
        report.append(
            {
                **target,
                "observed": sorted(observed),
                "observed_distinct": len(observed),
                "missing": missing,
                "distinct_shortfall": distinct_shortfall,
            }
        )
        passed = passed and not missing and distinct_shortfall == 0
    return passed, report


def _select_partitioned_records(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any],
    split: dict[str, Any],
) -> tuple[
    list[tuple[CanonicalRecord, dict[str, Any]]],
    dict[str, Any],
    dict[str, str],
]:
    units = _selection_units(candidates, selection)
    preferred_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    required_preferred_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
    }
    for preference in selection["preferred_unit_files"]:
        preference_path = Path(preference["path"])
        if not preference_path.is_file():
            raise FileNotFoundError(preference_path)
        for row in read_jsonl(preference_path):
            value = _path_value(row, preference["field"])
            if not isinstance(value, str) or not value:
                raise ConfigurationError(
                    f"preferred unit field {preference['field']} must resolve to non-empty text"
                )
            preferred_by_split[preference["split"]].add(value)
            if preference["require_all"]:
                required_preferred_by_split[preference["split"]].add(value)

    def includes_required_preferences(
        candidate_choice: dict[str, list[_SelectionUnit]],
    ) -> bool:
        return all(
            required_preferred_by_split[split_name].issubset(
                {unit.unit_id for unit in candidate_choice[split_name]}
            )
            for split_name in ("train", "validation")
        )
    split_strata = {
        name: [row for row in selection["strata"] if row["split"] == name]
        for name in ("train", "validation")
    }
    chosen_by_split: dict[str, list[_SelectionUnit]] | None = None
    available_report: dict[str, dict[str, int]] = {}
    selected_report: dict[str, dict[str, int]] = {}

    exact_required_selection = (
        split["strategy"] == "selection_partition_v1"
        and all(
            len(required_preferred_by_split[split_name])
            == sum(row["count"] for row in split_strata[split_name])
            for split_name in ("train", "validation")
        )
    )
    if exact_required_selection:
        units_by_id = {unit.unit_id: unit for unit in units}
        if len(units_by_id) != len(units):
            raise ConfigurationError(
                "required preferred selection needs globally unique selection unit IDs"
            )
        missing_by_split = {
            split_name: sorted(
                required_preferred_by_split[split_name] - set(units_by_id)
            )
            for split_name in ("train", "validation")
        }
        missing_by_split = {
            split_name: values
            for split_name, values in missing_by_split.items()
            if values
        }
        if missing_by_split:
            raise ConfigurationError(
                "required preferred selection units are absent from the candidate pool: "
                f"{missing_by_split}"
            )
        exact_choice = {
            split_name: [
                units_by_id[unit_id]
                for unit_id in sorted(required_preferred_by_split[split_name])
            ]
            for split_name in ("train", "validation")
        }
        train_groups = {
            unit.leakage_group_id for unit in exact_choice["train"]
        }
        validation_groups = {
            unit.leakage_group_id for unit in exact_choice["validation"]
        }
        overlapping_groups = sorted(train_groups & validation_groups)
        if overlapping_groups:
            raise ConfigurationError(
                "required preferred units cross train/validation leakage groups: "
                f"{overlapping_groups}"
            )
        validated_choice: dict[str, list[_SelectionUnit]] = {}
        for split_name in ("train", "validation"):
            result = _choose_quota_units(
                exact_choice[split_name],
                split_strata[split_name],
                selection,
                split_name=split_name,
                attempt=0,
                preferred_ids=required_preferred_by_split[split_name],
            )
            if result is None:
                raise ConfigurationError(
                    f"required preferred units do not satisfy {split_name} quotas or "
                    "resource constraints"
                )
            validated_choice[split_name], _, selected = result
            available_counts: Counter[str] = Counter()
            for unit in units:
                matched = _matching_stratum(unit, split_strata[split_name])
                if matched is not None:
                    available_counts[matched] += 1
            available_report[split_name] = {
                row["id"]: available_counts[row["id"]]
                for row in split_strata[split_name]
            }
            selected_report[split_name] = selected
        coverage_ok, _ = _coverage_report(validated_choice, selection)
        if not coverage_ok:
            raise ConfigurationError(
                "required preferred units do not satisfy configured coverage"
            )
        chosen_by_split = validated_choice
    elif split["strategy"] == "metadata_partition_v1":
        fixed: dict[str, list[_SelectionUnit]] = {"train": [], "validation": [], "holdout": []}
        group_splits: dict[str, str] = {}
        for unit in units:
            unit_splits = {_metadata_split_name(record, split) for record, _ in unit.rows}
            if len(unit_splits) != 1:
                raise ConfigurationError(
                    f"selection unit {unit.unit_id} crosses metadata partitions: {unit_splits}"
                )
            split_name = next(iter(unit_splits))
            previous = group_splits.setdefault(unit.leakage_group_id, split_name)
            if previous != split_name:
                raise ConfigurationError(
                    f"leakage group {unit.leakage_group_id} crosses metadata partitions"
                )
            fixed[split_name].append(unit)
        for attempt in range(selection["max_attempts"]):
            candidate_choice: dict[str, list[_SelectionUnit]] = {}
            failed = False
            for split_name in ("train", "validation"):
                result = _choose_quota_units(
                    fixed[split_name],
                    split_strata[split_name],
                    selection,
                    split_name=split_name,
                    attempt=attempt,
                    preferred_ids=preferred_by_split[split_name],
                )
                if result is None:
                    failed = True
                    break
                candidate_choice[split_name], available, selected = result
                available_report[split_name] = available
                selected_report[split_name] = selected
            if failed:
                continue
            coverage_ok, _ = _coverage_report(candidate_choice, selection)
            if coverage_ok and includes_required_preferences(candidate_choice):
                chosen_by_split = candidate_choice
                break
    elif split["strategy"] == "selection_partition_v1":
        matches = {
            split_name: {
                unit.unit_id: _matching_stratum(unit, split_strata[split_name])
                for unit in units
            }
            for split_name in ("train", "validation")
        }
        quotas = {
            split_name: {row["id"]: row["count"] for row in split_strata[split_name]}
            for split_name in ("train", "validation")
        }
        grouped: dict[str, list[_SelectionUnit]] = defaultdict(list)
        for unit in units:
            grouped[unit.leakage_group_id].append(unit)
        all_train_counts = Counter(
            matched for matched in matches["train"].values() if matched is not None
        )
        for attempt in range(selection["max_attempts"]):
            validation_groups: set[str] = set()
            validation_available: Counter[str] = Counter()
            remaining_train = Counter(all_train_counts)

            def can_move(group_id: str) -> bool:
                train_counts = Counter(
                    matches["train"][unit.unit_id]
                    for unit in grouped[group_id]
                    if matches["train"][unit.unit_id] is not None
                )
                return all(
                    remaining_train[stratum_id] - count >= quotas["train"][stratum_id]
                    for stratum_id, count in train_counts.items()
                )

            def move(group_id: str) -> None:
                validation_groups.add(group_id)
                for unit in grouped[group_id]:
                    train_match = matches["train"][unit.unit_id]
                    validation_match = matches["validation"][unit.unit_id]
                    if train_match is not None:
                        remaining_train[train_match] -= 1
                    if validation_match is not None:
                        validation_available[validation_match] += 1

            validation_coverage = [
                target for target in selection["coverage"] if target["split"] == "validation"
            ]
            coverage_candidates: list[tuple[int, str, str]] = []
            for target in validation_coverage:
                for value in target["values"]:
                    eligible_groups = {
                        unit.leakage_group_id
                        for unit in units
                        if matches["validation"][unit.unit_id] is not None
                        and str(_unit_path_value(unit, target["field"])) == str(value)
                    }
                    coverage_candidates.append((len(eligible_groups), target["field"], str(value)))
            coverage_failed = False
            for _, field, value in sorted(coverage_candidates):
                if any(
                    str(_unit_path_value(unit, field)) == value
                    for group_id in validation_groups
                    for unit in grouped[group_id]
                    if matches["validation"][unit.unit_id] is not None
                ):
                    continue
                eligible = [
                    group_id
                    for group_id, group_units in grouped.items()
                    if group_id not in validation_groups
                    and any(
                        matches["validation"][unit.unit_id] is not None
                        and str(_unit_path_value(unit, field)) == value
                        for unit in group_units
                    )
                    and can_move(group_id)
                ]
                if not eligible:
                    coverage_failed = True
                    break
                chosen_group = min(
                    eligible,
                    key=lambda group_id: stable_hash(
                        {
                            "strategy": selection["strategy"],
                            "seed": selection["seed"],
                            "attempt": attempt,
                            "coverage_field": field,
                            "coverage_value": value,
                            "group": group_id,
                        }
                    ),
                )
                move(chosen_group)
            if coverage_failed:
                continue

            ordered_groups = sorted(
                grouped,
                key=lambda group_id: (
                    sum(
                        unit.unit_id in preferred_by_split["train"]
                        for unit in grouped[group_id]
                    ),
                    -sum(
                        unit.unit_id in preferred_by_split["validation"]
                        for unit in grouped[group_id]
                    ),
                    stable_hash(
                        {
                            "strategy": selection["strategy"],
                            "seed": selection["seed"],
                            "attempt": attempt,
                            "group": group_id,
                        }
                    ),
                ),
            )
            for group_id in ordered_groups:
                if group_id in validation_groups:
                    continue
                if all(
                    validation_available[stratum_id] >= count
                    for stratum_id, count in quotas["validation"].items()
                ):
                    break
                contributes = any(
                    matches["validation"][unit.unit_id] is not None
                    and validation_available[matches["validation"][unit.unit_id]]
                    < quotas["validation"][matches["validation"][unit.unit_id]]
                    for unit in grouped[group_id]
                )
                if contributes and can_move(group_id):
                    move(group_id)
            if not all(
                validation_available[stratum_id] >= count
                for stratum_id, count in quotas["validation"].items()
            ):
                continue
            candidate_choice = {}
            failed = False
            for split_name, eligible_units in (
                (
                    "validation",
                    [unit for group_id in validation_groups for unit in grouped[group_id]],
                ),
                (
                    "train",
                    [
                        unit
                        for group_id, group_units in grouped.items()
                        if group_id not in validation_groups
                        for unit in group_units
                    ],
                ),
            ):
                result = _choose_quota_units(
                    eligible_units,
                    split_strata[split_name],
                    selection,
                    split_name=split_name,
                    attempt=attempt,
                    preferred_ids=preferred_by_split[split_name],
                )
                if result is None:
                    failed = True
                    break
                candidate_choice[split_name], available, selected = result
                available_report[split_name] = available
                selected_report[split_name] = selected
            if failed:
                continue
            coverage_ok, _ = _coverage_report(candidate_choice, selection)
            if coverage_ok and includes_required_preferences(candidate_choice):
                chosen_by_split = candidate_choice
                break
    else:
        raise ConfigurationError(
            "partitioned_group_quota_v1 requires metadata_partition_v1 or "
            "selection_partition_v1"
        )

    if chosen_by_split is None:
        raise ConfigurationError(
            "partitioned group quotas are infeasible after "
            f"{selection['max_attempts']} deterministic attempts"
        )

    selected: list[tuple[CanonicalRecord, dict[str, Any]]] = []
    sample_splits: dict[str, str] = {}
    selected_units_by_split: dict[str, list[_SelectionUnit]] = {
        "train": chosen_by_split["train"],
        "validation": chosen_by_split["validation"],
    }
    selected_sample_ids: set[str] = set()
    for split_name, split_units in selected_units_by_split.items():
        for unit in split_units:
            for candidate in unit.rows:
                sample_id = candidate[0].sample_id
                previous = sample_splits.setdefault(sample_id, split_name)
                if previous != split_name:
                    raise AssertionError(f"sample {sample_id} selected into multiple splits")
                if sample_id not in selected_sample_ids:
                    selected.append(candidate)
                    selected_sample_ids.add(sample_id)

    if "holdout" in selection["passthrough_splits"]:
        if split["strategy"] != "metadata_partition_v1":
            raise ConfigurationError("holdout passthrough requires metadata_partition_v1")
        for candidate in candidates:
            if _metadata_split_name(candidate[0], split) != "holdout":
                continue
            sample_id = candidate[0].sample_id
            if sample_id in selected_sample_ids:
                raise AssertionError(f"holdout sample {sample_id} was selected for training")
            selected.append(candidate)
            selected_sample_ids.add(sample_id)
            sample_splits[sample_id] = "holdout"

    coverage_ok, coverage = _coverage_report(selected_units_by_split, selection)
    if not coverage_ok:
        raise AssertionError("selected units failed configured coverage")
    resource_usage: dict[str, dict[str, dict[str, Any]]] = {}
    if selection.get("resource_capacities"):
        for split_name, split_units in selected_units_by_split.items():
            counts: Counter[str] = Counter(
                resource_id
                for unit in split_units
                for resource_id in unit.resource_ids
            )
            by_type: dict[str, dict[str, Any]] = {}
            for resource_type in selection["resource_capacities"][split_name]:
                type_counts = [
                    count
                    for resource_id, count in counts.items()
                    if _resource_type(resource_id) == resource_type
                ]
                by_type[resource_type] = {
                    "unique_resources": len(type_counts),
                    "total_uses": sum(type_counts),
                    "max_uses": max(type_counts, default=0),
                    "usage_histogram": dict(sorted(Counter(type_counts).items())),
                }
            resource_usage[split_name] = by_type
    report = {
        **selection,
        "available_units": available_report,
        "selected_units": selected_report,
        "selected_unit_totals": {
            split_name: len(split_units)
            for split_name, split_units in selected_units_by_split.items()
        },
        "selected_records": len(selected),
        "coverage_report": coverage,
        "preferred_units": {
            split_name: {
                "available": len(preferred_by_split[split_name]),
                "selected": sum(
                    unit.unit_id in preferred_by_split[split_name]
                    for unit in split_units
                ),
                "required": len(required_preferred_by_split[split_name]),
            }
            for split_name, split_units in selected_units_by_split.items()
        },
    }
    if resource_usage:
        report["resource_usage"] = resource_usage
    return selected, report, sample_splits


def _select_records(
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]],
    selection: dict[str, Any] | None,
    split: dict[str, Any] | None = None,
) -> tuple[
    list[tuple[CanonicalRecord, dict[str, Any]]],
    dict[str, Any] | None,
    dict[str, str],
]:
    if selection is None:
        return candidates, None, {}

    exclusion_files = selection.get("exclude_overlap_query_files", [])
    if exclusion_files:
        excluded_trajectory_ids: set[str] = set()
        excluded_leakage_groups: set[str] = set()
        for value in exclusion_files:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(path)
            for row in read_jsonl(path):
                trajectory_id = row.get("trajectory_id")
                leakage_group_id = row.get("leakage_group_id")
                if not isinstance(trajectory_id, str) or not trajectory_id:
                    raise ConfigurationError(
                        f"selection exclusion row in {path} lacks trajectory_id"
                    )
                if not isinstance(leakage_group_id, str) or not leakage_group_id:
                    raise ConfigurationError(
                        f"selection exclusion row in {path} lacks leakage_group_id"
                    )
                excluded_trajectory_ids.add(trajectory_id)
                excluded_leakage_groups.add(leakage_group_id)
        candidates = [
            candidate
            for candidate in candidates
            if candidate[0].trajectory_id not in excluded_trajectory_ids
            and candidate[0].leakage_group_id not in excluded_leakage_groups
        ]

    if selection["strategy"] == "partitioned_group_quota_v1":
        if split is None:
            raise ConfigurationError("partitioned selection requires a training split config")
        return _select_partitioned_records(candidates, selection, split)

    if selection["strategy"] == "deterministic_group_quota_v1":
        units = _selection_units(candidates, selection)
        units_by_stratum: dict[str, list[_SelectionUnit]] = {
            row["id"]: [] for row in selection["strata"]
        }
        membership: dict[str, str | None] = {}
        for unit in units:
            matched = _matching_stratum(unit, selection["strata"])
            membership[unit.unit_id] = matched
            if matched is not None:
                units_by_stratum[matched].append(unit)

        selected_units: dict[str, _SelectionUnit] = {}
        available_counts = {}
        selected_counts = {}
        for stratum in selection["strata"]:
            stratum_id = stratum["id"]
            available = units_by_stratum[stratum_id]
            available_counts[stratum_id] = len(available)
            if len(available) < stratum["count"]:
                raise ConfigurationError(
                    f"selection stratum {stratum_id!r} has {len(available)} units, "
                    f"below quota {stratum['count']}"
                )
            ordered = sorted(
                available,
                key=lambda unit: stable_hash(
                    {
                        "strategy": selection["strategy"],
                        "seed": selection["seed"],
                        "stratum": stratum_id,
                        "unit": unit.unit_id,
                    }
                ),
            )
            chosen = ordered[: stratum["count"]]
            for unit in chosen:
                if unit.exclusive_id in {
                    existing.exclusive_id for existing in selected_units.values()
                }:
                    raise ConfigurationError(
                        f"selection unit exclusivity collision: {unit.exclusive_id}"
                    )
                selected_units[unit.unit_id] = unit
            selected_counts[stratum_id] = len(chosen)

        selected = [
            candidate
            for unit in selected_units.values()
            for candidate in unit.rows
        ]
        report = {
            **selection,
            "available_units": dict(sorted(available_counts.items())),
            "selected_units": dict(sorted(selected_counts.items())),
            "total_available_units": len(units),
            "total_matching_units": sum(value is not None for value in membership.values()),
            "total_selected_units": len(selected_units),
            "selected_records": len(selected),
        }
        return selected, report, {}

    unit_strata: dict[str, str] = {}
    units_by_stratum: dict[str, set[str]] = {}
    for record, _ in candidates:
        record_mapping = record.to_dict()
        unit = str(_path_value(record_mapping, selection["unit_field"]))
        stratum = str(_path_value(record_mapping, selection["field"]))
        previous = unit_strata.setdefault(unit, stratum)
        if previous != stratum:
            raise ConfigurationError(
                f"selection unit {unit!r} appears in multiple strata: "
                f"{previous!r}, {stratum!r}"
            )
        units_by_stratum.setdefault(stratum, set()).add(unit)

    observed = set(units_by_stratum)
    configured = set(selection["caps"])
    if observed != configured:
        raise ConfigurationError(
            "selection strata do not match configured caps: "
            f"observed={sorted(observed)}, configured={sorted(configured)}"
        )

    selected_units: set[str] = set()
    available_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for stratum, cap in selection["caps"].items():
        units = units_by_stratum[stratum]
        available_counts[stratum] = len(units)
        if len(units) < cap:
            raise ConfigurationError(
                f"selection stratum {stratum!r} has {len(units)} units, below cap {cap}"
            )
        ordered = sorted(
            units,
            key=lambda unit: stable_hash(
                {
                    "strategy": selection["strategy"],
                    "seed": selection["seed"],
                    "stratum": stratum,
                    "unit": unit,
                }
            ),
        )
        chosen = ordered[:cap]
        selected_units.update(chosen)
        selected_counts[stratum] = len(chosen)

    selected = [
        (record, mixture)
        for record, mixture in candidates
        if record.trajectory_id in selected_units
    ]
    report = {
        **selection,
        "available_units": dict(sorted(available_counts.items())),
        "selected_units": dict(sorted(selected_counts.items())),
        "dropped_units": {
            stratum: available_counts[stratum] - selected_counts[stratum]
            for stratum in sorted(available_counts)
        },
        "total_available_units": sum(available_counts.values()),
        "total_selected_units": len(selected_units),
    }
    return selected, report, {}


def _group_split(group: str, split: dict[str, Any]) -> str:
    digest = stable_hash({"seed": split["seed"], "leakage_group_id": group})
    uniform = int(digest[:16], 16) / float(16**16)
    return "train" if uniform < split["train_fraction"] else "validation"


def _path_value(value: dict[str, Any], field: str) -> Any:
    current: Any = value
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigurationError(f"record_loss_weight_field not found: {field}")
        current = current[part]
    return current


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _summary(
    rows: list[dict[str, Any]],
    *,
    report_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    loss_weight_by_label: Counter[str] = Counter()
    loss_weight_by_level: Counter[str] = Counter()
    for row in rows:
        loss_weight_by_label[str(row["label_name"])] += float(row["loss_weight"])
        loss_weight_by_level[str(row["factor_level"])] += float(row["loss_weight"])
    summary = {
        "queries": len(rows),
        "labels": _counter(rows, "label_name"),
        "factor_levels": _counter(rows, "factor_level"),
        "cells": _counter(rows, "cell"),
        "families": _counter(rows, "family"),
        "sources": _counter(rows, "source"),
        "mixture_ids": _counter(rows, "mixture_id"),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "leakage_groups": len({row["leakage_group_id"] for row in rows}),
        "sampling_weight_sum": sum(float(row["sampling_weight"]) for row in rows),
        "loss_weight_sum": sum(float(row["loss_weight"]) for row in rows),
        "loss_weight_by_label": dict(sorted(loss_weight_by_label.items())),
        "loss_weight_by_factor_level": dict(sorted(loss_weight_by_level.items())),
    }
    for field in ("benchmark_source", "factor_scope", "negative_type", "condition"):
        values = [row.get("metadata", {}).get(field) for row in rows]
        if rows and all(value is not None for value in values):
            summary[f"metadata_{field}"] = dict(
                sorted(Counter(str(value) for value in values).items())
            )
    for field in report_fields:
        values = [_path_value(row, field) for row in rows]
        summary[f"report_{field.replace('.', '_')}"] = dict(
            sorted(Counter(str(value) for value in values).items())
        )
    factor_instances = {
        row.get("factor_instance_id")
        or row.get("metadata", {}).get("factor_instance_id")
        for row in rows
    }
    factor_instances.discard(None)
    if factor_instances:
        summary["factor_instances"] = len(factor_instances)
    return summary


def _factor_instance_id(row: dict[str, Any]) -> str:
    value = row.get("factor_instance_id")
    if not isinstance(value, str) or not value:
        metadata = row.get("metadata")
        value = metadata.get("factor_instance_id") if isinstance(metadata, dict) else None
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"query {row.get('query_id')} lacks factor_instance_id")
    return value


def _validate_factor_instance_level_sets(
    rows: list[dict[str, Any]],
    *,
    where: str,
    allowed_level_sets: tuple[frozenset[str], ...] = (frozenset({"L1", "L3"}),),
) -> None:
    instances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        instances[_factor_instance_id(row)].append(row)
    for factor_instance_id, instance_rows in instances.items():
        levels = {row["factor_level"] for row in instance_rows}
        labels = {row["label_id"] for row in instance_rows}
        trajectory_ids = {row["trajectory_id"] for row in instance_rows}
        leakage_groups = {row["leakage_group_id"] for row in instance_rows}
        splits = {str(row.get("split") or "query") for row in instance_rows}
        if (
            len(instance_rows) != len(levels)
            or frozenset(levels) not in allowed_level_sets
            or len(labels) != 1
            or len(trajectory_ids) != 1
            or len(leakage_groups) != 1
            or len(splits) != 1
        ):
            raise ConfigurationError(
                f"{where} factor instance {factor_instance_id} violates the declared "
                f"level policy {sorted(sorted(value) for value in allowed_level_sets)}"
            )


def _validate_strict_l1_l3_instances(rows: list[dict[str, Any]], *, where: str) -> None:
    _validate_factor_instance_level_sets(rows, where=where)


def _trajectory(record: CanonicalRecord) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "trajectory_id": record.trajectory_id,
        "task_id": record.task_id,
        "task_context": record.task_context,
        "model_cot": record.model_cot,
    }


def _query(
    record: CanonicalRecord,
    *,
    role: str,
    mixture: dict[str, Any],
    split: str | None,
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    query = {
        "schema_version": 1,
        "query_id": stable_hash(
            {
                "contract": CONTRACT_ID,
                "role": role,
                "sample_id": record.sample_id,
                "factor_level": record.factor_level,
            }
        ),
        "sample_id": record.sample_id,
        "trajectory_id": record.trajectory_id,
        "task_id": record.task_id,
        "leakage_group_id": record.leakage_group_id,
        "cell": record.cell,
        "family": record.family,
        "source": record.source,
        "mixture_id": mixture["mixture_id"],
        "factor_level": record.factor_level,
        "factor_spec": asdict(record.factor_spec),
        "label_id": record.label,
        "label_name": LABEL_NAMES[record.label],
        "label_provenance": record.label_provenance,
        "evidence_spans": list(record.evidence_spans),
        "metadata": record.metadata,
        "sampling_weight": mixture["sampling_weight"],
        "loss_weight": mixture["loss_weight"],
        "prompt_contract_id": CONTRACT_ID,
        "prompt_contract_hash": contract_hash(),
    }
    factor_instance_id_field = mixture.get("factor_instance_id_field")
    if factor_instance_id_field:
        source_identity = _path_value(record.to_dict(), factor_instance_id_field)
        if not isinstance(source_identity, str) or not source_identity:
            raise ConfigurationError(
                f"record {record.sample_id} factor instance field "
                f"{factor_instance_id_field!r} must resolve to non-empty text"
            )
        query["factor_instance_id"] = stable_hash(
            {
                "mixture_id": mixture["mixture_id"],
                "trajectory_id": record.trajectory_id,
                "source_identity": source_identity,
            }
        )
    record_weight_field = mixture.get("record_loss_weight_field")
    if record_weight_field:
        record_weight = _positive_number(
            _path_value(record.to_dict(), str(record_weight_field)),
            f"record {record.sample_id} loss weight",
        )
        query["loss_weight"] *= record_weight
        query["record_loss_weight"] = record_weight
    if split is not None:
        query["split"] = split
    if evaluation is not None:
        query["evaluation"] = evaluation
    return query


def _validate_factor_instances(
    rows: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> None:
    """Fail closed when a config declares semantic L1/L3 instance identities."""

    expected_levels = {
        source["mixture_id"]: set(source["include_levels"])
        for source in sources
        if source.get("factor_instance_id_field")
    }
    if not expected_levels:
        return
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["mixture_id"] not in expected_levels:
            continue
        factor_instance_id = row.get("factor_instance_id")
        if not isinstance(factor_instance_id, str) or not factor_instance_id:
            raise AssertionError(f"query {row['query_id']} lacks materialized factor_instance_id")
        grouped.setdefault((row["mixture_id"], factor_instance_id), []).append(row)

    for (mixture_id, factor_instance_id), instance_rows in grouped.items():
        levels = [str(row["factor_level"]) for row in instance_rows]
        expected = expected_levels[mixture_id]
        if len(levels) != len(expected) or set(levels) != expected:
            raise ConfigurationError(
                f"factor instance {factor_instance_id} from {mixture_id} must contain "
                f"exactly {sorted(expected)}; observed={levels}"
            )
        labels = {int(row["label_id"]) for row in instance_rows}
        trajectories = {str(row["trajectory_id"]) for row in instance_rows}
        leakage_groups = {str(row["leakage_group_id"]) for row in instance_rows}
        if len(labels) != 1 or len(trajectories) != 1 or len(leakage_groups) != 1:
            raise ConfigurationError(
                f"factor instance {factor_instance_id} violates paired-record invariants"
            )


def _composite_component(
    value: Any, index: int, *, config_dir: Path
) -> dict[str, Any]:
    where = f"components[{index}]"
    component = require_mapping(value, where)
    required = {"component_id", "directory"}
    require_keys(component, required, where)
    reject_unknown(component, required | {"split_selection"}, where)
    component_id = component["component_id"]
    if not isinstance(component_id, str) or not component_id.strip():
        raise ConfigurationError(f"{where}.component_id must be non-empty text")
    split_selection_value = component.get("split_selection")
    split_selection: dict[str, dict[str, Any]] | None = None
    if split_selection_value is not None:
        split_selection_mapping = require_mapping(
            split_selection_value, f"{where}.split_selection"
        )
        unexpected_splits = set(split_selection_mapping) - {
            "train",
            "validation",
            "holdout",
        }
        if unexpected_splits:
            raise ConfigurationError(
                f"{where}.split_selection has unknown splits: {sorted(unexpected_splits)}"
            )
        if not split_selection_mapping:
            raise ConfigurationError(f"{where}.split_selection must not be empty")
        split_selection = {}
        for split_name, raw_rule in split_selection_mapping.items():
            rule_where = f"{where}.split_selection.{split_name}"
            rule = require_mapping(raw_rule, rule_where)
            reject_unknown(
                rule,
                {"include_levels", "include_families", "exclude_overlap_with"},
                rule_where,
            )
            include_levels = rule.get("include_levels", list(LEVELS))
            if (
                not isinstance(include_levels, list)
                or not include_levels
                or any(level not in LEVELS for level in include_levels)
                or len(include_levels) != len(set(include_levels))
            ):
                raise ConfigurationError(
                    f"{rule_where}.include_levels must be a unique non-empty subset of "
                    f"{list(LEVELS)}"
                )
            exclude_overlap_with = rule.get("exclude_overlap_with", [])
            if (
                not isinstance(exclude_overlap_with, list)
                or any(
                    not isinstance(reference, str) or not reference.strip()
                    for reference in exclude_overlap_with
                )
                or len(exclude_overlap_with) != len(set(exclude_overlap_with))
            ):
                raise ConfigurationError(
                    f"{rule_where}.exclude_overlap_with must be a unique text list"
                )
            include_families = rule.get("include_families")
            if include_families is not None and (
                not isinstance(include_families, list)
                or not include_families
                or any(
                    not isinstance(family, str) or not family.strip()
                    for family in include_families
                )
                or len(include_families) != len(set(include_families))
            ):
                raise ConfigurationError(
                    f"{rule_where}.include_families must be a unique non-empty text list"
                )
            split_selection[split_name] = {
                "include_levels": list(include_levels),
                "include_families": (
                    [family.strip() for family in include_families]
                    if include_families is not None
                    else None
                ),
                "exclude_overlap_with": [
                    reference.strip() for reference in exclude_overlap_with
                ],
            }
    return {
        "component_id": component_id.strip(),
        "directory": resolve_path(component["directory"], relative_to=config_dir),
        "split_selection": split_selection,
    }


def _composite_factor_instance_level_sets(
    value: Any,
) -> tuple[frozenset[str], ...]:
    if value is None:
        return (frozenset({"L1", "L3"}),)
    if not isinstance(value, list) or not value:
        raise ConfigurationError("factor_instance_level_sets must be a non-empty list")
    parsed: list[frozenset[str]] = []
    for index, raw_levels in enumerate(value):
        if (
            not isinstance(raw_levels, list)
            or not raw_levels
            or any(level not in LEVELS for level in raw_levels)
            or len(raw_levels) != len(set(raw_levels))
        ):
            raise ConfigurationError(
                f"factor_instance_level_sets[{index}] must be a unique non-empty subset "
                f"of {list(LEVELS)}"
            )
        level_set = frozenset(raw_levels)
        if level_set in parsed:
            raise ConfigurationError("factor_instance_level_sets contains duplicates")
        parsed.append(level_set)
    return tuple(parsed)


def _compose_train_dataset(
    config_path: Path,
    config: dict[str, Any],
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Compose frozen train artifacts without reselecting or rewriting query semantics."""

    required = {"schema_version", "kind", "components", "required_families", "provenance", "output"}
    require_keys(config, required, "composite build config")
    reject_unknown(
        config,
        required | {"factor_instance_level_sets", "trajectory_id_collision_policy"},
        "composite build config",
    )
    config_dir = config_path.parent

    components_value = config["components"]
    if not isinstance(components_value, list) or not components_value:
        raise ConfigurationError("components must be a non-empty list")
    components = [
        _composite_component(value, index, config_dir=config_dir)
        for index, value in enumerate(components_value)
    ]
    component_ids = [component["component_id"] for component in components]
    if len(component_ids) != len(set(component_ids)):
        raise ConfigurationError("components contain duplicate component_id values")
    component_id_set = set(component_ids)
    for component in components:
        for split_name, rule in (component["split_selection"] or {}).items():
            for reference in rule["exclude_overlap_with"]:
                if reference == component["component_id"]:
                    raise ConfigurationError(
                        f"component {reference} cannot exclude overlap with itself"
                    )
                if reference not in component_id_set:
                    raise ConfigurationError(
                        f"component {component['component_id']} {split_name} references "
                        f"unknown overlap component {reference}"
                    )

    allowed_level_sets = _composite_factor_instance_level_sets(
        config.get("factor_instance_level_sets")
    )

    required_families = config["required_families"]
    if (
        not isinstance(required_families, list)
        or not required_families
        or any(not isinstance(value, str) or not value.strip() for value in required_families)
        or len(required_families) != len(set(required_families))
    ):
        raise ConfigurationError("required_families must be a unique non-empty text list")
    required_family_set = set(required_families)
    trajectory_id_collision_policy = config.get(
        "trajectory_id_collision_policy", "reject"
    )
    if trajectory_id_collision_policy not in {"reject", "namespace_cross_family"}:
        raise ConfigurationError(
            "trajectory_id_collision_policy must be reject or namespace_cross_family"
        )
    provenance = require_mapping(config["provenance"], "provenance")

    output = require_mapping(config["output"], "output")
    require_keys(output, {"directory"}, "output")
    reject_unknown(output, {"directory"}, "output")
    output_dir = resolve_path(output["directory"], relative_to=config_dir)
    expected_output_names = {
        "trajectories.jsonl",
        "train_queries.jsonl",
        "validation_queries.jsonl",
        "holdout_queries.jsonl",
        "effective_config.json",
        "summary.json",
    }
    if any(output_dir == component["directory"] for component in components):
        raise ConfigurationError("composite output directory must differ from every component")
    if output_dir.exists() and any(output_dir.iterdir()) and not dry_run:
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite")
        unexpected = sorted(
            path.name
            for path in output_dir.iterdir()
            if path.name not in expected_output_names and path.name not in MANAGED_OUTPUT_NAMES
        )
        if unexpected:
            raise FileExistsError(
                f"refusing to overwrite directory with unowned artifacts: {unexpected}"
            )

    child_trajectory_rows: dict[str, dict[str, dict[str, Any]]] = {}
    child_split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    child_config_hashes: dict[str, Any] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "holdout": [],
    }
    seen_query_ids: set[str] = set()
    component_reports: list[dict[str, Any]] = []
    component_inputs: list[dict[str, Any]] = []
    required_child_names = {
        "trajectories.jsonl",
        "train_queries.jsonl",
        "validation_queries.jsonl",
        "holdout_queries.jsonl",
        "effective_config.json",
        "summary.json",
    }

    for component in components:
        component_id = component["component_id"]
        directory = component["directory"]
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        missing = sorted(name for name in required_child_names if not (directory / name).is_file())
        if missing:
            raise FileNotFoundError(f"component {component_id} is missing files: {missing}")
        child_effective = load_config(directory / "effective_config.json")
        child_summary = load_config(directory / "summary.json")
        if child_effective.get("kind") not in {
            "train_dataset",
            COMPOSITE_KIND,
            LENGTH_PARTITION_KIND,
        }:
            raise ConfigurationError(
                f"component {component_id} is not a reusable train artifact"
            )

        child_outputs = require_mapping(child_summary.get("outputs"), f"{component_id} summary.outputs")
        file_inputs: dict[str, dict[str, Any]] = {}
        for name in sorted(required_child_names):
            path = directory / name
            digest = file_sha256(path)
            declared = child_outputs.get(name)
            if name != "summary.json":
                if not isinstance(declared, dict) or declared.get("sha256") != digest:
                    raise ConfigurationError(
                        f"component {component_id} file hash disagrees with its summary: {name}"
                    )
            file_inputs[name] = {"path": str(path), "sha256": digest}

        child_trajectories = read_jsonl(directory / "trajectories.jsonl")
        child_trajectory_ids: set[str] = set()
        child_trajectory_map: dict[str, dict[str, Any]] = {}
        for row in child_trajectories:
            trajectory_id = row.get("trajectory_id")
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise ConfigurationError(
                    f"component {component_id} has a trajectory without a valid trajectory_id"
                )
            if trajectory_id in child_trajectory_ids:
                raise ConfigurationError(
                    f"component {component_id} contains duplicate trajectory_id {trajectory_id}"
                )
            child_trajectory_ids.add(trajectory_id)
            child_trajectory_map[trajectory_id] = row

        raw_splits: dict[str, list[dict[str, Any]]] = {}
        for split_name in ("train", "validation", "holdout"):
            rows = read_jsonl(directory / f"{split_name}_queries.jsonl")
            raw_splits[split_name] = rows
            for row in rows:
                query_id = row.get("query_id")
                trajectory_id = row.get("trajectory_id")
                if not isinstance(query_id, str) or not query_id:
                    raise ConfigurationError(
                        f"component {component_id} has a query without a valid query_id"
                    )
                if row.get("split") != split_name:
                    raise ConfigurationError(
                        f"query {query_id} split disagrees with {split_name}_queries.jsonl"
                    )
                if trajectory_id not in child_trajectory_ids:
                    raise ConfigurationError(
                        f"query {query_id} references a trajectory outside component {component_id}"
                    )
                if row.get("prompt_contract_id") != CONTRACT_ID or row.get(
                    "prompt_contract_hash"
                ) != contract_hash():
                    raise ConfigurationError(f"query {query_id} uses a different prompt contract")
                _factor_instance_id(row)
        child_trajectory_rows[component_id] = child_trajectory_map
        child_split_rows[component_id] = raw_splits
        child_config_hashes[component_id] = child_effective.get("config_hash")
        component_inputs.append(
            {
                "component_id": component_id,
                "directory": str(directory),
                "split_selection": component["split_selection"],
                "files": file_inputs,
            }
        )

    base_selected_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for component in components:
        component_id = component["component_id"]
        rules = component["split_selection"]
        base_selected_rows[component_id] = {}
        for split_name in ("train", "validation", "holdout"):
            if rules is not None and split_name not in rules:
                base_selected_rows[component_id][split_name] = []
                continue
            include_levels = set(
                rules[split_name]["include_levels"] if rules is not None else LEVELS
            )
            include_families = (
                set(rules[split_name]["include_families"])
                if rules is not None
                and rules[split_name]["include_families"] is not None
                else None
            )
            base_selected_rows[component_id][split_name] = [
                row
                for row in child_split_rows[component_id][split_name]
                if row["factor_level"] in include_levels
                and (include_families is None or row["family"] in include_families)
            ]

    selected_trajectories: dict[str, dict[str, Any]] = {}
    selected_trajectory_families: dict[str, str] = {}
    namespaced_trajectory_count = 0
    for component in components:
        component_id = component["component_id"]
        rules = component["split_selection"]
        selected_counts: dict[str, int] = {}
        excluded_overlap_counts: dict[str, int] = {}
        component_selected_rows: list[dict[str, Any]] = []
        for split_name in ("train", "validation", "holdout"):
            selected = base_selected_rows[component_id][split_name]
            references = (
                rules[split_name]["exclude_overlap_with"]
                if rules is not None and split_name in rules
                else []
            )
            if references:
                excluded_trajectory_ids = {
                    row["trajectory_id"]
                    for reference in references
                    for row in base_selected_rows[reference][split_name]
                }
                excluded_leakage_groups = {
                    row["leakage_group_id"]
                    for reference in references
                    for row in base_selected_rows[reference][split_name]
                }
                before = len(selected)
                selected = [
                    row
                    for row in selected
                    if row["trajectory_id"] not in excluded_trajectory_ids
                    and row["leakage_group_id"] not in excluded_leakage_groups
                ]
                excluded_overlap_counts[split_name] = before - len(selected)
            selected_counts[split_name] = len(selected)
            for source_row in selected:
                row = dict(source_row)
                query_id = row["query_id"]
                if query_id in seen_query_ids:
                    raise ConfigurationError(
                        f"duplicate selected query_id across components: {query_id}"
                    )
                seen_query_ids.add(query_id)
                trajectory_id = row["trajectory_id"]
                trajectory = child_trajectory_rows[component_id][trajectory_id]
                previous = selected_trajectories.get(trajectory_id)
                if previous is not None and previous != trajectory:
                    raise ConfigurationError(
                        f"trajectory_id {trajectory_id} maps to conflicting component content"
                    )
                previous_family = selected_trajectory_families.get(trajectory_id)
                if previous_family is not None and previous_family != row["family"]:
                    if trajectory_id_collision_policy != "namespace_cross_family":
                        raise ConfigurationError(
                            f"trajectory_id {trajectory_id} crosses families "
                            f"{previous_family!r} and {row['family']!r}; select an explicit "
                            "collision policy"
                        )
                    namespaced_id = stable_hash(
                        {
                            "policy": "component_family_trajectory_namespace_v1",
                            "component_id": component_id,
                            "family": row["family"],
                            "trajectory_id": trajectory_id,
                        }
                    )
                    namespaced_trajectory = {
                        **trajectory,
                        "trajectory_id": namespaced_id,
                    }
                    existing_namespaced = selected_trajectories.get(namespaced_id)
                    if (
                        existing_namespaced is not None
                        and existing_namespaced != namespaced_trajectory
                    ):
                        raise AssertionError("namespaced trajectory ID collision")
                    row["trajectory_id"] = namespaced_id
                    trajectory = namespaced_trajectory
                    trajectory_id = namespaced_id
                    if existing_namespaced is None:
                        namespaced_trajectory_count += 1
                selected_trajectories[trajectory_id] = trajectory
                selected_trajectory_families[trajectory_id] = row["family"]
                split_rows[split_name].append(row)
                component_selected_rows.append(row)

        component_reports.append(
            {
                "component_id": component_id,
                "directory": str(component["directory"]),
                "config_hash": child_config_hashes[component_id],
                "input_queries": {
                    split_name: len(child_split_rows[component_id][split_name])
                    for split_name in ("train", "validation", "holdout")
                },
                "selected_queries": selected_counts,
                "excluded_overlap_queries": excluded_overlap_counts,
                "split_selection": rules,
                "input_trajectories": len(child_trajectory_rows[component_id]),
                "selected_trajectories": len(
                    {row["trajectory_id"] for row in component_selected_rows}
                ),
            }
        )

    if not split_rows["train"] or not split_rows["validation"]:
        raise ConfigurationError("composite train and validation splits must both be non-empty")
    for rows in split_rows.values():
        rows.sort(key=lambda row: row["query_id"])

    split_groups = {
        split_name: {row["leakage_group_id"] for row in rows}
        for split_name, rows in split_rows.items()
    }
    for left, right in (
        ("train", "validation"),
        ("train", "holdout"),
        ("validation", "holdout"),
    ):
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            raise ConfigurationError(
                f"composite leakage groups appear in both {left} and {right}: {len(overlap)}"
            )

    all_rows = [
        *split_rows["train"],
        *split_rows["validation"],
        *split_rows["holdout"],
    ]
    observed_train = {row["family"] for row in split_rows["train"]}
    observed_validation = {row["family"] for row in split_rows["validation"]}
    if observed_train != required_family_set or observed_validation != required_family_set:
        raise ConfigurationError(
            "composite required_families mismatch: "
            f"train={sorted(observed_train)}, validation={sorted(observed_validation)}, "
            f"required={sorted(required_family_set)}"
        )
    unexpected_families = {row["family"] for row in all_rows} - required_family_set
    if unexpected_families:
        raise ConfigurationError(
            f"composite contains unexpected families: {sorted(unexpected_families)}"
        )

    _validate_factor_instance_level_sets(
        all_rows,
        where="composite",
        allowed_level_sets=allowed_level_sets,
    )
    factor_instances = {_factor_instance_id(row) for row in all_rows}

    trajectory_rows = sorted(
        selected_trajectories.values(), key=lambda row: row["trajectory_id"]
    )
    serialized_level_sets = [sorted(value) for value in allowed_level_sets]
    effective = {
        "schema_version": 1,
        "kind": COMPOSITE_KIND,
        "config_path": str(config_path),
        "config_hash": stable_hash(config),
        "prompt_contract_id": CONTRACT_ID,
        "prompt_contract_hash": contract_hash(),
        "visibility_scope": "cot_only",
        "components": component_inputs,
        "required_families": list(required_families),
        "factor_instance_level_sets": serialized_level_sets,
        "trajectory_id_collision_policy": trajectory_id_collision_policy,
        "provenance": dict(provenance),
        "output_directory": str(output_dir),
        "repository": repository_state(config_path.parent),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "role": "train",
        "composition": (
            "select_frozen_component_splits_v1"
            if any(component["split_selection"] is not None for component in components)
            else "preserve_frozen_component_splits_v1"
        ),
        "components": component_reports,
        "factor_instance_level_sets": serialized_level_sets,
        "all_queries": _summary(all_rows),
        "splits": {
            split_name: _summary(rows)
            for split_name, rows in split_rows.items()
            if rows
        },
        "trajectories": len(trajectory_rows),
        "namespaced_cross_family_trajectories": namespaced_trajectory_count,
        "factor_instances": len(factor_instances),
        "leakage_overlap": {
            "train_validation": 0,
            "train_holdout": 0,
            "validation_holdout": 0,
        },
        "provenance": dict(provenance),
        "dry_run": dry_run,
    }
    if dry_run:
        return {"effective_config": effective, "summary": summary}

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(MANAGED_OUTPUT_NAMES - expected_output_names):
        stale = output_dir / name
        if stale.is_file():
            stale.unlink()
    write_jsonl(output_dir / "trajectories.jsonl", trajectory_rows)
    for split_name, rows in split_rows.items():
        write_jsonl(output_dir / f"{split_name}_queries.jsonl", rows)
    write_json(output_dir / "effective_config.json", effective)
    summary["outputs"] = {
        path.name: {"path": str(path), "sha256": file_sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    write_json(output_dir / "summary.json", summary)
    summary["outputs"]["summary.json"] = {
        "path": str(output_dir / "summary.json"),
        "sha256_note": "self-hash intentionally omitted",
    }
    write_json(output_dir / "summary.json", summary)
    return {"effective_config": effective, "summary": summary}


def _partition_train_dataset_by_lengths(
    config_path: Path,
    config: dict[str, Any],
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Materialize a frozen no-truncation train or evaluation partition."""

    partition_role = LENGTH_PARTITION_KINDS[config["kind"]]

    required = {
        "schema_version",
        "kind",
        "source",
        "tokenizer",
        "limits",
        "provenance",
        "outputs",
    }
    require_keys(config, required, "length partition config")
    reject_unknown(
        config,
        required | {"architecture", "taxonomy"},
        "length partition config",
    )
    if ("architecture" in config) == ("taxonomy" in config):
        raise ConfigurationError(
            "length partition config requires exactly one of architecture or taxonomy"
        )
    config_dir = config_path.parent

    source = require_mapping(config["source"], "source")
    require_keys(source, {"directory", "audit", "audit_sha256"}, "source")
    reject_unknown(source, {"directory", "audit", "audit_sha256"}, "source")
    source_dir = resolve_path(source["directory"], relative_to=config_dir)
    audit_path = resolve_path(source["audit"], relative_to=source_dir)
    if not source_dir.is_dir() or not audit_path.is_file():
        raise FileNotFoundError(f"length partition source or audit is missing: {source_dir}")
    expected_audit_sha = source["audit_sha256"]
    if not isinstance(expected_audit_sha, str) or file_sha256(audit_path) != expected_audit_sha:
        raise ConfigurationError("length partition audit SHA-256 does not match frozen config")
    audit = load_config(audit_path)

    architecture_path = None
    taxonomy_path = None
    validation_source: dict[str, Any]
    if "architecture" in config:
        architecture_path = resolve_path(config["architecture"], relative_to=config_dir)
        architecture_audit = require_mapping(audit.get("architecture"), "audit.architecture")
        if file_sha256(architecture_path) != architecture_audit.get("sha256"):
            raise ConfigurationError("architecture hash differs from the frozen length audit")
        validation_source = {
            "architecture": {
                "path": str(architecture_path),
                "sha256": file_sha256(architecture_path),
            }
        }
    else:
        taxonomy_path = resolve_path(config["taxonomy"], relative_to=config_dir)
        taxonomy_audit = require_mapping(audit.get("taxonomy"), "audit.taxonomy")
        if file_sha256(taxonomy_path) != taxonomy_audit.get("sha256"):
            raise ConfigurationError("taxonomy hash differs from the frozen length audit")
        validation_source = {
            "taxonomy": {
                "path": str(taxonomy_path),
                "sha256": file_sha256(taxonomy_path),
            }
        }

    tokenizer_config = require_mapping(config["tokenizer"], "tokenizer")
    require_keys(tokenizer_config, {"directory", "revision", "trust_remote_code"}, "tokenizer")
    reject_unknown(
        tokenizer_config,
        {"directory", "revision", "trust_remote_code", "pad_token"},
        "tokenizer",
    )
    tokenizer_path = resolve_path(tokenizer_config["directory"], relative_to=config_dir)
    revision = tokenizer_config["revision"]
    if not isinstance(revision, str) or not revision:
        raise ConfigurationError("tokenizer.revision must be non-empty text")
    if not isinstance(tokenizer_config["trust_remote_code"], bool):
        raise ConfigurationError("tokenizer.trust_remote_code must be boolean")
    pad_token = tokenizer_config.get("pad_token")
    if pad_token is not None and (not isinstance(pad_token, str) or not pad_token):
        raise ConfigurationError("tokenizer.pad_token must be non-empty text when provided")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)

    limits = require_mapping(config["limits"], "limits")
    encoding = limits.get("encoding", "factor_late_fusion")
    if encoding not in {"factor_late_fusion", "endpoint_early_fusion"}:
        raise ConfigurationError(
            "limits.encoding must be factor_late_fusion or endpoint_early_fusion"
        )
    required_limit_fields = {"truncation", "overlength_policy", "length_buckets"}
    allowed_limit_fields = set(required_limit_fields) | {"field_boundary_policy"}
    if encoding == "factor_late_fusion":
        if partition_role == "eval":
            raise ConfigurationError(
                "evaluation length partitions require endpoint_early_fusion encoding"
            )
        required_limit_fields.update({"max_evidence_tokens", "max_factor_tokens"})
        allowed_limit_fields.update(required_limit_fields)
        positive_limits = (
            ("max_evidence_tokens", limits["max_evidence_tokens"]),
            ("max_factor_tokens", limits["max_factor_tokens"]),
        )
        terminal_limit_name = "max_evidence_tokens"
    else:
        required_limit_fields.update({"encoding", "max_sequence_tokens"})
        allowed_limit_fields.update(
            required_limit_fields | {"endpoint_contract_id", "partition_unit"}
        )
        if partition_role == "eval":
            required_limit_fields.update({"endpoint_contract_id", "partition_unit"})
        positive_limits = (("max_sequence_tokens", limits["max_sequence_tokens"]),)
        terminal_limit_name = "max_sequence_tokens"
    require_keys(limits, required_limit_fields, "limits")
    reject_unknown(limits, allowed_limit_fields, "limits")
    field_policy = limits.get(
        "field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY
    )
    if field_policy not in FIELD_BOUNDARY_POLICIES:
        raise ConfigurationError(
            "limits.field_boundary_policy must be one of "
            f"{FIELD_BOUNDARY_POLICIES}"
        )
    for name, value in positive_limits:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"limits.{name} must be a positive integer")
    if limits["truncation"] != "forbidden":
        raise ConfigurationError("length partition truncation must be forbidden")
    expected_overlength_policy = (
        "exclude_to_phase2" if partition_role == "train" else "exclude_to_manifest"
    )
    if limits["overlength_policy"] != expected_overlength_policy:
        raise ConfigurationError(
            "length partition overlength_policy must be "
            f"{expected_overlength_policy} for {partition_role} artifacts"
        )
    if encoding == "endpoint_early_fusion":
        endpoint_contract_id = limits.get(
            "endpoint_contract_id", "endpoint_early_fusion_contract_v1"
        )
        if not isinstance(endpoint_contract_id, str) or not endpoint_contract_id:
            raise ConfigurationError("limits.endpoint_contract_id must be non-empty text")
        partition_unit = limits.get("partition_unit", "factor_instance")
        if partition_unit not in {"query", "factor_instance"}:
            raise ConfigurationError(
                "limits.partition_unit must be query or factor_instance"
            )
    buckets = limits["length_buckets"]
    if (
        not isinstance(buckets, list)
        or not buckets
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in buckets)
        or buckets != sorted(set(buckets))
        or buckets[-1] != limits[terminal_limit_name]
    ):
        raise ConfigurationError(
            "limits.length_buckets must be unique increasing positive integers ending at "
            f"{terminal_limit_name}"
        )

    outputs = require_mapping(config["outputs"], "outputs")
    output_fields = {"accepted_directory", "overlength_directory"}
    require_keys(outputs, output_fields, "outputs")
    reject_unknown(outputs, output_fields, "outputs")
    output_dirs = {
        "accepted": resolve_path(outputs["accepted_directory"], relative_to=config_dir),
        "overlength": resolve_path(outputs["overlength_directory"], relative_to=config_dir),
    }
    if len(set(output_dirs.values())) != 2 or source_dir in output_dirs.values():
        raise ConfigurationError("partition outputs must be distinct and differ from the source")
    split_files = (
        {
            "train": "train_queries.jsonl",
            "validation": "validation_queries.jsonl",
            "holdout": "holdout_queries.jsonl",
        }
        if partition_role == "train"
        else {"query": "eval_queries.jsonl"}
    )
    expected_names = {
        "trajectories.jsonl",
        *split_files.values(),
        "effective_config.json",
        "summary.json",
    }
    for output_dir in output_dirs.values():
        if output_dir.exists() and any(output_dir.iterdir()) and not dry_run:
            if not overwrite:
                raise FileExistsError(
                    f"output directory is not empty: {output_dir}; pass --overwrite"
                )
            unexpected = sorted(
                path.name for path in output_dir.iterdir() if path.name not in expected_names
            )
            if unexpected:
                raise FileExistsError(
                    f"refusing to overwrite directory with unowned artifacts: {unexpected}"
                )

    required_source_names = {
        "trajectories.jsonl",
        *split_files.values(),
        "effective_config.json",
        "summary.json",
    }
    audit_artifact = require_mapping(audit.get("artifact"), "audit.artifact")
    audit_files = require_mapping(audit_artifact.get("files"), "audit.artifact.files")
    missing_audit_files = required_source_names - set(audit_files)
    if missing_audit_files:
        raise ConfigurationError(
            "frozen audit does not cover the complete source artifact: missing "
            + ", ".join(sorted(missing_audit_files))
        )
    source_files: dict[str, dict[str, str]] = {}
    for name in sorted(required_source_names):
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = file_sha256(path)
        declared = require_mapping(audit_files.get(name), f"audit.artifact.files.{name}")
        if digest != declared.get("sha256"):
            raise ConfigurationError(f"source file hash differs from frozen audit: {name}")
        source_files[name] = {"path": str(path), "sha256": digest}

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "length partition requires the training dependencies: "
            "pip install -e 'factor_monitor[training]'"
        ) from exc
    from .training.backend import tokenizer_identity
    from .training.data import (
        endpoint_token_lengths,
        load_trajectory_groups,
        token_limit_factor_instance_partition,
    )

    architecture = None
    taxonomy = None
    if architecture_path is not None:
        from .training.config import load_architecture

        architecture = load_architecture(architecture_path)
    else:
        assert taxonomy_path is not None
        taxonomy = Taxonomy.load(taxonomy_path)
    trajectory_path = source_dir / "trajectories.jsonl"
    groups_by_split = {
        split: load_trajectory_groups(
            trajectory_path,
            source_dir / filename,
            architecture=architecture,
            taxonomy=taxonomy,
            allow_unlabelled=partition_role == "eval",
        )
        for split, filename in split_files.items()
        if (source_dir / filename).stat().st_size > 0
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=tokenizer_config["trust_remote_code"],
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("length partition requires a fast tokenizer with exact offsets")
    if pad_token is not None:
        vocabulary = tokenizer.get_vocab()
        if pad_token not in vocabulary:
            raise ConfigurationError(
                "tokenizer.pad_token must already exist in the tokenizer vocabulary: "
                f"{pad_token!r}"
            )
        tokenizer.pad_token = pad_token
    set_field_boundary_policy(tokenizer, field_policy)
    identity = tokenizer_identity(tokenizer, source=str(tokenizer_path), revision=revision)
    audit_identity = require_mapping(audit.get("tokenizer"), "audit.tokenizer")
    identity_fields = {
        "source",
        "revision",
        "class",
        "name_or_path",
        "vocabulary_hash",
        "special_tokens_map",
        "padding_side",
        "pad_token_id",
    }
    audit_field_policy = audit_identity.get(
        "field_boundary_policy", DEFAULT_FIELD_BOUNDARY_POLICY
    )
    if audit_field_policy != field_policy:
        raise ConfigurationError(
            "loaded tokenizer boundary policy differs from the frozen length audit"
        )
    if audit_identity.get("field_boundary_policy") is not None:
        identity_fields.add("field_boundary_policy")
    if {key: identity.get(key) for key in identity_fields} != {
        key: audit_identity.get(key) for key in identity_fields
    }:
        raise ConfigurationError("loaded tokenizer identity differs from the frozen length audit")

    source_split_rows = {
        split: read_jsonl(source_dir / filename)
        for split, filename in split_files.items()
    }
    source_rows = [row for split in source_split_rows.values() for row in split]
    if encoding == "factor_late_fusion":
        max_evidence_tokens = limits["max_evidence_tokens"]
        max_factor_tokens = limits["max_factor_tokens"]
        partition = token_limit_factor_instance_partition(
            tokenizer,
            groups_by_split,
            max_evidence_tokens=max_evidence_tokens,
            max_factor_tokens=max_factor_tokens,
        )
        scenario = next(
            (
                value
                for value in audit.get("filter_scenarios", [])
                if value.get("max_evidence_tokens") == max_evidence_tokens
                and value.get("max_factor_tokens") == max_factor_tokens
            ),
            None,
        )
        if not isinstance(scenario, dict):
            raise ConfigurationError(
                "frozen audit does not contain the selected length scenario"
            )
        if (
            partition["accepted"]["queries"] != scenario.get("remaining_queries")
            or partition["overlength"]["queries"] != scenario.get("dropped_queries")
            or partition["accepted"]["factor_instances"]
            != scenario.get("remaining_factor_instances")
            or partition["overlength"]["factor_instances"]
            != scenario.get("dropped_factor_instances")
            or partition["accepted"]["by_family"] != scenario.get("remaining_by_family")
            or partition["accepted"]["by_split"] != scenario.get("remaining_by_split")
        ):
            raise ConfigurationError(
                "materialized partition counts differ from the frozen audit"
            )
    else:
        if audit.get("encoding") != endpoint_contract_id:
            raise ConfigurationError(
                "endpoint partition contract differs from the frozen length audit"
            )
        max_sequence_tokens = limits["max_sequence_tokens"]
        audited_scenario = next(
            (
                value
                for value in audit.get("candidate_limits", [])
                if value.get("max_sequence_length") == max_sequence_tokens
            ),
            None,
        )
        if not isinstance(audited_scenario, dict):
            raise ConfigurationError(
                "frozen endpoint audit does not contain max_sequence_tokens"
            )
        lengths_by_query = {
            query_id: length
            for groups in groups_by_split.values()
            for query_id, length in endpoint_token_lengths(
                tokenizer, groups, contract_id=endpoint_contract_id
            )
        }
        if set(lengths_by_query) != {row["query_id"] for row in source_rows}:
            raise AssertionError("endpoint length rows do not cover every source query")
        directly_overlength_queries = {
            query_id
            for query_id, length in lengths_by_query.items()
            if length > max_sequence_tokens
        }
        if len(directly_overlength_queries) != audited_scenario.get("dropped_queries"):
            raise ConfigurationError(
                "materialized endpoint overlength count differs from the frozen audit"
            )
        if partition_unit == "factor_instance":
            query_selection = {
                row["query_id"]: _factor_instance_id(row) for row in source_rows
            }
        else:
            query_selection = {row["query_id"]: row["query_id"] for row in source_rows}
        all_selection_ids = set(query_selection.values())
        overlength_ids = {
            query_selection[query_id] for query_id in directly_overlength_queries
        }
        accepted_ids = all_selection_ids - overlength_ids

        def endpoint_partition_counts(selection_ids: set[str]) -> dict[str, Any]:
            selected = [
                row
                for row in source_rows
                if query_selection[row["query_id"]] in selection_ids
            ]
            return {
                "queries": len(selected),
                "factor_instances": len({_factor_instance_id(row) for row in selected}),
                "trajectories": len({row["trajectory_id"] for row in selected}),
                "by_split": dict(
                    sorted(Counter(str(row.get("split") or "query") for row in selected).items())
                ),
                "by_family": dict(sorted(Counter(row["family"] for row in selected).items())),
                "by_label": dict(
                    sorted(
                        Counter(
                            "applied" if row["label_id"] == 1 else "not_applied"
                            for row in selected
                        ).items()
                    )
                ),
                "by_factor_level": dict(
                    sorted(Counter(row["factor_level"] for row in selected).items())
                ),
            }

        partition = {
            "schema_version": 1,
            "policy": f"drop_{partition_unit}_no_truncation",
            "partition_unit": partition_unit,
            "encoding": endpoint_contract_id,
            "max_sequence_tokens": max_sequence_tokens,
            "directly_overlength_queries": len(directly_overlength_queries),
            "accepted_selection_ids": sorted(accepted_ids),
            "overlength_selection_ids": sorted(overlength_ids),
            "accepted": endpoint_partition_counts(accepted_ids),
            "overlength": endpoint_partition_counts(overlength_ids),
        }
    source_effective = load_config(source_dir / "effective_config.json")
    source_allowed_level_sets = _composite_factor_instance_level_sets(
        source_effective.get("factor_instance_level_sets")
    )
    if encoding == "factor_late_fusion":
        selection_unit = "factor_instance"
        accepted_ids = set(partition["accepted_factor_instance_ids"])
        overlength_ids = set(partition["overlength_factor_instance_ids"])
        selection_id = _factor_instance_id
    else:
        selection_unit = partition_unit
        accepted_ids = set(partition["accepted_selection_ids"])
        overlength_ids = set(partition["overlength_selection_ids"])
        selection_id = (
            _factor_instance_id
            if selection_unit == "factor_instance"
            else lambda row: row["query_id"]
        )
    if accepted_ids & overlength_ids:
        raise AssertionError("length partitions overlap")
    if accepted_ids | overlength_ids != {selection_id(row) for row in source_rows}:
        raise AssertionError(
            f"length partitions do not cover the source {selection_unit} units"
        )

    partition_rows = {
        "accepted": {
            split: [row for row in rows if selection_id(row) in accepted_ids]
            for split, rows in source_split_rows.items()
        },
        "overlength": {
            split: [row for row in rows if selection_id(row) in overlength_ids]
            for split, rows in source_split_rows.items()
        },
    }
    for role, split_rows in partition_rows.items():
        rows = [row for split in split_rows.values() for row in split]
        if selection_unit == "factor_instance":
            _validate_factor_instance_level_sets(
                rows,
                where=role,
                allowed_level_sets=source_allowed_level_sets,
            )
        if len(rows) != partition[role]["queries"]:
            raise AssertionError(f"{role} query count disagrees with token partition")

    source_trajectories = read_jsonl(trajectory_path)
    trajectory_ids = [row.get("trajectory_id") for row in source_trajectories]
    if any(not isinstance(value, str) or not value for value in trajectory_ids):
        raise ConfigurationError("source contains a trajectory without trajectory_id")
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ConfigurationError("source contains duplicate trajectory_id values")

    provenance = require_mapping(config["provenance"], "provenance")
    config_hash = stable_hash(config)
    artifact_results: dict[str, Any] = {}
    for role, split_rows in partition_rows.items():
        rows = [row for split in split_rows.values() for row in split]
        referenced_trajectories = {row["trajectory_id"] for row in rows}
        trajectory_rows = [
            row for row in source_trajectories if row["trajectory_id"] in referenced_trajectories
        ]
        if {row["trajectory_id"] for row in trajectory_rows} != referenced_trajectories:
            raise AssertionError(f"{role} partition has unresolved trajectory references")
        output_dir = output_dirs[role]
        effective = {
            "schema_version": 1,
            "kind": config["kind"],
            "partition": role,
            "config_path": str(config_path),
            "config_hash": config_hash,
            "source": {
                "directory": str(source_dir),
                "files": source_files,
                "length_audit": {"path": str(audit_path), "sha256": expected_audit_sha},
            },
            **validation_source,
            "tokenizer": identity,
            "limits": dict(limits),
            "visibility_scope": "cot_only",
            "provenance": dict(provenance),
            "evaluation": source_effective.get("evaluation"),
            "output_directory": str(output_dir),
            "repository": repository_state(config_path.parent),
        }
        summary = {
            "schema_version": 1,
            "role": (
                "train"
                if partition_role == "train" and role == "accepted"
                else "phase2_overlength_pool"
                if partition_role == "train"
                else "eval"
                if role == "accepted"
                else "eval_overlength_manifest"
            ),
            "partition": role,
            "all_queries": _summary(rows),
            "splits": {split: _summary(values) for split, values in split_rows.items()},
            "trajectories": len(trajectory_rows),
            "factor_instances": len({_factor_instance_id(row) for row in rows}),
            "visibility_scope": "cot_only",
            "model_final_output_included": False,
            "coverage": {
                "source_queries": len(source_rows),
                "selected_queries": len(rows),
                "ratio": len(rows) / len(source_rows),
            },
            "limits": dict(limits),
            "source": effective["source"],
            "provenance": dict(provenance),
            "dry_run": dry_run,
        }
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(output_dir / "trajectories.jsonl", trajectory_rows)
            for split, values in split_rows.items():
                write_jsonl(output_dir / split_files[split], values)
            write_json(output_dir / "effective_config.json", effective)
            summary["outputs"] = {
                path.name: {"path": str(path), "sha256": file_sha256(path)}
                for path in sorted(output_dir.iterdir())
                if path.is_file() and path.name != "summary.json"
            }
            write_json(output_dir / "summary.json", summary)
            summary["outputs"]["summary.json"] = {
                "path": str(output_dir / "summary.json"),
                "sha256_note": "self-hash intentionally omitted",
            }
            write_json(output_dir / "summary.json", summary)
        artifact_results[role] = {"effective_config": effective, "summary": summary}

    return {
        "schema_version": 1,
        "kind": config["kind"],
        "partition": {
            key: value
            for key, value in partition.items()
            if key
            not in {
                "accepted_factor_instance_ids",
                "overlength_factor_instance_ids",
                "accepted_selection_ids",
                "overlength_selection_ids",
            }
        },
        "artifacts": artifact_results,
    }


def build_dataset(config_path: Path, *, dry_run: bool, overwrite: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.get("schema_version") != 1:
        raise ConfigurationError("build config schema_version must be 1")
    if config.get("kind") == COMPOSITE_KIND:
        return _compose_train_dataset(
            config_path,
            config,
            dry_run=dry_run,
            overwrite=overwrite,
        )
    if config.get("kind") in LENGTH_PARTITION_KINDS:
        return _partition_train_dataset_by_lengths(
            config_path,
            config,
            dry_run=dry_run,
            overwrite=overwrite,
        )
    require_keys(config, {"kind", "taxonomy", "sources", "output"}, "build config")
    allowed_config = {"schema_version", "kind", "taxonomy", "sources", "output"}
    if config.get("kind") == "train_dataset":
        allowed_config.update({"split", "selection", "provenance"})
    elif config.get("kind") == "eval_dataset":
        allowed_config.add("evaluation")
    reject_unknown(config, allowed_config, "build config")
    kind = config["kind"]
    if kind not in BUILD_KINDS:
        raise ConfigurationError(f"build kind must be one of {sorted(BUILD_KINDS)}")
    role = BUILD_KINDS[kind]
    config_dir = config_path.parent
    taxonomy_path = resolve_path(config["taxonomy"], relative_to=config_dir)
    taxonomy = Taxonomy.load(taxonomy_path)
    sources_value = config["sources"]
    if not isinstance(sources_value, list) or not sources_value:
        raise ConfigurationError("sources must be a non-empty list")
    sources = [_source_config(value, index, config_dir=config_dir) for index, value in enumerate(sources_value)]
    mixture_ids = [source["mixture_id"] for source in sources]
    if len(mixture_ids) != len(set(mixture_ids)):
        raise ConfigurationError("sources contain duplicate mixture_id values")
    split = _split_config(config.get("split")) if role == "train" else None
    selection = (
        _selection_config(config.get("selection"), config_dir=config_dir)
        if role == "train"
        else None
    )
    provenance = None
    if role == "train" and "provenance" in config:
        provenance = require_mapping(config["provenance"], "provenance")
    if role == "eval" and "split" in config:
        raise ConfigurationError("formal eval datasets must not use the training split config")
    evaluation = None
    if role == "eval":
        evaluation_value = require_mapping(config.get("evaluation"), "evaluation")
        require_keys(evaluation_value, {"benchmark", "metric_adapter"}, "evaluation")
        reject_unknown(evaluation_value, {"benchmark", "metric_adapter"}, "evaluation")
        if any(
            not isinstance(evaluation_value[field], str)
            or not evaluation_value[field].strip()
            for field in ("benchmark", "metric_adapter")
        ):
            raise ConfigurationError("evaluation fields must be non-empty strings")
        evaluation = dict(evaluation_value)

    output = require_mapping(config["output"], "output")
    require_keys(output, {"directory"}, "output")
    reject_unknown(output, {"directory"}, "output")
    output_dir = resolve_path(output["directory"], relative_to=config_dir)
    expected_output_names = {
        "trajectories.jsonl",
        "effective_config.json",
        "summary.json",
    }
    if role == "train":
        expected_output_names.update(
            {"pool_queries.jsonl"}
            if split["strategy"] == "unsplit_pool_v1"
            else {"train_queries.jsonl", "validation_queries.jsonl", "holdout_queries.jsonl"}
        )
    else:
        expected_output_names.add("eval_queries.jsonl")
    if output_dir.exists() and any(output_dir.iterdir()) and not dry_run:
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite")
        unexpected = sorted(
            path.name
            for path in output_dir.iterdir()
            if path.name not in expected_output_names and path.name not in MANAGED_OUTPUT_NAMES
        )
        if unexpected:
            raise FileExistsError(
                f"refusing to overwrite directory with unowned artifacts: {unexpected}"
            )

    records: list[CanonicalRecord] = []
    record_mixtures: list[dict[str, Any]] = []
    input_files: list[dict[str, Any]] = []
    for source in sources:
        path = source["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        raw_rows = read_jsonl(path)
        selected_rows = [
            raw
            for raw in raw_rows
            if all(
                _path_value(raw, field) == expected
                for field, expected in source["filters"].items()
            )
        ]
        if not selected_rows:
            raise ConfigurationError(
                f"source {source['mixture_id']} filters selected zero records"
            )
        for index, raw in enumerate(selected_rows, start=1):
            try:
                record = CanonicalRecord.from_mapping(raw, taxonomy=taxonomy)
            except Exception as exc:
                raise type(exc)(f"{path}:{index}: {exc}") from exc
            records.append(record)
            record_mixtures.append(source)
        input_files.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(raw_rows),
                "selected_rows": len(selected_rows),
                "filters": dict(source["filters"]),
            }
        )

    source_summary = validate_record_collection(records)
    candidates: list[tuple[CanonicalRecord, dict[str, Any]]] = []
    for record, mixture in zip(records, record_mixtures):
        if record.factor_level not in mixture["include_levels"]:
            continue
        if record.annotation_status != "complete" or record.factor_spec is None:
            raise ValueError(
                f"selected record {record.sample_id} has pending Factor annotation"
            )
        candidates.append((record, mixture))

    selected_candidates, selection_report, selected_sample_splits = _select_records(
        candidates, selection, split
    )
    trajectories: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    group_splits: dict[str, str] = {}
    for record, mixture in selected_candidates:
        trajectory = _trajectory(record)
        previous = trajectories.get(record.trajectory_id)
        if previous is not None and previous != trajectory:
            raise ValueError(f"trajectory_id {record.trajectory_id!r} maps to conflicting content")
        trajectories[record.trajectory_id] = trajectory
        split_name = None
        if split is not None:
            if record.sample_id in selected_sample_splits:
                split_name = selected_sample_splits[record.sample_id]
            elif split["strategy"] == "unsplit_pool_v1":
                split_name = None
            elif split["strategy"] == "metadata_partition_v1":
                partition_map = {
                    split["train_value"]: "train",
                    split["validation_value"]: "validation",
                    split["holdout_value"]: "holdout",
                }
                partition = _path_value(record.to_dict(), split["field"])
                if partition not in partition_map:
                    raise ValueError(
                        f"record {record.sample_id} has unsupported metadata partition "
                        f"{partition!r}"
                    )
                split_name = partition_map[partition]
            else:
                split_name = group_splits.setdefault(
                    record.leakage_group_id, _group_split(record.leakage_group_id, split)
                )
        queries.append(
            _query(
                record,
                role=role,
                mixture=mixture,
                split=split_name,
                evaluation=evaluation,
            )
        )

    query_ids = [row["query_id"] for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("expanded queries contain duplicate query_id values")
    _validate_factor_instances(queries, sources)
    queries.sort(key=lambda row: row["query_id"])
    trajectory_rows = sorted(trajectories.values(), key=lambda row: row["trajectory_id"])
    effective = {
        "schema_version": 1,
        "kind": kind,
        "config_path": str(config_path),
        "config_hash": stable_hash(config),
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_sha256": file_sha256(taxonomy_path),
        "prompt_contract_id": CONTRACT_ID,
        "prompt_contract_hash": contract_hash(),
        "visibility_scope": "cot_only",
        "input_files": input_files,
        "sources": [
            {**source, "path": str(source["path"])} for source in sources
        ],
        "split": split,
        "selection": selection,
        "provenance": provenance,
        "evaluation": evaluation,
        "output_directory": str(output_dir),
        "repository": repository_state(config_path.parent),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "role": role,
        "source_records": source_summary,
        "all_queries": _summary(
            queries,
            report_fields=tuple(selection.get("report_fields", ())) if selection else (),
        ),
        "trajectories": len(trajectory_rows),
        "dry_run": dry_run,
    }
    if provenance is not None:
        summary["provenance"] = provenance
    if selection_report is not None:
        summary["selection"] = selection_report
    if role == "train":
        if split["strategy"] == "unsplit_pool_v1":
            pool_rows = queries
            train_rows = []
            validation_rows = []
            holdout_rows = []
            summary["pool"] = _summary(
                pool_rows,
                report_fields=tuple(selection.get("report_fields", ())) if selection else (),
            )
            summary["pool"]["assignment_status"] = "unsplit"
        else:
            pool_rows = []
            train_rows = [row for row in queries if row["split"] == "train"]
            validation_rows = [row for row in queries if row["split"] == "validation"]
            holdout_rows = [row for row in queries if row["split"] == "holdout"]
            if not train_rows or not validation_rows:
                raise ValueError("group split produced an empty train or validation split")
            if {row["leakage_group_id"] for row in train_rows} & {
                row["leakage_group_id"] for row in validation_rows
            }:
                raise AssertionError("leakage group appears in both train and validation")
            split_groups = {
                "train": {row["leakage_group_id"] for row in train_rows},
                "validation": {row["leakage_group_id"] for row in validation_rows},
                "holdout": {row["leakage_group_id"] for row in holdout_rows},
            }
            for left, right in (("train", "validation"), ("train", "holdout"), ("validation", "holdout")):
                overlap = split_groups[left] & split_groups[right]
                if overlap:
                    raise AssertionError(
                        f"leakage groups appear in both {left} and {right}: {len(overlap)}"
                    )
            summary["splits"] = {
                "train": _summary(
                    train_rows,
                    report_fields=tuple(selection.get("report_fields", ()))
                    if selection
                    else (),
                ),
                "validation": _summary(
                    validation_rows,
                    report_fields=tuple(selection.get("report_fields", ()))
                    if selection
                    else (),
                ),
            }
            if holdout_rows:
                summary["splits"]["holdout"] = _summary(
                    holdout_rows,
                    report_fields=tuple(selection.get("report_fields", ()))
                    if selection
                    else (),
                )
    else:
        pool_rows = []
        train_rows = []
        validation_rows = []
        holdout_rows = []

    if dry_run:
        return {"effective_config": effective, "summary": summary}

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(MANAGED_OUTPUT_NAMES - expected_output_names):
        stale = output_dir / name
        if stale.is_file():
            stale.unlink()
    write_jsonl(output_dir / "trajectories.jsonl", trajectory_rows)
    if role == "train":
        if split["strategy"] == "unsplit_pool_v1":
            write_jsonl(output_dir / "pool_queries.jsonl", pool_rows)
        else:
            write_jsonl(output_dir / "train_queries.jsonl", train_rows)
            write_jsonl(output_dir / "validation_queries.jsonl", validation_rows)
            write_jsonl(output_dir / "holdout_queries.jsonl", holdout_rows)
    else:
        write_jsonl(output_dir / "eval_queries.jsonl", queries)
    write_json(output_dir / "effective_config.json", effective)
    summary["outputs"] = {
        path.name: {"path": str(path), "sha256": file_sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    write_json(output_dir / "summary.json", summary)
    summary["outputs"]["summary.json"] = {
        "path": str(output_dir / "summary.json"),
        "sha256_note": "self-hash intentionally omitted",
    }
    write_json(output_dir / "summary.json", summary)
    return {"effective_config": effective, "summary": summary}
