"""Audited adapters from preserved legacy datasets into the canonical schema."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

from .config import (
    ConfigurationError,
    file_sha256,
    load_config,
    reject_unknown,
    repository_state,
    require_keys,
    require_mapping,
    resolve_path,
    stable_hash,
)
from .io import iter_jsonl, read_jsonl, write_json, write_jsonl
from .intervention import (
    stable_sycophancy_target_choice,
    sycophancy_factor_spec,
    sycophancy_targeted_l1_factor_spec,
)
from .schema import CanonicalRecord, Taxonomy, validate_record_collection


LEGACY_RELATIONS = {"following", "application", "influence", "contribution", "engagement"}
CANONICAL_RELATIONS = {"Extrinsic", "Intrinsic"}
DERIVED_FACTOR_HASH_FIELDS = {"factor_spec_sha256", "semantic_factor_spec_sha256"}


def _relation_for_cell(cell: str) -> str:
    if cell.endswith("_external"):
        return "Extrinsic"
    if cell.endswith("_internal"):
        return "Intrinsic"
    raise ConfigurationError(f"cannot infer relation from cell={cell!r}")


def _rewrite_relation_fields(value: Any, *, expected: str) -> tuple[Any, int]:
    """Rewrite only schema relation enum fields, never semantic text fields."""

    changed = 0
    if isinstance(value, list):
        result = []
        for item in value:
            rewritten, item_changed = _rewrite_relation_fields(item, expected=expected)
            result.append(rewritten)
            changed += item_changed
        return result, changed
    if not isinstance(value, dict):
        return value, 0
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"relation", "semantic_relation", "schema_relation"} and isinstance(item, str):
            if item in LEGACY_RELATIONS:
                result[key] = expected
                changed += 1
                continue
            if item in CANONICAL_RELATIONS and item != expected:
                raise ConfigurationError(
                    f"relation={item!r} conflicts with expected {expected!r}"
                )
        rewritten, item_changed = _rewrite_relation_fields(item, expected=expected)
        result[key] = rewritten
        changed += item_changed
    return result, changed


def _relation_neutral(value: Any) -> Any:
    """Normalize enum and its owned derived hashes for an invariance check."""

    if isinstance(value, list):
        return [_relation_neutral(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: "<RELATION_DERIVED>"
        if key in {
            "relation",
            "semantic_relation",
            "schema_relation",
            *DERIVED_FACTOR_HASH_FIELDS,
        }
        else _relation_neutral(item)
        for key, item in value.items()
    }


def _refresh_record_factor_hashes(value: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Refresh exact hash fields derived from the record's four-field Factor."""

    factor_spec = value.get("factor_spec")
    if not isinstance(factor_spec, dict):
        return value, 0
    expected = stable_hash(factor_spec)
    changed = 0

    def refresh(item: Any) -> Any:
        nonlocal changed
        if isinstance(item, list):
            return [refresh(child) for child in item]
        if not isinstance(item, dict):
            return item
        result: dict[str, Any] = {}
        for key, child in item.items():
            if key in DERIVED_FACTOR_HASH_FIELDS and isinstance(child, str):
                result[key] = expected
                changed += int(child != expected)
            else:
                result[key] = refresh(child)
        return result

    return refresh(value), changed


def _owned_relation_jsonl(datasets_root: Path, artifacts_root: Path) -> list[Path]:
    dataset_paths = sorted(datasets_root.glob("**/data/canonical/**/records*.jsonl"))
    artifact_paths = (
        sorted(
            path
            for path in artifacts_root.glob("**/*.jsonl")
            if path.name == "records.jsonl" or path.name.endswith("_queries.jsonl")
        )
        if artifacts_root.is_dir()
        else []
    )
    return dataset_paths + artifact_paths


def _refresh_primary_records_summary(
    records_path: Path, *, records_sha256: str, dry_run: bool
) -> dict[str, Any] | None:
    """Refresh only hashes owned by a sibling primary records.jsonl file."""

    if records_path.name != "records.jsonl":
        return None
    summary_path = records_path.with_name("summary.json")
    if not summary_path.is_file():
        return None
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError(f"summary must be a JSON object: {summary_path}")
    updated_fields: list[str] = []
    if "records_sha256" in value and value["records_sha256"] != records_sha256:
        value["records_sha256"] = records_sha256
        updated_fields.append("records_sha256")
    output = value.get("output")
    if (
        isinstance(output, dict)
        and "records_sha256" in output
        and output["records_sha256"] != records_sha256
    ):
        output["records_sha256"] = records_sha256
        updated_fields.append("output.records_sha256")
    if updated_fields and not dry_run:
        write_json(summary_path, value)
    return {
        "path": str(summary_path),
        "updated_fields": updated_fields,
    }


def migrate_relations(
    *,
    datasets_root: Path,
    artifacts_root: Path,
    taxonomy_path: Path,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Migrate persisted Factor records to the final locus-based relation enum."""

    taxonomy = Taxonomy.load(taxonomy_path.resolve())
    if set(taxonomy.relations) != CANONICAL_RELATIONS:
        raise ConfigurationError(
            "relation migration requires taxonomy.relations={Extrinsic, Intrinsic}"
        )
    if not dry_run and not overwrite:
        raise ConfigurationError("relation migration writes in place; pass --overwrite")
    paths = _owned_relation_jsonl(datasets_root.resolve(), artifacts_root.resolve())
    report: list[dict[str, Any]] = []
    for path in paths:
        row_count = 0
        changed_fields = 0
        changed_hash_fields = 0

        def migrated_rows() -> Iterable[dict[str, Any]]:
            nonlocal row_count, changed_fields, changed_hash_fields
            for index, row in enumerate(iter_jsonl(path)):
                cell = row.get("cell")
                if not isinstance(cell, str):
                    raise ConfigurationError(f"missing cell in {path}:{index + 1}")
                expected = _relation_for_cell(cell)
                if taxonomy.expected_relation(cell, str(row.get("family"))) != expected:
                    raise ConfigurationError(
                        f"taxonomy relation mismatch for {cell}/{row.get('family')}"
                    )
                rewritten, changed = _rewrite_relation_fields(row, expected=expected)
                rewritten, hash_changes = _refresh_record_factor_hashes(rewritten)
                if stable_hash(_relation_neutral(row)) != stable_hash(
                    _relation_neutral(rewritten)
                ):
                    raise ConfigurationError(
                        f"non-relation content changed during migration: {path}:{index + 1}"
                    )
                if "/data/canonical/" in path.as_posix():
                    CanonicalRecord.from_mapping(rewritten, taxonomy=taxonomy)
                row_count += 1
                changed_fields += changed
                changed_hash_fields += hash_changes
                yield rewritten

        before = file_sha256(path)
        if dry_run:
            for _ in migrated_rows():
                pass
        else:
            write_jsonl(path, migrated_rows())
        after = before if dry_run else file_sha256(path)
        summary_refresh = _refresh_primary_records_summary(
            path, records_sha256=after, dry_run=dry_run
        )
        report.append(
            {
                "path": str(path),
                "rows": row_count,
                "relation_fields_changed": changed_fields,
                "derived_factor_hash_fields_changed": changed_hash_fields,
                "before_sha256": before,
                "after_sha256": after,
                "summary_refresh": summary_refresh,
            }
        )
    return {
        "schema_version": 1,
        "kind": "factor_relation_migration",
        "mapping": {"external": "Extrinsic", "internal": "Intrinsic"},
        "dry_run": dry_run,
        "files": report,
        "files_scanned": len(report),
        "rows_scanned": sum(item["rows"] for item in report),
        "relation_fields_changed": sum(
            item["relation_fields_changed"] for item in report
        ),
        "derived_factor_hash_fields_changed": sum(
            item["derived_factor_hash_fields_changed"] for item in report
        ),
        "summary_hash_fields_updated": sum(
            len((item["summary_refresh"] or {}).get("updated_fields", []))
            for item in report
        ),
    }


def _legacy_rows(directory: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    for legacy_split in ("train", "validation", "test"):
        path = directory / f"{legacy_split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            yield legacy_split, row


def _partition(legacy_split: str) -> str:
    return "holdout" if legacy_split == "test" else legacy_split


def _trajectory_id(kind: str, legacy_id: str, task_context: str, model_cot: str) -> str:
    return stable_hash(
        {
            "schema": "canonical-factor-trajectory-v1",
            "kind": kind,
            "legacy_trajectory_id": legacy_id,
            "task_context": task_context,
            "model_cot": model_cot,
        }
    )


def _factor_spec(
    *, factor_text: str, relation: str, criteria: dict[str, Any]
) -> dict[str, str]:
    require_keys(criteria, {"applied_criterion", "not_applied_criterion"}, "criteria")
    return {
        "factor": factor_text,
        "relation": relation,
        "applied_criterion": str(criteria["applied_criterion"]),
        "not_applied_criterion": str(criteria["not_applied_criterion"]),
    }


def _output_guard(output_dir: Path, *, overwrite: bool, dry_run: bool) -> None:
    if not output_dir.exists() or not any(output_dir.iterdir()) or dry_run:
        return
    if not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite")
    unexpected = sorted(
        path.name for path in output_dir.iterdir() if path.name not in {"records.jsonl", "summary.json"}
    )
    if unexpected:
        raise FileExistsError(f"refusing to overwrite unowned migration artifacts: {unexpected}")


def _write_records(
    *,
    output_dir: Path,
    records: list[CanonicalRecord],
    summary: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    summary = {**summary, "record_summary": validate_record_collection(records), "dry_run": dry_run}
    if dry_run:
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    write_jsonl(records_path, (record.to_dict() for record in records))
    summary["output"] = {
        "records": str(records_path),
        "records_sha256": file_sha256(records_path),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _trajectory_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    mapped = {str(row.get("trajectory_id")): row for row in rows}
    if len(mapped) != len(rows) or "" in mapped:
        raise ValueError(f"missing or duplicate trajectory_id in {path}")
    return mapped


def _target_contrast_cards(
    l1_rows: list[tuple[str, dict[str, Any]]],
    intervention_by_id: dict[str, dict[str, Any]],
    config_value: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    config = require_mapping(config_value, "intervention target_contrast")
    required = {"contract_id", "seed", "source_partitions"}
    require_keys(config, required, "intervention target_contrast")
    reject_unknown(config, required, "intervention target_contrast")
    if config["contract_id"] != "sycophancy_l1_target_swap_v1":
        raise ConfigurationError("unsupported sycophancy target-contrast contract")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("intervention target_contrast.seed must be an integer")
    partitions = config["source_partitions"]
    if (
        not isinstance(partitions, list)
        or not partitions
        or len(partitions) != len(set(partitions))
        or any(value not in {"train", "validation"} for value in partitions)
    ):
        raise ConfigurationError(
            "intervention target_contrast.source_partitions must be a unique non-empty "
            "subset of train/validation"
        )

    cards: dict[str, dict[str, Any]] = {}
    balance_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for legacy_split, old in l1_rows:
        if old.get("label_id") != 1:
            continue
        if old.get("triplet_role") != "applied_x1_y1":
            raise ConfigurationError(
                f"positive intervention row is not an applied arm: {old.get('judgment_id')}"
            )
        cue_family = str(old.get("cue_family") or "")
        if cue_family == "sandbagging":
            continue
        partition = _partition(legacy_split)
        if partition not in partitions:
            continue
        intervention_id = str(old.get("intervention_id") or "")
        prompt = intervention_by_id[intervention_id]
        choices_value = prompt.get("choices")
        if not isinstance(choices_value, dict) or len(choices_value) < 2:
            raise ConfigurationError(
                f"{intervention_id} target contrast requires at least two structured choices"
            )
        choices = sorted(str(value) for value in choices_value)
        actual = str(prompt.get("intervention_target_answer") or "").strip()
        correct = str(prompt.get("correct_answer") or "").strip()
        if actual not in choices or correct not in choices:
            raise ConfigurationError(
                f"{intervention_id} target/correct answer is outside structured choices"
            )
        declared = str(old.get("target_correctness") or "")
        observed = "correct" if actual == correct else "wrong"
        if declared != observed:
            raise ConfigurationError(
                f"{intervention_id} target correctness drift: {declared!r} != {observed!r}"
            )
        judgment_id = str(old.get("judgment_id") or "")
        if not judgment_id or judgment_id in cards:
            raise ConfigurationError(
                f"missing or duplicate applied judgment identity: {judgment_id!r}"
            )
        card = {
            "judgment_id": judgment_id,
            "intervention_id": intervention_id,
            "triplet_group_id": str(old.get("triplet_group_id") or ""),
            "source": str(old.get("source_dataset") or ""),
            "partition": partition,
            "cue_family": cue_family,
            "actual_target": actual,
            "correct_target": correct,
            "actual_target_correctness": observed,
            "candidate_targets": choices,
            "candidate_targets_sha256": stable_hash(choices),
        }
        cards[judgment_id] = card
        if len(choices) > 2 and actual != correct:
            balance_groups[(card["source"], cue_family, partition)].append(card)

    balanced_mode: dict[str, str] = {}
    for group_key, group_cards in sorted(balance_groups.items()):
        ordered = sorted(
            group_cards,
            key=lambda card: stable_hash(
                {
                    "contract": config["contract_id"],
                    "seed": seed,
                    "balance_group": group_key,
                    "intervention_id": card["intervention_id"],
                }
            ),
        )
        for index, card in enumerate(ordered):
            wrong_alternates = [
                target
                for target in card["candidate_targets"]
                if target not in {card["actual_target"], card["correct_target"]}
            ]
            balanced_mode[card["judgment_id"]] = (
                "correct" if index % 2 == 0 or not wrong_alternates else "wrong"
            )

    for judgment_id, card in cards.items():
        actual = card["actual_target"]
        correct = card["correct_target"]
        candidates = [target for target in card["candidate_targets"] if target != actual]
        if len(candidates) == 1:
            mode = "binary_opposite"
            alternate = candidates[0]
        elif actual == correct:
            mode = "incorrect"
            alternate = stable_sycophancy_target_choice(
                [target for target in candidates if target != correct],
                seed=seed,
                intervention_id=card["intervention_id"],
                purpose="actual_correct_choose_incorrect",
            )
        elif balanced_mode[judgment_id] == "correct":
            mode = "correct"
            alternate = correct
        else:
            mode = "other_incorrect"
            alternate = stable_sycophancy_target_choice(
                [target for target in candidates if target != correct],
                seed=seed,
                intervention_id=card["intervention_id"],
                purpose="actual_wrong_choose_other_incorrect",
            )
        card["alternate_target"] = alternate
        card["alternate_target_correctness"] = (
            "correct" if alternate == correct else "wrong"
        )
        card["alternate_selection_mode"] = mode
        card["selection_seed"] = seed
        card["selection_sha256"] = stable_hash(
            {
                "contract_id": config["contract_id"],
                "seed": seed,
                "intervention_id": card["intervention_id"],
                "candidate_targets": card["candidate_targets"],
                "actual_target": actual,
                "correct_target": correct,
                "alternate_target": alternate,
                "mode": mode,
            }
        )

    summary = {
        "contract_id": config["contract_id"],
        "seed": seed,
        "source_partitions": list(partitions),
        "factor_level": "L1",
        "l3_policy": "unchanged_not_duplicated",
        "cards": len(cards),
        "by_partition": dict(sorted(Counter(card["partition"] for card in cards.values()).items())),
        "by_source": dict(sorted(Counter(card["source"] for card in cards.values()).items())),
        "by_actual_target_correctness": dict(
            sorted(Counter(card["actual_target_correctness"] for card in cards.values()).items())
        ),
        "by_alternate_target_correctness": dict(
            sorted(Counter(card["alternate_target_correctness"] for card in cards.values()).items())
        ),
        "by_selection_mode": dict(
            sorted(Counter(card["alternate_selection_mode"] for card in cards.values()).items())
        ),
    }
    return cards, summary


def _intervention_records(
    job: dict[str, Any], *, config_dir: Path, taxonomy: Taxonomy
) -> tuple[dict[str, list[CanonicalRecord]], dict[str, Any]]:
    required = {
        "kind",
        "l1_dataset",
        "l3_dataset",
        "prompt_pool",
        "specification_source",
        "relation",
        "l1_factor_templates",
        "l3_specs",
        "criteria",
        "target_contrast",
        "outputs",
    }
    require_keys(job, required, "intervention job")
    reject_unknown(job, required, "intervention job")
    l1_dir = resolve_path(job["l1_dataset"], relative_to=config_dir)
    l3_dir = resolve_path(job["l3_dataset"], relative_to=config_dir)
    prompt_pool = resolve_path(job["prompt_pool"], relative_to=config_dir)
    specification_source = resolve_path(job["specification_source"], relative_to=config_dir)
    trajectories_path = l1_dir / "trajectories.jsonl"
    trajectories = _trajectory_map(trajectories_path)
    l1_rows = list(_legacy_rows(l1_dir))
    l3_rows = list(_legacy_rows(l3_dir))
    l1_keys = {
        (str(row["judgment_id"]), str(row["trajectory_id"]), split): row
        for split, row in l1_rows
    }
    l3_keys = {
        (str(row["judgment_id"]), str(row["trajectory_id"]), split): row
        for split, row in l3_rows
    }
    if set(l1_keys) != set(l3_keys):
        raise ValueError("legacy Intervention L1/L3 row identities do not match")
    mismatched_labels = [key for key in l1_keys if l1_keys[key]["label_id"] != l3_keys[key]["label_id"]]
    if mismatched_labels:
        raise ValueError(f"legacy Intervention L1/L3 labels differ: {len(mismatched_labels)}")

    selected_sample_keys = {str(row["sample_key"]) for row in trajectories.values()}
    intervention_ids = {str(row["intervention_id"]) for _, row in l1_rows}
    prompt_by_sample: dict[str, dict[str, Any]] = {}
    intervention_by_id: dict[str, dict[str, Any]] = {}
    for prompt in iter_jsonl(prompt_pool):
        sample_key = str(prompt.get("sample_key") or "")
        if sample_key in selected_sample_keys:
            prompt_by_sample[sample_key] = prompt
        intervention_id = str(prompt.get("intervention_instance_id") or "")
        if intervention_id in intervention_ids and intervention_id not in intervention_by_id:
            intervention_by_id[intervention_id] = prompt
    if set(prompt_by_sample) != selected_sample_keys:
        raise ValueError("not every selected Intervention trajectory has its exact solver prompt")
    if set(intervention_by_id) != intervention_ids:
        raise ValueError("not every selected Intervention Factor has target metadata")

    target_contrast_cards, target_contrast_summary = _target_contrast_cards(
        l1_rows,
        intervention_by_id,
        job["target_contrast"],
    )

    relation = str(job["relation"])
    templates = require_mapping(job["l1_factor_templates"], "l1_factor_templates")
    l3_specs = require_mapping(job["l3_specs"], "l3_specs")
    criteria = require_mapping(job["criteria"], "criteria")
    result: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for legacy_split, old in l1_rows:
        legacy_trajectory = trajectories[str(old["trajectory_id"])]
        prompt_row = prompt_by_sample[str(legacy_trajectory["sample_key"])]
        full_task_context = str(prompt_row.get("prompt") or "")
        model_cot = str(legacy_trajectory["model_cot"])
        cue_family = str(old["cue_family"])
        family = "sandbagging" if cue_family == "sandbagging" else "sycophancy"
        source = str(old["source_dataset"])
        target_row = intervention_by_id[str(old["intervention_id"])]
        target = str(target_row.get("intervention_target_answer") or "")
        target_stance = (
            "the author is in the wrong"
            if target == "A"
            else "the author is not in the wrong"
            if target == "B"
            else f"answer {target} is correct"
        )
        if cue_family not in templates:
            raise ConfigurationError(f"missing L1 factor template for cue family {cue_family}")
        l1_text = str(templates[cue_family]).format(target=target, target_stance=target_stance)
        criterion_key = "sandbagging" if family == "sandbagging" else "default"
        l3_key = criterion_key
        legacy_id = str(old["trajectory_id"])
        trajectory_id = _trajectory_id(
            "intervention", legacy_id, full_task_context, model_cot
        )
        common_metadata = {
            "source_partition": _partition(legacy_split),
            "legacy": {
                "query_id": old.get("query_id"),
                "judgment_id": old.get("judgment_id"),
                "trajectory_id": legacy_id,
                "intervention_id": old.get("intervention_id"),
                "source_split": legacy_split,
            },
            "causal": {
                "intervention_x": old.get("intervention_x"),
                "behavioral_outcome_y": old.get("behavioral_outcome_y"),
                "te": old.get("te"),
                "p_y1_x0": old.get("p_y1_x0"),
                "p_y1_x1": old.get("p_y1_x1"),
                "triplet_role": old.get("triplet_role"),
                "triplet_group_id": old.get("triplet_group_id"),
            },
            "intervention": {
                "cue_family": cue_family,
                "cue_template_id": old.get("cue_template_id"),
                "target_answer": target,
                "target_correctness": old.get("target_correctness"),
                "target_relation": old.get("target_relation"),
            },
            "training": {"legacy_loss_weight": float(old["cycle_loss_weight"])},
        }
        provenance = {
            "method": str(old["label_source"]),
            "status": "gold",
            "teacher_model": old.get("teacher_model"),
            "legacy_judgment_id": old.get("judgment_id"),
        }
        factors = {
            "L1": _factor_spec(
                factor_text=l1_text,
                relation=relation,
                criteria=require_mapping(criteria[criterion_key], f"criteria.{criterion_key}"),
            ),
            "L3": _factor_spec(
                factor_text=str(require_mapping(l3_specs[l3_key], f"l3_specs.{l3_key}")["factor"]),
                relation=relation,
                criteria=require_mapping(criteria[criterion_key], f"criteria.{criterion_key}"),
            ),
        }
        if family == "sycophancy":
            shared_factors = {
                "L1": asdict(
                    sycophancy_factor_spec(
                        "L1", cue_family=cue_family, target_answer=target
                    )
                ),
                "L3": asdict(
                    sycophancy_factor_spec(
                        "L3", cue_family=cue_family, target_answer=target
                    )
                ),
            }
            if factors != shared_factors:
                raise ConfigurationError(
                    "legacy sycophancy config has drifted from the shared "
                    f"intervention Factor contract for cue_family={cue_family!r}"
                )
            factors = shared_factors
        for level in ("L1", "L3"):
            raw = {
                "schema_version": 1,
                "sample_id": stable_hash(
                    {
                        "migration": "intervention-triplet-canonical-v1",
                        "legacy_judgment_id": old["judgment_id"],
                        "factor_level": level,
                    }
                ),
                "trajectory_id": trajectory_id,
                "task_id": str(old["problem_id"]),
                "leakage_group_id": str(old["problem_id"]),
                "cell": "frontier_risk_external",
                "family": family,
                "source": source,
                "task_context": full_task_context,
                "model_cot": model_cot,
                "factor_level": level,
                "factor_spec": factors[level],
                "annotation_status": "complete",
                "label": int(old["label_id"]),
                "label_provenance": provenance,
                "evidence_spans": [],
                "metadata": common_metadata,
            }
            result[f"{family}/{source}"].append(
                CanonicalRecord.from_mapping(raw, taxonomy=taxonomy)
            )

        contrast_card = target_contrast_cards.get(str(old["judgment_id"]))
        if contrast_card is not None:
            if family != "sycophancy" or old.get("triplet_role") != "applied_x1_y1":
                raise ConfigurationError(
                    f"target contrast attached to an invalid source row: {old['judgment_id']}"
                )
            contrast_group_id = stable_hash(
                {
                    "contract": "sycophancy_l1_target_swap_v1",
                    "triplet_group_id": old["triplet_group_id"],
                    "trajectory_id": trajectory_id,
                }
            )
            for role, candidate_target, label in (
                ("actual_target_applied", contrast_card["actual_target"], 1),
                (
                    "alternate_target_not_applied",
                    contrast_card["alternate_target"],
                    0,
                ),
            ):
                factor_instance_id = stable_hash(
                    {
                        "contract": "sycophancy_l1_target_swap_factor_v1",
                        "contrast_group_id": contrast_group_id,
                        "candidate_target": candidate_target,
                    }
                )
                candidate_correctness = (
                    "correct"
                    if candidate_target == contrast_card["correct_target"]
                    else "wrong"
                )
                contrast_metadata = {
                    **common_metadata,
                    "intervention": {
                        **common_metadata["intervention"],
                        "target_answer": candidate_target,
                        "target_correctness": candidate_correctness,
                        "actual_cue_target_answer": contrast_card["actual_target"],
                        "actual_cue_target_correctness": contrast_card[
                            "actual_target_correctness"
                        ],
                    },
                    "factor_instance_id": factor_instance_id,
                    "factor_scope": "target_specific",
                    "negative_type": (
                        None if label == 1 else "alternate_target_negative"
                    ),
                    "target_contrast": {
                        "kind": "sycophancy_l1_target_swap_v1",
                        "contrast_group_id": contrast_group_id,
                        "role": role,
                        "actual_target": contrast_card["actual_target"],
                        "candidate_target": candidate_target,
                        "alternate_target": contrast_card["alternate_target"],
                        "correct_target": contrast_card["correct_target"],
                        "actual_target_correctness": contrast_card[
                            "actual_target_correctness"
                        ],
                        "candidate_target_correctness": candidate_correctness,
                        "candidate_targets": contrast_card["candidate_targets"],
                        "candidate_targets_sha256": contrast_card[
                            "candidate_targets_sha256"
                        ],
                        "alternate_selection_mode": contrast_card[
                            "alternate_selection_mode"
                        ],
                        "selection_seed": contrast_card["selection_seed"],
                        "selection_sha256": contrast_card["selection_sha256"],
                        "frozen_before_cot": True,
                    },
                }
                contrast_provenance = (
                    {
                        **provenance,
                        "projection_method": "actual_target_l1_projection_v1",
                    }
                    if label == 1
                    else {
                        "method": "deterministic_pre_cot_alternate_target_v1",
                        "status": "gold",
                        "source_judgment_id": old.get("judgment_id"),
                        "selection_sha256": contrast_card["selection_sha256"],
                    }
                )
                contrast_raw = {
                    "schema_version": 1,
                    "sample_id": stable_hash(
                        {
                            "migration": "sycophancy-l1-target-swap-v1",
                            "legacy_judgment_id": old["judgment_id"],
                            "role": role,
                            "candidate_target": candidate_target,
                        }
                    ),
                    "trajectory_id": trajectory_id,
                    "task_id": str(old["problem_id"]),
                    "leakage_group_id": str(old["problem_id"]),
                    "cell": "frontier_risk_external",
                    "family": "sycophancy",
                    "source": source,
                    "task_context": full_task_context,
                    "model_cot": model_cot,
                    "factor_level": "L1",
                    "factor_spec": asdict(
                        sycophancy_targeted_l1_factor_spec(
                            cue_family=cue_family,
                            target_answer=candidate_target,
                        )
                    ),
                    "annotation_status": "complete",
                    "label": label,
                    "label_provenance": contrast_provenance,
                    "evidence_spans": [],
                    "metadata": contrast_metadata,
                }
                result[f"{family}/{source}"].append(
                    CanonicalRecord.from_mapping(contrast_raw, taxonomy=taxonomy)
                )

    contrast_collision_labels: dict[str, set[int]] = defaultdict(set)
    contrast_collision_rows = 0
    for records in result.values():
        for record in records:
            if not isinstance(record.metadata.get("target_contrast"), dict):
                continue
            contrast_collision_rows += 1
            collision_key = stable_hash(
                {
                    "task_context": record.task_context,
                    "model_cot": record.model_cot,
                    "factor_spec": asdict(record.factor_spec),
                }
            )
            contrast_collision_labels[collision_key].add(record.label)
    mixed_collision_keys = sorted(
        key for key, labels in contrast_collision_labels.items() if len(labels) > 1
    )
    if mixed_collision_keys:
        raise ConfigurationError(
            "sycophancy L1 target contrast has identical Evidence/Factor inputs with "
            f"conflicting labels: {mixed_collision_keys[:5]}"
        )
    target_contrast_summary["collision_audit"] = {
        "records": contrast_collision_rows,
        "unique_evidence_factor_inputs": len(contrast_collision_labels),
        "conflicting_label_inputs": 0,
    }

    audit = {
        "legacy_base_rows": len(l1_rows),
        "legacy_l1_l3_identity_match": True,
        "legacy_l1_l3_label_mismatches": 0,
        "legacy_trajectories": len(trajectories),
        "solver_prompts_joined": len(prompt_by_sample),
        "factor_targets_joined": len(intervention_by_id),
        "target_contrast": target_contrast_summary,
        "source_files": {
            "l1_summary": {"path": str(l1_dir / "summary.json"), "sha256": file_sha256(l1_dir / "summary.json")},
            "l3_summary": {"path": str(l3_dir / "summary.json"), "sha256": file_sha256(l3_dir / "summary.json")},
            "l1_trajectories": {"path": str(trajectories_path), "sha256": file_sha256(trajectories_path)},
            "prompt_pool": {"path": str(prompt_pool), "sha256": file_sha256(prompt_pool)},
            "specification_source": {
                "path": str(specification_source),
                "sha256": file_sha256(specification_source),
            },
        },
    }
    return dict(result), audit


def _intervention_output_paths(
    value: Any, *, config_dir: Path
) -> dict[str, Path]:
    outputs = require_mapping(value, "intervention outputs")
    expected_families = {"sycophancy", "sandbagging"}
    require_keys(outputs, expected_families, "intervention outputs")
    reject_unknown(outputs, expected_families, "intervention outputs")
    flattened: dict[str, Path] = {}
    for family in sorted(expected_families):
        datasets = require_mapping(outputs[family], f"intervention outputs.{family}")
        if not datasets:
            raise ConfigurationError(f"intervention outputs.{family} cannot be empty")
        for dataset_id, path_value in datasets.items():
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ConfigurationError("intervention output dataset IDs must be non-empty")
            flattened[f"{family}/{dataset_id}"] = resolve_path(
                path_value, relative_to=config_dir
            )
    return flattened


def _process_records(
    job: dict[str, Any], *, config_dir: Path, taxonomy: Taxonomy
) -> tuple[list[CanonicalRecord], dict[str, Any]]:
    required = {
        "kind",
        "dataset",
        "manifest",
        "relation",
        "l1_factor_template",
        "criteria",
        "output",
    }
    require_keys(job, required, "process job")
    reject_unknown(job, required, "process job")
    data_dir = resolve_path(job["dataset"], relative_to=config_dir)
    manifest = resolve_path(job["manifest"], relative_to=config_dir)
    trajectories_path = data_dir / "trajectories.jsonl"
    trajectories = _trajectory_map(trajectories_path)
    criteria = require_mapping(job["criteria"], "process criteria")
    relation = str(job["relation"])
    template = str(job["l1_factor_template"])
    records: list[CanonicalRecord] = []
    legacy_counts: Counter[str] = Counter()
    for legacy_split, old in _legacy_rows(data_dir):
        legacy_counts[legacy_split] += 1
        trajectory = trajectories[str(old["trajectory_id"])]
        task_context = str(trajectory["task_context"])
        model_cot = str(trajectory["model_cot"])
        legacy_id = str(old["trajectory_id"])
        trajectory_id = _trajectory_id("process", legacy_id, task_context, model_cot)
        factor_text = template.format(
            factor_name=str(old["factor_name"]),
            factor_definition=str(old["factor_definition"]),
        )
        provenance = {
            "method": str(old["annotation_source"]),
            "status": str(old.get("judgment_status") or "targeted_silver"),
            "confidence": old.get("annotation_confidence"),
            "legacy_query_id": old.get("query_id"),
        }
        metadata = {
            "source_partition": _partition(legacy_split),
            "legacy": {
                "query_id": old.get("query_id"),
                "trajectory_id": legacy_id,
                "candidate_id": old.get("candidate_id"),
                "factor_surface_id": old.get("factor_surface_id"),
                "factor_name_group_id": old.get("factor_name_group_id"),
                "factor_concept_id": old.get("factor_concept_id"),
                "source_split": legacy_split,
            },
            "annotation": {
                "factor_name": old.get("factor_name"),
                "factor_definition": old.get("factor_definition"),
                "evidence_block_indices": old.get("evidence_block_indices"),
                "reason": old.get("annotation_reason"),
                "commitment_start_block": old.get("commitment_start_block"),
                "commitment_end_block": old.get("commitment_end_block"),
            },
            "training": {
                # Legacy training used the same explicit default for non-train rows.
                "legacy_loss_weight": float(old.get("cycle_loss_weight", 1.0)),
                "legacy_loss_weight_explicit": "cycle_loss_weight" in old,
                "trainer_validation": bool(old.get("trainer_validation", False)),
                "coverage_round": old.get("coverage_round"),
            },
        }
        common = {
            "schema_version": 1,
            "trajectory_id": trajectory_id,
            "task_id": str(old["candidate_id"]),
            "leakage_group_id": str(old["candidate_id"]),
            "cell": "capability_internal",
            "family": "reasoning_method",
            "source": "harp",
            "task_context": task_context,
            "model_cot": model_cot,
            "label": int(old["label_id"]),
            "label_provenance": provenance,
            "evidence_spans": [],
            "metadata": metadata,
        }
        for level, spec, status in (
            (
                "L1",
                _factor_spec(factor_text=factor_text, relation=relation, criteria=criteria),
                "complete",
            ),
            ("L3", None, "pending"),
        ):
            raw = {
                **common,
                "sample_id": stable_hash(
                    {
                        "migration": "harp-complete-canonical-v1",
                        "legacy_query_id": old["query_id"],
                        "factor_level": level,
                    }
                ),
                "factor_level": level,
                "factor_spec": spec,
                "annotation_status": status,
            }
            records.append(CanonicalRecord.from_mapping(raw, taxonomy=taxonomy))
    audit = {
        "legacy_rows": sum(legacy_counts.values()),
        "legacy_splits": dict(sorted(legacy_counts.items())),
        "legacy_trajectories": len(trajectories),
        "l1_complete_records": sum(record.factor_level == "L1" for record in records),
        "l3_pending_records": sum(record.annotation_status == "pending" for record in records),
        "source_files": {
            "summary": {"path": str(data_dir / "summary.json"), "sha256": file_sha256(data_dir / "summary.json")},
            "trajectories": {"path": str(trajectories_path), "sha256": file_sha256(trajectories_path)},
            "manifest": {"path": str(manifest), "sha256": file_sha256(manifest)},
        },
    }
    return records, audit


def migrate_legacy(
    config_path: Path,
    *,
    dry_run: bool,
    overwrite: bool,
    selected_job: str = "all",
    selected_family: str | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.get("schema_version") != 1 or config.get("kind") != "legacy_data_migration":
        raise ConfigurationError(
            "migration config requires schema_version=1 and kind=legacy_data_migration"
        )
    require_keys(config, {"taxonomy", "jobs"}, "migration config")
    reject_unknown(config, {"schema_version", "kind", "taxonomy", "jobs"}, "migration config")
    config_dir = config_path.parent
    taxonomy_path = resolve_path(config["taxonomy"], relative_to=config_dir)
    taxonomy = Taxonomy.load(taxonomy_path)
    jobs = require_mapping(config["jobs"], "jobs")
    require_keys(jobs, {"intervention", "process"}, "jobs")
    reject_unknown(jobs, {"intervention", "process"}, "jobs")
    intervention_job = require_mapping(jobs["intervention"], "jobs.intervention")
    process_job = require_mapping(jobs["process"], "jobs.process")
    if intervention_job.get("kind") != "legacy_intervention_triplets":
        raise ConfigurationError("intervention job kind must be legacy_intervention_triplets")
    if process_job.get("kind") != "legacy_harp_complete":
        raise ConfigurationError("process job kind must be legacy_harp_complete")

    if selected_job not in {"all", "intervention", "process"}:
        raise ConfigurationError(f"unsupported migration job: {selected_job!r}")
    if selected_family not in {None, "sycophancy", "sandbagging"}:
        raise ConfigurationError(
            f"unsupported intervention family selection: {selected_family!r}"
        )
    if selected_family is not None and selected_job != "intervention":
        raise ConfigurationError(
            "an intervention family can only be selected with --job intervention"
        )
    intervention: dict[str, list[CanonicalRecord]] = {}
    intervention_audit: dict[str, Any] = {}
    process: list[CanonicalRecord] = []
    process_audit: dict[str, Any] = {}
    if selected_job in {"all", "intervention"}:
        intervention, intervention_audit = _intervention_records(
            intervention_job, config_dir=config_dir, taxonomy=taxonomy
        )
    if selected_job in {"all", "process"}:
        process, process_audit = _process_records(
            process_job, config_dir=config_dir, taxonomy=taxonomy
        )
    intervention_outputs = _intervention_output_paths(
        intervention_job["outputs"], config_dir=config_dir
    )
    if selected_job in {"all", "intervention"} and set(intervention_outputs) != set(intervention):
        raise ConfigurationError(
            "configured Intervention dataset outputs do not match observed family/source "
            f"groups: configured={sorted(intervention_outputs)} observed={sorted(intervention)}"
        )
    if selected_family is not None:
        prefix = f"{selected_family}/"
        intervention = {
            key: value for key, value in intervention.items() if key.startswith(prefix)
        }
        intervention_outputs = {
            key: value
            for key, value in intervention_outputs.items()
            if key.startswith(prefix)
        }
    process_output = resolve_path(process_job["output"], relative_to=config_dir)
    output_paths = {}
    if selected_job in {"all", "intervention"}:
        output_paths.update(intervention_outputs)
    if selected_job in {"all", "process"}:
        output_paths["reasoning_method/harp"] = process_output
    for output_dir in output_paths.values():
        _output_guard(output_dir, overwrite=overwrite, dry_run=dry_run)

    common_summary = {
        "schema_version": 1,
        "migration_config": str(config_path),
        "migration_config_hash": stable_hash(config),
        "taxonomy": {"path": str(taxonomy_path), "sha256": file_sha256(taxonomy_path)},
        "repository": repository_state(config_path.parent),
    }
    summaries = {
        dataset_key: _write_records(
            output_dir=output_paths[dataset_key],
            records=records,
            summary={
                **common_summary,
                "migration_kind": "legacy_intervention_triplets",
                "dataset_key": dataset_key,
                "audit": intervention_audit,
            },
            dry_run=dry_run,
        )
        for dataset_key, records in sorted(intervention.items())
    }
    if selected_job in {"all", "process"}:
        summaries["reasoning_method/harp"] = _write_records(
            output_dir=process_output,
            records=process,
            summary={
                **common_summary,
                "migration_kind": "legacy_harp_complete",
                "family": "reasoning_method",
                "audit": process_audit,
                "pending_policy": "L3 factor_spec is null and excluded from builds until annotated",
            },
            dry_run=dry_run,
        )
    return summaries
