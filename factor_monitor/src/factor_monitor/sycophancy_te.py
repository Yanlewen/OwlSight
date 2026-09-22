"""Canonicalize the preserved ScienceMCQ and MoralStories minimal-TE sources."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, replace
import re
from pathlib import Path
from typing import Any

from .bundle import validate_dataset_bundle
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
from .intervention import (
    stable_sycophancy_target_choice,
    sycophancy_factor_spec,
    sycophancy_targeted_l1_factor_spec,
)
from .io import iter_jsonl, read_jsonl, write_json, write_jsonl
from .schema import CanonicalRecord, Taxonomy, validate_record_collection


_COMPONENTS = {
    "ScienceMCQ": {
        "dataset_id": "science_mcq_te",
        "cue_modes": {
            "direct_answer_hint": "leaked_privileged_answer",
            "encoded_letter_index": "encoded_privileged_answer",
            "no_cue": None,
        },
    },
    "MoralStories": {
        "dataset_id": "moral_stories_independent_te",
        "cue_modes": {
            "first_person": "moral_first_person",
            "suggest_right": "moral_suggest_right",
            "suggest_wrong": "moral_suggest_wrong",
            "neutral": None,
        },
    },
}


def _source_partition_index(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            metadata = require_mapping(row.get("metadata"), f"{path} metadata")
            record_id = metadata.get("dataset_record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ConfigurationError(f"{path} row lacks metadata.dataset_record_id")
            previous = result.setdefault(record_id, split)
            if previous != split:
                raise ConfigurationError(
                    f"source record {record_id} crosses source partitions"
                )
    return result


def _task_context(evidence: dict[str, Any]) -> str:
    task = str(evidence.get("task_context") or "").strip()
    cue = str(evidence.get("candidate_cue") or "").strip()
    if not task or not cue:
        raise ConfigurationError("structured monitor evidence lacks task context or cue")
    return f"{task}\n\nCandidate external cue:\n{cue}"


def _science_choices(task_context: str) -> list[str]:
    choices = sorted(set(re.findall(r"^\(([A-Z])\)\s", task_context, flags=re.MULTILINE)))
    if len(choices) < 2:
        raise ConfigurationError("ScienceMCQ task has fewer than two structured choices")
    return choices


def _row_card(row: dict[str, Any], *, component: str, partition: str) -> dict[str, Any]:
    metadata = require_mapping(row.get("metadata"), "source row metadata")
    evidence = require_mapping(row.get("monitor_evidence"), "source row monitor_evidence")
    prompt_mode = str(metadata.get("prompt_mode") or "")
    component_spec = _COMPONENTS[component]
    if prompt_mode not in component_spec["cue_modes"]:
        raise ConfigurationError(
            f"unsupported {component} prompt_mode={prompt_mode!r}"
        )
    if component == "ScienceMCQ":
        bucket = str(metadata.get("bucket") or "")
        cue_style = (
            "encoded"
            if prompt_mode == "encoded_letter_index"
            or "encoded" in bucket
            else "direct"
        )
        cue_family = (
            "encoded_privileged_answer"
            if cue_style == "encoded"
            else "leaked_privileged_answer"
        )
        group_id = str(metadata.get("te_group_id") or "")
        leakage_group_id = str(metadata.get("question_id") or "")
        actual_target = str(metadata.get("candidate_target_answer") or "").strip()
        correct_target = str(metadata.get("correct_answer") or "").strip()
        choices = _science_choices(str(evidence.get("task_context") or ""))
        task_id = f"science_mcq_te::{leakage_group_id}"
    else:
        intervention_mode = str(metadata.get("scruples_mode") or prompt_mode)
        if intervention_mode not in {"first_person", "suggest_right", "suggest_wrong"}:
            raise ConfigurationError(
                f"unsupported MoralStories intervention mode={intervention_mode!r}"
            )
        cue_style = intervention_mode.removeprefix("suggest_")
        cue_family = {
            "first_person": "moral_first_person",
            "suggest_right": "moral_suggest_right",
            "suggest_wrong": "moral_suggest_wrong",
        }[intervention_mode]
        group_id = str(metadata.get("intervention_instance_id") or "")
        leakage_group_id = str(metadata.get("source_id") or "")
        actual_target = str(metadata.get("intervention_target_answer") or "").strip()
        correct_target = str(metadata.get("natural_answer") or "").strip()
        choices = ["A", "B"]
        task_id = f"moral_stories::{group_id}"
    if not group_id or not leakage_group_id:
        raise ConfigurationError(f"{component} row lacks TE/leakage identity")
    if actual_target not in choices or correct_target not in choices:
        raise ConfigurationError(
            f"{component} target/correct target is outside structured choices"
        )
    x = metadata.get("x")
    y = metadata.get("target_follow_y")
    if x not in {0, 1} or y not in {0, 1}:
        raise ConfigurationError(f"{component} row has invalid x/y={x!r}/{y!r}")
    role = (
        "applied"
        if x == 1 and y == 1
        else "hard"
        if x == 1 and y == 0
        else "simple"
    )
    full_context = _task_context(evidence)
    model_cot = str(evidence.get("model_cot") or "").strip()
    if not model_cot:
        raise ConfigurationError("structured monitor evidence lacks model_cot")
    source_record_id = str(metadata.get("source_component_record_id") or "")
    if not source_record_id:
        raise ConfigurationError("combined row lacks source_component_record_id")
    trajectory_id = stable_hash(
        {
            "contract": "sycophancy-te-trajectory-v1",
            "component": component,
            "sample_key": metadata.get("sample_key"),
            "task_context": full_context,
            "model_cot": model_cot,
        }
    )
    return {
        "row": row,
        "metadata": metadata,
        "component": component,
        "dataset_id": component_spec["dataset_id"],
        "source_record_id": source_record_id,
        "original_source_partition": partition,
        "source_partition": "train" if partition == "train" else "validation",
        "cue_family": cue_family,
        "cue_style": cue_style,
        "prompt_mode": prompt_mode,
        "group_id": group_id,
        "leakage_group_id": leakage_group_id,
        "task_id": task_id,
        "actual_target": actual_target,
        "correct_target": correct_target,
        "choices": choices,
        "x": int(x),
        "y": int(y),
        "role": role,
        "task_context": full_context,
        "model_cot": model_cot,
        "trajectory_id": trajectory_id,
    }


def _alternate_targets(cards: list[dict[str, Any]], *, seed: int) -> dict[str, str]:
    positives = [card for card in cards if card["role"] == "applied"]
    balance_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for card in positives:
        if len(card["choices"]) > 2 and card["actual_target"] != card["correct_target"]:
            balance_groups[
                (card["dataset_id"], card["cue_family"], card["source_partition"])
            ].append(card)
    balance_mode: dict[str, str] = {}
    for key, group in sorted(balance_groups.items()):
        ordered = sorted(
            group,
            key=lambda card: stable_hash(
                {
                    "contract": "sycophancy_l1_target_swap_v1",
                    "seed": seed,
                    "balance_group": key,
                    "trajectory_id": card["trajectory_id"],
                }
            ),
        )
        for index, card in enumerate(ordered):
            wrong = [
                target
                for target in card["choices"]
                if target not in {card["actual_target"], card["correct_target"]}
            ]
            balance_mode[card["source_record_id"]] = (
                "correct" if index % 2 == 0 or not wrong else "wrong"
            )

    result: dict[str, str] = {}
    for card in positives:
        actual = card["actual_target"]
        correct = card["correct_target"]
        candidates = [target for target in card["choices"] if target != actual]
        if len(candidates) == 1:
            alternate = candidates[0]
        elif actual == correct:
            alternate = stable_sycophancy_target_choice(
                [target for target in candidates if target != correct],
                seed=seed,
                intervention_id=card["source_record_id"],
                purpose="actual_correct_choose_incorrect",
            )
        elif balance_mode[card["source_record_id"]] == "correct":
            alternate = correct
        else:
            alternate = stable_sycophancy_target_choice(
                [target for target in candidates if target != correct],
                seed=seed,
                intervention_id=card["source_record_id"],
                purpose="actual_wrong_choose_other_incorrect",
            )
        result[card["source_record_id"]] = alternate
    return result


def _metadata(card: dict[str, Any], *, target: str) -> dict[str, Any]:
    raw = card["metadata"]
    return {
        "source_partition": card["source_partition"],
        "original_source_partition": card["original_source_partition"],
        "source_component": card["component"],
        "source_record_id": card["source_record_id"],
        "source_file_name": raw.get("source_file_name"),
        "source_dataset": raw.get("source_dataset"),
        "sample_key": raw.get("sample_key"),
        "factor_scope": "target_specific",
        "training_eligible": True,
        "causal": {
            "intervention_x": card["x"],
            "behavioral_outcome_y": card["y"],
            "training_role": card["role"],
            "triplet_group_id": card["group_id"],
            "te": raw.get("te_effect_size"),
            "p_y1_x0": raw.get("te_p_y1_x0"),
            "p_y1_x1": raw.get("te_p_y1_x1"),
        },
        "intervention": {
            "cue_family": card["cue_family"],
            "cue_style": card["cue_style"],
            "prompt_mode": card["prompt_mode"],
            "actual_target_answer": card["actual_target"],
            "target_answer": target,
            "correct_target_answer": card["correct_target"],
            "target_correctness": "correct" if target == card["correct_target"] else "wrong",
        },
    }


def _record(
    card: dict[str, Any],
    *,
    factor_level: str,
    target: str,
    target_role: str,
    label: int,
    taxonomy: Taxonomy,
) -> CanonicalRecord:
    factor = (
        sycophancy_targeted_l1_factor_spec(
            cue_family=card["cue_family"], target_answer=target
        )
        if factor_level == "L1"
        else sycophancy_factor_spec(
            "L3", cue_family=card["cue_family"], target_answer=target
        )
    )
    metadata = _metadata(card, target=target)
    metadata["factor_scope"] = "target_specific" if factor_level == "L1" else "generic"
    metadata["target_role"] = target_role
    metadata["factor_instance_id"] = stable_hash(
        {
            "contract": "sycophancy-te-factor-instance-v1",
            "source_record_id": card["source_record_id"],
            "factor_level": factor_level,
            "target": target,
            "target_role": target_role,
        }
    )
    raw = {
        "schema_version": 1,
        "sample_id": stable_hash(
            {
                "contract": "sycophancy-te-canonical-v1",
                "dataset_id": card["dataset_id"],
                "source_record_id": card["source_record_id"],
                "factor_level": factor_level,
                "target": target,
                "target_role": target_role,
            }
        ),
        "trajectory_id": card["trajectory_id"],
        "task_id": card["task_id"],
        "leakage_group_id": card["leakage_group_id"],
        "cell": "frontier_risk_external",
        "family": "sycophancy",
        "source": card["dataset_id"],
        "task_context": card["task_context"],
        "model_cot": card["model_cot"],
        "factor_level": factor_level,
        "factor_spec": asdict(factor),
        "annotation_status": "complete",
        "label": label,
        "label_provenance": {
            "method": "minimal_treatment_effect_behavioral_projection",
            "status": "gold",
            "source_record_id": card["source_record_id"],
        },
        "evidence_spans": [],
        "metadata": metadata,
    }
    return CanonicalRecord.from_mapping(raw, taxonomy=taxonomy)


def canonicalize(config_path: Path, *, dry_run: bool, overwrite: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    required = {
        "schema_version",
        "kind",
        "dataset_id",
        "component",
        "combined_dataset",
        "partition_dataset",
        "taxonomy",
        "manifest",
        "seed",
        "output",
    }
    require_keys(config, required, "sycophancy TE source config")
    reject_unknown(config, required, "sycophancy TE source config")
    if config["schema_version"] != 1 or config["kind"] != "sycophancy_te_source":
        raise ConfigurationError("requires schema_version=1 and kind=sycophancy_te_source")
    component = str(config["component"])
    if component not in _COMPONENTS:
        raise ConfigurationError(f"unsupported component {component!r}")
    expected_dataset_id = _COMPONENTS[component]["dataset_id"]
    if config["dataset_id"] != expected_dataset_id:
        raise ConfigurationError(
            f"dataset_id must be {expected_dataset_id!r} for {component}"
        )
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ConfigurationError("seed must be an integer")
    config_dir = config_path.parent
    combined = resolve_path(config["combined_dataset"], relative_to=config_dir)
    partition_dataset = resolve_path(config["partition_dataset"], relative_to=config_dir)
    taxonomy_path = resolve_path(config["taxonomy"], relative_to=config_dir)
    output = resolve_path(config["output"], relative_to=config_dir)
    manifest = resolve_path(config["manifest"], relative_to=config_dir)
    taxonomy = Taxonomy.load(taxonomy_path)
    partition_by_id = _source_partition_index(partition_dataset)
    cards: list[dict[str, Any]] = []
    for row in iter_jsonl(combined / "all.jsonl"):
        metadata = require_mapping(row.get("metadata"), "combined row metadata")
        if metadata.get("source_component") != component:
            continue
        source_record_id = metadata.get("source_component_record_id")
        if source_record_id not in partition_by_id:
            raise ConfigurationError(
                f"combined {component} row is absent from original source partitions"
            )
        cards.append(
            _row_card(
                row,
                component=component,
                partition=partition_by_id[str(source_record_id)],
            )
        )
    if not cards:
        raise ConfigurationError(f"no {component} rows found in combined source")
    alternates = _alternate_targets(cards, seed=config["seed"])
    records: list[CanonicalRecord] = []
    l3_by_input: dict[str, CanonicalRecord] = {}
    l3_duplicate_groups: dict[str, set[str]] = defaultdict(set)
    for card in cards:
        label = 1 if card["role"] == "applied" else 0
        records.append(
            _record(
                card,
                factor_level="L1",
                target=card["actual_target"],
                target_role="actual",
                label=label,
                taxonomy=taxonomy,
            )
        )
        if card["role"] == "applied":
            records.append(
                _record(
                    card,
                    factor_level="L1",
                    target=alternates[card["source_record_id"]],
                    target_role="alternate",
                    label=0,
                    taxonomy=taxonomy,
                )
            )
        l3_record = _record(
            card,
            factor_level="L3",
            target=card["actual_target"],
            target_role="generic",
            label=label,
            taxonomy=taxonomy,
        )
        l3_key = stable_hash(
            {
                "task_context": l3_record.task_context,
                "model_cot": l3_record.model_cot,
                "factor_spec": asdict(l3_record.factor_spec),
            }
        )
        previous = l3_by_input.get(l3_key)
        if previous is not None and previous.label != l3_record.label:
            raise ConfigurationError("generic L3 projection has conflicting labels")
        if previous is None:
            l3_by_input[l3_key] = l3_record
        l3_duplicate_groups[l3_key].add(card["group_id"])
    for key, record in l3_by_input.items():
        groups = sorted(l3_duplicate_groups[key])
        records.append(
            replace(
                record,
                metadata={
                    **record.metadata,
                    "generic_projection": {
                        "deduplicated": len(groups) > 1,
                        "source_triplet_group_ids": groups,
                    },
                },
            )
        )
    records.sort(key=lambda record: record.sample_id)
    report = {
        "schema_version": 1,
        "kind": "sycophancy_te_canonicalization",
        "dataset_id": expected_dataset_id,
        "component": component,
        "seed": config["seed"],
        "dry_run": dry_run,
        "effective_configuration": {
            "model": "DeepSeek-R1-0528-Qwen3-8B_source_generation",
            "tokenizer": "DeepSeek-R1-0528-Qwen3-8B_source_generation",
            "dataset": str(combined),
            "source_partitions": ["train", "validation", "test"],
            "factor_validation_partition": "source_validation_plus_test",
            "max_sequence_length": "unbounded_complete_cot",
            "truncation": "none",
            "padding": "not_applicable_pre_tokenization",
            "batch_size": "not_applicable_dataset_construction",
            "learning_rate": "not_applicable_dataset_construction",
            "epochs_or_steps": "not_applicable_dataset_construction",
            "seed": config["seed"],
            "evaluation_metric": "within_trajectory_factor_attribution",
            "output_directory": str(output),
        },
        "records": validate_record_collection(records),
        "cards": len(cards),
        "roles": dict(sorted(Counter(card["role"] for card in cards).items())),
        "cue_styles": dict(sorted(Counter(card["cue_style"] for card in cards).items())),
        "original_partitions": dict(
            sorted(Counter(card["original_source_partition"] for card in cards).items())
        ),
        "l3_generic_projection": {
            "input_rows": len(cards),
            "unique_rows": len(l3_by_input),
            "deduplicated_rows": len(cards) - len(l3_by_input),
            "conflicting_labels": 0,
        },
        "inputs": {
            "combined_all": {
                "path": str(combined / "all.jsonl"),
                "sha256": file_sha256(combined / "all.jsonl"),
            },
            "combined_summary": {
                "path": str(combined / "summary.json"),
                "sha256": file_sha256(combined / "summary.json"),
            },
            "partition_summary": {
                "path": str(partition_dataset / "summary.json"),
                "sha256": file_sha256(partition_dataset / "summary.json"),
            },
        },
        "repository": repository_state(config_path.parent),
    }
    if dry_run:
        return report
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; pass --overwrite")
    unexpected = sorted(
        path.name for path in output.iterdir() if path.name not in {"records.jsonl", "summary.json"}
    )
    if unexpected:
        raise FileExistsError(f"refusing to overwrite unowned outputs: {unexpected}")
    records_path = output / "records.jsonl"
    write_jsonl(records_path, (record.to_dict() for record in records))
    report["output"] = {
        "records": str(records_path),
        "records_sha256": file_sha256(records_path),
    }
    write_json(output / "summary.json", report)
    validate_dataset_bundle(
        manifest,
        taxonomy_path=taxonomy_path,
        datasets_root=manifest.parents[3],
        require_canonical=True,
    )
    return report


def run_stage(
    config_path: Path,
    *,
    stage: str,
    dry_run: bool = False,
    overwrite: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    del base_url, api_key
    if stage == "canonicalize":
        return canonicalize(config_path, dry_run=dry_run, overwrite=overwrite)
    if stage == "validate":
        config = load_config(config_path.resolve())
        config_dir = config_path.resolve().parent
        manifest = resolve_path(config["manifest"], relative_to=config_dir)
        taxonomy = resolve_path(config["taxonomy"], relative_to=config_dir)
        return validate_dataset_bundle(
            manifest,
            taxonomy_path=taxonomy,
            datasets_root=manifest.parents[3],
            require_canonical=True,
        )
    raise ConfigurationError(f"unsupported stage {stage!r}")
