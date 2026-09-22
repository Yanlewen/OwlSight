"""Command line entrypoint for the unified factor-monitor framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bundle import validate_dataset_bundle
from .build import build_dataset
from .config import file_sha256
from .data_ops import run_judge_batch
from .eval import score_binary_artifact
from .full_eval import (
    audit_coverage,
    load_suite,
    materialize_bound_suite,
    merge_suite,
    plan_workers,
    run_worker,
)
from .io import read_jsonl, write_json
from .metrics import score_summary, select_balanced_threshold
from .migrate import migrate_legacy, migrate_relations
from .openai_intervention import score_artifact
from .process_eval import score_process_artifact
from .prompt_contract import contract_spec
from .processing import resolve_vllm_job, run_dataset_stage
from .runtime import resolve_run_config, run_from_config
from .schema import CanonicalRecord, Taxonomy, validate_record_collection
from .stratified_calibration import run_stratified_calibration
from .threshold_diagnostics import run_threshold_decomposition
from .training.encoding import FIELD_BOUNDARY_POLICIES, set_field_boundary_policy


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _validate_source(args: argparse.Namespace) -> None:
    taxonomy = Taxonomy.load(args.taxonomy)
    records = [
        CanonicalRecord.from_mapping(row, taxonomy=taxonomy)
        for row in read_jsonl(args.input)
    ]
    _print(validate_record_collection(records))


def _build(args: argparse.Namespace) -> None:
    _print(build_dataset(args.config, dry_run=args.dry_run, overwrite=args.overwrite))


def _audit_lengths(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "length audit requires the training dependencies: pip install -e 'factor_monitor[training]'"
        ) from exc

    from .training.backend import tokenizer_identity
    from .training.data import (
        causal_choice_token_length_audit,
        endpoint_token_length_audit,
        load_trajectory_groups,
        token_length_audit,
    )

    artifact = args.artifact.resolve()
    tokenizer_path = args.tokenizer.resolve()
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)
    architecture = None
    taxonomy = None
    validation_source: dict[str, Any]
    if args.architecture is not None:
        from .training.config import load_architecture

        architecture = load_architecture(args.architecture)
        validation_source = {
            "architecture": {
                "path": str(args.architecture.resolve()),
                "sha256": file_sha256(args.architecture.resolve()),
            }
        }
    else:
        taxonomy = Taxonomy.load(args.taxonomy)
        validation_source = {
            "taxonomy": {
                "path": str(args.taxonomy.resolve()),
                "sha256": file_sha256(args.taxonomy.resolve()),
            }
        }
    trajectories = artifact / "trajectories.jsonl"
    groups_by_split = {}
    for split in ("train", "validation", "holdout"):
        query_path = artifact / f"{split}_queries.jsonl"
        if query_path.is_file() and query_path.stat().st_size > 0:
            groups_by_split[split] = load_trajectory_groups(
                trajectories,
                query_path,
                architecture=architecture,
                taxonomy=taxonomy,
            )
    eval_queries = artifact / "eval_queries.jsonl"
    if eval_queries.is_file() and eval_queries.stat().st_size > 0:
        if groups_by_split:
            raise RuntimeError(
                "length audit artifact may contain split queries or eval_queries.jsonl, not both"
            )
        groups_by_split["query"] = load_trajectory_groups(
            trajectories,
            eval_queries,
            architecture=architecture,
            taxonomy=taxonomy,
            allow_unlabelled=True,
        )
    if not groups_by_split:
        raise FileNotFoundError(
            f"no nonempty train/validation/holdout/eval query file found in {artifact}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("length audit requires a fast tokenizer with exact offsets")
    if args.pad_token is not None:
        vocabulary = tokenizer.get_vocab()
        if args.pad_token not in vocabulary:
            raise RuntimeError(
                "--pad-token must already exist in the tokenizer vocabulary: "
                f"{args.pad_token!r}"
            )
        tokenizer.pad_token = args.pad_token
    set_field_boundary_policy(tokenizer, args.field_boundary_policy)
    if args.encoding == "endpoint_early_fusion":
        report = endpoint_token_length_audit(
            tokenizer,
            groups_by_split,
            sequence_thresholds=args.evidence_thresholds,
            contract_id=args.endpoint_contract_id,
        )
    elif args.encoding == "causal_lm_choice":
        report = causal_choice_token_length_audit(
            tokenizer,
            groups_by_split,
            sequence_thresholds=args.evidence_thresholds,
            chat_template_kwargs={"enable_thinking": False},
        )
    else:
        report = token_length_audit(
            tokenizer,
            groups_by_split,
            evidence_thresholds=args.evidence_thresholds,
            factor_limits=args.factor_limits,
        )
    report.update(
        {
            "artifact": {
                "directory": str(artifact),
                "files": {
                    path.name: {"path": str(path), "sha256": file_sha256(path)}
                    for path in sorted(artifact.iterdir())
                    if path.is_file() and path.name != "length_audit.json"
                },
            },
            **validation_source,
            "tokenizer": tokenizer_identity(
                tokenizer,
                source=str(tokenizer_path),
                revision="local_snapshot",
            ),
            "truncation": "none",
        }
    )
    if args.output is not None:
        output = args.output.resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"length audit output exists: {output}; pass --overwrite")
        write_json(output, report)
    _print(report)


def _validate_dataset(args: argparse.Namespace) -> None:
    _print(
        validate_dataset_bundle(
            args.manifest,
            taxonomy_path=args.taxonomy,
            datasets_root=args.datasets_root,
            require_canonical=args.require_canonical,
        )
    )


def _migrate(args: argparse.Namespace) -> None:
    _print(
        migrate_legacy(
            args.config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            selected_job=args.job,
            selected_family=args.family,
        )
    )


def _migrate_relations(args: argparse.Namespace) -> None:
    _print(
        migrate_relations(
            datasets_root=args.datasets_root,
            artifacts_root=args.artifacts_root,
            taxonomy_path=args.taxonomy,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    )


def _run(args: argparse.Namespace) -> None:
    validation_checkpoint = getattr(args, "validation_checkpoint", None)
    output_directory = getattr(args, "output_directory", None)
    model_bundle = getattr(args, "model_bundle", None)
    threshold_source = getattr(args, "threshold_source", None)
    architecture = getattr(args, "architecture", None)
    native_checkpoint_threshold = getattr(args, "native_checkpoint_threshold", None)
    long_cot_aggregation = getattr(args, "long_cot_aggregation", None)
    direct_reference_scores = getattr(args, "direct_reference_scores", None)
    mil_short_validation_auroc_floor = getattr(
        args, "mil_short_validation_auroc_floor", None
    )
    effective = resolve_run_config(
        args.config,
        validation_checkpoint=validation_checkpoint,
        output_directory=output_directory,
        model_bundle=model_bundle,
        threshold_source=threshold_source,
        architecture=architecture,
        native_checkpoint_threshold=native_checkpoint_threshold,
        long_cot_aggregation=long_cot_aggregation,
        direct_reference_scores=direct_reference_scores,
        mil_short_validation_auroc_floor=mil_short_validation_auroc_floor,
    )
    expected = "train_run" if args.command == "train" else "eval_run"
    if effective["kind"] != expected:
        raise ValueError(f"{args.command} requires kind={expected}, got {effective['kind']}")
    _print(
        run_from_config(
            args.config,
            dry_run=args.dry_run,
            validation_checkpoint=validation_checkpoint,
            output_directory=output_directory,
            model_bundle=model_bundle,
            threshold_source=threshold_source,
            architecture=architecture,
            native_checkpoint_threshold=native_checkpoint_threshold,
            long_cot_aggregation=long_cot_aggregation,
            direct_reference_scores=direct_reference_scores,
            mil_short_validation_auroc_floor=mil_short_validation_auroc_floor,
        )
    )


def _score(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.scores)
    threshold = select_balanced_threshold(rows) if args.select_threshold else args.threshold
    if threshold is None:
        raise ValueError("provide --threshold or --select-threshold")
    result = score_summary(rows, float(threshold))
    if args.output is not None:
        write_json(args.output, result)
    _print(result)


def _score_binary_artifact(args: argparse.Namespace) -> None:
    _print(
        score_binary_artifact(
            artifact=args.artifact,
            predictions=args.predictions,
            monitor_effective_config=args.monitor_effective_config,
            output_directory=args.output_directory,
            overwrite=args.overwrite,
        )
    )


def _score_intervention(args: argparse.Namespace) -> None:
    _print(
        score_artifact(
            artifact=args.artifact,
            predictions=args.predictions,
            monitor_effective_config=args.monitor_effective_config,
            output_directory=args.output_directory,
            overwrite=args.overwrite,
        )
    )


def _score_process(args: argparse.Namespace) -> None:
    _print(
        score_process_artifact(
            artifact=args.artifact,
            predictions=args.predictions,
            monitor_effective_config=args.monitor_effective_config,
            output_directory=args.output_directory,
            overwrite=args.overwrite,
        )
    )


def _audit_thresholds(args: argparse.Namespace) -> None:
    _print(run_threshold_decomposition(args.config, overwrite=args.overwrite))


def _calibrate_thresholds(args: argparse.Namespace) -> None:
    _print(run_stratified_calibration(args.config, overwrite=args.overwrite))


def _process_dataset(args: argparse.Namespace) -> None:
    _print(
        run_dataset_stage(
            args.manifest,
            stage=args.stage,
            config_path=args.config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    )


def _plan_vllm(args: argparse.Namespace) -> None:
    result = resolve_vllm_job(
        args.manifest,
        args.runtime_config,
        stage=args.stage,
        pipeline_config_path=args.pipeline_config,
    )
    if args.output is not None:
        write_json(args.output, result)
    _print(result)


def _judge_batch(args: argparse.Namespace) -> None:
    _print(
        run_judge_batch(
            profile=args.profile,
            requests_path=args.requests,
            output_path=args.output,
            errors_path=args.errors_output,
            summary_path=args.summary_output,
            base_url=args.base_url,
            api_key=args.api_key,
            dry_run=args.dry_run,
        )
    )


def _evaluate_vllm_baseline(args: argparse.Namespace) -> None:
    from .vllm_baseline import run_suite

    result = run_suite(args.config, dry_run=args.dry_run)
    if args.effective_output is not None:
        effective = result.get("effective_config")
        if effective is None:
            raise ValueError("--effective-output is only available with --dry-run")
        write_json(args.effective_output, effective)
    _print(result)


def _full_eval(args: argparse.Namespace) -> None:
    if args.action == "materialize":
        if args.model_output is None:
            raise ValueError("full-eval materialize requires --model-output")
        _print(
            materialize_bound_suite(
                args.config,
                model_output=args.model_output,
                checkpoint=args.checkpoint,
                threshold_source=args.threshold_source,
                output_root=args.output_root,
                overwrite=args.overwrite,
            )
        )
        return
    suite = load_suite(args.config)
    if args.action == "plan":
        _print(plan_workers(suite))
    elif args.action == "validate":
        _print({"suite": suite, "coverage": audit_coverage(suite)})
    elif args.action == "worker":
        if not args.benchmark or not args.stage:
            raise ValueError("full-eval worker requires --benchmark and --stage")
        _print(run_worker(args.config, args.benchmark, args.stage, dry_run=args.dry_run))
    else:
        _print(merge_suite(suite, overwrite=args.overwrite))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="factor-monitor")
    commands = root.add_subparsers(dest="command", required=True)

    contract = commands.add_parser("contract", help="print the frozen prompt contract")
    contract.set_defaults(handler=lambda _: _print(contract_spec()))

    validate = commands.add_parser("validate-source", help="validate canonical source JSONL")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--taxonomy", type=Path, required=True)
    validate.set_defaults(handler=_validate_source)

    validate_dataset = commands.add_parser(
        "validate-dataset", help="validate a neutral dataset bundle and its canonical pool"
    )
    validate_dataset.add_argument("--manifest", type=Path, required=True)
    validate_dataset.add_argument("--taxonomy", type=Path, required=True)
    validate_dataset.add_argument("--datasets-root", type=Path, required=True)
    validate_dataset.add_argument("--require-canonical", action="store_true")
    validate_dataset.set_defaults(handler=_validate_dataset)

    build = commands.add_parser("build", help="build train or formal-eval queries from YAML")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(handler=_build)

    length_audit = commands.add_parser(
        "audit-lengths",
        help="audit exact rendered lengths and candidate no-truncation filters",
    )
    length_audit.add_argument("--artifact", type=Path, required=True)
    audit_schema = length_audit.add_mutually_exclusive_group(required=True)
    audit_schema.add_argument("--architecture", type=Path)
    audit_schema.add_argument(
        "--taxonomy",
        type=Path,
        help="validate data families/relations without selecting training routes",
    )
    length_audit.add_argument("--tokenizer", type=Path, required=True)
    length_audit.add_argument(
        "--pad-token",
        help="explicit existing vocabulary token used by runtime padding",
    )
    length_audit.add_argument(
        "--field-boundary-policy",
        choices=FIELD_BOUNDARY_POLICIES,
        default="strict",
        help="tokenizer-facing structural separator policy for field closing tags",
    )
    length_audit.add_argument(
        "--evidence-thresholds", type=int, nargs="+", default=[32768, 49152, 65536]
    )
    length_audit.add_argument("--factor-limits", type=int, nargs="+", default=[256, 512, 1024])
    length_audit.add_argument(
        "--encoding",
        choices=("factor_late_fusion", "endpoint_early_fusion", "causal_lm_choice"),
        default="factor_late_fusion",
    )
    length_audit.add_argument(
        "--endpoint-contract-id",
        default="endpoint_early_fusion_contract_v1",
        choices=(
            "endpoint_early_fusion_contract_v1",
            "endpoint_factor_suffix_contract_v1",
            "endpoint_factor_suffix_relay_contract_v1",
            "endpoint_factor_bookends_contract_v1",
            "endpoint_factor_suffix_criterion_difference_contract_v1",
        ),
        help="registered endpoint renderer used when --encoding=endpoint_early_fusion",
    )
    length_audit.add_argument("--trust-remote-code", action="store_true")
    length_audit.add_argument("--output", type=Path)
    length_audit.add_argument("--overwrite", action="store_true")
    length_audit.set_defaults(handler=_audit_lengths)

    migrate = commands.add_parser(
        "migrate", help="migrate audited legacy sources into canonical records"
    )
    migrate.add_argument("--config", type=Path, required=True)
    migrate.add_argument("--job", choices=("all", "intervention", "process"), default="all")
    migrate.add_argument("--family", choices=("sycophancy", "sandbagging"))
    migrate.add_argument("--dry-run", action="store_true")
    migrate.add_argument("--overwrite", action="store_true")
    migrate.set_defaults(handler=_migrate)

    relation_migrate = commands.add_parser(
        "migrate-relations",
        help="rewrite persisted Factor relation enums from locus, in place",
    )
    relation_migrate.add_argument("--datasets-root", type=Path, required=True)
    relation_migrate.add_argument("--artifacts-root", type=Path, required=True)
    relation_migrate.add_argument("--taxonomy", type=Path, required=True)
    relation_migrate.add_argument("--dry-run", action="store_true")
    relation_migrate.add_argument("--overwrite", action="store_true")
    relation_migrate.set_defaults(handler=_migrate_relations)

    process = commands.add_parser(
        "process-dataset", help="run one stage through the adapter in dataset.yaml"
    )
    process.add_argument("--manifest", type=Path, required=True)
    process.add_argument("--stage", required=True)
    process.add_argument("--config", type=Path)
    process.add_argument("--base-url")
    process.add_argument("--api-key")
    process.add_argument("--dry-run", action="store_true")
    process.add_argument("--overwrite", action="store_true")
    process.set_defaults(handler=_process_dataset)

    plan_vllm = commands.add_parser(
        "plan-vllm", help="resolve and validate a local-vLLM dataset-stage job"
    )
    plan_vllm.add_argument("--manifest", type=Path, required=True)
    plan_vllm.add_argument("--runtime-config", type=Path, required=True)
    plan_vllm.add_argument("--pipeline-config", type=Path)
    plan_vllm.add_argument(
        "--stage",
        required=True,
        help="registered dataset stage with a vLLM runtime mapping",
    )
    plan_vllm.add_argument("--output", type=Path)
    plan_vllm.set_defaults(handler=_plan_vllm)

    judge_batch = commands.add_parser(
        "judge-batch",
        help="run dataset-built Judge requests through one shared model profile",
    )
    judge_batch.add_argument("--profile", type=Path, required=True)
    judge_batch.add_argument("--requests", type=Path, required=True)
    judge_batch.add_argument("--output", type=Path, required=True)
    judge_batch.add_argument("--errors-output", type=Path, required=True)
    judge_batch.add_argument("--summary-output", type=Path, required=True)
    judge_batch.add_argument("--base-url")
    judge_batch.add_argument("--api-key")
    judge_batch.add_argument("--dry-run", action="store_true")
    judge_batch.set_defaults(handler=_judge_batch)

    vllm_baseline = commands.add_parser(
        "evaluate-vllm-baseline",
        help="run a config-driven zero-shot Factor suite with offline vLLM forced choice",
    )
    vllm_baseline.add_argument("--config", type=Path, required=True)
    vllm_baseline.add_argument("--dry-run", action="store_true")
    vllm_baseline.add_argument(
        "--effective-output",
        type=Path,
        help="write the resolved effective config during a dry run",
    )
    vllm_baseline.set_defaults(handler=_evaluate_vllm_baseline)

    full_eval = commands.add_parser(
        "full-eval",
        help="materialize, plan, validate, run workers, or CPU-merge a complete-source suite",
    )
    full_eval.add_argument("action", choices=("materialize", "plan", "validate", "worker", "merge"))
    full_eval.add_argument("--config", type=Path, required=True)
    full_eval.add_argument("--benchmark")
    full_eval.add_argument("--stage", choices=("direct", "window"))
    full_eval.add_argument("--model-output", type=Path)
    full_eval.add_argument("--checkpoint")
    full_eval.add_argument("--threshold-source", type=Path)
    full_eval.add_argument("--output-root", type=Path)
    full_eval.add_argument("--dry-run", action="store_true")
    full_eval.add_argument("--overwrite", action="store_true")
    full_eval.set_defaults(handler=_full_eval)

    for name in ("train", "evaluate"):
        run = commands.add_parser(name, help=f"validate and run a {name} backend")
        run.add_argument("--config", type=Path, required=True)
        if name == "train":
            run.add_argument(
                "--validation-checkpoint",
                type=Path,
                help="evaluate this immutable training checkpoint without any optimizer step",
            )
            run.add_argument(
                "--output-directory",
                type=Path,
                help="independent output directory for a validation-only checkpoint run",
            )
            run.add_argument(
                "--mil-short-validation-auroc-floor",
                type=float,
                help=(
                    "validation-only override for the MIL short-validation AUROC "
                    "guard; intended for a model-specific guard such as Llama"
                ),
            )
        else:
            run.add_argument(
                "--model-bundle",
                type=Path,
                help="saved candidate model bundle replacing the eval config default",
            )
            run.add_argument(
                "--threshold-source",
                type=Path,
                help="frozen validation threshold bound to the candidate model",
            )
            run.add_argument(
                "--architecture",
                type=Path,
                help="saved architecture config matching the candidate model bundle",
            )
            run.add_argument(
                "--native-checkpoint-threshold",
                type=float,
                help=(
                    "diagnostic-only threshold used for the native-checkpoint metrics; "
                    "must be in [0,1]"
                ),
            )
            run.add_argument(
                "--output-directory",
                type=Path,
                help="independent candidate evaluation output directory",
            )
            run.add_argument(
                "--long-cot-aggregation",
                choices=("top2_mean", "top2_logmeanexp"),
                help="single preregistered aggregation for a long-CoT eval override",
            )
            run.add_argument(
                "--direct-reference-scores",
                type=Path,
                help="candidate ordinary-eval scores matched to a sliding eval",
            )
        run.add_argument("--dry-run", action="store_true")
        run.set_defaults(handler=_run)

    score = commands.add_parser("score", help="compute binary metrics from label_id/score JSONL")
    score.add_argument("--scores", type=Path, required=True)
    threshold_group = score.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument("--threshold", type=float)
    threshold_group.add_argument("--select-threshold", action="store_true")
    score.add_argument("--output", type=Path)
    score.set_defaults(handler=_score)

    binary_score = commands.add_parser(
        "score-binary-artifact",
        help="score a frozen binary artifact from persisted predictions",
    )
    binary_score.add_argument("--artifact", type=Path, required=True)
    binary_score.add_argument("--predictions", type=Path, required=True)
    binary_score.add_argument("--monitor-effective-config", type=Path, required=True)
    binary_score.add_argument("--output-directory", type=Path, required=True)
    binary_score.add_argument("--overwrite", action="store_true")
    binary_score.set_defaults(handler=_score_binary_artifact)

    intervention_score = commands.add_parser(
        "score-intervention-plan",
        help="score Factor predictions with a copied OpenAI intervention cross-fit plan",
    )
    intervention_score.add_argument("--artifact", type=Path, required=True)
    intervention_score.add_argument("--predictions", type=Path, required=True)
    intervention_score.add_argument("--monitor-effective-config", type=Path, required=True)
    intervention_score.add_argument("--output-directory", type=Path, required=True)
    intervention_score.add_argument("--overwrite", action="store_true")
    intervention_score.set_defaults(handler=_score_intervention)

    process_score = commands.add_parser(
        "score-process",
        help="score Composite Process Factors and trajectory-level any-path application",
    )
    process_score.add_argument("--artifact", type=Path, required=True)
    process_score.add_argument("--predictions", type=Path, required=True)
    process_score.add_argument("--monitor-effective-config", type=Path, required=True)
    process_score.add_argument("--output-directory", type=Path, required=True)
    process_score.add_argument("--overwrite", action="store_true")
    process_score.set_defaults(handler=_score_process)

    threshold_audit = commands.add_parser(
        "audit-thresholds",
        help="decompose frozen-threshold and ranking/separability error from saved scores",
    )
    threshold_audit.add_argument("--config", type=Path, required=True)
    threshold_audit.add_argument("--overwrite", action="store_true")
    threshold_audit.set_defaults(handler=_audit_thresholds)

    calibration = commands.add_parser(
        "calibrate-thresholds",
        help="fit validation-only hierarchical thresholds and audit saved eval scores",
    )
    calibration.add_argument("--config", type=Path, required=True)
    calibration.add_argument("--overwrite", action="store_true")
    calibration.set_defaults(handler=_calibrate_thresholds)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)
