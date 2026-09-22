"""CLI for independent scoring and manifest inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .infer import run_infer
from .io import read_json, write_json
from .manifest import sync_artifacts, validate_manifest
from .paper_eval import run_paper_eval
from .report import write_report
from .runner import run_suite
from .scoring import score_binary_artifact
from .smoke import run_smoke


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="factor-monitor-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--path", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--artifact", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output-directory", type=Path, required=True)
    score.add_argument("--threshold", type=float)
    score.add_argument("--overwrite", action="store_true")
    snapshot = commands.add_parser("sync-datasets")
    snapshot.add_argument("--manifest", type=Path, required=True)
    snapshot.add_argument("--destination", type=Path, required=True)
    snapshot.add_argument("--overwrite", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--predictions-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--monitor-config-root", type=Path)
    run.add_argument("--threshold-root", type=Path)
    run.add_argument("--thresholds-file", type=Path)
    run.add_argument("--overwrite", action="store_true")
    paper = commands.add_parser(
        "paper-eval",
        help="rescore all frozen paper models with every manifest metric adapter",
    )
    paper.add_argument("--manifest", type=Path, required=True)
    paper.add_argument("--models-config", type=Path, required=True)
    paper.add_argument("--output-root", type=Path, required=True)
    paper.add_argument("--models", nargs="+", help="optional subset of model ids")
    paper.add_argument("--overwrite", action="store_true")
    smoke = commands.add_parser("smoke", help="call one first query per active benchmark")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--dataset-root", type=Path, required=True)
    smoke.add_argument("--base-url", required=True)
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--api-key-env", default="OPENAI_API_KEY")
    smoke.add_argument("--timeout", type=float, default=60)
    smoke.add_argument("--max-tokens", type=int, default=16)
    smoke.add_argument("--max-prompt-chars", type=int, default=16000)
    smoke.add_argument("--ids", nargs="+", help="optional subset of benchmark ids, in order")
    smoke.add_argument("--thinking-off-mode", choices=("thinking_disabled", "reasoning_effort_none"))
    smoke.add_argument("--output", type=Path)
    infer = commands.add_parser("infer", help="call every query in the selected benchmarks")
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--dataset-root", type=Path, required=True)
    infer.add_argument("--base-url", required=True)
    infer.add_argument("--model", required=True)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--api-key-env", default="OPENAI_API_KEY")
    infer.add_argument("--timeout", type=float, default=60)
    infer.add_argument("--max-tokens", type=int, default=16)
    infer.add_argument("--max-prompt-chars", type=int, default=16000)
    infer.add_argument("--workers", type=int, default=8)
    infer.add_argument("--ids", nargs="+")
    infer.add_argument("--thinking-off-mode", choices=("thinking_disabled", "reasoning_effort_none"))
    infer.add_argument("--limit", type=int, help="optional per-benchmark cap; omit for the full suite")
    args = parser.parse_args(argv)
    if args.command == "manifest":
        result = validate_manifest(args.path)
    elif args.command == "sync-datasets":
        result = sync_artifacts(args.manifest, args.destination, args.overwrite)
    elif args.command == "run":
        result = run_suite(manifest_path=args.manifest, predictions_root=args.predictions_root, output_root=args.output_root, monitor_config_root=args.monitor_config_root, threshold_root=args.threshold_root, thresholds_file=args.thresholds_file, overwrite=args.overwrite)
    elif args.command == "paper-eval":
        result = run_paper_eval(
            manifest_path=args.manifest,
            models_config=args.models_config,
            output_root=args.output_root,
            model_ids=args.models,
            overwrite=args.overwrite,
        )
    elif args.command == "smoke":
        result = run_smoke(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_prompt_chars=args.max_prompt_chars,
            ids=args.ids,
            thinking_off_mode=args.thinking_off_mode,
            output=args.output,
        )
    elif args.command == "infer":
        result = run_infer(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            base_url=args.base_url,
            model=args.model,
            output_root=args.output_root,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_prompt_chars=args.max_prompt_chars,
            workers=args.workers,
            ids=args.ids,
            thinking_off_mode=args.thinking_off_mode,
            limit=args.limit,
        )
    else:
        result = score_binary_artifact(artifact=args.artifact, predictions=args.predictions, output_directory=args.output_directory, threshold=args.threshold, overwrite=args.overwrite)
        write_report(result, args.output_directory / "report.md")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "run" and not result.get("complete", True):
        raise SystemExit(2)
    if args.command == "paper-eval" and not result.get("complete", True):
        raise SystemExit(2)
    if args.command == "infer" and any(item.get("request_errors", 0) for item in result.get("benchmarks", [])):
        raise SystemExit(2)
