"""Pinned, filtered acquisition for the two MALT Factor candidate pools."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

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
from .io import iter_jsonl, write_json, write_jsonl


STAGES = ("download", "prepare", "audit", "validate")

POSITIVE_CONFIG_FIELDS = {
    "schema_version",
    "kind",
    "dataset_id",
    "source",
    "selection",
    "output",
    "repository",
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be non-empty text")
    return value.strip()


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be boolean")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _load_pipeline_config(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = load_config(path)
    fields = {
        *POSITIVE_CONFIG_FIELDS,
        "negative_pools",
        "negative_output",
    }
    require_keys(config, fields, "MALT pipeline config")
    reject_unknown(config, fields, "MALT pipeline config")
    if config["schema_version"] != 1 or config["kind"] != "malt_candidate_snapshot":
        raise ConfigurationError(
            "MALT config requires schema_version=1 and kind=malt_candidate_snapshot"
        )
    dataset_id = _text(config["dataset_id"], "dataset_id")
    bundle_dir = path.parent.parent
    if bundle_dir.name != dataset_id:
        raise ConfigurationError(
            f"MALT config dataset_id={dataset_id!r} disagrees with bundle {bundle_dir.name!r}"
        )

    source = require_mapping(config["source"], "source")
    source_fields = {
        "repo_id",
        "config_name",
        "split",
        "revision",
        "gated",
        "parquet_pattern",
        "expected_shards",
        "expected_source_rows",
        "batch_size",
        "cache_directory",
        "raw_directory",
    }
    require_keys(source, source_fields, "source")
    reject_unknown(source, source_fields, "source")
    for name in ("repo_id", "config_name", "split", "revision"):
        _text(source[name], f"source.{name}")
    if source["gated"] is not True:
        raise ConfigurationError("MALT source must record gated=true")
    parquet_pattern = _text(source["parquet_pattern"], "source.parquet_pattern")
    if parquet_pattern != "data/*.parquet":
        raise ConfigurationError(
            "MALT acquisition is restricted to the default data/*.parquet files"
        )
    _positive_integer(source["expected_shards"], "source.expected_shards")
    _positive_integer(source["expected_source_rows"], "source.expected_source_rows")
    _positive_integer(source["batch_size"], "source.batch_size")

    selection = require_mapping(config["selection"], "selection")
    selection_fields = {
        "run_source",
        "require_has_chain_of_thought",
        "require_manually_reviewed",
        "any_labels",
        "expected_rows",
    }
    require_keys(selection, selection_fields, "selection")
    reject_unknown(selection, selection_fields, "selection")
    _text(selection["run_source"], "selection.run_source")
    _boolean(
        selection["require_has_chain_of_thought"],
        "selection.require_has_chain_of_thought",
    )
    _boolean(
        selection["require_manually_reviewed"],
        "selection.require_manually_reviewed",
    )
    labels = selection["any_labels"]
    if (
        not isinstance(labels, list)
        or not labels
        or len(labels) != len(set(labels))
        or any(not isinstance(label, str) or not label.strip() for label in labels)
    ):
        raise ConfigurationError("selection.any_labels must contain unique non-empty strings")
    _positive_integer(selection["expected_rows"], "selection.expected_rows")

    output = require_mapping(config["output"], "output")
    require_keys(output, {"filename", "summary_filename"}, "output")
    reject_unknown(output, {"filename", "summary_filename"}, "output")
    filename = _text(output["filename"], "output.filename")
    summary_filename = _text(output["summary_filename"], "output.summary_filename")
    if Path(filename).name != filename or Path(summary_filename).name != summary_filename:
        raise ConfigurationError("MALT output names must be plain filenames")

    negative_pools = config["negative_pools"]
    if not isinstance(negative_pools, list) or not negative_pools:
        raise ConfigurationError("negative_pools must be a non-empty list")
    pool_ids: list[str] = []
    negative_filenames: list[str] = []
    for index, value in enumerate(negative_pools):
        pool = require_mapping(value, f"negative_pools[{index}]")
        pool_fields = {
            "pool_id",
            "quality_tier",
            "description",
            "filename",
            "selection",
        }
        require_keys(pool, pool_fields, f"negative_pools[{index}]")
        reject_unknown(pool, pool_fields, f"negative_pools[{index}]")
        pool_id = _text(pool["pool_id"], f"negative_pools[{index}].pool_id")
        pool_ids.append(pool_id)
        _text(pool["quality_tier"], f"negative_pools[{index}].quality_tier")
        _text(pool["description"], f"negative_pools[{index}].description")
        negative_filename = _text(
            pool["filename"], f"negative_pools[{index}].filename"
        )
        if Path(negative_filename).name != negative_filename or not negative_filename.endswith(
            ".parquet"
        ):
            raise ConfigurationError("negative pool filenames must be plain .parquet names")
        negative_filenames.append(negative_filename)
        negative_selection = require_mapping(
            pool["selection"], f"negative_pools[{index}].selection"
        )
        negative_selection_fields = {
            "run_source",
            "require_has_chain_of_thought",
            "require_manually_reviewed",
            "exact_labels",
            "models",
            "expected_rows",
        }
        require_keys(
            negative_selection,
            negative_selection_fields,
            f"negative_pools[{index}].selection",
        )
        reject_unknown(
            negative_selection,
            negative_selection_fields,
            f"negative_pools[{index}].selection",
        )
        _text(
            negative_selection["run_source"],
            f"negative_pools[{index}].selection.run_source",
        )
        for boolean_field in (
            "require_has_chain_of_thought",
            "require_manually_reviewed",
        ):
            _boolean(
                negative_selection[boolean_field],
                f"negative_pools[{index}].selection.{boolean_field}",
            )
        for list_field, allow_empty in (("exact_labels", False), ("models", True)):
            items = negative_selection[list_field]
            if (
                not isinstance(items, list)
                or (not allow_empty and not items)
                or len(items) != len(set(items))
                or any(not isinstance(item, str) or not item.strip() for item in items)
            ):
                raise ConfigurationError(
                    f"negative_pools[{index}].selection.{list_field} is invalid"
                )
        _positive_integer(
            negative_selection["expected_rows"],
            f"negative_pools[{index}].selection.expected_rows",
        )
    if len(pool_ids) != len(set(pool_ids)):
        raise ConfigurationError("negative pool_id values must be unique")
    if len(negative_filenames) != len(set(negative_filenames)):
        raise ConfigurationError("negative pool filenames must be unique")

    negative_output = require_mapping(config["negative_output"], "negative_output")
    require_keys(negative_output, {"summary_filename", "compression"}, "negative_output")
    reject_unknown(negative_output, {"summary_filename", "compression"}, "negative_output")
    negative_summary_filename = _text(
        negative_output["summary_filename"], "negative_output.summary_filename"
    )
    if Path(negative_summary_filename).name != negative_summary_filename:
        raise ConfigurationError("negative summary filename must be a plain filename")
    if negative_output["compression"] != "zstd":
        raise ConfigurationError("MALT negative Parquet compression must be zstd")

    base = path.parent
    raw = resolve_path(source["raw_directory"], relative_to=base)
    paths = {
        "raw": raw,
        "cache": resolve_path(source["cache_directory"], relative_to=base),
        "records": raw / filename,
        "summary": raw / summary_filename,
        "negative_summary": raw / negative_summary_filename,
        "negative_records": {
            pool["pool_id"]: raw / pool["filename"] for pool in negative_pools
        },
        "repository": resolve_path(config["repository"], relative_to=base),
    }
    return config, paths


def _selected(row: dict[str, Any], selection: dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return False
    labels = metadata.get("labels")
    if not isinstance(labels, list):
        return False
    return (
        metadata.get("run_source") == selection["run_source"]
        and metadata.get("has_chain_of_thought")
        is selection["require_has_chain_of_thought"]
        and metadata.get("manually_reviewed")
        is selection["require_manually_reviewed"]
        and bool(set(labels) & set(selection["any_labels"]))
    )


def _negative_selected(metadata: Any, selection: dict[str, Any]) -> bool:
    if not isinstance(metadata, dict):
        return False
    labels = metadata.get("labels")
    if not isinstance(labels, list):
        return False
    models = selection["models"]
    return (
        metadata.get("run_source") == selection["run_source"]
        and metadata.get("has_chain_of_thought")
        is selection["require_has_chain_of_thought"]
        and metadata.get("manually_reviewed")
        is selection["require_manually_reviewed"]
        and set(labels) == set(selection["exact_labels"])
        and (not models or metadata.get("model") in models)
    )


def task_context_from_row(row: dict[str, Any]) -> str:
    """Render the initial MALT task context without progress or tool messages."""

    samples = row.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ConfigurationError("MALT row has no samples")
    first_input = samples[0].get("input")
    if not isinstance(first_input, list) or not first_input:
        raise ConfigurationError("MALT row has no initial task context")
    messages: list[str] = []
    for message in first_input:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str) or not content.strip():
            continue
        if content.lstrip().startswith("So far in this attempt at the task"):
            continue
        messages.append(f"[{role.upper()}-{len(messages) + 1:03d}] {content.strip()}")
    if not messages:
        raise ConfigurationError("MALT row has no usable task context")
    return "\n\n".join(messages)


def reasoning_lines_from_row(row: dict[str, Any]) -> list[str]:
    """Return every private-reasoning branch in MALT sample order."""

    samples = row.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ConfigurationError("MALT row has no samples")
    lines: list[str] = []
    for sample_index, sample in enumerate(samples, start=1):
        output_groups = sample.get("output")
        if not isinstance(output_groups, list) or not output_groups:
            raise ConfigurationError(f"MALT sample {sample_index} has no output branches")
        for output_group in output_groups:
            if not isinstance(output_group, list) or not output_group:
                raise ConfigurationError(f"MALT sample {sample_index} has an empty output branch")
            for message in output_group:
                metadata = message.get("metadata") or {}
                content = message.get("content")
                if metadata.get("completion_type") == "reasoning_completion" and isinstance(content, str):
                    lines.extend(line.strip() for line in content.splitlines() if line.strip())
    if not lines:
        raise ConfigurationError("MALT row has no reasoning_completion content")
    return lines


def reasoning_steps_from_row(row: dict[str, Any]) -> list[list[str]]:
    """Return MALT private-reasoning branches as ordered step blocks.

    MALT stores each solver turn in ``samples`` and may contain multiple
    alternative ``output`` branches for a turn.  The old line extractor
    flattened all branches together, which silently mixed independent
    trajectories.  This helper preserves one branch per trajectory and one
    reasoning completion per assistant step; callers can duplicate the task
    label/Factor for each branch.
    """

    samples = row.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ConfigurationError("MALT row has no samples")
    branches: list[list[str]] = []
    for sample_index, sample in enumerate(samples, start=1):
        output_groups = sample.get("output")
        if not isinstance(output_groups, list) or not output_groups:
            raise ConfigurationError(f"MALT sample {sample_index} has no output branches")
        if sample_index == 1:
            branches = [[] for _ in output_groups]
        elif len(output_groups) != len(branches):
            raise ConfigurationError(
                "MALT output branch count changes across samples; cannot align branches"
            )
        for branch_index, output_group in enumerate(output_groups):
            if not isinstance(output_group, list) or not output_group:
                raise ConfigurationError(
                    f"MALT sample {sample_index} branch {branch_index} is empty"
                )
            for message in output_group:
                metadata = message.get("metadata") or {}
                content = message.get("content")
                if metadata.get("completion_type") == "reasoning_completion" and isinstance(content, str):
                    text = "\n".join(line.strip() for line in content.splitlines() if line.strip())
                    if text:
                        branches[branch_index].append(text)
    branches = [branch for branch in branches if branch]
    if not branches:
        raise ConfigurationError("MALT row has no reasoning_completion content")
    return branches


def reasoning_step_blocks_from_row(row: dict[str, Any]) -> list[str]:
    """Return each MALT reasoning completion as one ordered CoT block.

    Output branches are retained in source order because MALT's branching
    samples do not expose stable parent identifiers.  We therefore avoid
    inventing cross-sample branch joins while still preserving every assistant
    reasoning turn as an explicit step for the monitor.
    """

    samples = row.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ConfigurationError("MALT row has no samples")
    blocks: list[str] = []
    for sample_index, sample in enumerate(samples, start=1):
        output_groups = sample.get("output")
        if not isinstance(output_groups, list) or not output_groups:
            raise ConfigurationError(f"MALT sample {sample_index} has no output branches")
        for output_group in output_groups:
            if not isinstance(output_group, list) or not output_group:
                raise ConfigurationError(f"MALT sample {sample_index} has an empty output branch")
            for message in output_group:
                metadata = message.get("metadata") or {}
                content = message.get("content")
                if metadata.get("completion_type") == "reasoning_completion" and isinstance(content, str):
                    text = "\n".join(line.strip() for line in content.splitlines() if line.strip())
                    if text:
                        blocks.append(text)
    if not blocks:
        raise ConfigurationError("MALT row has no reasoning_completion content")
    return blocks


def _validate_snapshot(
    records_path: Path,
    *,
    selection: dict[str, Any],
) -> dict[str, Any]:
    rows = 0
    run_ids: list[Any] = []
    labels: Counter[str] = Counter()
    models: Counter[str] = Counter()
    task_ids: set[str] = set()
    for row in iter_jsonl(records_path):
        rows += 1
        if not _selected(row, selection):
            raise ConfigurationError(f"MALT snapshot row {rows} violates selection contract")
        metadata = require_mapping(row.get("metadata"), f"MALT row {rows}.metadata")
        run_id = metadata.get("run_id")
        if isinstance(run_id, bool) or not isinstance(run_id, int):
            raise ConfigurationError(f"MALT row {rows} has invalid run_id")
        run_ids.append(run_id)
        task_ids.add(_text(metadata.get("task_id"), f"MALT row {rows}.task_id"))
        models[_text(metadata.get("model"), f"MALT row {rows}.model")] += 1
        labels.update(str(label) for label in metadata["labels"])
    expected = int(selection["expected_rows"])
    if rows != expected:
        raise ConfigurationError(f"MALT snapshot has {rows} rows, expected {expected}")
    if len(run_ids) != len(set(run_ids)):
        raise ConfigurationError("MALT snapshot contains duplicate run_id values")
    return {
        "rows": rows,
        "unique_run_ids": len(set(run_ids)),
        "unique_task_ids": len(task_ids),
        "labels": dict(sorted(labels.items())),
        "models": dict(sorted(models.items())),
        "sha256": file_sha256(records_path),
    }


def _snapshot_parquet(
    source: dict[str, Any], cache_directory: Path
) -> tuple[Path, list[Path]]:
    """Download or reuse only the pinned default-config Parquet shards."""

    local_snapshot = (
        cache_directory
        / f"datasets--{source['repo_id'].replace('/', '--')}"
        / "snapshots"
        / source["revision"]
    )
    local_shards = sorted(local_snapshot.glob(source["parquet_pattern"]))
    expected = int(source["expected_shards"])
    if len(local_shards) == expected:
        return local_snapshot, local_shards

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("MALT download requires huggingface_hub") from exc
    snapshot = Path(
        snapshot_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            allow_patterns=[source["parquet_pattern"]],
            cache_dir=str(cache_directory),
            token=True,
        )
    )
    shards = sorted(snapshot.glob(source["parquet_pattern"]))
    if len(shards) != expected:
        raise ConfigurationError(
            f"MALT snapshot has {len(shards)} Parquet shards, expected {expected}"
        )
    return snapshot, shards


def _iter_parquet_rows(shards: Iterable[Path], *, batch_size: int) -> Iterator[dict[str, Any]]:
    for batch in _iter_parquet_batches(shards, batch_size=batch_size):
        yield from batch.to_pylist()


def _export_selected_snapshot(
    rows: Iterable[dict[str, Any]],
    *,
    output_path: Path,
    selection: dict[str, Any],
    expected_source_rows: int,
) -> dict[str, int]:
    source_rows_seen = 0
    selected_rows = 0

    def selected() -> Iterator[dict[str, Any]]:
        nonlocal source_rows_seen, selected_rows
        for row in rows:
            source_rows_seen += 1
            if _selected(row, selection):
                selected_rows += 1
                yield row
        if source_rows_seen != expected_source_rows:
            raise ConfigurationError(
                f"MALT source has {source_rows_seen} rows, expected {expected_source_rows}"
            )
        expected_selected = int(selection["expected_rows"])
        if selected_rows != expected_selected:
            raise ConfigurationError(
                f"MALT selection has {selected_rows} rows, expected {expected_selected}"
            )

    write_jsonl(output_path, selected())
    return {"source_rows_seen": source_rows_seen, "rows": selected_rows}


def _negative_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "kind": config["kind"],
        "dataset_id": config["dataset_id"],
        "source": config["source"],
        "negative_pools": config["negative_pools"],
        "negative_output": config["negative_output"],
        "repository": config["repository"],
    }


def _positive_config(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config[field] for field in POSITIVE_CONFIG_FIELDS}


def _validate_negative_parquet(
    records_path: Path, *, selection: dict[str, Any]
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("MALT negative validation requires pyarrow") from exc

    rows = 0
    run_ids: list[Any] = []
    labels: Counter[str] = Counter()
    models: Counter[str] = Counter()
    task_ids: set[str] = set()
    parquet_file = parquet.ParquetFile(records_path)
    for batch in parquet_file.iter_batches(columns=["metadata"]):
        for value in batch.column(0).to_pylist():
            rows += 1
            if not _negative_selected(value, selection):
                raise ConfigurationError(
                    f"MALT negative snapshot row {rows} violates selection contract"
                )
            metadata = require_mapping(value, f"MALT negative row {rows}.metadata")
            run_id = metadata.get("run_id")
            if isinstance(run_id, bool) or not isinstance(run_id, int):
                raise ConfigurationError(f"MALT negative row {rows} has invalid run_id")
            run_ids.append(run_id)
            task_ids.add(_text(metadata.get("task_id"), f"MALT negative row {rows}.task_id"))
            models[_text(metadata.get("model"), f"MALT negative row {rows}.model")] += 1
            labels.update(str(label) for label in metadata["labels"])
    expected = int(selection["expected_rows"])
    if rows != expected:
        raise ConfigurationError(f"MALT negative snapshot has {rows} rows, expected {expected}")
    if len(run_ids) != len(set(run_ids)):
        raise ConfigurationError("MALT negative snapshot contains duplicate run_id values")
    return {
        "rows": rows,
        "unique_run_ids": len(set(run_ids)),
        "unique_task_ids": len(task_ids),
        "labels": dict(sorted(labels.items())),
        "models": dict(sorted(models.items())),
        "bytes": records_path.stat().st_size,
        "sha256": file_sha256(records_path),
    }


def _export_negative_pools(
    shards: list[Path],
    *,
    pools: list[dict[str, Any]],
    output_paths: dict[str, Path],
    expected_source_rows: int,
    compression: str,
    batch_size: int,
) -> dict[str, int]:
    try:
        import pyarrow as arrow
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("MALT negative export requires pyarrow") from exc

    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: dict[str, Path] = {}
    writers: dict[str, Any] = {}
    selected_counts = {pool["pool_id"]: 0 for pool in pools}
    source_rows_seen = 0
    try:
        schema = parquet.ParquetFile(shards[0]).schema_arrow
        for pool in pools:
            pool_id = pool["pool_id"]
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{output_paths[pool_id].name}.",
                dir=output_paths[pool_id].parent,
            )
            os.close(descriptor)
            temporary_paths[pool_id] = Path(temporary)
            writers[pool_id] = parquet.ParquetWriter(
                temporary,
                schema,
                compression=compression,
                use_dictionary=True,
            )
        for row in _iter_parquet_batches(shards, batch_size=batch_size):
            source_rows_seen += row.num_rows
            metadata_index = row.schema.get_field_index("metadata")
            if metadata_index < 0:
                raise ConfigurationError("MALT Parquet batch is missing metadata")
            metadata = row.column(metadata_index).to_pylist()
            for pool in pools:
                pool_id = pool["pool_id"]
                selection = require_mapping(pool["selection"], f"negative pool {pool_id}.selection")
                indices = [
                    index
                    for index, value in enumerate(metadata)
                    if _negative_selected(value, selection)
                ]
                if not indices:
                    continue
                selected_counts[pool_id] += len(indices)
                selected = row.take(arrow.array(indices, type=arrow.int64()))
                writers[pool_id].write_table(arrow.Table.from_batches([selected]))
        if source_rows_seen != expected_source_rows:
            raise ConfigurationError(
                f"MALT source has {source_rows_seen} rows, expected {expected_source_rows}"
            )
        for pool in pools:
            pool_id = pool["pool_id"]
            expected = int(pool["selection"]["expected_rows"])
            if selected_counts[pool_id] != expected:
                raise ConfigurationError(
                    f"MALT negative pool {pool_id} has {selected_counts[pool_id]} rows, "
                    f"expected {expected}"
                )
        for writer in writers.values():
            writer.close()
        writers.clear()
        for pool_id, temporary in temporary_paths.items():
            os.chmod(temporary, 0o664)
            os.replace(temporary, output_paths[pool_id])
        temporary_paths.clear()
    finally:
        for writer in writers.values():
            writer.close()
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()
    return {"source_rows_seen": source_rows_seen, **selected_counts}


def _iter_parquet_batches(shards: Iterable[Path], *, batch_size: int) -> Iterator[Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("MALT filtering requires pyarrow") from exc
    for shard in shards:
        yield from parquet.ParquetFile(shard).iter_batches(batch_size=batch_size)


def _source_snapshot_summary(snapshot: Path, shards: list[Path]) -> dict[str, Any]:
    files = [
        {
            "path": str(shard.relative_to(snapshot)),
            "bytes": shard.stat().st_size,
            "blob_id": shard.resolve().name,
        }
        for shard in shards
    ]
    return {
        "path": str(snapshot),
        "shards": len(files),
        "compressed_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "manifest_sha256": stable_hash(files),
    }


def _download(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    source = require_mapping(config["source"], "source")
    selection = require_mapping(config["selection"], "selection")
    plan = {
        "schema_version": 1,
        "operation": "huggingface_parquet_filtered_snapshot",
        "repo_id": source["repo_id"],
        "config_name": source["config_name"],
        "split": source["split"],
        "revision": source["revision"],
        "parquet_pattern": source["parquet_pattern"],
        "expected_shards": source["expected_shards"],
        "expected_source_rows": source["expected_source_rows"],
        "batch_size": source["batch_size"],
        "cache_directory": str(paths["cache"]),
        "output": str(paths["records"]),
        "summary_output": str(paths["summary"]),
        "selection": dict(selection),
        "dataset_id": config["dataset_id"],
        "expected_rows": selection["expected_rows"],
        "gated": source["gated"],
        "effective_config_hash": stable_hash(_positive_config(config)),
        "dry_run": dry_run,
    }
    if dry_run:
        return plan
    records_exist = paths["records"].is_file()
    summary_exists = paths["summary"].is_file()
    if records_exist and summary_exists and not overwrite:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if (
            summary.get("effective_config_hash") != stable_hash(_positive_config(config))
            or summary.get("sha256") != file_sha256(paths["records"])
        ):
            raise ConfigurationError(
                "existing MALT snapshot differs from the current config; "
                "inspect it before using --overwrite"
            )
        return {**summary, "reused": True}
    if (records_exist or summary_exists) and not overwrite:
        raise ConfigurationError(
            "incomplete MALT snapshot exists; inspect it before using --overwrite"
        )

    snapshot, shards = _snapshot_parquet(source, paths["cache"])
    transport = _export_selected_snapshot(
        _iter_parquet_rows(shards, batch_size=int(source["batch_size"])),
        output_path=paths["records"],
        selection=selection,
        expected_source_rows=int(source["expected_source_rows"]),
    )
    audit = _validate_snapshot(paths["records"], selection=selection)
    summary = {
        **plan,
        "dry_run": False,
        **transport,
        "sha256": audit["sha256"],
        "source_snapshot": _source_snapshot_summary(snapshot, shards),
        "audit": audit,
        "repository": repository_state(paths["repository"]),
    }
    write_json(paths["summary"], summary)
    return summary


def _negative_plan(config: dict[str, Any], paths: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    source = require_mapping(config["source"], "source")
    pools = config["negative_pools"]
    return {
        "schema_version": 1,
        "operation": "malt_negative_parquet_pools",
        "dataset_id": config["dataset_id"],
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "expected_shards": source["expected_shards"],
        "expected_source_rows": source["expected_source_rows"],
        "cache_directory": str(paths["cache"]),
        "summary_output": str(paths["negative_summary"]),
        "compression": config["negative_output"]["compression"],
        "pools": [
            {
                **pool,
                "output": str(paths["negative_records"][pool["pool_id"]]),
            }
            for pool in pools
        ],
        "effective_config_hash": stable_hash(_negative_config(config)),
        "dry_run": dry_run,
    }


def _export_negatives(
    config: dict[str, Any],
    paths: dict[str, Any],
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    plan = _negative_plan(config, paths, dry_run=dry_run)
    if dry_run:
        return plan
    records = paths["negative_records"]
    summary_path = paths["negative_summary"]
    existing_records = {pool_id: path.is_file() for pool_id, path in records.items()}
    if all(existing_records.values()) and summary_path.is_file() and not overwrite:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("effective_config_hash") != plan["effective_config_hash"]:
            raise ConfigurationError(
                "existing MALT negative pools differ from the current config; "
                "inspect them before using --overwrite"
            )
        for pool in summary.get("pools", []):
            path = Path(pool["output"])
            if pool.get("audit", {}).get("sha256") != file_sha256(path):
                raise ConfigurationError(f"existing MALT negative pool hash differs: {path}")
        return {**summary, "reused": True}
    if (any(existing_records.values()) or summary_path.is_file()) and not overwrite:
        raise ConfigurationError(
            "incomplete MALT negative export exists; inspect it before using --overwrite"
        )

    source = require_mapping(config["source"], "source")
    snapshot, shards = _snapshot_parquet(source, paths["cache"])
    pools = config["negative_pools"]
    transport = _export_negative_pools(
        shards,
        pools=pools,
        output_paths=records,
        expected_source_rows=int(source["expected_source_rows"]),
        compression=config["negative_output"]["compression"],
        batch_size=int(source["batch_size"]),
    )
    pool_summaries = []
    for pool in pools:
        pool_id = pool["pool_id"]
        audit = _validate_negative_parquet(
            records[pool_id],
            selection=require_mapping(pool["selection"], f"negative pool {pool_id}.selection"),
        )
        pool_summaries.append(
            {
                **pool,
                "output": str(records[pool_id]),
                "audit": audit,
            }
        )
    summary = {
        **plan,
        "dry_run": False,
        "source_rows_seen": transport.pop("source_rows_seen"),
        "selected_rows": transport,
        "source_snapshot": _source_snapshot_summary(snapshot, shards),
        "pools": pool_summaries,
        "repository": repository_state(paths["repository"]),
    }
    write_json(summary_path, summary)
    return summary


def _validate_negatives(config: dict[str, Any], paths: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    pools = config["negative_pools"]
    missing = [
        str(paths["negative_records"][pool["pool_id"]])
        for pool in pools
        if not paths["negative_records"][pool["pool_id"]].is_file()
    ]
    if missing:
        raise ConfigurationError(f"MALT negative pools do not exist: {missing}")
    if dry_run:
        return {
            "stage": "audit",
            "dry_run": True,
            "dataset_id": config["dataset_id"],
            "pools": [pool["pool_id"] for pool in pools],
        }
    return {
        "stage": "audit",
        "dry_run": False,
        "dataset_id": config["dataset_id"],
        "pools": {
            pool["pool_id"]: _validate_negative_parquet(
                paths["negative_records"][pool["pool_id"]],
                selection=require_mapping(
                    pool["selection"], f"negative pool {pool['pool_id']}.selection"
                ),
            )
            for pool in pools
        },
    }


def run_stage(
    config_path: Path,
    *,
    stage: str,
    dry_run: bool = False,
    overwrite: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run the source-only MALT acquisition or validation stage."""

    del base_url, api_key
    config_path = config_path.resolve()
    config, paths = _load_pipeline_config(config_path)
    if stage not in STAGES:
        raise ConfigurationError(f"unsupported MALT stage {stage!r}")
    if stage == "download":
        return _download(config, paths, dry_run=dry_run, overwrite=overwrite)
    if stage == "prepare":
        return _export_negatives(config, paths, dry_run=dry_run, overwrite=overwrite)
    if stage == "audit":
        return _validate_negatives(config, paths, dry_run=dry_run)
    if not paths["records"].is_file():
        raise ConfigurationError(f"MALT snapshot does not exist: {paths['records']}")
    if dry_run:
        return {
            "stage": "validate",
            "dry_run": True,
            "dataset_id": config["dataset_id"],
            "records": str(paths["records"]),
            "selection": config["selection"],
        }
    return {
        "stage": "validate",
        "dry_run": False,
        "dataset_id": config["dataset_id"],
        "audit": _validate_snapshot(
            paths["records"], selection=require_mapping(config["selection"], "selection")
        ),
    }
