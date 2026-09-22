"""Config-driven dataset-stage and local-vLLM orchestration helpers."""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path
from typing import Any, Callable

from .bundle import DatasetBundle
from .config import (
    ConfigurationError,
    file_sha256,
    load_config,
    reject_unknown,
    repository_state,
    require_keys,
    require_mapping,
    resolve_path,
)
from .schema import Taxonomy


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number")
    result = float(value)
    if positive and result <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _framework_paths(manifest_path: Path) -> tuple[Path, Path, Path]:
    manifest_path = manifest_path.resolve()
    datasets_root = next(
        (
            parent
            for parent in manifest_path.parents
            if parent.name == "datasets" and parent.parent.name == "factor_monitor"
        ),
        None,
    )
    if datasets_root is None:
        raise ConfigurationError(
            "dataset manifest must live below factor_monitor/datasets"
        )
    framework_root = datasets_root.parent
    repository = framework_root.parent
    return framework_root, datasets_root, repository


def load_dataset_bundle(manifest_path: Path) -> DatasetBundle:
    framework_root, datasets_root, _ = _framework_paths(manifest_path)
    taxonomy = Taxonomy.load(framework_root / "configs" / "taxonomy.yaml")
    return DatasetBundle.load(
        manifest_path, taxonomy=taxonomy, datasets_root=datasets_root
    )


def processing_config_path(bundle: DatasetBundle) -> Path:
    processing = require_mapping(bundle.manifest["processing"], "processing")
    if "config" not in processing:
        raise ConfigurationError(
            f"dataset {bundle.dataset_id} has no processing.config"
        )
    path = resolve_path(processing["config"], relative_to=bundle.bundle_dir)
    if not path.is_file():
        raise ConfigurationError(f"processing config does not exist: {path}")
    return path


def _adapter_callable(bundle: DatasetBundle) -> Callable[..., dict[str, Any]]:
    if bundle.adapter_path is not None:
        module_name = "_factor_monitor_adapter_" + re.sub(
            r"[^a-zA-Z0-9_]", "_", bundle.dataset_id
        )
        spec = importlib.util.spec_from_file_location(module_name, bundle.adapter_path)
        if spec is None or spec.loader is None:
            raise ConfigurationError(f"cannot import adapter: {bundle.adapter_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module_name = bundle.adapter_reference.split(":", 1)[0]
        module = importlib.import_module(module_name)
    entrypoint = getattr(module, bundle.adapter_entrypoint, None)
    if not callable(entrypoint):
        raise ConfigurationError(
            f"adapter entrypoint is not callable: {bundle.adapter_reference}"
        )
    return entrypoint


def run_dataset_stage(
    manifest_path: Path,
    *,
    stage: str,
    config_path: Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Load the adapter registered by a dataset manifest and run one stage."""

    bundle = load_dataset_bundle(manifest_path)
    if stage not in bundle.stages:
        raise ConfigurationError(
            f"stage {stage!r} is not registered for {bundle.dataset_id}: {bundle.stages}"
        )
    selected_config = (
        processing_config_path(bundle)
        if config_path is None
        else config_path.resolve()
    )
    if not selected_config.is_file():
        raise ConfigurationError(f"processing config does not exist: {selected_config}")
    return _adapter_callable(bundle)(
        selected_config,
        stage=stage,
        dry_run=dry_run,
        overwrite=overwrite,
        base_url=base_url,
        api_key=api_key,
    )


def resolve_vllm_job(
    manifest_path: Path,
    runtime_config_path: Path,
    *,
    stage: str,
    pipeline_config_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve and validate one local-vLLM dataset-stage job."""

    bundle = load_dataset_bundle(manifest_path)
    if stage not in bundle.stages:
        raise ConfigurationError(
            f"local vLLM jobs require a registered stage; found {stage!r}"
        )
    processing = require_mapping(bundle.manifest["processing"], "processing")
    registered_vllm_stages = require_mapping(
        processing.get("vllm_stages", {}), "processing.vllm_stages"
    )
    stage_spec = registered_vllm_stages.get(stage)
    if stage_spec is None:
        if stage not in {"generate", "judge"}:
            raise ConfigurationError(
                f"stage {stage!r} requires an explicit processing.vllm_stages entry"
            )
        section_name = "generation" if stage == "generate" else "judge"
        output_section_name = section_name
    else:
        stage_spec = require_mapping(
            stage_spec, f"processing.vllm_stages.{stage}"
        )
        section_name = _text(
            stage_spec.get("settings_section"),
            f"processing.vllm_stages.{stage}.settings_section",
        )
        output_section_name = _text(
            stage_spec.get("output_section"),
            f"processing.vllm_stages.{stage}.output_section",
        )
    selected_pipeline_path = (
        processing_config_path(bundle)
        if pipeline_config_path is None
        else pipeline_config_path.resolve()
    )
    pipeline = load_config(selected_pipeline_path)
    section = require_mapping(pipeline.get(section_name), section_name)
    output_section = require_mapping(
        pipeline.get(output_section_name), output_section_name
    )
    require_keys(section, {"model", "transport"}, section_name)

    runtime_config_path = runtime_config_path.resolve()
    runtime = load_config(runtime_config_path)
    require_keys(runtime, {"schema_version", "kind", "server", "cluster", "python"}, "vLLM runtime")
    reject_unknown(
        runtime,
        {"schema_version", "kind", "server", "cluster", "python"},
        "vLLM runtime",
    )
    if runtime["schema_version"] != 1 or runtime["kind"] != "factor_vllm_runtime":
        raise ConfigurationError(
            "vLLM runtime requires schema_version=1 and kind=factor_vllm_runtime"
        )

    server = require_mapping(runtime["server"], "server")
    server_fields = {
        "model",
        "served_model_name",
        "reasoning_parser",
        "tool_call_parser",
        "enable_auto_tool_choice",
        "tensor_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
        "port",
        "enable_prefix_caching",
        "trust_remote_code",
        "startup_timeout_seconds",
    }
    require_keys(server, server_fields, "server")
    reject_unknown(server, server_fields, "server")
    normalized_server = {
        "model": _text(server["model"], "server.model"),
        "served_model_name": _text(
            server["served_model_name"], "server.served_model_name"
        ),
        "reasoning_parser": _text(
            server["reasoning_parser"], "server.reasoning_parser"
        ),
        "tool_call_parser": _text(
            server["tool_call_parser"], "server.tool_call_parser"
        ),
        "enable_auto_tool_choice": _boolean(
            server["enable_auto_tool_choice"], "server.enable_auto_tool_choice"
        ),
        "tensor_parallel_size": _integer(
            server["tensor_parallel_size"], "server.tensor_parallel_size", positive=True
        ),
        "max_model_len": _integer(
            server["max_model_len"], "server.max_model_len", positive=True
        ),
        "gpu_memory_utilization": _number(
            server["gpu_memory_utilization"],
            "server.gpu_memory_utilization",
            positive=True,
        ),
        "port": _integer(server["port"], "server.port", positive=True),
        "enable_prefix_caching": _boolean(
            server["enable_prefix_caching"], "server.enable_prefix_caching"
        ),
        "trust_remote_code": _boolean(
            server["trust_remote_code"], "server.trust_remote_code"
        ),
        "startup_timeout_seconds": _integer(
            server["startup_timeout_seconds"],
            "server.startup_timeout_seconds",
            positive=True,
        ),
    }

    cluster = require_mapping(runtime["cluster"], "cluster")
    cluster_fields = {
        "gpu",
        "cpu",
        "memory_mb",
        "namespace",
        "charged_group",
        "private_machine",
        "image",
        "host_network",
        "mounts",
    }
    require_keys(cluster, cluster_fields, "cluster")
    reject_unknown(cluster, cluster_fields, "cluster")
    mounts = cluster["mounts"]
    if not isinstance(mounts, list) or not mounts:
        raise ConfigurationError("cluster.mounts must be a non-empty list")
    normalized_cluster = {
        "gpu": _integer(cluster["gpu"], "cluster.gpu", positive=True),
        "cpu": _integer(cluster["cpu"], "cluster.cpu", positive=True),
        "memory_mb": _integer(
            cluster["memory_mb"], "cluster.memory_mb", positive=True
        ),
        "namespace": _text(cluster["namespace"], "cluster.namespace"),
        "charged_group": _text(cluster["charged_group"], "cluster.charged_group"),
        "private_machine": _text(
            cluster["private_machine"], "cluster.private_machine"
        ),
        "image": _text(cluster["image"], "cluster.image"),
        "host_network": _boolean(cluster["host_network"], "cluster.host_network"),
        "mounts": [_text(value, "cluster.mounts item") for value in mounts],
    }

    python = require_mapping(runtime["python"], "python")
    require_keys(python, {"executable", "packages"}, "python")
    reject_unknown(python, {"executable", "packages"}, "python")
    packages = python["packages"]
    if not isinstance(packages, list):
        raise ConfigurationError("python.packages must be a list")
    normalized_packages: list[dict[str, str]] = []
    for index, value in enumerate(packages):
        package = require_mapping(value, f"python.packages[{index}]")
        require_keys(package, {"distribution", "import"}, f"python.packages[{index}]")
        reject_unknown(
            package,
            {"distribution", "import", "version"},
            f"python.packages[{index}]",
        )
        normalized = {
            "distribution": _text(
                package["distribution"], f"python.packages[{index}].distribution"
            ),
            "import": _text(
                package["import"], f"python.packages[{index}].import"
            ),
        }
        if "version" in package:
            normalized["version"] = _text(
                package["version"], f"python.packages[{index}].version"
            )
        normalized_packages.append(normalized)

    pipeline_model = _text(section["model"], f"{section_name}.model")
    if pipeline_model != normalized_server["served_model_name"]:
        raise ConfigurationError(
            f"{section_name}.model={pipeline_model!r} does not match "
            f"server.served_model_name={normalized_server['served_model_name']!r}"
        )
    pipeline_tool_parser = section.get("tool_call_parser")
    if (
        pipeline_tool_parser is not None
        and _text(pipeline_tool_parser, f"{section_name}.tool_call_parser")
        != normalized_server["tool_call_parser"]
    ):
        raise ConfigurationError(
            f"{section_name}.tool_call_parser={pipeline_tool_parser!r} does not match "
            f"server.tool_call_parser={normalized_server['tool_call_parser']!r}"
        )
    pipeline_auto_tools = section.get("enable_auto_tool_choice")
    if pipeline_auto_tools is not None:
        pipeline_auto_tools = _boolean(
            pipeline_auto_tools, f"{section_name}.enable_auto_tool_choice"
        )
        if pipeline_auto_tools and not normalized_server["enable_auto_tool_choice"]:
            raise ConfigurationError(
                f"{section_name} requires automatic tool choice but the server disables it"
            )
    context_length = section.get("context_length")
    if context_length is not None:
        context_length = _integer(
            context_length, f"{section_name}.context_length", positive=True
        )
        if context_length > normalized_server["max_model_len"]:
            raise ConfigurationError(
                f"{section_name}.context_length exceeds server.max_model_len"
            )

    pipeline_packages = section.get("runtime_packages", [])
    if not isinstance(pipeline_packages, list):
        raise ConfigurationError(f"{section_name}.runtime_packages must be a list")
    packages_by_import = {row["import"]: row for row in normalized_packages}
    for index, value in enumerate(pipeline_packages):
        package = require_mapping(
            value, f"{section_name}.runtime_packages[{index}]"
        )
        require_keys(
            package,
            {"distribution", "import"},
            f"{section_name}.runtime_packages[{index}]",
        )
        reject_unknown(
            package,
            {"distribution", "import", "version"},
            f"{section_name}.runtime_packages[{index}]",
        )
        normalized = {
            "distribution": _text(
                package["distribution"],
                f"{section_name}.runtime_packages[{index}].distribution",
            ),
            "import": _text(
                package["import"],
                f"{section_name}.runtime_packages[{index}].import",
            ),
        }
        if "version" in package:
            normalized["version"] = _text(
                package["version"],
                f"{section_name}.runtime_packages[{index}].version",
            )
        existing = packages_by_import.get(normalized["import"])
        if existing is not None and existing != normalized:
            raise ConfigurationError(
                f"runtime package import {normalized['import']!r} has conflicting "
                f"distributions: {existing['distribution']!r} and "
                f"{normalized['distribution']!r}"
            )
        if existing is None:
            normalized_packages.append(normalized)
            packages_by_import[normalized["import"]] = normalized

    output_value = output_section.get("output_directory")
    run_directory = (
        resolve_path(output_value, relative_to=selected_pipeline_path.parent)
        if output_value is not None
        else bundle.paths["intermediate"] / "runtime" / stage
    )
    _, _, repository = _framework_paths(manifest_path)
    return {
        "schema_version": 1,
        "kind": "factor_vllm_job",
        "dataset_id": bundle.dataset_id,
        "stage": stage,
        "section": section_name,
        "output_section": output_section_name,
        "manifest_path": str(bundle.manifest_path),
        "manifest_sha256": file_sha256(bundle.manifest_path),
        "pipeline_config_path": str(selected_pipeline_path),
        "pipeline_config_sha256": file_sha256(selected_pipeline_path),
        "runtime_config_path": str(runtime_config_path),
        "runtime_config_sha256": file_sha256(runtime_config_path),
        "adapter": bundle.adapter_reference,
        "request_model": pipeline_model,
        "request_settings": {
            key: section.get(key)
            for key in (
                "tokenizer",
                "context_length",
                "truncation",
                "padding",
                "round_seed",
                "temperature",
                "top_p",
                "max_tokens",
                "tool_call_parser",
                "enable_auto_tool_choice",
            )
            if key in section
        },
        "server": normalized_server,
        "cluster": normalized_cluster,
        "python": {
            "executable": _text(python["executable"], "python.executable"),
            "packages": normalized_packages,
        },
        "base_url": f"http://127.0.0.1:{normalized_server['port']}/v1",
        "run_directory": str(run_directory),
        "job_name": f"fm-{bundle.dataset_id.replace('_', '-')}-{stage}",
        "repository": repository_state(repository),
    }
