"""Neutral dataset-bundle manifests and six-cell path validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    ConfigurationError,
    file_sha256,
    load_config,
    reject_unknown,
    require_keys,
    require_mapping,
    resolve_path,
    stable_hash,
)
from .io import iter_jsonl
from .schema import CanonicalRecord, Taxonomy, validate_record_collection


STAGES = (
    "download",
    "prepare",
    "prepare-contrast",
    "prepare-contrast-smoke",
    "prepare-contrast-topup-pilot",
    "prepare-contrast-topup-smoke",
    "prepare-contrast-topup-scale",
    "prepare-contrast-topup-scale-smoke",
    "prepare-contrast-topup-recovery",
    "prepare-contrast-topup-recovery-smoke",
    "replace-invalid-contrast",
    "build-runner",
    "generate",
    "generate-contrast",
    "generate-contrast-smoke",
    "generate-contrast-topup-pilot",
    "generate-contrast-topup-smoke",
    "generate-contrast-topup-scale",
    "generate-contrast-topup-scale-smoke",
    "generate-contrast-topup-recovery",
    "generate-contrast-topup-recovery-smoke",
    "generate-contrast-topup-scale-shard-0",
    "generate-contrast-topup-scale-shard-1",
    "generate-contrast-topup-scale-shard-2",
    "decoy-smoke-prepare",
    "decoy-smoke-generate",
    "decoy-smoke-analyze",
    "decoy-prepare",
    "decoy-spec-judge-smoke",
    "decoy-spec-judge",
    "decoy-freeze",
    "decoy-generation-smoke",
    "decoy-application-judge-smoke",
    "decoy-generate",
    "decoy-judge-first",
    "decoy-retry-generate",
    "decoy-judge-final",
    "decoy-curate",
    "decoy-audit",
    "relational-audit",
    "relational-topup-audit",
    "relational-topup-prepare",
    "relational-topup-generation-smoke",
    "relational-topup-judge-smoke",
    "relational-topup-generate",
    "relational-topup-judge",
    "relational-finalize",
    "relational-curate",
    "verify",
    "extract-factor",
    "extract-factor-contrast",
    "extract-factor-contrast-topup-pilot",
    "extract-factor-contrast-topup-scale",
    "extract-factor-contrast-topup-recovery",
    "label-application",
    "judge",
    "judge-contrast",
    "judge-contrast-smoke",
    "judge-contrast-topup-pilot",
    "judge-contrast-topup-scale",
    "judge-contrast-topup-recovery",
    "judge_recheck",
    "audit_positive",
    "audit_disagreements",
    "canonicalize",
    "audit",
    "audit-contrast",
    "audit-application",
    "curate",
    "curate-contrast-topup-pilot",
    "curate-contrast-topup-scale",
    "curate-contrast-topup-recovery",
    "merge-contrast-topup-scale-shards",
    "materialize-contrast-training",
    "stance",
    "label",
    "validate",
)
FIXED_DATA_PATHS = {
    "raw": "data/raw",
    "intermediate": "data/intermediate",
    "canonical": "data/canonical",
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class DatasetBundle:
    manifest_path: Path
    bundle_dir: Path
    datasets_root: Path
    dataset_id: str
    cell: str
    family: str
    description: str
    source: dict[str, str]
    adapter_reference: str
    adapter_path: Path | None
    adapter_entrypoint: str
    stages: tuple[str, ...]
    paths: dict[str, Path]
    manifest: dict[str, Any]

    @classmethod
    def load(
        cls, manifest_path: Path, *, taxonomy: Taxonomy, datasets_root: Path
    ) -> "DatasetBundle":
        manifest_path = manifest_path.resolve()
        datasets_root = datasets_root.resolve()
        manifest = load_config(manifest_path)
        required = {
            "schema_version",
            "kind",
            "dataset_id",
            "cell",
            "family",
            "description",
            "source",
            "processing",
            "paths",
        }
        require_keys(manifest, required, "dataset manifest")
        reject_unknown(manifest, required, "dataset manifest")
        if manifest["schema_version"] != 1 or manifest["kind"] != "factor_dataset":
            raise ConfigurationError(
                "dataset manifest requires schema_version=1 and kind=factor_dataset"
            )

        dataset_id = _text(manifest["dataset_id"], "dataset_id")
        cell = _text(manifest["cell"], "cell")
        family = _text(manifest["family"], "family")
        taxonomy.expected_relation(cell, family)
        bundle_dir = manifest_path.parent
        try:
            relative = bundle_dir.relative_to(datasets_root)
        except ValueError as exc:
            raise ConfigurationError(
                f"dataset bundle must be below datasets root {datasets_root}"
            ) from exc
        expected_parts = (cell, family, dataset_id)
        if relative.parts != expected_parts:
            raise ConfigurationError(
                "dataset bundle path must be "
                f"datasets/<cell>/<family>/<dataset_id>; expected {expected_parts}, "
                f"found {relative.parts}"
            )

        source = require_mapping(manifest["source"], "source")
        source_fields = {"name", "origin", "revision", "license"}
        require_keys(source, source_fields, "source")
        reject_unknown(source, source_fields, "source")
        normalized_source = {field: _text(source[field], f"source.{field}") for field in source_fields}

        processing = require_mapping(manifest["processing"], "processing")
        require_keys(processing, {"adapter", "stages"}, "processing")
        reject_unknown(
            processing,
            {"adapter", "stages", "config", "vllm_stages"},
            "processing",
        )
        adapter = _text(processing["adapter"], "processing.adapter")
        if adapter.count(":") != 1:
            raise ConfigurationError(
                "processing.adapter must be '<relative Python path>:<entrypoint>'"
            )
        adapter_value, adapter_entrypoint = adapter.split(":", 1)
        adapter_path: Path | None = None
        if adapter_value.endswith(".py"):
            adapter_path = resolve_path(adapter_value, relative_to=bundle_dir)
            try:
                adapter_path.relative_to(bundle_dir)
            except ValueError as exc:
                raise ConfigurationError("local dataset adapter must live inside its bundle") from exc
            if not adapter_path.is_file():
                raise ConfigurationError(f"dataset adapter does not exist: {adapter_path}")
        elif "/" in adapter_value or not all(
            part.isidentifier() for part in adapter_value.split(".")
        ):
            raise ConfigurationError(
                "shared adapter must be a Python module path, for example "
                "factor_monitor.migrate:run_stage"
            )
        _text(adapter_entrypoint, "processing adapter entrypoint")
        stages_value = processing["stages"]
        if not isinstance(stages_value, list) or not stages_value:
            raise ConfigurationError("processing.stages must be a non-empty list")
        stages = tuple(_text(stage, "processing stage") for stage in stages_value)
        if len(stages) != len(set(stages)) or any(stage not in STAGES for stage in stages):
            raise ConfigurationError(
                f"processing.stages must be unique values from {list(STAGES)}"
            )
        vllm_stages = processing.get("vllm_stages", {})
        if not isinstance(vllm_stages, dict):
            raise ConfigurationError("processing.vllm_stages must be a mapping")
        for stage, value in vllm_stages.items():
            stage = _text(stage, "processing.vllm_stages stage")
            if stage not in stages:
                raise ConfigurationError(
                    f"processing.vllm_stages references unregistered stage {stage!r}"
                )
            spec = require_mapping(
                value, f"processing.vllm_stages.{stage}"
            )
            fields = {"settings_section", "output_section"}
            require_keys(spec, fields, f"processing.vllm_stages.{stage}")
            reject_unknown(spec, fields, f"processing.vllm_stages.{stage}")
            for field in fields:
                _text(spec[field], f"processing.vllm_stages.{stage}.{field}")

        paths_value = require_mapping(manifest["paths"], "paths")
        require_keys(paths_value, set(FIXED_DATA_PATHS), "paths")
        reject_unknown(paths_value, set(FIXED_DATA_PATHS), "paths")
        paths: dict[str, Path] = {}
        for name, fixed in FIXED_DATA_PATHS.items():
            if paths_value[name] != fixed:
                raise ConfigurationError(f"paths.{name} must be {fixed!r}")
            paths[name] = resolve_path(paths_value[name], relative_to=bundle_dir)

        return cls(
            manifest_path=manifest_path,
            bundle_dir=bundle_dir,
            datasets_root=datasets_root,
            dataset_id=dataset_id,
            cell=cell,
            family=family,
            description=_text(manifest["description"], "description"),
            source=normalized_source,
            adapter_reference=adapter,
            adapter_path=adapter_path,
            adapter_entrypoint=adapter_entrypoint,
            stages=stages,
            paths=paths,
            manifest=manifest,
        )

    def summary(self, *, taxonomy: Taxonomy, require_canonical: bool) -> dict[str, Any]:
        records_path = self.paths["canonical"] / "records.jsonl"
        canonical: dict[str, Any]
        if records_path.is_file():
            records = [
                CanonicalRecord.from_mapping(row, taxonomy=taxonomy)
                for row in iter_jsonl(records_path)
            ]
            wrong_owner = [
                record.sample_id
                for record in records
                if record.cell != self.cell
                or record.family != self.family
                or record.source != self.dataset_id
            ]
            if wrong_owner:
                raise ConfigurationError(
                    f"canonical records disagree with bundle ownership: {wrong_owner[:10]}"
                )
            canonical = {
                "status": "present",
                "path": str(records_path),
                "sha256": file_sha256(records_path),
                "summary": validate_record_collection(records),
            }
        elif require_canonical:
            raise ConfigurationError(f"canonical records do not exist: {records_path}")
        else:
            canonical = {"status": "not_built", "path": str(records_path)}
        return {
            "schema_version": 1,
            "kind": "factor_dataset_bundle_validation",
            "dataset_id": self.dataset_id,
            "cell": self.cell,
            "family": self.family,
            "manifest_path": str(self.manifest_path),
            "manifest_hash": stable_hash(self.manifest),
            "adapter": (
                {
                    "kind": "bundle_local",
                    "reference": self.adapter_reference,
                    "path": str(self.adapter_path),
                    "entrypoint": self.adapter_entrypoint,
                    "sha256": file_sha256(self.adapter_path),
                }
                if self.adapter_path is not None
                else {
                    "kind": "shared",
                    "reference": self.adapter_reference,
                    "entrypoint": self.adapter_entrypoint,
                }
            ),
            "stages": list(self.stages),
            "canonical": canonical,
        }


def validate_dataset_bundle(
    manifest_path: Path,
    *,
    taxonomy_path: Path,
    datasets_root: Path,
    require_canonical: bool,
) -> dict[str, Any]:
    taxonomy = Taxonomy.load(taxonomy_path.resolve())
    bundle = DatasetBundle.load(
        manifest_path, taxonomy=taxonomy, datasets_root=datasets_root
    )
    return bundle.summary(taxonomy=taxonomy, require_canonical=require_canonical)
