"""Canonical data schema and six-cell taxonomy validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import (
    ConfigurationError,
    load_config,
    reject_unknown,
    require_keys,
    require_mapping,
)


LEVELS = ("L1", "L3")
LABEL_NAMES = {0: "not_applied", 1: "applied"}


class SchemaError(ValueError):
    """Raised when a canonical record violates the data contract."""


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return value.strip()


def _binary_label(value: Any, name: str = "label") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in LABEL_NAMES:
        raise SchemaError(f"{name} must be integer 0 or 1")
    return value


@dataclass(frozen=True)
class FactorSpec:
    factor: str
    relation: str
    applied_criterion: str
    not_applied_criterion: str

    @classmethod
    def from_mapping(cls, value: Any, name: str) -> "FactorSpec":
        mapping = require_mapping(value, name)
        allowed = {"factor", "relation", "applied_criterion", "not_applied_criterion"}
        require_keys(mapping, allowed, name)
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise SchemaError(f"{name} has unknown fields: {unknown}")
        return cls(
            factor=_nonempty_text(mapping["factor"], f"{name}.factor"),
            relation=_nonempty_text(mapping["relation"], f"{name}.relation"),
            applied_criterion=_nonempty_text(
                mapping["applied_criterion"], f"{name}.applied_criterion"
            ),
            not_applied_criterion=_nonempty_text(
                mapping["not_applied_criterion"], f"{name}.not_applied_criterion"
            ),
        )


@dataclass(frozen=True)
class Taxonomy:
    cells: dict[str, dict[str, Any]]
    relations: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        value = load_config(path)
        if value.get("schema_version") != 1:
            raise ConfigurationError("taxonomy schema_version must be 1")
        require_keys(value, {"axes", "relations", "cells"}, "taxonomy")
        reject_unknown(value, {"schema_version", "axes", "relations", "cells"}, "taxonomy")
        relations_value = value["relations"]
        if not isinstance(relations_value, list) or not relations_value:
            raise ConfigurationError("taxonomy.relations must be a non-empty list")
        relations = tuple(_nonempty_text(item, "taxonomy relation") for item in relations_value)
        if len(relations) != len(set(relations)):
            raise ConfigurationError("taxonomy.relations contains duplicates")
        cells = require_mapping(value["cells"], "taxonomy.cells")
        if len(cells) != 6:
            raise ConfigurationError(f"taxonomy must define exactly six cells, found {len(cells)}")
        for cell_name, cell_value in cells.items():
            cell = require_mapping(cell_value, f"taxonomy.cells.{cell_name}")
            require_keys(cell, {"domain", "locus", "families"}, f"taxonomy.cells.{cell_name}")
            families = require_mapping(cell["families"], f"taxonomy.cells.{cell_name}.families")
            if not families:
                raise ConfigurationError(f"taxonomy cell {cell_name} has no families")
            for family_name, family_value in families.items():
                family = require_mapping(
                    family_value, f"taxonomy.cells.{cell_name}.families.{family_name}"
                )
                require_keys(family, {"relation"}, f"taxonomy family {family_name}")
                if family["relation"] not in relations:
                    raise ConfigurationError(
                        f"taxonomy family {cell_name}/{family_name} uses unknown relation "
                        f"{family['relation']!r}"
                    )
        return cls(cells=cells, relations=relations)

    def expected_relation(self, cell: str, family: str) -> str:
        if cell not in self.cells:
            raise SchemaError(f"unknown taxonomy cell: {cell}")
        families = self.cells[cell]["families"]
        if family not in families:
            raise SchemaError(f"unknown family {family!r} for cell {cell!r}")
        return str(families[family]["relation"])


_RECORD_FIELDS = {
    "schema_version",
    "sample_id",
    "trajectory_id",
    "task_id",
    "leakage_group_id",
    "cell",
    "family",
    "source",
    "task_context",
    "model_cot",
    "factor_level",
    "factor_spec",
    "annotation_status",
    "label",
    # Some published canonical artifacts retain explicit aliases for the
    # binary label.  They are accepted only as consistency-checked metadata;
    # CanonicalRecord continues to use `label` as the single source of truth.
    "label_id",
    "label_name",
    "label_provenance",
    "evidence_spans",
    "metadata",
}


@dataclass(frozen=True)
class CanonicalRecord:
    schema_version: int
    sample_id: str
    trajectory_id: str
    task_id: str
    leakage_group_id: str
    cell: str
    family: str
    source: str
    task_context: str
    model_cot: str
    factor_level: str
    factor_spec: FactorSpec | None
    annotation_status: str
    label: int
    label_provenance: dict[str, Any]
    evidence_spans: tuple[str, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Any, *, taxonomy: Taxonomy) -> "CanonicalRecord":
        row = require_mapping(value, "canonical record")
        unknown = sorted(set(row) - _RECORD_FIELDS)
        if unknown:
            raise SchemaError(f"canonical record has unknown fields: {unknown}")
        required = _RECORD_FIELDS - {
            "evidence_spans",
            "metadata",
            "label_id",
            "label_name",
        }
        require_keys(row, required, "canonical record")
        if row["schema_version"] != 1:
            raise SchemaError("canonical record schema_version must be 1")
        cell = _nonempty_text(row["cell"], "cell")
        family = _nonempty_text(row["family"], "family")
        expected_relation = taxonomy.expected_relation(cell, family)
        factor_level = _nonempty_text(row["factor_level"], "factor_level")
        if factor_level not in LEVELS:
            raise SchemaError(f"factor_level must be one of {list(LEVELS)}")
        annotation_status = _nonempty_text(row["annotation_status"], "annotation_status")
        if annotation_status not in {"complete", "pending"}:
            raise SchemaError("annotation_status must be complete or pending")
        factor_spec_value = row["factor_spec"]
        factor_spec = None
        if annotation_status == "complete":
            factor_spec = FactorSpec.from_mapping(factor_spec_value, "factor_spec")
            if factor_spec.relation != expected_relation:
                raise SchemaError(
                    f"factor_spec.relation={factor_spec.relation!r} does not match taxonomy "
                    f"relation {expected_relation!r} for {cell}/{family}"
                )
        elif factor_spec_value is not None:
            raise SchemaError("pending records must use factor_spec=null")
        label = _binary_label(row["label"])
        if "label_id" in row and _binary_label(row["label_id"], "label_id") != label:
            raise SchemaError("label_id must match label")
        if "label_name" in row:
            label_name = _nonempty_text(row["label_name"], "label_name")
            if label_name != LABEL_NAMES[label]:
                raise SchemaError("label_name must match label")
        model_cot = _nonempty_text(row["model_cot"], "model_cot")
        spans_value = row.get("evidence_spans", [])
        if not isinstance(spans_value, list):
            raise SchemaError("evidence_spans must be a list")
        evidence_spans = tuple(_nonempty_text(span, "evidence span") for span in spans_value)
        missing_spans = [span for span in evidence_spans if span not in model_cot]
        if missing_spans:
            raise SchemaError("every evidence span must be an exact substring of model_cot")

        provenance = require_mapping(row["label_provenance"], "label_provenance")
        require_keys(provenance, {"method"}, "label_provenance")
        _nonempty_text(provenance["method"], "label_provenance.method")
        metadata = require_mapping(row.get("metadata", {}), "metadata")

        return cls(
            schema_version=1,
            sample_id=_nonempty_text(row["sample_id"], "sample_id"),
            trajectory_id=_nonempty_text(row["trajectory_id"], "trajectory_id"),
            task_id=_nonempty_text(row["task_id"], "task_id"),
            leakage_group_id=_nonempty_text(row["leakage_group_id"], "leakage_group_id"),
            cell=cell,
            family=family,
            source=_nonempty_text(row["source"], "source"),
            task_context=_nonempty_text(row["task_context"], "task_context"),
            model_cot=model_cot,
            factor_level=factor_level,
            factor_spec=factor_spec,
            annotation_status=annotation_status,
            label=label,
            label_provenance=dict(provenance),
            evidence_spans=evidence_spans,
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_spans"] = list(self.evidence_spans)
        return value


def validate_record_collection(records: Iterable[CanonicalRecord]) -> dict[str, Any]:
    rows = list(records)
    sample_ids = [row.sample_id for row in rows]
    duplicates = sorted(name for name, count in Counter(sample_ids).items() if count > 1)
    if duplicates:
        raise SchemaError(f"duplicate sample_id values: {duplicates[:10]}")
    return {
        "records": len(rows),
        "labels": dict(sorted(Counter(LABEL_NAMES[row.label] for row in rows).items())),
        "cells": dict(sorted(Counter(row.cell for row in rows).items())),
        "families": dict(sorted(Counter(row.family for row in rows).items())),
        "sources": dict(sorted(Counter(row.source for row in rows).items())),
        "factor_levels": dict(sorted(Counter(row.factor_level for row in rows).items())),
        "annotation_status": dict(
            sorted(Counter(row.annotation_status for row in rows).items())
        ),
        "label_provenance_methods": dict(
            sorted(Counter(str(row.label_provenance["method"]) for row in rows).items())
        ),
        "trajectories": len({row.trajectory_id for row in rows}),
        "leakage_groups": len({row.leakage_group_id for row in rows}),
    }
