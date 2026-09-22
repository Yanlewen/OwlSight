"""Manifest validation and artifact snapshot utilities."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .io import sha256, write_json


def eval_root() -> Path:
    """Return the `factor_monitor_eval/` directory that contains `manifests/` and `datasets/`."""
    return Path(__file__).resolve().parents[2]


def resolve_eval_path(path: str | Path, *, manifest_path: Path | None = None) -> Path:
    """Resolve artifact paths from either the repo root or a standalone eval folder.

    Manifests may use `datasets/...` or `factor_monitor_eval/datasets/...`. Absolute
    paths are left unchanged. The first existing candidate wins; if none exist, the
    first candidate is returned so callers can raise a path-specific error.
    """
    raw = Path(str(path)).expanduser()
    if raw.is_absolute():
        return raw
    roots: list[Path] = [Path.cwd(), eval_root()]
    if manifest_path is not None:
        manifest_dir = Path(manifest_path).expanduser().resolve().parent
        roots.extend((manifest_dir, manifest_dir.parent))
    relatives = [raw]
    if raw.parts and raw.parts[0] == "factor_monitor_eval":
        relatives.append(Path(*raw.parts[1:]))
    else:
        relatives.append(Path("factor_monitor_eval") / raw)
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        for relative in relatives:
            candidate = (root / relative).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
            if candidate.exists():
                return candidate
    return candidates[0]


def load_manifest(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("benchmarks"), list):
        raise ValueError("manifest must contain a benchmarks list")
    ids = [item.get("id") for item in value["benchmarks"]]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("benchmark IDs must be unique non-empty strings")
    return value


def select_benchmarks(manifest: dict[str, Any], ids: list[str] | None = None) -> list[dict[str, Any]]:
    entries = list(manifest["benchmarks"])
    if not ids:
        return entries
    wanted = list(ids)
    by_id = {str(entry["id"]): entry for entry in entries}
    missing = [item for item in wanted if item not in by_id]
    if missing:
        raise ValueError(f"unknown benchmark ids: {missing}")
    return [by_id[item] for item in wanted]


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = load_manifest(path)
    checked = []
    for entry in manifest["benchmarks"]:
        artifact = resolve_eval_path(entry["artifact"], manifest_path=path)
        if not artifact.is_dir():
            checked.append({"id": entry["id"], "status": "missing", "artifact": str(artifact)})
            continue
        files = {}
        for name in ("eval_queries.jsonl", "effective_config.json", "summary.json"):
            path_value = artifact / name
            if path_value.is_file(): files[name] = {"path": str(path_value), "sha256": sha256(path_value)}
        checked.append({"id": entry["id"], "status": "ok" if "eval_queries.jsonl" in files else "invalid", "artifact": str(artifact), "files": files})
    return {"schema_version": 1, "manifest": str(path.resolve()), "benchmarks": checked}


def sync_artifacts(manifest_path: Path, destination: Path, overwrite: bool = False) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in manifest["benchmarks"]:
        source = resolve_eval_path(entry["artifact"], manifest_path=manifest_path)
        if not source.is_dir(): raise FileNotFoundError(source)
        target = destination / str(entry["id"])
        if target.exists():
            if not overwrite: raise FileExistsError(target)
            shutil.rmtree(target)
        shutil.copytree(source, target)
        complete_snapshot = None
        complete = entry.get("complete_eval")
        if isinstance(complete, dict):
            complete_snapshot = {}
            for name in ("accepted_artifact", "overlength_artifact"):
                value = complete.get(name)
                if not value:
                    continue
                complete_source = resolve_eval_path(value, manifest_path=manifest_path)
                if not complete_source.is_dir():
                    raise FileNotFoundError(complete_source)
                complete_target = target / "complete_eval" / ("accepted" if name == "accepted_artifact" else "overlength")
                complete_target.parent.mkdir(parents=True, exist_ok=True)
                if complete_target.exists():
                    if not overwrite:
                        raise FileExistsError(complete_target)
                    shutil.rmtree(complete_target)
                shutil.copytree(complete_source, complete_target)
                complete_snapshot[name] = {
                    "source": str(complete_source),
                    "snapshot": str(complete_target),
                    "files": {p.name: sha256(p) for p in complete_target.iterdir() if p.is_file()},
                }
        entries.append({"id": entry["id"], "source": str(source), "snapshot": str(target), "files": {p.name: sha256(p) for p in target.iterdir() if p.is_file()}, "complete_eval": complete_snapshot})
    result = {"schema_version": 1, "kind": "factor_monitor_eval_dataset_snapshot", "entries": entries}
    write_json(destination / "snapshot_manifest.json", result)
    return result
