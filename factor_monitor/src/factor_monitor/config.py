"""Config, hashing, and reproducibility helpers shared by every pipeline stage."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a config is incomplete or internally inconsistent."""


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def require_keys(value: dict[str, Any], keys: set[str], name: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ConfigurationError(f"{name} missing required fields: {missing}")


def reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"{name} has unknown fields: {unknown}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge JSON-safe mappings without mutating either input."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load YAML with optional ``extends`` inheritance resolved relative to the file."""

    path = path.resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ConfigurationError(f"cyclic config inheritance at {path}")
    seen.add(path)
    if not path.is_file():
        raise ConfigurationError(f"config does not exist: {path}")
    raw = require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    parent_value = raw.pop("extends", None)
    if parent_value is None:
        return raw
    parent_path = resolve_path(parent_value, relative_to=path.parent)
    return deep_merge(load_config(parent_path, seen), raw)


def resolve_path(value: Any, *, relative_to: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"path must be a non-empty string, got {value!r}")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_state(repo_dir: Path) -> dict[str, Any]:
    """Capture bounded git provenance without enumerating repository contents."""

    repo_dir = repo_dir.resolve()
    synthetic_snapshot = os.environ.get("FACTOR_MONITOR_SYNTHETIC_GIT") == "1"
    bundle_value = os.environ.get("FACTOR_MONITOR_SOURCE_BUNDLE")
    bundle_path = Path(bundle_value).resolve() if bundle_value else None
    if synthetic_snapshot and bundle_path is None:
        raise ConfigurationError(
            "FACTOR_MONITOR_SYNTHETIC_GIT=1 requires FACTOR_MONITOR_SOURCE_BUNDLE"
        )
    if bundle_path is not None and not bundle_path.is_file():
        raise ConfigurationError(
            f"FACTOR_MONITOR_SOURCE_BUNDLE does not exist: {bundle_path}"
        )
    bundle_fields = {
        "source_bundle_path": str(bundle_path) if bundle_path is not None else None,
        "source_bundle_sha256": file_sha256(bundle_path) if bundle_path is not None else None,
    }

    def git(*args: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ["git", "-c", f"safe.directory={repo_dir}", "-C", str(repo_dir), *args],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return None

    root = git("rev-parse", "--show-toplevel")
    if root is None:
        return {
            "repository_root": None,
            "git_commit": None,
            "git_state": "unavailable",
            "git_dirty": None,
            "git_status_sha256": None,
            "git_diff_sha256": None,
            "git_untracked_entries": None,
            **bundle_fields,
        }
    if root.returncode != 0:
        return {
            "repository_root": None,
            "git_commit": None,
            "git_state": "not_a_repository",
            "git_dirty": None,
            "git_status_sha256": None,
            "git_diff_sha256": None,
            "git_untracked_entries": None,
            **bundle_fields,
        }
    head = git("rev-parse", "--verify", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    commit = head.stdout.strip() if head is not None and head.returncode == 0 else None
    status_text = status.stdout if status is not None and status.returncode == 0 else None
    diff = git("diff", "--binary", "HEAD") if commit else None
    diff_text = diff.stdout if diff is not None and diff.returncode == 0 else None
    untracked = (
        sum(1 for line in status_text.splitlines() if line.startswith("?? "))
        if status_text is not None
        else None
    )
    return {
        "repository_root": str(Path(root.stdout.strip()).resolve()),
        "git_commit": commit,
        "git_state": (
            "synthetic_snapshot"
            if commit and synthetic_snapshot and bundle_path is not None
            else "committed"
            if commit
            else "unavailable"
            if head is None
            else "unborn"
        ),
        "git_dirty": bool(status_text.strip()) if status_text is not None else None,
        "git_status_sha256": (
            hashlib.sha256(status_text.encode("utf-8")).hexdigest()
            if status_text is not None
            else None
        ),
        "git_diff_sha256": (
            hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
            if diff_text is not None
            else None
        ),
        "git_untracked_entries": untracked,
        **bundle_fields,
    }
