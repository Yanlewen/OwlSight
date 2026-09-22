"""Resolve frozen operating thresholds without selecting them on eval data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        return None
    return value


def _extract(value: Any, *, path: Path) -> float:
    if isinstance(value, Mapping):
        for key in ("threshold", "threshold_value", "eval_threshold"):
            candidate = _number(value.get(key))
            if candidate is not None:
                return candidate
        nested: list[float] = []
        for key in ("evaluation", "monitor", "calibration", "thresholds"):
            if key in value:
                try:
                    nested.append(_extract(value[key], path=path))
                except ValueError:
                    pass
        if nested and all(item == nested[0] for item in nested):
            return nested[0]
    raise ValueError(f"{path} does not contain one frozen numeric threshold in [0,1]")


def _load(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"threshold source does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid threshold JSON: {path}") from exc
    return _extract(value, path=path)


def resolve_threshold(
    entry: Mapping[str, Any], *, manifest_path: Path,
    thresholds_file: Path | None = None, threshold_root: Path | None = None,
    monitor_config_root: Path | None = None,
) -> dict[str, Any] | None:
    """Return a declared frozen threshold and its source.

    Resolution order is explicit: manifest entry, mapping file, threshold
    directory, then monitor config directory.  There is deliberately no
    fallback to 0.5 and no threshold search over evaluation rows.
    """
    benchmark_id = str(entry["id"])
    if _number(entry.get("threshold")) is not None:
        return {"value": _number(entry["threshold"]), "source": "manifest"}

    if entry.get("threshold_source"):
        source = Path(str(entry["threshold_source"])).expanduser()
        if not source.is_absolute():
            source = manifest_path.parent / source
        return {"value": _load(source), "source": str(source.resolve())}

    if thresholds_file is not None:
        value = json.loads(thresholds_file.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or benchmark_id not in value:
            raise ValueError(f"threshold mapping has no entry for {benchmark_id}: {thresholds_file}")
        threshold = _number(value[benchmark_id])
        if threshold is None:
            raise ValueError(f"invalid threshold for {benchmark_id}: {thresholds_file}")
        return {"value": threshold, "source": str(thresholds_file.resolve())}

    candidates: list[Path] = []
    for root in (threshold_root, monitor_config_root):
        if root is not None:
            candidates.extend((root / f"{benchmark_id}.json", root / benchmark_id / "effective_config.json"))
    for candidate in candidates:
        if candidate.is_file():
            return {"value": _load(candidate), "source": str(candidate.resolve())}

    return None

