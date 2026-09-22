"""Model-agnostic binary metrics for persisted decisions."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable


def _counts(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, int], int]:
    counts = Counter(tp=0, fp=0, tn=0, fn=0)
    invalid = 0
    values = list(rows)
    if not values:
        raise ValueError("cannot score an empty collection")
    for row in values:
        label = row.get("label_id")
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("label_id must be integer 0 or 1")
        decision = row.get("decision")
        if decision in (0, 1) and not isinstance(decision, bool):
            prediction = int(decision)
        elif decision in {"applied", "not_applied"}:
            prediction = int(decision == "applied")
        else:
            invalid += 1
            prediction = None
        if prediction is None:
            # An invalid response is an error for either ground-truth class.
            counts["fn" if label == 1 else "fp"] += 1
        elif label == 1 and prediction == 1:
            counts["tp"] += 1
        elif label == 1:
            counts["fn"] += 1
        elif prediction == 1:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return dict(counts), invalid


def _summary(counts: dict[str, int], *, rows: int, invalid: int, threshold: float | None) -> dict[str, Any]:
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    gmean2 = tpr * tnr if tpr is not None and tnr is not None else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if tpr is not None and precision + tpr else 0.0
    return {
        **counts,
        "rows": rows,
        "invalid": invalid,
        "invalid_rate": invalid / rows,
        "positive_rows": tp + fn,
        "negative_rows": tn + fp,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "gmean": math.sqrt(gmean2) if gmean2 is not None else None,
        "gmean2": gmean2,
        "f1": f1,
        "threshold": threshold,
    }


def _score_values(rows: Iterable[dict[str, Any]]) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        label = row.get("label_id")
        score = row.get("score")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"row {index} label_id must be integer 0 or 1")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"row {index} score must be finite numeric")
        values.append((int(label), float(score)))
    if not values:
        raise ValueError("cannot score an empty collection")
    return values


def auroc(rows: Iterable[dict[str, Any]]) -> float | None:
    values = _score_values(rows)
    positives = sum(label for label, _ in values)
    negatives = len(values) - positives
    if not positives or not negatives:
        return None
    grouped: dict[float, list[int]] = {}
    for label, score in values:
        grouped.setdefault(score, [0, 0])[label] += 1
    negatives_below = 0
    wins = 0.0
    for score in sorted(grouped):
        negative_count, positive_count = grouped[score]
        wins += positive_count * negatives_below + 0.5 * positive_count * negative_count
        negatives_below += negative_count
    return wins / (positives * negatives)


def average_precision(rows: Iterable[dict[str, Any]]) -> float | None:
    values = sorted(_score_values(rows), key=lambda item: item[1], reverse=True)
    positives = sum(label for label, _ in values)
    if not positives:
        return None
    true_positives = false_positives = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    while index < len(values):
        score = values[index][1]
        labels: list[int] = []
        while index < len(values) and values[index][1] == score:
            labels.append(values[index][0])
            index += 1
        true_positives += sum(labels)
        false_positives += len(labels) - sum(labels)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def score_decisions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    counts, invalid = _counts(values)
    return _summary(counts, rows=len(values), invalid=invalid, threshold=None)


def score_rows(rows: Iterable[dict[str, Any]], threshold: float) -> dict[str, Any]:
    values = list(rows)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be numeric")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0,1]")
    values = list(rows)
    converted = []
    for row in values:
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            converted.append({"label_id": row.get("label_id"), "decision": "invalid"})
        else:
            converted.append({"label_id": row.get("label_id"), "decision": int(float(score) >= threshold)})
    counts, invalid = _counts(converted)
    ranking_valid = all(
        row.get("label_id") in (0, 1)
        and not isinstance(row.get("label_id"), bool)
        and isinstance(row.get("score"), (int, float))
        and not isinstance(row.get("score"), bool)
        and math.isfinite(float(row.get("score")))
        for row in values
    )
    return {
        **_summary(counts, rows=len(values), invalid=invalid, threshold=threshold),
        "auroc": auroc(values) if ranking_valid else None,
        "auprc": average_precision(values) if ranking_valid else None,
    }
