"""Backend-independent binary monitor metrics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Sequence


def _validated(rows: Iterable[dict[str, Any]]) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        label = row.get("label_id")
        score = row.get("score")
        if isinstance(label, bool) or not isinstance(label, int) or label not in {0, 1}:
            raise ValueError(f"row {index} label_id must be integer 0 or 1")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"row {index} score must be numeric")
        score_value = float(score)
        if not math.isfinite(score_value):
            raise ValueError(f"row {index} score must be finite")
        values.append((label, score_value))
    if not values:
        raise ValueError("cannot score an empty collection")
    return values


def _metrics_from_counts(*, threshold: float, tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr if tpr is not None else 0.0
    gmean2 = tpr * tnr if tpr is not None and tnr is not None else None
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "gmean": math.sqrt(gmean2) if gmean2 is not None else None,
        "gmean2": gmean2,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def threshold_metrics(rows: Iterable[dict[str, Any]], threshold: float) -> dict[str, Any]:
    values = _validated(rows)
    tp = fp = tn = fn = 0
    for label, score in values:
        prediction = int(score >= threshold)
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 0:
            tn += 1
        else:
            fn += 1
    return _metrics_from_counts(threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn)


def threshold_curve(
    rows: Iterable[dict[str, Any]], *, extra_thresholds: Iterable[float] = ()
) -> list[dict[str, Any]]:
    """Return every distinct binary operating point in O(n log n) time.

    Predictions follow the package-wide ``score >= threshold`` convention.
    Extra thresholds are useful for placing a frozen operating point on the
    same curve without changing the set of oracle candidates.
    """

    values = _validated(rows)
    grouped: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for label, score in values:
        grouped[score][label] += 1
    scores = sorted(grouped)
    thresholds = {
        scores[0] - 1e-12,
        *scores,
        scores[-1] + 1e-12,
    }
    for threshold in extra_thresholds:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("extra thresholds must be numeric")
        value = float(threshold)
        if not math.isfinite(value):
            raise ValueError("extra thresholds must be finite")
        thresholds.add(value)

    positives = sum(label for label, _ in values)
    negatives = len(values) - positives
    tp, fp, tn, fn = positives, negatives, 0, 0
    curve: list[dict[str, Any]] = []
    score_index = 0
    for threshold in sorted(thresholds):
        while score_index < len(scores) and scores[score_index] < threshold:
            negative_count, positive_count = grouped[scores[score_index]]
            tp -= positive_count
            fn += positive_count
            fp -= negative_count
            tn += negative_count
            score_index += 1
        curve.append(
            _metrics_from_counts(threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn)
        )
    return curve


def best_threshold_metrics(
    rows: Iterable[dict[str, Any]], *, metric: str, extra_thresholds: Iterable[float] = ()
) -> dict[str, Any]:
    if metric not in {"balanced_accuracy", "gmean2"}:
        raise ValueError(f"unsupported threshold selection metric: {metric}")
    curve = threshold_curve(rows, extra_thresholds=extra_thresholds)
    defined = [row for row in curve if row[metric] is not None]
    if not defined:
        raise ValueError(f"cannot select {metric} threshold without both classes")
    return max(
        defined,
        key=lambda row: (float(row[metric]), -abs(float(row["threshold"]) - 0.5), float(row["threshold"])),
    )


def auroc(rows: Iterable[dict[str, Any]]) -> float | None:
    values = _validated(rows)
    positives = sum(label for label, _ in values)
    negatives = len(values) - positives
    if not positives or not negatives:
        return None
    grouped: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for label, score in values:
        grouped[score][label] += 1
    negatives_below = 0
    wins = 0.0
    for score in sorted(grouped):
        negative_count, positive_count = grouped[score]
        wins += positive_count * negatives_below + 0.5 * positive_count * negative_count
        negatives_below += negative_count
    return wins / (positives * negatives)


def average_precision(rows: Iterable[dict[str, Any]]) -> float | None:
    values = sorted(_validated(rows), key=lambda item: item[1], reverse=True)
    positives = sum(label for label, _ in values)
    if not positives:
        return None
    true_positives = false_positives = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    while index < len(values):
        score = values[index][1]
        group_labels: list[int] = []
        while index < len(values) and values[index][1] == score:
            group_labels.append(values[index][0])
            index += 1
        true_positives += sum(group_labels)
        false_positives += len(group_labels) - sum(group_labels)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def select_balanced_threshold(rows: Iterable[dict[str, Any]]) -> float:
    return float(best_threshold_metrics(rows, metric="balanced_accuracy")["threshold"])


def score_summary(rows: Iterable[dict[str, Any]], threshold: float) -> dict[str, Any]:
    values = list(rows)
    return {
        **threshold_metrics(values, threshold),
        "auroc": auroc(values),
        "auprc": average_precision(values),
        "rows": len(values),
    }


def grouped_fold_partitions(
    rows: Sequence[dict[str, Any]], *, group_field: str, folds: int, seed: int
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Return deterministic label-presence-stratified grouped folds.

    Every group is held out exactly once.  Groups are stratified by whether
    they contain negative labels, positive labels, or both, matching the
    threshold diagnostic's established split contract.
    """

    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = row.get(group_field)
        if not isinstance(group, str) or not group:
            raise ValueError(f"row {index} lacks nonempty cross-fit group {group_field}")
        grouped[group].append(row)
    if len(grouped) < folds:
        raise ValueError(f"cross-fit requires at least {folds} groups, found {len(grouped)}")

    strata: dict[tuple[bool, bool], list[str]] = defaultdict(list)
    for group, items in grouped.items():
        labels = {label for label, _ in _validated(items)}
        strata[(0 in labels, 1 in labels)].append(group)
    assignments: list[list[str]] = [[] for _ in range(folds)]
    for stratum, groups in sorted(strata.items(), key=str):
        ordered = sorted(groups)
        random.Random(f"{seed}:{stratum}").shuffle(ordered)
        for index, group in enumerate(ordered):
            assignments[index % folds].append(group)

    all_groups = set(grouped)
    return [
        (
            [
                row
                for group in sorted(all_groups - set(held_out))
                for row in grouped[group]
            ],
            [row for group in held_out for row in grouped[group]],
        )
        for held_out in assignments
    ]
