"""Segment-level scores for comparing cough detectors.

A detector is judged by events, not clips. Two coughs 0.3 s apart must not be
scored as one event, and a detector that splits one cough into two must not win
on recall. Matching is one-to-one greedy by IoU at 0.5, the community default.
"""

from __future__ import annotations


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_events(
    truth: list[tuple[float, float]],
    predicted: list[tuple[float, float]],
    threshold: float = 0.5,
) -> tuple[int, int]:
    """Return ``(true_positives, false_positives)`` under one-to-one IoU matching."""
    remaining = list(predicted)
    true_positives = 0
    for event in truth:
        best_index, best_iou = -1, threshold
        for index, candidate in enumerate(remaining):
            score = iou(event, candidate)
            if score >= best_iou:
                best_index, best_iou = index, score
        if best_index >= 0:
            remaining.pop(best_index)
            true_positives += 1
    return true_positives, len(remaining)


def segment_f1(
    truths: list[list[tuple[float, float]]],
    predictions: list[list[tuple[float, float]]],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Corpus-level precision, recall and F1 over many recordings."""
    tp = fp = fn = 0
    for truth, predicted in zip(truths, predictions):
        hits, false_alarms = match_events(truth, predicted, threshold)
        tp += hits
        fp += false_alarms
        fn += len(truth) - hits
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def false_alarms_per_hour(
    predictions: list[list[tuple[float, float]]], duration_hours: float
) -> float:
    count = sum(len(p) for p in predictions)
    return count / duration_hours if duration_hours > 0 else 0.0
