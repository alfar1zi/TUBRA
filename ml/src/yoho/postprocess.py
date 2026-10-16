"""YOHO label encoding, decoding and smoothing.

YOHO (Venkatesh, Moffat & Miranda, 2022) turns sound event detection into a
regression problem. At each output step the network emits three numbers per
class: a presence score, and the start and end of the event *relative to the
step*, both in ``[0, 1]``.

Every step that sees part of the event predicts the same start and end for it.
Decoding therefore needs no non-maximum suppression: any step with presence
above the threshold votes for a ``(start, end)`` pair, and neighbouring votes
are merged. That is why the paper reports inference that is several times
faster than segmentation-by-classification, and why smoothing is only a
duration/merge rule rather than a median filter over a dense frame grid.

Time base
---------
All functions take times in seconds. Step boundaries are defined by
``n_steps`` and ``duration_s``, so a model trained on 8 s clips works on a
15 s take with no change beyond the number of steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COUGH = 0

# Defaults are deliberately looser than the paper's music/speech smoothing.
# A cough is 0.2-0.9 s, so a lower duration floor and a tighter merge gap keep
# two coughs 0.3 s apart as two events instead of fusing them into one.
@dataclass(frozen=True)
class DecodeConfig:
    presence_threshold: float = 0.5
    min_event_duration_s: float = 0.08
    max_event_duration_s: float = 1.5
    merge_gap_s: float = 0.15
    end_before_start_tolerance_s: float = 0.05


def step_bounds(n_steps: int, duration_s: float) -> np.ndarray:
    """``(n_steps + 1,)`` array of step boundaries in seconds."""
    if n_steps <= 0:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(0.0, float(duration_s), n_steps + 1, dtype=np.float64)


def encode_labels(
    events: list[tuple[float, float]],
    n_steps: int,
    duration_s: float,
    event_class: int = COUGH,
) -> np.ndarray:
    """Encode event spans as YOHO targets of shape ``(n_steps, 1 + 2 * n_classes)``.

    A step is positive when it overlaps an event at all. Positive steps carry
    the event's start and end expressed as a fraction of that step's own span,
    clipped to ``[0, 1]``. Negative steps carry zeros and their two regression
    targets are masked out by the loss, exactly as in equation (1) of the
    paper.
    """
    n_classes = max(event_class + 1, 1)
    targets = np.zeros((max(n_steps, 0), 1 + 2 * n_classes), dtype=np.float32)
    if n_steps <= 0:
        return targets

    bounds = step_bounds(n_steps, duration_s)
    for start_s, end_s in events:
        if end_s <= start_s:
            continue
        first = int(np.searchsorted(bounds, start_s, side="right") - 1)
        last = int(np.searchsorted(bounds, end_s, side="left") - 1)
        for step in range(max(first, 0), min(last, n_steps - 1) + 1):
            lo, hi = bounds[step], bounds[step + 1]
            span = hi - lo
            if span <= 0:
                continue
            targets[step, 1 + 2 * event_class] = np.clip((start_s - lo) / span, 0.0, 1.0)
            targets[step, 2 + 2 * event_class] = np.clip((end_s - lo) / span, 0.0, 1.0)
        targets[max(first, 0) : min(last, n_steps - 1) + 1, event_class] = 1.0
    return targets


def _merge_votes(
    votes: list[tuple[float, float, float]], merge_gap_s: float
) -> list[tuple[float, float, float]]:
    """Merge votes that agree on the same event.

    A vote is ``(start, end, score)``. Two votes belong to the same event when
    their spans overlap or sit within ``merge_gap_s`` of each other. Overlap
    has to be measured between spans, not between start points: an event 0.4 s
    long covers two whole steps, and those two steps vote for the same start
    and end, so comparing only starts merged unrelated coughs while a short
    event seen from a single step stayed two separate votes.
    """
    if not votes:
        return []
    votes.sort(key=lambda v: (v[0], v[1]))
    merged = [votes[0]]
    for start_s, end_s, score in votes[1:]:
        best_start, best_end, best_score = merged[-1]
        overlaps = start_s <= best_end + merge_gap_s
        if overlaps:
            merged[-1] = (
                min(best_start, start_s),
                max(best_end, end_s),
                max(best_score, score),
            )
        else:
            merged.append((start_s, end_s, score))
    return merged


def decode(
    predictions: np.ndarray,
    duration_s: float,
    config: DecodeConfig = DecodeConfig(),
    event_class: int = COUGH,
) -> list[tuple[float, float]]:
    """Turn ``(n_steps, 3)` predictions into ``[(start_s, end_s), ...]``.

    ``predictions[:, 0]`` is the presence probability and ``predictions[:, 1:3]``
    are start/end as a fraction of the step. Confidence is the presence score
    multiplied by how well the vote's span overlaps the step, so a step that
    only clips the edge of an event cannot outvote the steps at its centre.

    Every positive step votes for the full event, so the merged votes that
    decode into events are already the event spans. They are not widened to
    step boundaries afterwards: a 0.2 s cough inside one 0.2 s step would
    otherwise gain the width of the step it happened to fall in.
    """
    if predictions.size == 0:
        return []
    pred = np.asarray(predictions, dtype=np.float64)
    n_steps = pred.shape[0]
    bounds = step_bounds(n_steps, duration_s)

    votes: list[tuple[float, float, float]] = []
    for step in range(n_steps):
        score = float(pred[step, 0])
        if score < config.presence_threshold:
            continue
        lo, hi = bounds[step], bounds[step + 1]
        span = hi - lo
        if span <= 0:
            continue
        start_s = lo + float(pred[step, 1]) * span
        end_s = lo + float(pred[step, 2]) * span
        if end_s < start_s:
            if start_s - end_s <= config.end_before_start_tolerance_s:
                start_s, end_s = end_s, start_s
            else:
                continue
        start_s = max(start_s, 0.0)
        end_s = min(end_s, duration_s)
        overlap = max(0.0, min(end_s, hi) - max(start_s, lo))
        votes.append((start_s, end_s, score * (overlap / span)))

    events = []
    for start_s, end_s, _score in _merge_votes(votes, config.merge_gap_s):
        length = end_s - start_s
        if config.min_event_duration_s <= length <= config.max_event_duration_s:
            events.append((round(float(start_s), 4), round(float(end_s), 4)))
    return events


def decode_output(
    output: np.ndarray,
    duration_s: float,
    config: DecodeConfig = DecodeConfig(),
    event_class: int = COUGH,
) -> list[tuple[float, float]]:
    """Decode a full network output of shape ``(n_steps, 1 + 2 * n_classes)``."""
    out = np.asarray(output)
    if out.ndim != 2:
        raise ValueError(f"expected a 2-D output, got shape {out.shape}")
    columns = [event_class, 1 + 2 * event_class, 2 + 2 * event_class]
    if max(columns) >= out.shape[1]:
        raise ValueError(
            f"output has {out.shape[1]} columns, not enough for class {event_class}"
        )
    return decode(out[:, columns], duration_s, config, event_class)


def presence_change_points(predictions: np.ndarray) -> list[int]:
    """Steps where the presence classifier flips, for debugging and metrics."""
    presence = np.asarray(predictions)[:, 0] >= DecodeConfig().presence_threshold
    if presence.size == 0:
        return []
    flips = np.nonzero(presence[1:] != presence[:-1])[0] + 1
    return [int(index) for index in flips]
