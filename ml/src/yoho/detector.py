"""Streaming YOHO detector: the runtime that both the API and the browser use.

Reference recordings are short (0.5 s CODA crops, a few seconds of app takes),
so the detector never runs a long mel spectrogram. It works chunk by chunk:

1. find cough *nuclei* with cheap frame descriptors (RMS above an adaptive
   floor, plus spectral flatness for the broadband burst a cough starts with);
2. group nuclei that are closer than ``min_event_duration_s``;
3. cut a window around each group, run YOHO on it, and keep the highest
   presence step whose predicted boundary is close to the nucleus;
4. fall back to the energy detector when the YOHO output is not supplied, so
   the API keeps working if the model file is missing.

Only numpy is imported here on purpose: the serving image does not install
torch, and the browser port mirrors this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frontend import HOP_LENGTH, SAMPLE_RATE, log_mel, mfcc, pad_to_window
from .postprocess import _merge_votes

# 8 s context at a 12.5 ms hop. The network downsamples time 16x, so one
# output step is 0.2 s and a window has 40 steps: fine enough to place the
# start of a 0.2 s cough without running a dense frame classifier.
CONTEXT_SAMPLES = 8 * SAMPLE_RATE
CONTEXT_STEPS = 40
# Votes from overlapping windows that agree on the same cough are merged when
# their spans sit this close, the gap the segment decode also uses.
MERGE_GAP_S = 0.2
# No single cough in the training data is longer than this. A longer event is
# several coughs fused by the vote and is not counted.
MAX_EVENT_DURATION_S = 2.0
# A nucleus this close to an event is that event; a nucleus left over becomes
# the segment itself, a cough the network missed.
NUCLEUS_TOLERANCE_S = 0.3
FALLBACK_LEAD_S = 0.05
FALLBACK_TAIL_S = 0.25
# Exactly 16 mel frames per step, so the network returns CONTEXT_STEPS steps
# instead of one more from rounding the last partial window.
WINDOW_MEL_FRAMES = CONTEXT_STEPS * 16


@dataclass(frozen=True)
class NucleusConfig:
    frame_len: int = 512
    hop: int = 256
    floor_percentile: float = 20.0
    onset_db: float = 9.0
    flatness_min: float = 0.05
    merge_gap_s: float = 0.2
    max_nuclei: int = 6


@dataclass(frozen=True)
class Nucleus:
    centre_s: float
    rms_db: float
    flatness: float


def frame_features(audio: np.ndarray, config: NucleusConfig = NucleusConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame RMS (dB) and spectral flatness for ``audio``."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size < config.frame_len:
        samples = np.pad(samples, (0, config.frame_len - samples.size))
    n_frames = 1 + (samples.size - config.frame_len) // config.hop
    index = np.arange(config.frame_len)[None, :] + config.hop * np.arange(n_frames)[:, None]
    frames = samples[index]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(config.frame_len), axis=1)) + 1e-10
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)
    return 20.0 * np.log10(rms), flatness.astype(np.float32)


def find_nuclei(
    audio: np.ndarray, config: NucleusConfig = NucleusConfig()
) -> list[Nucleus]:
    """Loud broadband frames, merged into candidate cough centres."""
    rms_db, flatness = frame_features(audio, config)
    floor = np.percentile(rms_db, config.floor_percentile)
    active = (rms_db >= floor + config.onset_db) & (flatness >= config.flatness_min)
    if not active.any():
        return []

    frame_time = np.arange(active.size) * config.hop / SAMPLE_RATE
    # Take local maxima of loudness inside each active run so that a burst
    # followed by a decay (the "hiss" tail of a cough) counts as one nucleus.
    groups: list[list[int]] = []
    for index in np.flatnonzero(active):
        if groups and frame_time[index] - frame_time[groups[-1][-1]] <= config.merge_gap_s:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])

    nuclei = []
    for group in groups:
        loudest = group[int(np.argmax(rms_db[group]))]
        nuclei.append(
            Nucleus(
                centre_s=float(frame_time[loudest]),
                rms_db=float(rms_db[loudest]),
                flatness=float(flatness[loudest]),
            )
        )
    nuclei.sort(key=lambda n: n.rms_db, reverse=True)
    return sorted(nuclei[: config.max_nuclei], key=lambda n: n.centre_s)


def context_window(audio: np.ndarray, centre_s: float) -> tuple[np.ndarray, float]:
    """Cut an 8 s window centred just after ``centre_s``.

    Coughs stay 0.05-0.25 s after their attack, so the window starts 0.4 s
    before the nucleus to leave room for a first, shorter cough.
    """
    start = int(round(centre_s * SAMPLE_RATE)) - int(0.4 * SAMPLE_RATE)
    start = max(start, 0)
    end = start + CONTEXT_SAMPLES
    if end > audio.shape[0]:
        end = audio.shape[0]
        start = max(end - CONTEXT_SAMPLES, 0)
    return np.asarray(audio[start:end], dtype=np.float32), start / SAMPLE_RATE


def window_features(audio: np.ndarray, centre_s: float) -> np.ndarray:
    """``(1, 1, N_MELS, WINDOW_MEL_FRAMES)`` input for the YOHO network."""
    window, _ = context_window(audio, centre_s)
    features = pad_to_window(log_mel(window), WINDOW_MEL_FRAMES)
    return features.T[None, None, :, :].astype(np.float32)


def steps_by_planes(head: np.ndarray) -> np.ndarray:
    """Turn a ``(3, 40)`` class-major head into ``(40, 3)`` steps x planes.

    Plane 0 holds 40 presence scores, plane 1 holds the start targets of the
    same 40 steps and plane 2 the end targets, so the planes are stacked, not
    interleaved: reading the first 120 values three at a time would mix a
    presence score with a start target. A head that already has ``40`` rows,
    or any other shape, is passed through unchanged.
    """
    if head.ndim != 2:
        return head
    if head.shape[0] == CONTEXT_STEPS:
        return head
    if head.shape[1] == CONTEXT_STEPS:
        return np.ascontiguousarray(head.T)
    if head.shape[0] == 3:
        return np.ascontiguousarray(head.T)
    return head


def step_time_start(window_start_s: float, step: int) -> float:
    """Time where output step ``step`` begins, in the original recording."""
    step_s = CONTEXT_SAMPLES / SAMPLE_RATE / CONTEXT_STEPS
    return window_start_s + step * step_s


def nucleus_segments(
    audio: np.ndarray,
    predictions: list[np.ndarray],
    centres_s: list[float],
    window_starts_s: list[float],
    presence_threshold: float = 0.5,
) -> list[tuple[float, float]]:
    """Find each cough near a nucleus from the boundary votes of its windows.

    ``predictions[i]`` is the YOHO output ``(40, 3)`` for the window that
    starts at ``window_starts_s[i]`` and was cut for ``centres_s[i]``. The
    windows of the nuclei in a recording overlap, so the same cough is seen
    several times with slightly different votes. Votes are merged across the
    whole recording, and an event only counts when a nucleus is next to it
    that no stronger event has answered yet. A nucleus left over becomes a
    segment of its own, because an event the network missed is still a cough.
    This keeps the count tied to the nuclei instead of letting weak votes in
    the background add coughs on their own.
    """
    del audio
    step_s = CONTEXT_SAMPLES / SAMPLE_RATE / CONTEXT_STEPS
    votes: list[tuple[float, float, float]] = []
    for predictions_i, window_start in zip(predictions, window_starts_s):
        pred = np.asarray(predictions_i, dtype=np.float64)
        for step in range(pred.shape[0]):
            score = float(pred[step, 0])
            if score < presence_threshold:
                continue
            lo = window_start + step * step_s
            start_s = lo + float(pred[step, 1]) * step_s
            end_s = lo + float(pred[step, 2]) * step_s
            if end_s < start_s:
                if start_s - end_s > END_BEFORE_START_TOLERANCE_S:
                    continue
                start_s, end_s = end_s, start_s
            if start_s < window_start:
                continue
            overlap = max(0.0, min(end_s, lo + step_s) - max(start_s, lo))
            votes.append((start_s, end_s, score * (overlap / step_s)))

    merged = sorted(
        _merge_votes(votes, MERGE_GAP_S), key=lambda event: event[2], reverse=True
    )
    events: list[tuple[float, float, float]] = []
    unmatched = list(centres_s)
    for start_s, end_s, score in merged:
        if end_s - start_s > MAX_EVENT_DURATION_S:
            continue
        near = [
            centre_s
            for centre_s in unmatched
            if start_s - NUCLEUS_TOLERANCE_S <= centre_s <= end_s + NUCLEUS_TOLERANCE_S
        ]
        if near:
            events.append((start_s, end_s, score))
            for centre_s in near:
                unmatched.remove(centre_s)
    for centre_s in unmatched:
        events.append((centre_s - FALLBACK_LEAD_S, centre_s + FALLBACK_TAIL_S, 0.0))
    accepted = [
        (round(max(start_s, 0.0), 3), round(end_s, 3))
        for start_s, end_s, _score in sorted(events)
    ]
    return accepted


def energy_fallback(audio: np.ndarray, config: NucleusConfig = NucleusConfig()) -> list[tuple[float, float]]:
    """Boundary from the nucleus alone, used when no YOHO output exists."""
    events = []
    for nucleus in find_nuclei(audio, config):
        events.append((round(max(nucleus.centre_s - 0.05, 0.0), 3), round(nucleus.centre_s + 0.25, 3)))
    return events


def detect_cough_segments(
    audio: np.ndarray,
    config: NucleusConfig = NucleusConfig(),
    predict=None,
) -> list[tuple[float, float]]:
    """Full detector: nuclei -> YOHO windows -> segments.

    ``predict`` takes a ``(1, 1, 64, 640)`` array and returns ``(1, 3, 40)``;
    the API wires it to ONNX Runtime and the tests to a stub. Without it the
    energy fallback is used.
    """
    samples = np.asarray(audio, dtype=np.float32)
    nuclei = find_nuclei(samples, config)
    if not nuclei:
        return []
    if predict is None:
        return energy_fallback(samples, config)
    predictions = []
    starts = []
    for nucleus in nuclei:
        _, window_start = context_window(samples, nucleus.centre_s)
        output = np.asarray(predict(window_features(samples, nucleus.centre_s)))
        head = output[0] if output.shape[0] == 1 else output
        predictions.append(steps_by_planes(head))
        starts.append(window_start)
    return nucleus_segments(
        samples, predictions, [n.centre_s for n in nuclei], starts
    )
