"""Tests for the YOHO front end and streaming detector.

The detector is what the browser and the API actually run, so it is tested
without torch: a stub predictor stands in for the network. The contract is
that a stub which agrees with the labels reproduces the labels, and that a
missing predictor falls back to energy instead of crashing.
"""

from __future__ import annotations

import numpy as np

from ml.src.yoho.detector import (
    CONTEXT_SAMPLES,
    CONTEXT_STEPS,
    WINDOW_MEL_FRAMES,
    detect_cough_segments,
    energy_fallback,
    find_nuclei,
    frame_features,
    nucleus_segments,
    steps_by_planes,
    window_features,
)
from ml.src.yoho.frontend import HOP_LENGTH, N_MELS, SAMPLE_RATE, log_mel, mfcc, pad_to_window
from ml.src.yoho.metrics import iou, match_events, segment_f1


def burst(seconds: float = 0.35, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    n = int(seconds * SAMPLE_RATE)
    noise = rng.normal(0.0, 1.0, n)
    envelope = np.exp(-np.linspace(0.0, 6.0, n))
    return (0.6 * noise * envelope).astype(np.float32)


def room(seconds: float = 8.0, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(1)
    return (0.002 * rng.normal(0.0, 1.0, int(seconds * SAMPLE_RATE))).astype(np.float32)


def place(audio: np.ndarray, sound: np.ndarray, at_s: float) -> np.ndarray:
    out = audio.copy()
    start = int(at_s * SAMPLE_RATE)
    out[start : start + sound.size] += sound
    return out


def test_log_mel_shape_and_scale():
    features = log_mel(room(1.0))
    # Centre padding gives one extra frame at the end, like librosa.
    assert features.shape == (1 + SAMPLE_RATE // HOP_LENGTH, N_MELS)
    assert np.isfinite(features).all()
    assert features.max() <= 0.0


def test_log_mel_is_quiet_for_digital_silence():
    features = log_mel(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert features.min() >= -80.0 - 1e-3


def test_pad_to_window_edge_pads():
    features = np.ones((10, N_MELS), dtype=np.float32)
    padded = pad_to_window(features, 20)
    assert padded.shape == (20, N_MELS)
    assert (padded[10:] == 1.0).all()


def test_mfcc_is_compact():
    assert mfcc(log_mel(room(0.5))).shape[1] == 20


def test_frame_features_find_loud_frames():
    audio = place(room(2.0), burst(), 1.0)
    rms_db, flatness = frame_features(audio)
    assert rms_db.shape == flatness.shape
    assert rms_db.max() - np.median(rms_db) > 15.0


def test_nuclei_found_for_two_coughs():
    audio = place(place(room(8.0), burst(), 1.0), burst(), 4.0)
    nuclei = find_nuclei(audio)
    centres = [round(n.centre_s, 1) for n in nuclei]
    assert any(abs(c - 1.0) < 0.1 for c in centres)
    assert any(abs(c - 4.0) < 0.1 for c in centres)


def test_no_nuclei_in_room_noise():
    assert find_nuclei(room(8.0)) == []


def test_window_features_have_the_network_shape():
    audio = place(room(8.0), burst(), 3.0)
    features = window_features(audio, 3.1)
    assert features.shape == (1, 1, N_MELS, WINDOW_MEL_FRAMES)
    assert features.dtype == np.float32


def test_stub_predictor_reproduces_a_cough_at_the_start():
    audio = place(room(4.0), burst(), 0.05)
    centre = find_nuclei(audio)[0].centre_s
    audio = audio[:CONTEXT_SAMPLES]

    def predict(features: np.ndarray) -> np.ndarray:
        assert features.shape == (1, 1, N_MELS, WINDOW_MEL_FRAMES)
        output = np.zeros((1, CONTEXT_STEPS, 3), dtype=np.float32)
        step = int(centre // (8.0 / CONTEXT_STEPS))
        output[0, step] = [0.9, max(0.0, (centre - 0.05) / (8.0 / CONTEXT_STEPS) - step), 0.5]
        return output

    events = detect_cough_segments(audio, predict=predict)
    assert len(events) == 1
    start_s, end_s = events[0]
    assert start_s <= centre <= end_s


def test_missing_predictor_uses_energy_fallback():
    audio = place(room(4.0), burst(), 2.0)
    assert detect_cough_segments(audio) == energy_fallback(audio)


def test_empty_audio_has_no_events():
    assert detect_cough_segments(np.zeros(0, dtype=np.float32)) == []


def test_steps_by_planes_stacks_the_planes():
    planes = np.arange(120, dtype=np.float32).reshape(3, 40)
    steps = steps_by_planes(planes)
    assert steps.shape == (40, 3)
    np.testing.assert_array_equal(steps[:, 0], planes[0])
    np.testing.assert_array_equal(steps[:, 1], planes[1])
    np.testing.assert_array_equal(steps[:, 2], planes[2])
    assert steps_by_planes(np.zeros((1, 3, 40))).shape == (1, 3, 40)
    np.testing.assert_array_equal(steps_by_planes(steps), steps)

def test_nucleus_segments_keeps_each_cough_once():
    # Three coughs at 0.05, 0.65 and 1.25 s, each seen by every overlapping
    # nucleus window with a slightly longer end.
    planes = np.zeros((3, CONTEXT_STEPS), dtype=np.float64)
    step_s = 8.0 / CONTEXT_STEPS
    for start_s, end_s in ((0.05, 0.33), (0.65, 0.93), (1.25, 1.6)):
        for step in range(CONTEXT_STEPS):
            lo = step * step_s
            if lo < end_s and lo + step_s > start_s:
                planes[0, step] = 1.0
                planes[1, step] = (start_s - lo) / step_s
                planes[2, step] = min((end_s - lo) / step_s, 1.0)
    predictions = [steps_by_planes(planes) for _ in range(3)]
    events = nucleus_segments(np.zeros(CONTEXT_SAMPLES, dtype=np.float32), predictions, [0.1, 0.7, 1.3], [0.0, 0.0, 0.0])
    assert len(events) == 3
    for event, truth in zip(events, ((0.05, 0.33), (0.65, 0.93), (1.25, 1.6))):
        assert iou(event, truth) >= 0.5

def test_nucleus_segments_adds_a_cough_the_network_missed():
    planes = np.zeros((3, CONTEXT_STEPS), dtype=np.float64)
    planes[0, 1] = 1.0
    planes[1, 1] = 0.0
    planes[2, 1] = 1.0
    step_s = 8.0 / CONTEXT_STEPS
    events = nucleus_segments(
        np.zeros(CONTEXT_SAMPLES, dtype=np.float32), [steps_by_planes(planes)], [0.1, 5.0], [0.0, 4.6]
    )
    assert len(events) == 2
    assert events[0] == (0.2, 0.4)
    assert events[1] == (4.95, 5.25)

def test_nucleus_segments_ignores_votes_without_a_nucleus():
    planes = np.zeros((3, CONTEXT_STEPS), dtype=np.float64)
    planes[0, 20] = 0.99
    planes[1, 20] = 0.0
    planes[2, 20] = 1.0
    events = nucleus_segments(np.zeros(CONTEXT_SAMPLES, dtype=np.float32), [steps_by_planes(planes)], [2.0], [0.0])
    assert events == [(1.95, 2.25)]

def test_matching_one_to_one():
    truth = [(0.0, 0.5), (2.0, 2.5)]
    predicted = [(0.02, 0.49), (2.05, 2.6), (7.0, 7.3)]
    hits, false_alarms = match_events(truth, predicted)
    assert hits == 2
    assert false_alarms == 1
    assert abs(iou((0.0, 1.0), (0.5, 1.5)) - 1 / 3) < 1e-9


def test_segment_f1_penalises_splitting():
    truth = [[(1.0, 1.4)]]
    split = [[(1.0, 1.2), (1.2, 1.4)]]
    scores = segment_f1(truth, split)
    assert scores["recall"] == 1.0
    assert scores["precision"] == 0.5
    assert abs(scores["f1"] - 2 / 3) < 1e-9
