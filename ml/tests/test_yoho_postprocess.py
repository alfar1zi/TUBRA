"""Tests for YOHO label encoding and decoding.

The cases below are the ones that broke a frame-classification counter before,
so they are kept as regressions: a cough at the start of a take, two coughs
close together, a held sound longer than a cough, and a perfect prediction that
must round-trip within one hop.
"""

from __future__ import annotations

import numpy as np

from ml.src.yoho.postprocess import (
    DecodeConfig,
    decode,
    decode_output,
    encode_labels,
    presence_change_points,
    step_bounds,
)


def test_step_bounds_cover_the_whole_clip():
    bounds = step_bounds(26, 8.0)
    assert bounds.shape == (27,)
    assert bounds[0] == 0.0
    assert np.isclose(bounds[-1], 8.0)
    assert np.isclose(np.diff(bounds).max(), 8.0 / 26)


def test_positive_steps_carry_relative_offsets():
    targets = encode_labels([(0.2, 4.3)], n_steps=26, duration_s=8.0)
    # 0.2 s falls inside the first 0.307 s step, so start is 0.2 / 0.307.
    assert targets[0, 0] == 1.0
    assert np.isclose(targets[0, 1], 0.2 / (8.0 / 26), atol=1e-3)
    assert targets[-1, 0] == 0.0


def test_negative_steps_leave_regression_targets_at_zero():
    targets = encode_labels([(1.0, 1.2)], n_steps=26, duration_s=8.0)
    negatives = targets[:, 0] == 0.0
    assert negatives.any()
    assert np.all(targets[negatives, 1:] == 0.0)


def test_two_classes_use_separate_neuron_groups():
    targets = encode_labels([(1.0, 1.5)], n_steps=10, duration_s=4.0, event_class=1)
    assert targets.shape[1] == 5
    assert targets[:, 0].sum() == 0.0
    assert targets[:, 1].sum() > 0.0
    assert targets[:, 3].sum() > 0.0


def test_a_perfect_prediction_round_trips_within_one_step():
    events = [(1.0, 1.5), (3.2, 3.9)]
    targets = encode_labels(events, n_steps=26, duration_s=8.0)
    decoded = decode(targets[:, :3], duration_s=8.0)
    step = 8.0 / 26
    assert len(decoded) == len(events)
    for (want_start, want_end), (got_start, got_end) in zip(events, decoded):
        assert abs(want_start - got_start) <= step
        assert abs(want_end - got_end) <= step


def test_decoded_event_is_not_widened_to_the_step_grid():
    # Votes carry the event span itself. Turning them back into step bounds
    # made every cough at least one step (0.2 s at 40 steps over 8 s) long,
    # which buried a 0.2 s cough in a 0.4 s prediction.
    targets = encode_labels([(1.0, 1.2)], n_steps=40, duration_s=8.0)
    (start_s, end_s), = decode(targets[:, :3], duration_s=8.0)
    assert abs(start_s - 1.0) < 0.01
    assert abs(end_s - 1.2) < 0.01

def test_votes_merge_on_span_overlap_not_on_start_distance():
    # One vote at 1.04-1.8 and another at 1.8-2.8: their starts are 0.8 s
    # apart, but their spans touch, so they must stay two events.
    predictions = np.zeros((40, 3))
    predictions[5, 0] = 0.9
    predictions[5, 1:3] = (0.2, 0.9)
    predictions[9, 0] = 0.9
    predictions[9, 1:3] = (0.0, 0.5)
    decoded = decode(predictions, duration_s=8.0)
    assert len(decoded) == 2
    assert abs(decoded[0][0] - 1.04) < 0.01
    assert abs(decoded[1][0] - 1.8) < 0.01

def test_a_cough_at_the_very_start_is_kept():
    # The live counter used to lose a cough that began in the first 10 ms.
    events = [(0.0, 0.4)]
    targets = encode_labels(events, n_steps=26, duration_s=8.0)
    assert decode(targets[:, :3], duration_s=8.0) == [(0.0, 0.4)]


def test_two_coughs_close_together_stay_separate():
    events = [(1.0, 1.35), (1.65, 2.0)]
    targets = encode_labels(events, n_steps=39, duration_s=12.0)
    decoded = decode(targets[:, :3], duration_s=12.0)
    assert len(decoded) == 2


def test_a_held_sound_longer_than_a_cough_is_dropped():
    targets = encode_labels([(1.0, 5.0)], n_steps=26, duration_s=8.0)
    assert decode(targets[:, :3], duration_s=8.0) == []


def test_a_blip_shorter_than_the_floor_is_dropped():
    config = DecodeConfig(min_event_duration_s=0.1)
    predictions = np.zeros((26, 3), dtype=np.float32)
    predictions[5] = [0.9, 0.0, 0.2]  # 61 ms inside a 307 ms step
    assert decode(predictions, duration_s=8.0, config=config) == []


def test_low_presence_never_produces_an_event():
    predictions = np.full((26, 3), 0.49, dtype=np.float32)
    predictions[:, 1] = 0.1
    predictions[:, 2] = 0.0
    with_config = DecodeConfig(end_before_start_tolerance_s=0.0, presence_threshold=0.5)
    assert decode(predictions, duration_s=8.0, config=with_config) == []


def test_votes_from_neighbouring_steps_merge_into_one_event():
    predictions = np.zeros((26, 3), dtype=np.float32)
    for step in range(20, 24):
        predictions[step] = [0.9, 0.2, 1.0]
    # Every vote is anchored on its own step, so their spans chain together.
    decoded = decode(predictions, duration_s=8.0)
    assert len(decoded) == 1


def test_decode_output_selects_the_requested_class():
    output = np.zeros((10, 5), dtype=np.float32)
    output[:, [1, 3, 4]] = [0.9, 0.1, 0.9]


def test_decode_output_rejects_a_short_output():
    try:
        decode_output(np.zeros((10, 2)), duration_s=4.0)
    except ValueError as exc:
        assert "not enough" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_decode_rejects_a_negative_duration_event():
    predictions = np.full((26, 3), 0.9, dtype=np.float32)
    predictions[:, 1] = 0.9
    predictions[:, 2] = 0.1
    assert decode(predictions, duration_s=8.0) == []


def test_presence_change_points_track_the_classifier():
    predictions = np.zeros((6, 3), dtype=np.float32)
    predictions[1:4, 0] = 1.0
    assert presence_change_points(predictions) == [1, 4]


def test_empty_input_decodes_to_nothing():
    assert decode(np.zeros((0, 3)), duration_s=0.0) == []
    assert encode_labels([], n_steps=0, duration_s=0.0).shape == (0, 3)
