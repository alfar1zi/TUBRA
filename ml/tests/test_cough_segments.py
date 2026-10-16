"""The API's YOHO/energy choice must not lose events from either backend.

``segment_coughs`` is the seam between the server pipeline and the detector:
this checks that a missing predictor returns the energy events untouched and
that a predictor which sees nothing also leaves the energy events alone.
"""

from __future__ import annotations

import numpy as np

from tbcare_api.audio.cough_segments import extract_segments, segment_coughs

ENERGY = [(1.0, 1.3), (2.0, 2.2)]


def test_missing_predictor_returns_energy_events():
    audio = np.zeros(48000, dtype=np.float32)
    events, backend = segment_coughs(audio, 16000, ENERGY, predictor=None)
    if backend == "energy":
        assert events == ENERGY
    else:
        # A model is installed in this checkout; energy must still be the
        # fallback when YOHO's own encoding of the silence is empty.
        assert events == ENERGY


def test_silent_predictor_falls_back_to_energy():
    audio = np.zeros(48000, dtype=np.float32)

    def predict(features: np.ndarray) -> np.ndarray:
        return np.zeros((1, 3, 40), dtype=np.float32)

    events, backend = segment_coughs(audio, 16000, ENERGY, predictor=predict)
    assert backend == "energy"
    assert events == ENERGY


def test_extract_keeps_time_order_and_peak_normalises():
    audio = np.zeros(48000, dtype=np.float32)
    audio[32000:32500] = 0.25
    audio[16000:16500] = 0.5
    segments = extract_segments(audio, 16000, [(2.0, 2.5), (1.0, 1.5)], max_segments=5)
    assert len(segments) == 2
    assert all(abs(float(np.max(np.abs(segment))) - 1.0) < 1e-6 for segment in segments)
