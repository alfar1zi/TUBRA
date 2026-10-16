"""Cough segmentation for the API: YOHO first, the energy gate as fallback.

The energy detector in :mod:`tbcare_api.audio.events` is kept as the fallback.
YOHO only replaces the boundary decision, not the quality checks: the counts
and segments it returns flow through the same pipeline the server already used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tbcare_api.config import config as app_config

from ml.src.yoho.detector import NucleusConfig, detect_cough_segments
from ml.src.yoho.runtime import Predictor, load_predictor

_PREDICTOR: Predictor | None = None
_LOADED = False


def yoho_predictor() -> Predictor | None:
    """The ONNX predictor shipped under ``TBCARE_ARTIFACTS/yoho``, or ``None``.

    Loaded once per process. ``None`` is the normal case on a checkout without
    the export, and it routes every call to the energy gate.
    """
    global _PREDICTOR, _LOADED
    if not _LOADED:
        model_dir = Path(app_config.TBCARE_ARTIFACTS) / "yoho"
        _PREDICTOR = load_predictor(model_dir)
        _LOADED = True
    return _PREDICTOR


def segment_coughs(
    audio: np.ndarray,
    sample_rate: int,
    energy_events: list[tuple[float, float]],
    predictor: Predictor | None = None,
) -> tuple[list[tuple[float, float]], str]:
    """Return ``(events, backend)`` where backend is ``yoho`` or ``energy``.

    ``energy_events`` are the events the energy gate already found. They are
    used verbatim when YOHO is unavailable or, on this take, finds nothing
    inside a region the energy gate did flag: an empty YOHO result next to a
    clean energy consensus is more likely a network miss on a take the model
    was not trained on (16 kHz browser audio is not CODA-TB audio) than a real
    absence of coughs, and the take should not be rejected for it.
    """
    from .resample import ensure_sample_rate

    if predictor is None:
        predictor = yoho_predictor()
    if predictor is None:
        return list(energy_events), "energy"

    target = ensure_sample_rate(audio, sample_rate, 16000)[0].astype(np.float32)
    events = detect_cough_segments(target, NucleusConfig(), predictor)
    if not events:
        return list(energy_events), "energy"
    return events, "yoho"


def extract_segments(
    audio: np.ndarray,
    sample_rate: int,
    events: list[tuple[float, float]],
    max_segments: int = 5,
) -> list[np.ndarray]:
    """Cut and peak-normalise up to ``max_segments`` events in time order."""
    segments = []
    for start_s, end_s in sorted(events)[:max_segments]:
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        if end_idx > start_idx and end_idx <= len(audio):
            segment = audio[start_idx:end_idx]
            peak = np.max(np.abs(segment)) + 1e-12
            segments.append(segment / peak)
    return segments
