"""Hand-rolled log-mel front end for YOHO.

The serving API cannot import torch, so this module is written in numpy plus
scipy and is shared by training and inference. It follows the same recipe as
``apps/api/tbcare_api/audio/logmel.py`` (16 kHz, 64 mel bins, n_fft 1024, hop
400) and is checked against it by ``ml/tests/test_yoho_frontend.py``.

Two normalisation choices matter for live audio:

* the floor is absolute (dB re 1.0), not per-utterance, so a quiet take does
  not get amplified to the same loudness as a cough;
* no per-utterance mean subtraction, because that would erase the level cue
  that separates "speech at 60 dB" from "cough at 70 dB" in a stream.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dct
from scipy.signal import get_window

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 200
N_MELS = 64
F_MIN = 60.0
F_MAX = 7600.0
TOP_DB = 80.0
LOG_EPS = 1e-6


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    """HTK mel filterbank, ``(N_MELS, N_FFT // 2 + 1)``, area-normalised."""
    mel_points = np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2)
    bins = np.floor((N_FFT + 1) * _mel_to_hz(mel_points) / SAMPLE_RATE).astype(int)
    bank = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float64)
    for m in range(1, N_MELS + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        if centre > left:
            bank[m - 1, left:centre] = (np.arange(left, centre) - left) / (centre - left)
        if right > centre:
            bank[m - 1, centre:right] = (right - np.arange(centre, right)) / (right - centre)
    return bank.astype(np.float32)


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    return get_window("hann", N_FFT, fftbins=True).astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """``(n_samples,)`` float32 audio -> ``(n_frames, N_MELS)`` float32 dB.

    Centred frames like librosa, so frame ``t`` sits at ``t * hop / sr`` seconds.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size < N_FFT:
        samples = np.pad(samples, (0, N_FFT - samples.size))
    padded = np.pad(samples, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (padded.shape[0] - N_FFT) // HOP_LENGTH
    index = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    frames = padded[index] * _window()
    power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
    mel = power @ mel_filterbank().T
    db = 10.0 * np.log10(np.maximum(mel, LOG_EPS))
    peak = float(db.max()) if db.size else -TOP_DB
    return np.maximum(db, peak - TOP_DB).astype(np.float32)


def mfcc(logmel: np.ndarray, n_coeffs: int = 20) -> np.ndarray:
    """Compact frame descriptor used to find cough nuclei in long recordings."""
    coeffs = dct(logmel, type=2, axis=1, norm="ortho")[:, 1 : n_coeffs + 1]
    return coeffs.astype(np.float32)


def pad_to_window(features: np.ndarray, n_frames: int) -> np.ndarray:
    """Edge-pad or crop ``(T, F)`` to exactly ``n_frames``."""
    if features.shape[0] >= n_frames:
        return features[:n_frames]
    missing = n_frames - features.shape[0]
    return np.pad(features, ((0, missing), (0, 0)), mode="edge")
