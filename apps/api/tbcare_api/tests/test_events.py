import numpy as np
import pytest
from tbcare_api.audio.events import (
    rms_energy_frames,
    detect_cough_events,
    estimate_snr,
    estimate_clipping_ratio,
    quality_gate,
)
from tbcare_api.audio.config import AudioConfig


def make_cough_signal(
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    cough_times: list[tuple[float, float]] | None = None,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Generate synthetic audio with cough-like events."""
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    signal = np.random.normal(0, noise_level, n).astype(np.float32)

    if cough_times is None:
        cough_times = [(0.5, 0.8), (2.0, 2.3), (4.0, 4.3)]

    for start, end in cough_times:
        s = int(start * sample_rate)
        e = int(end * sample_rate)
        if e > n:
            e = n
        if s >= e:
            continue
        cough_t = t[s:e]
        # Cough-like: burst of noise with envelope
        envelope = np.exp(-5 * (cough_t - cough_t[0]))
        burst = np.random.normal(0, 0.5, len(cough_t)) * envelope
        signal[s:e] += burst.astype(np.float32)

    # Normalize to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.8

    return signal


def test_rms_energy_frames():
    audio = np.ones(16000, dtype=np.float32) * 0.5
    energies = rms_energy_frames(audio, 1024, 512)
    assert len(energies) > 0
    assert np.allclose(energies, 0.5, rtol=0.01)


def test_detect_cough_events_basic():
    config = AudioConfig(sample_rate=16000, frame_length=1024, hop_length=512)
    audio = make_cough_signal(duration_s=5.0, sample_rate=16000)
    events, energy_db, noise_floor = detect_cough_events(audio, 16000, config)

    assert len(events) >= 3
    for start, end in events:
        assert 0 <= start < end <= 5.0
        assert end - start >= config.min_cough_duration_s
        assert end - start <= config.max_cough_duration_s


def test_detect_cough_events_pure_noise():
    config = AudioConfig(sample_rate=16000, frame_length=1024, hop_length=512)
    # Pure low-level noise, no coughs
    audio = np.random.normal(0, 0.001, 16000 * 3).astype(np.float32)
    events, energy_db, noise_floor = detect_cough_events(audio, 16000, config)

    assert len(events) == 0


def test_detect_cough_events_clipped_signal():
    config = AudioConfig(sample_rate=16000, frame_length=1024, hop_length=512)
    # Signal at clipping level
    audio = np.ones(16000 * 2, dtype=np.float32) * 0.99
    events, energy_db, noise_floor = detect_cough_events(audio, 16000, config)

    # Should detect as continuous event (or none due to no energy variation)
    # Just verify it doesn't crash
    assert isinstance(events, list)


def test_estimate_snr():
    config = AudioConfig(sample_rate=16000)
    audio = make_cough_signal(duration_s=3.0, sample_rate=16000)
    events, _, noise_floor = detect_cough_events(audio, 16000, config)

    snr = estimate_snr(audio, events, 16000, noise_floor)
    assert snr > 10.0  # Should be well above noise


def test_estimate_snr_no_events():
    snr = estimate_snr(np.zeros(16000), [], 16000, -60.0)
    assert snr <= -100.0  # Very low


def test_estimate_clipping_ratio():
    # No clipping
    audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
    ratio = estimate_clipping_ratio(audio, 0.99)
    assert ratio == 0.0

    # Some clipping
    audio = np.concatenate([
        np.random.normal(0, 0.1, 9000).astype(np.float32),
        np.ones(1000, dtype=np.float32),  # Clipped
    ])
    ratio = estimate_clipping_ratio(audio, 0.99)
    assert 0.09 < ratio < 0.11


def test_quality_gate_pass():
    config = AudioConfig(
        sample_rate=16000,
        min_coughs_required=3,
        snr_threshold_db=5.0,
        clipping_threshold_ratio=0.01,
    )
    audio = make_cough_signal(duration_s=5.0, sample_rate=16000)
    passed, details = quality_gate(audio, 16000, config)

    assert passed is True
    assert details["num_coughs"] >= 3
    assert details["snr_db"] > 5.0
    assert details["clipping_ratio"] < 0.01


def test_quality_gate_too_few_coughs():
    config = AudioConfig(
        sample_rate=16000,
        min_coughs_required=5,
        snr_threshold_db=5.0,
        clipping_threshold_ratio=0.01,
    )
    audio = make_cough_signal(duration_s=3.0, sample_rate=16000, cough_times=[(0.5, 0.8), (1.0, 1.3)])
    passed, details = quality_gate(audio, 16000, config)

    assert passed is False
    assert details["num_coughs"] < 5
    assert "too_few_coughs" in " ".join(details["reasons"])


def test_quality_gate_too_noisy():
    config = AudioConfig(
        sample_rate=16000,
        min_coughs_required=3,
        snr_threshold_db=20.0,  # Very high threshold
        clipping_threshold_ratio=0.01,
    )
    audio = make_cough_signal(duration_s=5.0, sample_rate=16000, noise_level=0.1)
    passed, details = quality_gate(audio, 16000, config)

    assert passed is False
    assert "too_noisy" in " ".join(details["reasons"])


def test_quality_gate_clipped():
    config = AudioConfig(
        sample_rate=16000,
        min_coughs_required=1,
        snr_threshold_db=-50.0,
        clipping_threshold_ratio=0.001,
    )
    # Use 1.0 which is > 0.99 threshold
    audio = np.ones(16000 * 2, dtype=np.float32) * 1.0
    passed, details = quality_gate(audio, 16000, config)

    assert passed is False
    assert "clipped" in " ".join(details["reasons"])


def test_detect_cough_events_short_cough_rejected():
    config = AudioConfig(
        sample_rate=16000,
        frame_length=1024,
        hop_length=512,
        min_cough_duration_s=0.2,
    )
    # Very short cough (50ms)
    audio = make_cough_signal(duration_s=2.0, sample_rate=16000, cough_times=[(0.5, 0.55)])
    events, _, _ = detect_cough_events(audio, 16000, config)

    assert len(events) == 0


def test_detect_cough_events_long_cough_rejected():
    config = AudioConfig(
        sample_rate=16000,
        frame_length=1024,
        hop_length=512,
        max_cough_duration_s=1.0,
        noise_percentile=10.0,
        noise_margin_db=6.0,
    )
    # Sustained high energy for 2 seconds (above threshold the whole time)
    n = 3 * 16000
    audio = np.random.normal(0, 0.01, n).astype(np.float32)
    # Add sustained cough from 0.5s to 2.5s (2 seconds)
    s, e = int(0.5 * 16000), int(2.5 * 16000)
    audio[s:e] += 0.5 * np.random.normal(0, 1, e - s).astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8

    events, _, _ = detect_cough_events(audio, 16000, config)

    # Should be rejected because duration > max_cough_duration_s
    assert len(events) == 0