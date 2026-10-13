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
    seed: int = 0,
) -> np.ndarray:
    """Generate synthetic audio with cough-like events.

    Coughs are modelled as band-limited (200-4000 Hz) noise bursts so their
    spectral centroid and zero-crossing rate resemble real recordings. A
    plain white-noise burst has ZCR ~0.5 and is (correctly) rejected by the
    spectral gate in ``detect_cough_events``.
    """
    from scipy import signal as _signal

    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    signal = rng.normal(0, noise_level, n).astype(np.float32)

    if cough_times is None:
        cough_times = [(0.5, 0.8), (2.0, 2.3), (4.0, 4.3)]

    band = _signal.butter(
        4,
        [200 / (sample_rate / 2), 4000 / (sample_rate / 2)],
        btype="band",
        output="sos",
    )
    for start, end in cough_times:
        s = int(start * sample_rate)
        e = int(end * sample_rate)
        if e > n:
            e = n
        if s >= e:
            continue
        length = e - s
        burst = rng.normal(0, 0.5, length).astype(np.float32)
        burst = _signal.sosfilt(band, burst).astype(np.float32)
        peak = np.max(np.abs(burst)) + 1e-12
        burst = burst / peak * np.hanning(length)
        signal[s:e] += (burst * 0.8).astype(np.float32)

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

def _noisy(signal: np.ndarray, level: float = 0.01, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (signal + rng.normal(0, level, len(signal))).astype(np.float32)


def test_spectral_gate_rejects_pure_tone():
    """A sustained high tone is loud but is not a cough."""
    config = AudioConfig(sample_rate=16000)
    sr = 16000
    t = np.arange(sr * 4) / sr
    tone = (0.4 * np.sin(2 * np.pi * 2500 * t)).astype(np.float32)
    events, _, _ = detect_cough_events(_noisy(tone), sr, config)
    assert events == []


def test_spectral_gate_rejects_music_and_speech():
    """Harmonic music/speech is low-ZCR-ish and must not count as coughs."""
    config = AudioConfig(sample_rate=16000)
    sr = 16000
    t = np.arange(sr * 4) / sr
    music = sum(
        (0.2 * np.sin(2 * np.pi * f * t) for f in (110, 220, 440, 880, 1760, 3520))
    ).astype(np.float32)
    speech = sum(
        (0.2 * np.sin(2 * np.pi * f * t) for f in (300, 900, 1800, 2600))
    ).astype(np.float32)
    assert detect_cough_events(_noisy(music), sr, config)[0] == []
    assert detect_cough_events(_noisy(speech), sr, config)[0] == []


def test_spectral_gate_rejects_low_frequency_thud():
    """A muffled door-slam (heavily low-passed) is not a cough."""
    from scipy import signal as _signal

    config = AudioConfig(sample_rate=16000)
    sr = 16000
    rng = np.random.default_rng(2)
    audio = rng.normal(0, 0.01, sr * 4).astype(np.float32)
    length = int(0.4 * sr)
    thud = rng.normal(0, 0.9, length)
    thud = _signal.sosfilt(
        _signal.butter(4, 120 / (sr / 2), btype="low", output="sos"), thud
    )
    # Smooth envelope (no click edge) so the thud is purely low-frequency.
    env = np.hanning(length) ** 2
    thud = thud / (np.max(np.abs(thud)) + 1e-12) * env
    audio[sr: sr + length] += thud.astype(np.float32)
    assert detect_cough_events(audio, sr, config)[0] == []


def test_spectral_gate_keeps_realistic_coughs():
    """Band-limited cough bursts survive the gate, even with ambient noise."""
    from scipy import signal as _signal

    config = AudioConfig(sample_rate=16000)
    sr = 16000
    rng = np.random.default_rng(3)
    audio = rng.normal(0, 0.05, sr * 6).astype(np.float32)
    band = _signal.butter(
        4, [200 / (sr / 2), 4000 / (sr / 2)], btype="band", output="sos"
    )
    length = int(0.3 * sr)
    for start_s in (1, 2, 3, 4):
        burst = rng.normal(0, 0.5, length)
        burst = _signal.sosfilt(band, burst)
        burst = burst / (np.max(np.abs(burst)) + 1e-12) * np.hanning(length)
        s = int(start_s * sr)
        audio[s:s + length] += (burst * 0.8).astype(np.float32)
    events, _, _ = detect_cough_events(audio, sr, config)
    assert len(events) >= 3


def test_spectral_gate_can_be_disabled():
    """With the gate off, energy-only detection accepts a loud mid tone."""
    config = AudioConfig(sample_rate=16000, use_spectral_gate=False)
    sr = 16000
    rng = np.random.default_rng(4)
    audio = rng.normal(0, 0.005, sr * 4).astype(np.float32)
    buf = np.arange(int(0.3 * sr)) / sr
    burst = (0.5 * np.sin(2 * np.pi * 1500 * buf) * np.hanning(len(buf))).astype(
        np.float32
    )
    for start_s in (1.0, 2.0):
        s = int(start_s * sr)
        audio[s: s + len(burst)] += burst
    assert len(detect_cough_events(audio, sr, config)[0]) >= 1


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
        # Disable the single-clear-burst escape hatch so this test exercises
        # the count check itself, not the fallback.
        strong_burst_margin_db=200.0,
    )
    audio = make_cough_signal(duration_s=3.0, sample_rate=16000, cough_times=[(0.5, 0.8), (1.0, 1.3)])
    passed, details = quality_gate(audio, 16000, config)

    assert passed is False
    assert details["num_coughs"] < 5
    assert "too_few_coughs" in " ".join(details["reasons"])


def test_quality_gate_single_burst_clears_a_high_count_requirement():
    config = AudioConfig(
        sample_rate=16000,
        min_coughs_required=5,
        snr_threshold_db=5.0,
        clipping_threshold_ratio=0.01,
        strong_burst_margin_db=15.0,
    )
    audio = make_cough_signal(duration_s=3.0, sample_rate=16000, cough_times=[(1.0, 1.3)])
    passed, details = quality_gate(audio, 16000, config)

    assert details["num_coughs"] >= 1
    assert details["peak_margin_db"] >= 15.0
    assert details["passed"] is True


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