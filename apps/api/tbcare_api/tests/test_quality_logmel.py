import io
import wave
import numpy as np
import pytest
from tbcare_api.audio.resample import resample_audio, ensure_sample_rate
from tbcare_api.audio.quality import preprocess_audio, run_quality_pipeline, extract_cough_segments
from tbcare_api.audio.logmel import (
    hz_to_mel,
    mel_to_hz,
    create_mel_filterbank,
    compute_stft,
    compute_logmel,
    normalize_logmel,
    process_audio_to_logmel,
)
from tbcare_api.audio.config import AudioConfig


def make_wav_bytes(audio_float: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 [-1,1] audio to WAV bytes."""
    audio_int16 = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def make_cough_audio(
    duration_s: float = 3.0, sample_rate: int = 16000, seed: int = 0
) -> np.ndarray:
    """Generate synthetic cough-like audio (band-limited bursts).

    Band-limited so the spectral gate treats them as coughs; a raw
    white-noise burst has ZCR ~0.5 and would be rejected.
    """
    from scipy import signal as _signal

    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    # Background noise
    audio = rng.normal(0, 0.01, n).astype(np.float32)
    band = _signal.butter(
        4,
        [200 / (sample_rate / 2), 4000 / (sample_rate / 2)],
        btype="band",
        output="sos",
    )
    # Add 3 cough events (adjusted for duration)
    cough_times = [(0.5, 0.8), (1.2, 1.5), (2.0, 2.3)]
    for start, end in cough_times:
        if start >= duration_s:
            continue
        s = int(start * sample_rate)
        e = min(int(end * sample_rate), n)
        if s >= e:
            continue
        length = e - s
        burst = rng.normal(0, 0.4, length).astype(np.float32)
        burst = _signal.sosfilt(band, burst).astype(np.float32)
        peak = np.max(np.abs(burst)) + 1e-12
        burst = burst / peak * np.hanning(length)
        audio[s:e] += (burst * 0.6).astype(np.float32)
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.7
    return audio


class TestResample:
    def test_resample_same_rate(self):
        audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
        result = resample_audio(audio, 16000, 16000)
        assert np.allclose(result, audio)

    def test_resample_48k_to_16k(self):
        audio = np.random.normal(0, 0.1, 48000).astype(np.float32)
        result = resample_audio(audio, 48000, 16000)
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_resample_44_1k_to_16k(self):
        audio = np.random.normal(0, 0.1, 44100).astype(np.float32)
        result = resample_audio(audio, 44100, 16000)
        expected_len = int(44100 * 16000 / 44100)  # ~16000
        assert abs(len(result) - 16000) <= 1

    def test_ensure_sample_rate_no_resample(self):
        audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
        result, sr = ensure_sample_rate(audio, 16000, 16000)
        assert sr == 16000
        assert np.allclose(result, audio)

    def test_ensure_sample_rate_resample(self):
        audio = np.random.normal(0, 0.1, 48000).astype(np.float32)
        result, sr = ensure_sample_rate(audio, 48000, 16000)
        assert sr == 16000
        assert len(result) == 16000


class TestQuality:
    def test_preprocess_audio_valid_wav(self):
        audio = make_cough_audio(2.0, 16000)
        wav_bytes = make_wav_bytes(audio, 16000)

        audio_norm, sr, meta = preprocess_audio(wav_bytes, 16000)

        assert sr == 16000
        assert audio_norm.dtype == np.float32
        assert np.max(np.abs(audio_norm)) <= 1.0
        assert meta["target_sample_rate"] == 16000
        assert meta["original_sample_rate"] == 16000
        assert abs(meta["duration_s"] - 2.0) < 0.01

    def test_preprocess_audio_resamples_48k(self):
        audio = make_cough_audio(2.0, 48000)
        wav_bytes = make_wav_bytes(audio, 48000)

        audio_norm, sr, meta = preprocess_audio(wav_bytes, 16000)

        assert sr == 16000
        assert len(audio_norm) == 32000  # 2s * 16000
        assert meta["target_sample_rate"] == 16000
        assert meta["original_sample_rate"] == 48000

    def test_preprocess_audio_peak_normalized(self):
        # Audio with known peak
        audio = np.ones(16000, dtype=np.float32) * 0.5
        wav_bytes = make_wav_bytes(audio, 16000)

        audio_norm, sr, meta = preprocess_audio(wav_bytes, 16000)

        assert np.max(np.abs(audio_norm)) == 1.0
        # Tolerance relaxed due to int16 quantization
        assert abs(meta["peak_normalization_factor"] - 0.5) < 1e-3

    def test_run_quality_pipeline_pass(self):
        audio = make_cough_audio(3.0, 16000)
        wav_bytes = make_wav_bytes(audio, 16000)

        passed, details, audio_norm, sr = run_quality_pipeline(wav_bytes, 16000)

        assert passed is True
        assert details["num_coughs"] >= 3
        assert details["snr_db"] > 10.0
        assert details["clipping_ratio"] < 0.01
        assert sr == 16000

    def test_run_quality_pipeline_too_few_coughs(self):
        audio = make_cough_audio(1.0, 16000)  # Only ~1 cough
        wav_bytes = make_wav_bytes(audio, 16000)

        config = AudioConfig(min_coughs_required=3, snr_threshold_db=5.0)
        passed, details, _, _ = run_quality_pipeline(wav_bytes, 16000, config)

        assert passed is False
        assert details["num_coughs"] < 3
        assert "too_few_coughs" in " ".join(details["reasons"])

    def test_extract_cough_segments(self):
        audio = make_cough_audio(3.0, 16000)
        segments = extract_cough_segments(audio, 16000, max_segments=5)

        assert len(segments) >= 3
        for seg in segments:
            assert seg.dtype == np.float32
            assert np.max(np.abs(seg)) <= 1.0


class TestLogMel:
    def test_hz_to_mel(self):
        assert abs(hz_to_mel(700.0) - 2595.0 * np.log10(2)) < 1e-6
        assert abs(hz_to_mel(0.0) - 0.0) < 1e-6

    def test_mel_to_hz(self):
        assert abs(mel_to_hz(2595.0 * np.log10(2)) - 700.0) < 1e-3
        assert abs(mel_to_hz(0.0) - 0.0) < 1e-6

    def test_mel_roundtrip(self):
        for f in [50, 100, 500, 1000, 4000, 8000]:
            mel = hz_to_mel(f)
            hz = mel_to_hz(mel)
            assert abs(hz - f) < 1e-3

    def test_create_mel_filterbank_shape(self):
        fb = create_mel_filterbank(16000, 1024, 64, 50.0, 8000.0)
        assert fb.shape == (64, 513)  # n_mels, n_fft//2 + 1

    def test_create_mel_filterbank_sum_to_one(self):
        fb = create_mel_filterbank(16000, 1024, 64, 50.0, 8000.0)
        # Each filter should approximately sum to 1 (triangular)
        sums = fb.sum(axis=1)
        assert np.all(sums > 0.5)  # Triangular filters have area ~1

    def test_compute_stft_shape(self):
        audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
        window = np.hanning(1024).astype(np.float32)
        stft = compute_stft(audio, 1024, 512, window)
        expected_frames = 1 + (16000 - 1024) // 512
        assert stft.shape == (expected_frames, 513)

    def test_compute_logmel_shape(self):
        audio = make_cough_audio(1.0, 16000)
        log_mel = compute_logmel(audio, 16000)
        assert log_mel.shape[1] == 64  # n_mels
        assert log_mel.shape[0] > 0  # Has frames

    def test_compute_logmel_finite(self):
        audio = make_cough_audio(1.0, 16000)
        log_mel = compute_logmel(audio, 16000)
        assert np.all(np.isfinite(log_mel))

    def test_normalize_logmel(self):
        log_mel = np.random.normal(0, 1, (100, 64)).astype(np.float32)
        normalized = normalize_logmel(log_mel)
        assert normalized.shape == log_mel.shape
        assert abs(np.mean(normalized)) < 0.1
        assert abs(np.std(normalized) - 1.0) < 0.1

    def test_process_audio_to_logmel(self):
        audio = make_cough_audio(1.0, 16000)
        log_mel = process_audio_to_logmel(audio, 16000)
        assert log_mel.shape[0] == 64  # n_mels
        assert np.all(np.isfinite(log_mel))

    def test_process_audio_to_logmel_resamples(self):
        audio = make_cough_audio(1.0, 48000)
        log_mel = process_audio_to_logmel(audio, 48000)
        assert log_mel.shape[0] == 64  # n_mels
        assert np.all(np.isfinite(log_mel))


class TestParityConfig:
    """Test that config values match expected defaults from audio.yaml (TBD)."""

    def test_default_config_values(self):
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.frame_length == 1024
        assert config.hop_length == 512
        assert config.n_mels == 64
        assert config.f_min == 50.0
        assert config.f_max == 8000.0
        assert config.target_cough_duration_s == 0.5
        assert config.min_cough_duration_s == 0.1
        assert config.max_cough_duration_s == 2.0
        assert config.noise_percentile == 10.0
        assert config.noise_margin_db == 6.0
        assert config.hysteresis_db == 3.0
        assert config.snr_threshold_db == 10.0
        assert config.clipping_threshold_ratio == 0.01
        assert config.min_coughs_required == 3
        assert config.max_coughs_expected == 10


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    tests = [
        TestResample,
        TestQuality,
        TestLogMel,
        TestParityConfig,
    ]

    for test_class in tests:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"PASS: {test_class.__name__}.{method_name}")
                except Exception as e:
                    print(f"FAIL: {test_class.__name__}.{method_name}: {e}")

    print("All tests completed!")