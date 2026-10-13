"""Resampler regression tests.

`ml/src/features/audio.py::_anti_aliased_resample` previously used a
hand-rolled Kaiser kernel that was not scaled by the upsampling factor, so it
silently attenuated the signal by roughly 50x. Every 44.1 kHz training clip was
mangled relative to the serving path. The fix delegates to
`scipy.signal.resample_poly`, the same primitive serving uses.

These tests pin the gain and the train/serve agreement so the bug cannot come
back unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest


def _band_limited_noise(seed: int, n: int, rate: int, std: float = 0.3) -> np.ndarray:
    """Noise band-limited below 3 kHz, so downsampling preserves its energy.

    Plain white noise loses RMS when downsampled (content above the new
    Nyquist is filtered out), which would make a gain assertion meaningless.
    """
    from scipy import signal

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float64)
    cutoff = min(3000.0, 0.45 * rate)
    sos = signal.butter(4, cutoff / (rate / 2.0), btype="low", output="sos")
    x = signal.sosfilt(sos, x)
    x = x / (np.max(np.abs(x)) + 1e-12) * std
    return x.astype(np.float32)


@pytest.mark.parametrize(
    "source_rate,target_rate",
    [(44100, 16000), (48000, 16000), (22050, 16000), (8000, 16000)],
)
def test_resampler_preserves_signal_amplitude(source_rate, target_rate):
    """Resampling must not change the signal's scale (no silent attenuation)."""
    from ml.src.features.audio import _anti_aliased_resample

    samples = _band_limited_noise(0, source_rate, source_rate, std=0.3)
    out = _anti_aliased_resample(samples, source_rate, target_rate)

    # Resampling changes the sample count; the *scale* must survive.
    ratio = out.std() / samples.std()
    assert 0.9 < ratio < 1.1, (
        f"resampler changed signal amplitude by {ratio:.4f}x "
        f"({source_rate} -> {target_rate}); expected ~1.0x"
    )


def test_resampler_matches_serving_primitive():
    """Training resampler must equal the serving implementation exactly."""
    from ml.src.features.audio import _anti_aliased_resample
    from tbcare_api.audio.resample import resample_audio

    samples = _band_limited_noise(1, 44100, 44100, std=0.3)
    train = _anti_aliased_resample(samples, 44100, 16000)
    serve = resample_audio(samples, 44100, 16000)

    n = min(len(train), len(serve))
    assert len(train) == len(serve)
    max_diff = float(np.max(np.abs(train[:n] - serve[:n])))
    assert max_diff < 1e-5, f"train/serve resampler mismatch: max|diff|={max_diff}"


def test_resampler_is_identity_at_same_rate():
    from ml.src.features.audio import _anti_aliased_resample

    samples = _band_limited_noise(2, 16000, 16000, std=0.3)
    out = _anti_aliased_resample(samples, 16000, 16000)
    assert np.array_equal(out, samples.astype(np.float32))


def test_resampler_output_length():
    from ml.src.features.audio import _anti_aliased_resample

    samples = _band_limited_noise(3, 44100, 44100, std=0.3)
    out = _anti_aliased_resample(samples, 44100, 16000)
    expected = int(round(len(samples) * 16000 / 44100))
    assert abs(len(out) - expected) <= 1
