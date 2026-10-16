"""Log-mel parity tests.

AGENTS.md section 4.5: "Pra-pemrosesan training dan serving harus identik. Uji
paritas log-Mel (selisih maksimum 1e-4) wajib lulus sebelum model apa pun
dirilis."

The training pipeline lives in `ml/src/features/audio.py::extract_logmel` and the
serving pipeline in `apps/api/tbcare_api/audio/logmel.py::process_audio_to_logmel`.
This module compares them on the same synthetic signal and asserts they agree
within the documented tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

TOLERANCE = 1e-4


def _audio_services_importable() -> bool:
    try:
        import tbcare_api.audio.logmel  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _audio_services_importable(),
    reason="apps/api not importable (run with PYTHONPATH=.:apps/api)",
)


def _training_logmel(samples, sample_rate, config):
    """Training-side log-mel, shape (n_mels, target_frames).

    Uses `extract_logmel_from_wav`, the same entry point the training data
    loader uses (`ml/scripts/data_loader.py`), so the 16-bit quantization made
    on the way in matches what the model actually saw during training.
    """
    from ml.src.features.audio import extract_logmel_from_wav

    feat = extract_logmel_from_wav(_wav_bytes(samples, sample_rate), config)
    # extract_logmel returns (1, n_mels, frames); drop the channel axis.
    if feat.ndim == 3:
        feat = feat[0]
    return feat


def _serving_logmel(samples, sample_rate, target_frames):
    """Serving-side log-mel through the *real* serving entry point.

    Production serving runs `preprocess_audio` (validate, resample, peak
    normalize) and then `process_audio_to_logmel`. Comparing against the full
    path is the only way this test can catch a normalization skew, so it does
    not call `process_audio_to_logmel` on raw audio directly.
    """
    from tbcare_api.audio.quality import preprocess_audio
    from tbcare_api.audio.logmel import process_audio_to_logmel

    audio_norm, sr, _meta = preprocess_audio(_wav_bytes(samples, sample_rate))
    return process_audio_to_logmel(audio_norm, sr, target_frames=target_frames)


def _wav_bytes(samples, sample_rate):
    """Encode float samples as 16-bit mono WAV bytes."""
    import io
    import wave

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_training_and_serving_shapes_match(synthetic_cough, audio_config):
    tr = _training_logmel(synthetic_cough, 16000, audio_config)
    sv = _serving_logmel(synthetic_cough, 16000, audio_config.target_frames)
    assert tr.shape == sv.shape == (audio_config.n_mels, audio_config.target_frames)


def test_training_and_serving_preprocessing_parity(synthetic_cough, audio_config):
    """The core AGENTS.md 4.5 gate: max abs diff must be <= 1e-4."""
    tr = _training_logmel(synthetic_cough, 16000, audio_config)
    sv = _serving_logmel(synthetic_cough, 16000, audio_config.target_frames)

    max_abs_diff = float(np.max(np.abs(tr.astype(np.float64) - sv.astype(np.float64))))
    assert max_abs_diff <= TOLERANCE, (
        "train/serve log-mel mismatch: "
        f"max|diff|={max_abs_diff:.6f} > {TOLERANCE}. "
        "Serving features are out-of-distribution relative to training."
    )


def test_parity_helper_reports_same_verdict(synthetic_wav_bytes, audio_config):
    """`audio.check_parity` must agree with this test's tolerance."""
    from ml.src.features.audio import check_parity

    passed, max_diff = check_parity(
        synthetic_wav_bytes,
        serving_config=audio_config,
        training_config=audio_config,
        tolerance=TOLERANCE,
    )
    # Both sides use the same config here, so parity must hold exactly.
    assert passed is True
    assert max_diff <= TOLERANCE
