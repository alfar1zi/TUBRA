"""Shared fixtures for the TBCare ML test suite.

These tests must run WITHOUT training data: `ml/data/` is gitignored and never
shipped. Every fixture here builds synthetic signals or tiny in-memory frames,
so the suite is fast, deterministic, and safe to run in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable so `import ml...` works from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def audio_config():
    """The energy_v2 audio config the exported artifacts were trained with.

    Mirrors model_manifest.json -> audio_config (n_fft=1024, hop=256, mel=64,
    f_min=0.0, f_max=None). Keep in lockstep with
    apps/api/tbcare_api/audio/logmel.py; the parity test is only meaningful when
    both sides describe the *real* training recipe.
    """
    from ml.src.features.audio import AudioFeatureConfig

    return AudioFeatureConfig(
        recipe="energy_v2",
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        f_min=0.0,
        f_max=None,
        target_frames=64,
    )


@pytest.fixture
def synthetic_cough() -> np.ndarray:
    """A deterministic cough-like signal: 8 noise bursts over 3 seconds."""
    sr = 16000
    n = sr * 3
    t = np.arange(n) / sr
    rng = np.random.RandomState(0)
    sig = np.zeros(n)
    for start in rng.randint(0, n - sr // 4, 8):
        length = sr // 8
        sig[start : start + length] += (
            np.hanning(length) * np.sin(2 * np.pi * 300 * t[:length]) * 0.5
        )
    sig += rng.randn(n) * 0.01
    return sig.astype(np.float32)


@pytest.fixture
def synthetic_wav_bytes(synthetic_cough: np.ndarray) -> bytes:
    """16-bit mono WAV bytes derived from the synthetic cough."""
    import io
    import wave

    pcm = np.clip(synthetic_cough * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="session")
def artifacts_dir(project_root: Path) -> Path:
    """Trained artifact directory. Skips the test if artifacts are absent."""
    d = project_root / "ml" / "artifacts"
    has_backbone = (d / "backbone.onnx").exists() or bool(list(d.glob("backbone_seed*.onnx")))
    if not has_backbone:
        pytest.skip("ml/artifacts/ not built; run export_artifacts.py first")
    return d
