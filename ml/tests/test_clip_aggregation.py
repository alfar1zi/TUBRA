"""Clip-aggregation tests.

AGENTS.md section 4.5 and the proposal docs/proposal/training_improvement.md (P1):
padded (all-zero) clips must not influence a patient's score. These tests pin
that behaviour for the torch experts and for the serving ensemble.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _toy_audio(n_clips: int, n_mels: int = 64, t: int = 64, seed: int = 0):
    """Return (1, n_clips, 1, n_mels, t) with identical clips."""
    base = torch.randn(1, 1, 1, n_mels, t, generator=torch.Generator().manual_seed(seed))
    return base.repeat(1, n_clips, 1, 1, 1)


@pytest.fixture
def audio_model():
    from ml.src.models.residual_fusion import AudioOnlyClassifier

    torch.manual_seed(0)
    model = AudioOnlyClassifier()
    model.eval()
    return model


def test_audio_only_padding_does_not_change_score(audio_model):
    """One real clip + padding must equal one real clip alone."""
    real = _toy_audio(1)
    padded = torch.cat([real, torch.zeros(1, 4, 1, 64, 64)], dim=1)

    with torch.no_grad():
        single = audio_model(real)[0, 1].item()
        with_pad = audio_model(padded)[0, 1].item()

    assert abs(single - with_pad) < 1e-6, (
        f"padding changed P(TB): single={single:.6f} padded={with_pad:.6f}"
    )


def test_audio_only_masks_only_real_clips(audio_model):
    """5 identical real clips must equal 1 real clip (mask keeps the mean)."""
    one = _toy_audio(1)
    five = one.repeat(1, 5, 1, 1, 1)

    with torch.no_grad():
        s1 = audio_model(one)[0, 1].item()
        s5 = audio_model(five)[0, 1].item()

    assert abs(s1 - s5) < 1e-6


def test_fusion_padding_does_not_change_score():
    from ml.src.models.residual_fusion import ResidualFusionClassifier

    torch.manual_seed(0)
    model = ResidualFusionClassifier()
    model.eval()

    real = _toy_audio(1)
    padded = torch.cat([real, torch.zeros(1, 4, 1, 64, 64)], dim=1)
    clinical = torch.randn(1, 26, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        single = model(real, clinical)[0, 1].item()
        with_pad = model(padded, clinical)[0, 1].item()

    assert abs(single - with_pad) < 1e-5, (
        f"padding changed fusion P(TB): single={single:.6f} padded={with_pad:.6f}"
    )


def test_serving_ensemble_ignores_zero_clips(artifacts_dir, monkeypatch):
    """The serving ensemble must skip all-zero clips when averaging."""
    pytest.importorskip("onnxruntime")
    try:
        from tbcare_api.inference.ensemble import create_ensemble_from_artifacts
    except Exception:
        pytest.skip("apps/api not importable; run with PYTHONPATH=.:apps/api")

    monkeypatch.setenv("DEPLOYMENT_ENV", "staging")
    monkeypatch.setenv("ALLOW_BLOCKED_CANDIDATE", "true")
    ensemble = create_ensemble_from_artifacts(str(artifacts_dir))

    record = {
        "age": 40.0,
        "sex": 1.0,
        "cough_duration": 1.0,
        "hemoptysis": 1.0,
        "fever": 1.0,
        "night_sweats": 1.0,
        "weight_loss": 1.0,
        "prior_tb": 0.0,
        "smoking": 0.0,
    }
    clip = np.random.RandomState(0).rand(64, 64).astype(np.float32)

    one = ensemble.predict(clip, record)["prob_audio"]
    padded = ensemble.predict(np.stack([clip, np.zeros_like(clip)]), record)["prob_audio"]

    assert abs(one - padded) < 1e-6, (
        f"zero clip changed prob_audio: one={one} padded={padded}"
    )
