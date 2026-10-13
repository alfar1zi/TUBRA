"""Artifact-contract tests.

ARCHITECTURE.md section 6.3 and the README document what `ml/artifacts/` must
contain and what the model card keys are. These tests assert the contract holds
for the artifacts actually on disk. They skip when artifacts are absent so the
suite stays runnable without training.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# Files that must exist for serving. The audio backbone may be either a single
# ``backbone.onnx`` or a seed-ensemble of ``backbone_seed*.onnx`` members; that
# is checked separately below.
REQUIRED_ARTIFACTS = (
    "head.npz",
    "clinical_model.joblib",
    "supervisor_model.joblib",
    "clinical_preprocessor.joblib",
    "model_manifest.json",
    "model_card.json",
    "cv_results.json",
)

# Keys the web app reads: apps/web/src/screens/S11HowItWorks.tsx.
WEB_METRIC_KEYS = (
    "roc_auc",
    "pauc_90",
    "sens_at_A",
    "spec_at_A",
    "ece",
    "threshold_A",
    "threshold_B",
)


def test_all_required_artifacts_present(artifacts_dir):
    missing = [n for n in REQUIRED_ARTIFACTS if not (artifacts_dir / n).exists()]
    assert not missing, f"missing artifacts: {missing}"


def test_audio_backbone_present(artifacts_dir):
    """Either a single backbone or a seed-ensemble must be present."""
    has_single = (artifacts_dir / "backbone.onnx").exists()
    has_seeds = bool(list(artifacts_dir.glob("backbone_seed*.onnx")))
    assert has_single or has_seeds, "no audio backbone (backbone.onnx / backbone_seed*.onnx)"


def test_model_card_has_web_metric_keys(artifacts_dir):
    card = json.loads((artifacts_dir / "model_card.json").read_text())
    metrics = card.get("metrics", {})
    missing = [k for k in WEB_METRIC_KEYS if k not in metrics]
    assert not missing, f"model_card.metrics missing web keys: {missing}"


def test_model_card_metric_values_are_number_or_null(artifacts_dir):
    """Every metric value is a number, a short list (CI), a dict, or null."""
    card = json.loads((artifacts_dir / "model_card.json").read_text())

    def ok(value) -> bool:
        if value is None or isinstance(value, (int, float)):
            return True
        if isinstance(value, list):
            return all(x is None or isinstance(x, (int, float)) for x in value)
        if isinstance(value, dict):
            return all(ok(x) for x in value.values())
        return False

    for key, value in card["metrics"].items():
        assert ok(value), f"model_card.metrics[{key}] has bad type: {type(value)}"


def test_manifest_status_is_known(artifacts_dir):
    manifest = json.loads((artifacts_dir / "model_manifest.json").read_text())
    assert manifest["status"] in {"unavailable", "candidate", "validated"}


def test_manifest_carries_tier_thresholds(artifacts_dir):
    manifest = json.loads((artifacts_dir / "model_manifest.json").read_text())
    metrics = manifest.get("metrics", {})
    assert "threshold_A" in metrics and "threshold_B" in metrics
    assert metrics["threshold_A"] > metrics["threshold_B"], (
        "Tier A threshold must sit above Tier B"
    )


def test_onnx_backbone_accepts_dynamic_batch(artifacts_dir):
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(
        str(artifacts_dir / "backbone.onnx"), providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0]
    for batch in (1, 3, 5):
        x = np.random.RandomState(0).randn(batch, 1, 64, 64).astype(np.float32)
        out = sess.run(None, {inp.name: x})[0]
        assert out.shape == (batch, 2), f"batch {batch} produced {out.shape}"


def _sample_record() -> dict:
    return {
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


def test_ensemble_predict_returns_valid_tier(artifacts_dir, monkeypatch):
    pytest.importorskip("onnxruntime")
    try:
        from tbcare_api.inference.ensemble import create_ensemble_from_artifacts
    except Exception:
        pytest.skip("apps/api not importable; run with PYTHONPATH=.:apps/api")

    # Artifacts ship as CANDIDATE, which prod inference blocks by design.
    # Use the documented staging gate to exercise the real prediction path.
    monkeypatch.setenv("DEPLOYMENT_ENV", "staging")
    monkeypatch.setenv("ALLOW_BLOCKED_CANDIDATE", "true")

    ensemble = create_ensemble_from_artifacts(str(artifacts_dir))
    logmel = np.random.RandomState(0).rand(64, 64).astype(np.float32)
    result = ensemble.predict(logmel, _sample_record())
    assert result["risk_tier"] in ("A", "B", "C")


def test_audio_branch_is_deterministic(artifacts_dir):
    pytest.importorskip("onnxruntime")
    try:
        from tbcare_api.inference.ensemble import ONNXAudioModel
    except Exception:
        pytest.skip("apps/api not importable; run with PYTHONPATH=.:apps/api")

    model = ONNXAudioModel(
        str(artifacts_dir / "backbone.onnx"), str(artifacts_dir / "head.npz")
    )
    x = np.random.RandomState(42).randn(64, 64).astype(np.float32)
    np.testing.assert_array_equal(model.predict_proba(x), model.predict_proba(x))


def test_audio_branch_ensemble_duplicate_members_match_single(artifacts_dir, tmp_path):
    """Two identical backbones must average to the same logits as one.

    Averaging logits over N members is the serving contract for seed-ensembles;
    with N identical members the result must be numerically identical to the
    single-member model (up to float32 rounding).
    """
    pytest.importorskip("onnxruntime")
    try:
        from tbcare_api.inference.ensemble import ONNXAudioModel
    except Exception:
        pytest.skip("apps/api not importable; run with PYTHONPATH=.:apps/api")

    src = artifacts_dir / "backbone.onnx"
    head = str(artifacts_dir / "head.npz")
    # Duplicate the graph (+ its external weights) under a second name.
    dup = tmp_path / src.name
    dup.write_bytes(src.read_bytes())
    for ext in artifacts_dir.glob("backbone.onnx.data"):
        (tmp_path / ext.name).write_bytes(ext.read_bytes())

    x = np.random.RandomState(7).randn(64, 64).astype(np.float32)
    single = ONNXAudioModel(str(src), head).predict_proba(x, use_tta=False)
    double = ONNXAudioModel([str(src), str(dup)], head).predict_proba(x, use_tta=False)
    np.testing.assert_allclose(single, double, rtol=1e-5, atol=1e-6)


def test_factory_discovers_seed_members(artifacts_dir, tmp_path, monkeypatch):
    """``create_ensemble_from_artifacts`` must load ``backbone_seed*.onnx`` when present."""
    pytest.importorskip("onnxruntime")
    try:
        from tbcare_api.inference.ensemble import create_ensemble_from_artifacts
    except Exception:
        pytest.skip("apps/api not importable; run with PYTHONPATH=.:apps/api")

    monkeypatch.setenv("DEPLOYMENT_ENV", "staging")
    monkeypatch.setenv("ALLOW_BLOCKED_CANDIDATE", "true")

    # Copy the full artifact dir and add two seed backbones.
    import shutil

    adir = tmp_path / "artifacts"
    shutil.copytree(artifacts_dir, adir)
    src = adir / "backbone.onnx"
    for k in range(2):
        (adir / f"backbone_seed{k}.onnx").write_bytes(src.read_bytes())

    ensemble = create_ensemble_from_artifacts(str(adir))
    assert ensemble.audio_model.n_encoders == 2
