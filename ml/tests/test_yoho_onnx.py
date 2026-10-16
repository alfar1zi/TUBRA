"""Torch and ONNX Runtime must agree on the same log-mel window.

The API never imports torch: it loads ``yoho.onnx`` through ONNX Runtime. If the
export drifted from the trained graph, the server would decode different
boundaries than the ones measured during evaluation, so this test exports a
randomly initialised network and compares both paths on one 8 s window.

The exported graph takes exactly one window (batch 1) because the API scores a
single context window at a time; that keeps the time axis fixed at one step
count and removes an axis-shape ambiguity the server would otherwise hit.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from ml.src.yoho.detector import CONTEXT_STEPS, WINDOW_MEL_FRAMES  # noqa: E402
from ml.src.yoho.frontend import N_MELS  # noqa: E402
from ml.src.yoho.model import MobileNetYoho  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    model = MobileNetYoho(n_mels=N_MELS)
    model.eval()
    path = tmp_path_factory.mktemp("onnx") / "yoho.onnx"
    sample = torch.zeros(1, 1, N_MELS, WINDOW_MEL_FRAMES, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            str(path),
            input_names=["logmel"],
            output_names=["yoho"],
            opset_version=17,
            do_constant_folding=True,
        )
    return model, path


def test_onnx_matches_torch_on_one_window(exported):
    model, path = exported
    features = np.random.default_rng(0).normal(-40.0, 8.0, (1, 1, N_MELS, WINDOW_MEL_FRAMES)).astype(np.float32)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"logmel": features})[0]
    with torch.no_grad():
        torch_output = model(torch.from_numpy(features)).numpy()
    assert torch_output.shape == (1, 3, CONTEXT_STEPS)
    assert onnx_output.shape == torch_output.shape
    assert float(np.max(np.abs(torch_output - onnx_output))) < 1e-4


def test_onnx_output_groups_one_head_per_class(exported):
    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    features = np.zeros((1, 1, N_MELS, WINDOW_MEL_FRAMES), dtype=np.float32)
    output = session.run(None, {"logmel": features})[0]
    assert output.shape == (1, 3, CONTEXT_STEPS)
    assert np.isfinite(output).all()
    assert ((output >= 0.0) & (output <= 1.0)).all()
