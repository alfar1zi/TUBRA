"""Serve a trained YOHO network from ONNX Runtime, with an energy fallback.

The API must not import torch (the runtime image does not ship it), so the
trained graph travels as ``yoho.onnx`` and is scored through ONNX Runtime. If
the file is missing or fails to load, the detector keeps working through the
old energy segmentation instead of dropping coughs entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

Predictor = Callable[[np.ndarray], np.ndarray]


class YohoPredictor:
    """Wrap an ONNX Runtime session as a callable ``features -> output``.

    Bit-exactness with torch is checked at export time
    (``ml/scripts/export_yoho.py``), so nothing here re-validates the graph.
    The class is also the seam tests use to substitute a stub predictor.
    """

    def __init__(self, session, input_name: str, output_name: str) -> None:
        self._session = session
        self._input_name = input_name
        self._output_name = output_name

    def __call__(self, features: np.ndarray) -> np.ndarray:
        batch = np.ascontiguousarray(features, dtype=np.float32)
        output = self._session.run([self._output_name], {self._input_name: batch})[0]
        return np.asarray(output, dtype=np.float32)


def load_predictor(model_dir: Path | str) -> Predictor | None:
    """Load ``yoho.onnx`` from ``model_dir``; ``None`` means fall back to energy.

    Missing ``onnxruntime`` and an unreadable graph both return ``None`` on
    purpose: a deployment without the export should still accept a recording,
    and the old energy detector keeps that promise.
    """
    model_path = Path(model_dir) / "yoho.onnx"
    if not model_path.is_file():
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception:
        return None
    return YohoPredictor(session, session.get_inputs()[0].name, session.get_outputs()[0].name)
