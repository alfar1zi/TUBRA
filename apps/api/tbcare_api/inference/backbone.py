import os
import numpy as np
import onnxruntime as ort
from pathlib import Path


class BackboneONNX:
    """ONNX Runtime wrapper for the backbone model."""

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            # Default: ml/artifacts/backbone.onnx relative to project root
            base = Path(__file__).resolve().parent.parent.parent.parent.parent / "ml" / "artifacts"
            model_path = str(base / "backbone.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Backbone model not found: {model_path}")

        # Use CPU execution provider
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Expected: embedding, feature_map
        assert len(self.output_names) == 2, f"Expected 2 outputs, got {len(self.output_names)}"

    def run(self, logmel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on log-Mel spectrograms.

        Args:
            logmel: Shape (N, 1, n_mels, n_frames) or (N, n_mels, n_frames)
                    where N = number of cough segments

        Returns:
            embeddings: (N, 512)
            feature_maps: (N, 512, H, W)
        """
        # Ensure correct shape: (N, 1, H, W)
        if logmel.ndim == 3:
            logmel = logmel[:, np.newaxis, :, :]  # Add channel dim
        elif logmel.ndim != 4:
            raise ValueError(f"logmel must be 3D or 4D, got {logmel.ndim}D")

        # ONNX Runtime expects float32
        logmel = logmel.astype(np.float32)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: logmel},
        )

        embeddings = outputs[0]  # (N, 512)
        feature_maps = outputs[1]  # (N, 512, H, W)

        return embeddings, feature_maps


# Singleton instance
_backbone: BackboneONNX | None = None


def get_backbone() -> BackboneONNX:
    """Get or create the singleton backbone instance."""
    global _backbone
    if _backbone is None:
        _backbone = BackboneONNX()
    return _backbone