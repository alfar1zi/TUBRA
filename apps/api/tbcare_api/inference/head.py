import numpy as np
from pathlib import Path


class HeadModel:
    """NumPy implementation of the classification head."""

    def __init__(self, head_path: str | None = None):
        if head_path is None:
            base = Path(__file__).resolve().parent.parent.parent.parent.parent / "ml" / "artifacts"
            head_path = str(base / "head.npz")

        data = np.load(head_path)
        self.W1 = data["W1"]      # (hidden_dim, input_dim)
        self.b1 = data["b1"]      # (hidden_dim,)
        self.w2 = data["w2"]      # (hidden_dim,)
        self.b2 = data["b2"]      # (1,)
        self.T = float(data["T"]) # scalar

        self.hidden_dim = int(data["hidden_dim"])
        self.input_dim = int(data["input_dim"])

        # Sanity check
        assert self.W1.shape == (self.hidden_dim, self.input_dim)
        assert self.b1.shape == (self.hidden_dim,)
        assert self.w2.shape == (self.hidden_dim,)

    def forward(self, embeddings: np.ndarray, symptoms: np.ndarray, availability: np.ndarray) -> dict:
        """
        Forward pass through the head.

        Args:
            embeddings: (N, 512) - mean pooled to (512,)
            symptoms: (9,) - symptom values
            availability: (9,) - 0/1 mask

        Returns:
            dict with logit, probability, and intermediate values
        """
        # Mean pooling over cough segments: H = mean_i(h_i)
        H = embeddings.mean(axis=0)  # (512,)

        # Concatenate: [H; C; m]
        # C = symptoms (9,), m = availability (1,) - overall availability
        m = np.array([availability.mean()], dtype=np.float32)  # overall availability
        x = np.concatenate([H, symptoms, m])  # (512 + 9 + 1,) = (522,)

        # u = W1 @ x + b1
        u = self.W1 @ x + self.b1  # (hidden_dim,)

        # a = ReLU(u)
        a = np.maximum(u, 0)

        # z = w2 @ a + b2
        z = float(self.w2 @ a + self.b2)

        # p = sigmoid(z / T)
        p = 1.0 / (1.0 + np.exp(-z / self.T))

        return {
            "logit": z,
            "probability": p,
            "H": H,
            "u": u,
            "a": a,
            "x": x,
            "m": m,
        }

    def forward_with_intermediates(self, embeddings: np.ndarray, symptoms: np.ndarray,
                                    availability: np.ndarray) -> dict:
        """Forward pass returning all intermediates for Grad-CAM."""
        return self.forward(embeddings, symptoms, availability)


# Singleton
_head: HeadModel | None = None


def get_head() -> HeadModel:
    global _head
    if _head is None:
        _head = HeadModel()
    return _head