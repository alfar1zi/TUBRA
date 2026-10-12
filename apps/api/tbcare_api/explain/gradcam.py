"""
Grad-CAM computation without autograd (manual backward).
Based on ADR-004: since last backbone layer is Global Average Pooling,
we can compute Grad-CAM analytically through the head.

Formula:
  H = mean_i(h_i)                    # (512,)
  u = W1 @ [H; C; m] + b1            # (hidden_dim,)
  a = ReLU(u)
  z = w2 @ a + b2                    # logit

  ∂z/∂H = W1_H^T @ (1[u>0] ⊙ w2)     # (512,)
  ∂z/∂h_i = (1/N) * ∂z/∂H            # (512,) per segment

  L_i = ReLU( sum_k α_k * A_i^k )    # where α_k = ∂z/∂h_i[k]
"""

import numpy as np


def compute_gradcam(
    feature_maps: np.ndarray,
    head_W1: np.ndarray,
    head_w2: np.ndarray,
    head_b1: np.ndarray,
    symptoms: np.ndarray,
    availability: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmaps for each cough segment.

    Args:
        feature_maps: (N, 512, H, W) - from backbone
        head_W1: (hidden_dim, 522) - first layer weights
        head_w2: (hidden_dim,) - second layer weights
        head_b1: (hidden_dim,) - first layer bias
        symptoms: (9,) - symptom values
        availability: (9,) - availability mask
        embeddings: (N, 512) - for computing H

    Returns:
        heatmaps: (N, H, W) - Grad-CAM per segment, normalized to [0, 1]
    """
    N, C, H, W = feature_maps.shape
    assert C == 512, f"Expected 512 channels, got {C}"

    # Compute H = mean_i(h_i)
    H_vec = embeddings.mean(axis=0)  # (512,)

    # Compute overall availability m
    m = np.array([availability.mean()], dtype=np.float32)

    # x = [H; C; m]
    x = np.concatenate([H_vec, symptoms, m])  # (522,)

    # u = W1 @ x + b1
    u = head_W1 @ x + head_b1  # (hidden_dim,)

    # ReLU mask
    relu_mask = (u > 0).astype(np.float32)  # (hidden_dim,)

    # W1_H: first 512 columns of W1 (corresponding to H)
    W1_H = head_W1[:, :512]  # (hidden_dim, 512)

    # ∂z/∂H = W1_H^T @ (relu_mask * w2)
    dz_dH = W1_H.T @ (relu_mask * head_w2)  # (512,)

    # ∂z/∂h_i = (1/N) * ∂z/∂H
    dz_dh = dz_dH / N  # (512,)

    # For each segment, compute weighted combination of feature maps
    heatmaps = []
    for i in range(N):
        # α_k = ∂z/∂h_i[k]
        alpha = dz_dh  # (512,)

        # L_i = ReLU( sum_k α_k * A_i^k )
        # A_i^k is feature_maps[i, k, :, :]
        cam = np.sum(alpha[:, np.newaxis, np.newaxis] * feature_maps[i], axis=0)
        cam = np.maximum(cam, 0)  # ReLU

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        heatmaps.append(cam)

    return np.stack(heatmaps, axis=0)  # (N, H, W)


def resize_heatmap(heatmap: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize heatmap to target size using simple interpolation.
    For production, use cv2.resize or PIL.
    """
    from scipy.ndimage import zoom

    h, w = heatmap.shape
    target_h, target_w = target_size
    zoom_h = target_h / h
    zoom_w = target_w / w
    return zoom(heatmap, (zoom_h, zoom_w), order=1)


def aggregate_heatmaps(heatmaps: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Aggregate per-segment heatmaps into single heatmap.

    Args:
        heatmaps: (N, H, W)
        method: "mean" | "max"

    Returns:
        (H, W) aggregated heatmap
    """
    if method == "mean":
        return heatmaps.mean(axis=0)
    elif method == "max":
        return heatmaps.max(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")