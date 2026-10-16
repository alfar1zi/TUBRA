"""Test-Time Augmentation (TTA) for the audio branch — shared by training and serving.

The *same* deterministic transforms are applied during offline evaluation
(`ml/scripts/train.py::evaluate_model`) and online inference
(`apps/api/tbcare_api/inference/ensemble.py::ONNXAudioModel`). Because the
transforms contain no randomness, the two paths are provably identical; this is
what keeps AGENTS.md section 4.5 (train/serve parity) satisfied for TTA.

Design constraints:
- Pure NumPy. No torch, no librosa (serving must not import them).
- Deterministic: a given input always yields the same augmented views.
- Conservative: research (SpecMix/Calibration papers) shows aggressive masks
  hurt at test time; we only use small time-shifts and gain changes.
- Shape-preserving: the augmented tensor keeps the exact shape the model
  expects, so models that aggregate across clips (e.g. `AudioOnlyClassifier`)
  behave identically on every view.

This is the training-side copy; `apps/api/tbcare_api/inference/tta.py` mirrors
it and `ml/tests/test_tta_parity.py` asserts the copies never drift.
"""

from __future__ import annotations

import numpy as np

# Augmentation grid: (time_shift_fraction, gain_db). The identity view
# (0.0, 0.0) is always included so TTA can never be worse than the base model
# by more than the averaging noise.
TTA_GRID: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),      # identity
    (0.05, 0.0),     # shift right 5%
    (-0.05, 0.0),    # shift left 5%
    (0.0, 2.0),      # +2 dB
    (0.0, -2.0),     # -2 dB
)


def _shift_time(view: np.ndarray, frac: float) -> np.ndarray:
    """Circularly shift a (n_mels, n_frames) view along the time axis."""
    if frac == 0.0:
        return view
    n_frames = view.shape[-1]
    shift = int(round(frac * n_frames))
    if shift == 0:
        return view
    return np.roll(view, shift, axis=-1)


def tta_views(logmel: np.ndarray) -> list[np.ndarray]:
    """Return the deterministic augmented views for a single log-mel clip.

    Args:
        logmel: (n_mels, n_frames); a leading singleton channel axis is stripped.

    Returns:
        List of (n_mels, n_frames) arrays, one per ``TTA_GRID`` entry, in a
        fixed order (index 0 is always the unmodified identity view).
    """
    arr = np.asarray(logmel, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"tta_views expects (n_mels, n_frames), got {arr.shape}")

    views: list[np.ndarray] = []
    for shift_frac, gain_db in TTA_GRID:
        # Log-mel is in dB scale, so a gain change is a plain additive offset.
        v = arr + float(gain_db)
        v = _shift_time(v, shift_frac)
        views.append(v.astype(np.float32, copy=False))
    return views


def tta_average_logits(run_batch, logmel_batch: np.ndarray) -> np.ndarray:
    """Average logits over TTA views, preserving the model's batch structure.

    Handles both the serving layout ``(N, 1, n_mels, T)`` and the training
    layout ``(B, K, 1, n_mels, T)`` (B patients, K clips each). The augmented
    tensor is passed to ``run_batch`` with exactly the same shape as the input,
    so a model that aggregates across clips sees the correct structure on every
    view.

    Args:
        run_batch: callable mapping an array shaped like ``logmel_batch`` to a
            ``(B, C)`` logits array (torch in training, onnxruntime in serving).
        logmel_batch: 4-D ``(N, 1, n_mels, T)`` or 5-D ``(B, K, 1, n_mels, T)``
            float array. A 3-D ``(N, n_mels, T)`` is promoted to 4-D.

    Returns:
        ``(B, C)`` averaged logits.
    """
    arr = np.asarray(logmel_batch, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]

    acc = None
    for view_idx in range(len(TTA_GRID)):
        batch_view = arr.copy()
        if arr.ndim == 4:
            # (B, 1, n_mels, T)
            for i in range(arr.shape[0]):
                batch_view[i, 0] = tta_views(arr[i, 0])[view_idx]
        elif arr.ndim == 5:
            # (B, K, 1, n_mels, T)
            for i in range(arr.shape[0]):
                for k in range(arr.shape[1]):
                    batch_view[i, k, 0] = tta_views(arr[i, k, 0])[view_idx]
        else:
            raise ValueError(
                f"tta_average_logits expects 4-D or 5-D input, got {arr.shape}"
            )
        logits = np.asarray(run_batch(batch_view), dtype=np.float32)
        acc = logits if acc is None else acc + logits
    return acc / float(len(TTA_GRID))
