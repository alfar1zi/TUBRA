"""Test-Time Augmentation (TTA) for the audio branch — serving side.

This is a deliberate mirror of ``ml/src/features/tta.py``. The serving package
must not import the training package (and must not import torch/librosa), so the
logic is duplicated here. ``ml/tests/test_tta_parity.py`` asserts that both
copies produce bit-identical views, so they can never silently drift.

Keep any change to ``TTA_GRID`` or the transform order in sync with the
training-side module.
"""

from __future__ import annotations

import numpy as np

TTA_GRID: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),      # identity
    (0.05, 0.0),     # shift right 5%
    (-0.05, 0.0),    # shift left 5%
    (0.0, 2.0),      # +2 dB
    (0.0, -2.0),     # -2 dB
)


def _shift_time(view: np.ndarray, frac: float) -> np.ndarray:
    if frac == 0.0:
        return view
    n_frames = view.shape[-1]
    shift = int(round(frac * n_frames))
    if shift == 0:
        return view
    return np.roll(view, shift, axis=-1)


def tta_views(logmel: np.ndarray) -> list[np.ndarray]:
    """Return deterministic augmented views for one log-mel clip (n_mels, T)."""
    arr = np.asarray(logmel, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"tta_views expects (n_mels, n_frames), got {arr.shape}")

    views: list[np.ndarray] = []
    for shift_frac, gain_db in TTA_GRID:
        v = arr + float(gain_db)
        v = _shift_time(v, shift_frac)
        views.append(v.astype(np.float32, copy=False))
    return views


def tta_average_logits(run_batch, logmel_batch: np.ndarray) -> np.ndarray:
    """Average logits over TTA views, preserving the model's batch structure.

    ``run_batch`` maps an array shaped like ``logmel_batch`` to a ``(B, C)``
    logits array.
    """
    arr = np.asarray(logmel_batch, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]

    acc = None
    for view_idx in range(len(TTA_GRID)):
        batch_view = arr.copy()
        if arr.ndim == 4:
            for i in range(arr.shape[0]):
                batch_view[i, 0] = tta_views(arr[i, 0])[view_idx]
        elif arr.ndim == 5:
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
