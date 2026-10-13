"""TTA parity tests.

The training-side (``ml/src/features/tta.py``) and serving-side
(``apps/api/tbcare_api/inference/tta.py``) TTA implementations are two copies of
the same logic, because the serving package must not import the training package.
These tests guarantee the copies cannot drift: identical inputs must produce
bit-identical outputs.

They also pin the determinism property that makes TTA honest: the same input
always yields the same views (no RNG), so OOF metrics measured offline match
online behaviour.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_serving_tta():
    """Import the serving TTA module without requiring tbcare_api on sys.path."""
    root = Path(__file__).resolve().parents[2]
    api_dir = root / "apps" / "api"
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))
    return importlib.import_module("tbcare_api.inference.tta")


def test_grids_identical():
    from ml.src.features.tta import TTA_GRID as train_grid

    serving = _load_serving_tta()
    assert tuple(train_grid) == tuple(serving.TTA_GRID)


def test_views_bit_identical():
    from ml.src.features.tta import tta_views as train_views

    serving = _load_serving_tta()
    rng = np.random.default_rng(0)
    clip = rng.standard_normal((64, 64)).astype(np.float32)

    tr = train_views(clip)
    sv = serving.tta_views(clip)
    assert len(tr) == len(sv)
    for a, b in zip(tr, sv):
        assert a.shape == b.shape
        assert np.array_equal(a, b)


def test_identity_view_is_first_and_lossless():
    from ml.src.features.tta import tta_views

    rng = np.random.default_rng(1)
    clip = rng.standard_normal((64, 64)).astype(np.float32)
    views = tta_views(clip)
    # The first view must be the untouched clip (identity), so TTA can never be
    # worse than the base model beyond averaging noise.
    assert np.array_equal(views[0], clip)


def test_tta_is_deterministic():
    from ml.src.features.tta import tta_views

    rng = np.random.default_rng(2)
    clip = rng.standard_normal((64, 64)).astype(np.float32)
    first = tta_views(clip)
    second = tta_views(clip)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_tta_average_matches_manual_mean():
    from ml.src.features.tta import tta_average_logits, tta_views

    rng = np.random.default_rng(3)
    batch = rng.standard_normal((3, 1, 64, 64)).astype(np.float32)

    def fake_run(x):
        # Emulate a model returning, per sample, two logits both equal to the
        # mean over the (n_mels, T) plane. x is (N, 1, n_mels, T).
        per_sample = x[:, 0].reshape(x.shape[0], -1).mean(axis=1)  # (N,)
        return np.stack([per_sample, per_sample], axis=1)          # (N, 2)

    avg = tta_average_logits(fake_run, batch)

    manual = []
    for i in range(batch.shape[0]):
        per_view = [v.reshape(-1).mean() for v in tta_views(batch[i, 0])]
        manual.append([float(np.mean(per_view))] * 2)
    assert np.allclose(avg, np.asarray(manual), atol=1e-6)


def test_tta_average_handles_multiclip_5d():
    """The training layout (B, K, 1, n_mels, T) must work and preserve shape.

    This is the exact case that crashed the first export run: the audio tensor
    is 5-D (patients x clips), not 4-D.
    """
    from ml.src.features.tta import tta_average_logits, tta_views

    rng = np.random.default_rng(4)
    batch = rng.standard_normal((2, 3, 1, 64, 64)).astype(np.float32)
    seen_shapes = []

    def fake_run(x):
        seen_shapes.append(x.shape)
        # Aggregate over clips by summing, then produce 2 logits per patient.
        per_patient = x.reshape(x.shape[0], -1).mean(axis=1)
        return np.stack([per_patient, per_patient], axis=1)

    avg = tta_average_logits(fake_run, batch)
    assert avg.shape == (2, 2)
    # Every view must have been passed with the original 5-D shape.
    assert all(s == batch.shape for s in seen_shapes)
    assert len(seen_shapes) == 5


def test_tta_average_5d_matches_manual_mean():
    from ml.src.features.tta import tta_average_logits, tta_views

    rng = np.random.default_rng(5)
    batch = rng.standard_normal((2, 2, 1, 64, 64)).astype(np.float32)

    def fake_run(x):
        per_patient = x.reshape(x.shape[0], -1).mean(axis=1)
        return np.stack([per_patient, per_patient], axis=1)

    avg = tta_average_logits(fake_run, batch)

    manual = []
    for i in range(batch.shape[0]):
        per_view_means = []
        for view_idx in range(5):
            clip_means = [
                tta_views(batch[i, k, 0])[view_idx].reshape(-1).mean()
                for k in range(batch.shape[1])
            ]
            per_view_means.append(np.mean(clip_means))
        manual.append([float(np.mean(per_view_means))] * 2)
    assert np.allclose(avg, np.asarray(manual), atol=1e-6)
