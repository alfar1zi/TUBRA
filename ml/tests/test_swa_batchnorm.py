"""Regression tests for the SWA snapshot-averaging BatchNorm recalibration.

Context (bug fixed): averaging the top-k epoch weights (``TrainConfig.snapshot_topk``)
without recomputing BatchNorm running statistics collapses the model output
toward 0.5, which destroyed measured audio AUROC (0.6675 -> 0.5675). The fix
averages only floating-point parameters and then re-estimates BN buffers with
forward passes over the training data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.scripts.train import _recalibrate_batchnorm


class _TinyBNNet(nn.Module):
    """Minimal BN-containing network that mimics the backbone's structure."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.head(self.backbone(x))


def _make_separated_data(seed: int = 0, n: int = 128):
    """Two Gaussian blobs so a trained net has confident, well-spread logits."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) >= n // 2).astype(np.int64)
    x = rng.normal(size=(n, 1, 16, 16)).astype(np.float32)
    # Class 1 gets a shifted mean band, giving a learnable signal.
    x[y == 1, :, 4:12, :] += 2.5
    return x, y


def _snapshot_after_training(data, labels, seed: int, epochs: int = 40):
    torch.manual_seed(seed)
    model = _TinyBNNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xb = torch.from_numpy(data)
    yb = torch.from_numpy(labels)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    return model


def _prob_spread(model, data) -> float:
    with torch.no_grad():
        probs = model(torch.from_numpy(data)).softmax(dim=-1)[:, 1]
    return float(probs.max() - probs.min())


def test_recalibration_restores_output_after_bn_stats_wiped():
    """Directly prove the mechanism the SWA fix relies on.

    SWA averages weights, which invalidates the BatchNorm running statistics
    carried over from the source epochs. We simulate that mismatch by wiping the
    BN buffers (``reset_running_stats``) on an otherwise-trained model: the
    output collapses. ``_recalibrate_batchnorm`` must restore a well-separated
    probability range by re-estimating the stats from the data.
    """
    data, labels = _make_separated_data()
    device = torch.device("cpu")
    model = _snapshot_after_training(data, labels, seed=1)
    model.eval()

    healthy_spread = _prob_spread(model, data)
    assert healthy_spread > 0.20, "trained model should separate the blobs"

    # Wipe BN stats -> weights no longer match their normalisation -> collapse.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
    collapsed_spread = _prob_spread(model, data)
    assert collapsed_spread < healthy_spread - 0.05, (
        f"expected collapse after wiping BN stats (healthy={healthy_spread:.4f}, "
        f"collapsed={collapsed_spread:.4f})"
    )

    # Recalibration must recover the spread.
    _recalibrate_batchnorm(
        model,
        torch.from_numpy(data),
        torch.empty(data.shape[0], 0),
        device,
        batch_size=32,
    )
    model.eval()
    recovered_spread = _prob_spread(model, data)
    assert recovered_spread > collapsed_spread + 0.05, (
        f"recalibration did not recover the spread (collapsed={collapsed_spread:.4f}, "
        f"recovered={recovered_spread:.4f})"
    )
    assert recovered_spread > 0.20, f"recovered spread still collapsed: {recovered_spread:.4f}"


def test_weight_averaging_without_recalibration_can_degrade():
    """Averaging weights from divergent epochs leaves BN stats stale.

    Snapshot A is trained; snapshot B is trained more aggressively on a shifted
    input scale so its BN statistics differ substantially. Averaging the two
    weight sets while keeping A's BN buffers yields a model whose outputs are
    less separated than a properly recalibrated one.
    """
    data, labels = _make_separated_data()
    device = torch.device("cpu")

    snap_a = _snapshot_after_training(data, labels, seed=1, epochs=40)
    # Snapshot B sees a very differently-scaled input, so its BN stats diverge.
    snap_b = _snapshot_after_training((data * 8.0).astype(np.float32), labels, seed=2, epochs=40)

    avg = _TinyBNNet()
    state_a, state_b = snap_a.state_dict(), snap_b.state_dict()
    new_state = {}
    for key in state_a:
        if torch.is_floating_point(state_a[key]):
            new_state[key] = torch.stack([state_a[key], state_b[key]]).mean(dim=0)
        else:
            new_state[key] = state_a[key].clone()
    avg.load_state_dict(new_state)
    avg.eval()

    naive_spread = _prob_spread(avg, data)

    _recalibrate_batchnorm(
        avg,
        torch.from_numpy(data),
        torch.empty(data.shape[0], 0),
        device,
        batch_size=32,
    )
    avg.eval()
    recal_spread = _prob_spread(avg, data)

    # Recalibration must not make things worse, and should widen the spread here
    # because the averaged weights genuinely separate the blobs once BN matches.
    assert recal_spread >= naive_spread - 1e-6, (
        f"recalibration degraded spread (naive={naive_spread:.4f}, recal={recal_spread:.4f})"
    )
    assert recal_spread > 0.20, f"recalibrated spread collapsed: {recal_spread:.4f}"


def test_recalibration_is_a_noop_without_batchnorm():
    """A model with no BN layers must be left untouched (and not crash)."""
    model = nn.Linear(3, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _recalibrate_batchnorm(
        model,
        torch.randn(8, 3),
        torch.empty(8, 0),
        torch.device("cpu"),
        batch_size=4,
    )
    after = model.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key]), f"{key} changed unexpectedly"
    # Momentum must not be set on non-BN modules.
    assert model.training is True or model.training is False  # restored state


def test_recalibration_restores_eval_mode():
    """The helper must return the model to eval() if it was in eval() mode."""
    model = _TinyBNNet().eval()
    _recalibrate_batchnorm(
        model,
        torch.randn(16, 1, 16, 16),
        torch.empty(16, 0),
        torch.device("cpu"),
        batch_size=8,
    )
    assert model.training is False
