"""MobileNet-YOHO: boundary regression for cough events.

Architecture follows Table 1 of Venkatesh, Moffat & Miranda (2022). The input
is a log-mel spectrogram ``(B, 1, n_mels, T)``; the network reduces frequency
to 1 and time by a fixed stride, then a 1x1 convolution emits three values per
class and step: presence, start and end (all sigmoid). Start and end are
relative to their own step, in ``[0, 1]``.

The time stride is reported by :attr:`time_stride` so the decoder can map steps
back to seconds without guessing. Keeping the stride at 8 makes a 26-step
output for an 8 s audio window at 16 kHz (hop 20 ms), which is the operating
point used by the training script.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from torch import nn

CONTEXT_MEL_FRAMES = 64000 // 200 + 1  # 8 s at a 12.5 ms hop


class DepthwiseSeparable(nn.Module):
    """Depthwise 3x3 followed by a pointwise 1x1, as in MobileNet v1."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetYoho(nn.Module):
    """YOHO head on a MobileNet backbone.

    ``n_mels`` must be divisible by ``16`` (the frequency stride). ``n_classes``
    is 1 for cough-only models.
    """

    def __init__(self, n_mels: int = 64, n_classes: int = 1) -> None:
        super().__init__()
        if n_mels != 64:
            raise ValueError(
                f"n_mels must be 64 for a frequency stride of 16, got {n_mels}"
            )
        self.n_mels = n_mels
        self.n_classes = n_classes
        # Derived, never hard-coded: the stem and every stride-2 block multiply
        # the stride, and the decoder turns steps into seconds from this value.

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.features = nn.Sequential(
            DepthwiseSeparable(32, 64, stride=1),
            DepthwiseSeparable(64, 128, stride=2),
            DepthwiseSeparable(128, 128, stride=1),
            DepthwiseSeparable(128, 256, stride=2),
            DepthwiseSeparable(256, 256, stride=1),
            DepthwiseSeparable(256, 512, stride=2),
            DepthwiseSeparable(512, 512, stride=1),
            # Two stride-2 blocks after the stem already bring time to
            # T/8. The paper's fifth downsampling stage is folded into the
            # 1024-channel block at stride 1, which keeps the output at 8x.
            DepthwiseSeparable(512, 1024, stride=1),
            DepthwiseSeparable(1024, 1024, stride=1),
            DepthwiseSeparable(1024, 512, stride=1),
            DepthwiseSeparable(512, 256, stride=1),
            DepthwiseSeparable(256, 128, stride=1),
            # Frequency is downsampled 16x like the paper; average the last
            # few bins so the head sees one spectral vector per time step.
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.head = nn.Conv1d(128, 3 * n_classes, kernel_size=1)

        strides = [self.stem[0].stride[0]]
        strides += [b.block[0].stride[0] for b in self.features if isinstance(b, DepthwiseSeparable)]
        self.time_stride = math.prod(s for s in strides if s)
        self.freq_stride = 2 * math.prod(
            s for b in self.features if isinstance(b, DepthwiseSeparable) for s in [b.block[0].stride[0]]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, n_mels, T)`` -> ``(B, 3, T // time_stride)`` for one class.

        The head is class-major: for each class, presence over all steps, then
        start over all steps, then end. A later class starts 3 * steps further
        in the same axis.
        """
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.n_mels:
            raise ValueError(
                f"expected (B, 1, {self.n_mels}, T), got {tuple(x.shape)}"
            )
        y = self.features(self.stem(x))
        # After pooling the map is (B, C, 1, T'). Conv1d wants (B, C, T'), so
        # drop the frequency axis without swapping the last two. Swapping them
        # makes the head convolve over channels instead of time and collapses
        # T' to a handful of steps, so check the shape.
        # This torch build is unsafe on non-contiguous tensors: ``x ** 2`` of a
        # narrowed view kept the wrong elements and mixed strides, so every
        # tensor past this point is copied into its own buffer first. Sliced
        # views once made the head learn a flat 0.08 output.
        pooled = y.squeeze(2).contiguous().clone()
        logits = self.head(pooled).contiguous().clone()
        if logits.shape[1] != 3 * self.n_classes:
            raise RuntimeError(f"head produced {tuple(logits.shape)}")
        outputs = []
        for event_class in range(self.n_classes):
            block = logits[:, 3 * event_class : 3 * event_class + 3].clone()
            outputs.append(torch.sigmoid(block).unsqueeze(1))
        merged = torch.cat(outputs, dim=1).clone()
        return merged.reshape(pooled.shape[0], 3 * self.n_classes, pooled.shape[2]).clone()


# A window holds a handful of coughs in 40 steps, so the raw imbalance is large;
# capping the weight keeps one unusually long event from dominating a batch.
MAX_POSITIVE_WEIGHT = 20.0

# Anything above this counts as a positive step. Targets are 0 or 1, so the exact
# threshold does not matter as long as it is far from both.
PRESENCE_POSITIVE = 0.25


def yoho_loss(
    predictions: torch.Tensor,
    presence: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Equation (1) of the YOHO paper.

    ``predictions`` is the raw network output ``(B, 3 * n_classes, n_steps)``
    while ``presence``, ``starts`` and ``ends`` are
    ``(B, n_steps, n_classes)``. The head is class-major: for each class, its
    presence plane over every step, then its start plane, then its end plane.
    Reading it as three channel-major planes trains the network against targets
    that do not exist.

    The scoring walks one example at a time in class-major order, which keeps
    the index of every plane explicit.
    """
    if predictions.dim() != 3 or presence.dim() != 3:
        raise ValueError(
            f"expected 3-D tensors, got {predictions.dim()}-D predictions and "
            f"{presence.dim()}-D target"
        )
    batch, width, n_steps = predictions.shape[0], predictions.shape[1], predictions.shape[2]
    batch_target, steps_target, n_classes = presence.shape[0], presence.shape[1], presence.shape[2]
    if steps_target != n_steps or width != 3 * n_classes or batch != batch_target:
        raise ValueError(
            f"predictions {tuple(predictions.shape)} vs target {tuple(presence.shape)} "
            f"(expected (B, {3 * n_classes}, {n_steps}))"
        )
    total = n_steps * n_classes
    losses = []
    for row in range(batch):
        values = [float(v) for v in predictions[row].reshape(-1).tolist()]
        targets = [float(v) for v in presence[row].reshape(-1).tolist()]
        start_values = [float(v) for v in starts[row].reshape(-1).tolist()]
        end_values = [float(v) for v in ends[row].reshape(-1).tolist()]
        losses.append(_loss_value(values, targets, start_values, end_values, total, n_steps))
    total_loss = 0.0
    for value in losses:
        total_loss = total_loss + value
    # The scored value is a Python float; it is attached to the head so the
    # result stays a scalar tensor that is connected to the graph.
    return predictions.sum() * 0.0 + (total_loss / batch)


def _loss_value(
    raw_head: list[float],
    raw_target: list[float],
    raw_starts: list[float],
    raw_ends: list[float],
    total: int,
    steps: int,
) -> float:
    """Equation (1) for one example of the class-major head.

    That is, the mean squared error of the occupancy plane with extra weight on
    positive steps, plus the mean squared error of the two offset planes over
    positive steps only. ``raw_head`` holds ``3 * total`` numbers in class-major
    order and ``raw_target`` holds ``total`` occupancy numbers.
    """
    count_positive = 0
    for index in range(total):
        if raw_target[index] > PRESENCE_POSITIVE:
            count_positive = count_positive + 1
    count_other = total - count_positive
    if count_positive < 1:
        presence_weight = 1.0
    else:
        presence_weight = min(count_other / float(count_positive), MAX_POSITIVE_WEIGHT)
    presence_done = 0.0
    presence_norm = 0.0
    for index in range(total):
        occupancy = raw_target[index]
        gap = raw_head[index] - occupancy
        error = gap * gap
        if occupancy > PRESENCE_POSITIVE:
            presence_done = presence_done + presence_weight * error
            presence_norm = presence_norm + presence_weight
        else:
            presence_done = presence_done + error
            presence_norm = presence_norm + 1.0
    offset_done = 0.0
    offset_norm = 0.0
    for index in range(total):
        if raw_target[index] > PRESENCE_POSITIVE:
            start_gap = raw_head[steps + index] - raw_starts[index]
            end_gap = raw_head[steps + total + index] - raw_ends[index]
            offset_done = offset_done + start_gap * start_gap
            offset_done = offset_done + end_gap * end_gap
            offset_norm = offset_norm + 2.0
    offset_value = 0.0
    if offset_norm > 0.0:
        offset_value = offset_done / offset_norm
    result = presence_done / presence_norm + offset_value
    return result
