from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


# =============================================================================
# Model Configurations
# =============================================================================

@dataclass(frozen=True)
class AudioCNNConfig:
    """Configuration for Residual Spectrogram CNN."""
    input_channels: int = 1
    base_channels: int = 32
    n_mels: int = 64
    target_frames: int = 64
    dropout: float = 0.3
    embedding_dim: int = 128


@dataclass(frozen=True)
class ClinicalEncoderConfig:
    """Configuration for Clinical Tabular Encoder."""
    input_dim: int = 26
    hidden_dims: Tuple[int, ...] = (128, 64)
    embedding_dim: int = 32
    dropout: float = 0.2


@dataclass(frozen=True)
class GatedFusionConfig:
    """Configuration for Gated Fusion (FiLM-style)."""
    audio_embedding_dim: int = 128
    clinical_embedding_dim: int = 32
    fused_dim: int = 128
    dropout: float = 0.2


@dataclass(frozen=True)
class SupervisorConfig:
    """Configuration for Supervisor (fusion) model."""
    input_dim: int = 2 + 26  # prob_audio + prob_clinical + clinical_features
    hidden_dim: int = 64
    dropout: float = 0.2


# =============================================================================
# Residual Blocks
# =============================================================================

class ResidualBlock2D(nn.Module):
    """2D Residual block with GroupNorm and Swish activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(min(groups, out_channels), out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(groups, out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.silu(out)


class ResidualBlock1D(nn.Module):
    """1D Residual block for clinical encoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        return F.silu(out)


# =============================================================================
# Audio Encoder: Residual CNN V3
# =============================================================================

class ResidualSpectrogramEncoder(nn.Module):
    """
    Residual CNN for log-mel spectrograms (V3 architecture).
    Input: (B, 1, n_mels, target_frames) -> Output: (B, embedding_dim)
    """

    def __init__(self, config: AudioCNNConfig):
        super().__init__()
        self.config = config

        c = config.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(),
        )

        # Stage 1: c -> c
        self.stage1 = nn.Sequential(
            ResidualBlock2D(c, c, stride=1),
            ResidualBlock2D(c, c, stride=1),
        )
        # Stage 2: c -> 2c (downsample)
        self.stage2 = nn.Sequential(
            ResidualBlock2D(c, 2 * c, stride=2),
            ResidualBlock2D(2 * c, 2 * c, stride=1),
        )
        # Stage 3: 2c -> 4c (downsample)
        self.stage3 = nn.Sequential(
            ResidualBlock2D(2 * c, 4 * c, stride=2),
            ResidualBlock2D(4 * c, 4 * c, stride=1),
        )
        # Stage 4: 4c -> 8c (downsample)
        self.stage4 = nn.Sequential(
            ResidualBlock2D(4 * c, 8 * c, stride=2),
            ResidualBlock2D(8 * c, 8 * c, stride=1),
        )

        # Global average pooling + projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(8 * c, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, target_frames)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)  # (B, 8*c)
        x = self.projection(x)  # (B, embedding_dim)
        x = self.dropout(x)
        return x


# =============================================================================
# Clinical Encoder: Tabular V4
# =============================================================================

class ClinicalTabularEncoder(nn.Module):
    """
    Wider, properly normalized clinical branch (V4 architecture).
    Input: (B, 38) -> Output: (B, embedding_dim)
    """

    def __init__(self, config: ClinicalEncoderConfig):
        super().__init__()
        self.config = config

        layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(ResidualBlock1D(prev_dim, hidden_dim, config.dropout))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Linear(prev_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 38)
        x = self.backbone(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


# =============================================================================
# Gated Fusion (FiLM-style)
# =============================================================================

class GatedFusion(nn.Module):
    """
    Audio-conditioned modulation of clinical embedding (FiLM-style).
    Implements: modulated = gamma * clinical + beta, where gamma, beta = f(audio)
    """

    def __init__(self, config: GatedFusionConfig):
        super().__init__()
        self.config = config

        # Conditioning network: audio_embedding -> (gamma, beta) for clinical
        self.condition = nn.Sequential(
            nn.Linear(config.audio_embedding_dim, config.fused_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fused_dim, 2 * config.clinical_embedding_dim),
        )

        # Joint processing after modulation
        self.joint = nn.Sequential(
            nn.Linear(config.audio_embedding_dim + config.clinical_embedding_dim, config.fused_dim),
            nn.LayerNorm(config.fused_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fused_dim, config.fused_dim),
            nn.LayerNorm(config.fused_dim),
            nn.SiLU(),
        )

    def forward(self, audio_emb: torch.Tensor, clinical_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_emb: (B, audio_embedding_dim)
            clinical_emb: (B, clinical_embedding_dim)
        Returns:
            fused: (B, fused_dim)
        """
        # Generate FiLM parameters from audio
        film_params = self.condition(audio_emb)  # (B, 2 * clinical_embedding_dim)
        gamma, beta = film_params.chunk(2, dim=-1)

        # Modulate clinical embedding
        modulated = gamma * clinical_emb + beta  # (B, clinical_embedding_dim)

        # Joint processing
        combined = torch.cat([audio_emb, modulated], dim=-1)  # (B, audio + clinical)
        fused = self.joint(combined)  # (B, fused_dim)

        return fused


# =============================================================================
# Full Fusion Classifier: ResidualFusionCNN V3
# =============================================================================

class ResidualFusionClassifier(nn.Module):
    """
    Complete fusion model: Audio CNN + Clinical Encoder + Gated Fusion + Classifier.
    Architecture version: RESIDUAL_FUSION_CNN_V3
    """

    ARCHITECTURE_VERSION = "residual_fusion_cnn_v3"

    def __init__(
        self,
        audio_config: Optional[AudioCNNConfig] = None,
        clinical_config: Optional[ClinicalEncoderConfig] = None,
        fusion_config: Optional[GatedFusionConfig] = None,
        num_classes: int = 2,
    ):
        super().__init__()
        self.audio_config = audio_config or AudioCNNConfig()
        self.clinical_config = clinical_config or ClinicalEncoderConfig()
        self.fusion_config = fusion_config or GatedFusionConfig()
        self.num_classes = num_classes

        # Submodules
        self.audio_encoder = ResidualSpectrogramEncoder(self.audio_config)
        self.clinical_encoder = ClinicalTabularEncoder(self.clinical_config)
        self.fusion = GatedFusion(self.fusion_config)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_config.fused_dim, self.fusion_config.fused_dim // 2),
            nn.LayerNorm(self.fusion_config.fused_dim // 2),
            nn.SiLU(),
            nn.Dropout(self.fusion_config.dropout),
            nn.Linear(self.fusion_config.fused_dim // 2, num_classes),
        )

        # For patient-level aggregation (multiple clips per patient)
        self.clip_aggregation = "mean_logprob"  # Options: mean_logprob, mean_prob, max_prob

    def forward(
        self,
        audio_clips: torch.Tensor,      # (B, n_clips, 1, n_mels, target_frames)
        clinical: torch.Tensor,         # (B, 38) or (B, n_clips, 38)
    ) -> torch.Tensor:
        """
        Forward pass with patient-level clip aggregation.

        Args:
            audio_clips: Batch of patients, each with n_clips spectrograms
            clinical: Clinical features (same for all clips of a patient, or per-clip)

        Returns:
            log_probs: (B, num_classes) - log probabilities per patient
        """
        batch_size, n_clips = audio_clips.shape[:2]

        # Flatten clips for batch processing
        audio_flat = audio_clips.view(batch_size * n_clips, *audio_clips.shape[2:])  # (B*n_clips, 1, n_mels, T)
        audio_emb_flat = self.audio_encoder(audio_flat)  # (B*n_clips, audio_emb_dim)

        # Handle clinical: either (B, 38) -> broadcast, or (B, n_clips, 38)
        if clinical.dim() == 2:
            clinical = clinical.unsqueeze(1).expand(-1, n_clips, -1)  # (B, n_clips, 38)
        clinical_flat = clinical.reshape(batch_size * n_clips, -1)  # (B*n_clips, clinical_dim)
        clinical_emb_flat = self.clinical_encoder(clinical_flat)  # (B*n_clips, clinical_emb_dim)

        # Fusion per clip
        fused_flat = self.fusion(audio_emb_flat, clinical_emb_flat)  # (B*n_clips, fused_dim)
        logits_flat = self.classifier(fused_flat)  # (B*n_clips, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)  # (B*n_clips, num_classes)

        # Patient-level aggregation. Only average over real clips: padded
        # clips are all zeros (data_loader.prepare_fold_data), and including
        # them would dilute the patient score for short recordings.
        log_probs = log_probs_flat.view(batch_size, n_clips, self.num_classes)
        clip_valid = (audio_clips.abs().sum(dim=(2, 3, 4)) > 0).float()  # (B, n_clips)
        denom = clip_valid.sum(dim=1, keepdim=True).clamp(min=1.0)

        if self.clip_aggregation == "mean_logprob":
            patient_log_probs = (
                (log_probs * clip_valid.unsqueeze(-1)).sum(dim=1) / denom
            )
        elif self.clip_aggregation == "mean_prob":
            probs = log_probs.exp()
            mean_prob = (probs * clip_valid.unsqueeze(-1)).sum(dim=1) / denom
            patient_log_probs = mean_prob.clamp(min=1e-12).log()
        elif self.clip_aggregation == "max_prob":
            probs = log_probs.exp()
            # Ignore padded clips when taking the max.
            masked = probs.masked_fill(clip_valid.unsqueeze(-1) == 0, -1.0)
            patient_log_probs = masked.max(dim=1).values.clamp(min=1e-12).log()
        else:
            patient_log_probs = (
                (log_probs * clip_valid.unsqueeze(-1)).sum(dim=1) / denom
            )

        return patient_log_probs

    def forward_single(
        self,
        audio: torch.Tensor,      # (B, 1, n_mels, target_frames)
        clinical: torch.Tensor,   # (B, 38)
    ) -> torch.Tensor:
        """Single clip forward (for ONNX export simplicity)."""
        audio_emb = self.audio_encoder(audio)
        clinical_emb = self.clinical_encoder(clinical)
        fused = self.fusion(audio_emb, clinical_emb)
        logits = self.classifier(fused)
        return F.log_softmax(logits, dim=-1)

    def get_embeddings(
        self,
        audio: torch.Tensor,
        clinical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get intermediate embeddings for analysis."""
        audio_emb = self.audio_encoder(audio)
        clinical_emb = self.clinical_encoder(clinical)
        fused = self.fusion(audio_emb, clinical_emb)
        return audio_emb, clinical_emb, fused


# =============================================================================
# Expert Models (for Stacked Ensemble)
# =============================================================================

class AudioOnlyClassifier(nn.Module):
    """Audio-only expert: Residual CNN -> Classifier."""

    ARCHITECTURE_VERSION = "residual_spectrogram_cnn_v3"

    def __init__(self, audio_config: Optional[AudioCNNConfig] = None, num_classes: int = 2):
        super().__init__()
        self.audio_config = audio_config or AudioCNNConfig()
        self.num_classes = num_classes

        self.audio_encoder = ResidualSpectrogramEncoder(self.audio_config)

        self.classifier = nn.Sequential(
            nn.Linear(self.audio_config.embedding_dim, self.audio_config.embedding_dim // 2),
            nn.LayerNorm(self.audio_config.embedding_dim // 2),
            nn.SiLU(),
            nn.Dropout(self.audio_config.dropout),
            nn.Linear(self.audio_config.embedding_dim // 2, num_classes),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, 1, n_mels, target_frames) or (B, n_clips, 1, n_mels, target_frames)"""
        if audio.dim() == 5:
            # Multi-clip: aggregate only over real clips. Padded clips are all
            # zeros (see data_loader.prepare_fold_data); averaging them in
            # would dilute the signal for patients with fewer than max_clips.
            batch_size, n_clips = audio.shape[:2]
            audio_flat = audio.view(batch_size * n_clips, *audio.shape[2:])
            audio_emb = self.audio_encoder(audio_flat)
            logits = self.classifier(audio_emb)
            log_probs = F.log_softmax(logits, dim=-1).view(batch_size, n_clips, self.num_classes)

            # A clip is real if it is not entirely zero.
            clip_valid = (audio.abs().sum(dim=(2, 3, 4)) > 0).float()  # (B, n_clips)
            denom = clip_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            masked = (log_probs * clip_valid.unsqueeze(-1)).sum(dim=1) / denom
            return masked
        else:
            audio_emb = self.audio_encoder(audio)
            logits = self.classifier(audio_emb)
            return F.log_softmax(logits, dim=-1)


class ClinicalOnlyClassifier(nn.Module):
    """Clinical-only expert: Tabular Encoder -> Classifier."""

    ARCHITECTURE_VERSION = "residual_tabular_clinical_v4"

    def __init__(self, clinical_config: Optional[ClinicalEncoderConfig] = None, num_classes: int = 2):
        super().__init__()
        self.clinical_config = clinical_config or ClinicalEncoderConfig()
        self.num_classes = num_classes

        self.clinical_encoder = ClinicalTabularEncoder(self.clinical_config)

        self.classifier = nn.Sequential(
            nn.Linear(self.clinical_config.embedding_dim, self.clinical_config.embedding_dim // 2),
            nn.LayerNorm(self.clinical_config.embedding_dim // 2),
            nn.SiLU(),
            nn.Dropout(self.clinical_config.dropout),
            nn.Linear(self.clinical_config.embedding_dim // 2, num_classes),
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        """clinical: (B, 38)"""
        clinical_emb = self.clinical_encoder(clinical)
        logits = self.classifier(clinical_emb)
        return F.log_softmax(logits, dim=-1)


# =============================================================================
# Model Builders (for Training Scripts)
# =============================================================================

def build_audio_model(config: Optional[AudioCNNConfig] = None) -> AudioOnlyClassifier:
    """Build audio-only expert model."""
    return AudioOnlyClassifier(config)


def build_clinical_model(config: Optional[ClinicalEncoderConfig] = None) -> ClinicalOnlyClassifier:
    """Build clinical-only expert model."""
    return ClinicalOnlyClassifier(config)


def build_fusion_model(
    audio_config: Optional[AudioCNNConfig] = None,
    clinical_config: Optional[ClinicalEncoderConfig] = None,
    fusion_config: Optional[GatedFusionConfig] = None,
) -> ResidualFusionClassifier:
    """Build fusion model."""
    return ResidualFusionClassifier(audio_config, clinical_config, fusion_config)


# =============================================================================
# ONNX Export Helpers
# =============================================================================

def export_audio_encoder_onnx(
    model: AudioOnlyClassifier,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 64, 64),
    opset: int = 17,
) -> None:
    """Export audio encoder to ONNX."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model.audio_encoder,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["logmel"],
        output_names=["embedding"],
        dynamic_axes={"logmel": {0: "batch"}, "embedding": {0: "batch"}},
        do_constant_folding=True,
    )


def export_fusion_onnx(
    model: ResidualFusionClassifier,
    output_path: str,
    audio_shape: Tuple[int, ...] = (1, 1, 64, 64),
    clinical_shape: Tuple[int, ...] = (1, 38),
    opset: int = 17,
) -> None:
    """Export fusion model (single clip) to ONNX."""
    model.eval()
    dummy_audio = torch.randn(*audio_shape)
    dummy_clinical = torch.randn(*clinical_shape)
    torch.onnx.export(
        model,
        (dummy_audio, dummy_clinical),
        output_path,
        opset_version=opset,
        input_names=["logmel", "clinical"],
        output_names=["log_probs"],
        dynamic_axes={
            "logmel": {0: "batch"},
            "clinical": {0: "batch"},
            "log_probs": {0: "batch"},
        },
        do_constant_folding=True,
    )


def export_head_weights(model: ResidualFusionClassifier, output_path: str) -> None:
    """Export classifier head weights for manual inference (npz)."""
    # Get the classifier head state dict
    head_state = model.classifier.state_dict()
    npz_dict = {k: v.cpu().numpy() for k, v in head_state.items()}
    np.savez(output_path, **npz_dict)


# =============================================================================
# Model Loading (for Serving)
# =============================================================================

def load_fusion_model(
    checkpoint_path: str,
    audio_config: Optional[AudioCNNConfig] = None,
    clinical_config: Optional[ClinicalEncoderConfig] = None,
    fusion_config: Optional[GatedFusionConfig] = None,
    device: str = "cpu",
) -> ResidualFusionClassifier:
    """Load fusion model from checkpoint."""
    model = ResidualFusionClassifier(audio_config, clinical_config, fusion_config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device)
    model.eval()
    return model