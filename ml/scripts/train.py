#!/usr/bin/env python3
"""
Nested cross-validation training pipeline for TBCare.
Trains audio-only, clinical-only, and fusion models with participant-level splits.

Usage:
    python ml/scripts/train.py --config ml/configs/train.yaml
    python ml/scripts/train.py --coda-root ml/data/coda_tb --output ml/artifacts
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import os as _os

# Torch's default intra-op thread count does not scale with the core count, and
# on a machine shared with other applications letting torch grab every core is
# *slower* than a moderate count (measured here: 4 threads ~130s/epoch vs 8
# threads ~180s/epoch under contention). Default to 4 and allow an override via
# TBCARE_TORCH_THREADS for a dedicated machine.
torch.set_num_threads(int(_os.environ.get("TBCARE_TORCH_THREADS", "4")))
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.src.features.audio import AudioFeatureConfig, SpecAugmentConfig, spec_augment
from ml.src.features.tta import tta_average_logits
from ml.src.features.clinical import ClinicalPreprocessor, CLINICAL_FEATURE_ORDER
from ml.src.models.residual_fusion import (
    AudioCNNConfig,
    ClinicalEncoderConfig,
    GatedFusionConfig,
    build_audio_model,
    build_clinical_model,
    build_fusion_model,
    AudioOnlyClassifier,
    ClinicalOnlyClassifier,
    ResidualFusionClassifier,
    export_audio_encoder_onnx,
    export_fusion_onnx,
    export_head_weights,
)
from ml.scripts.data_loader import (
    PatientExample,
    load_coda_tb_patients,
    load_tbscreen_patients,
    create_nested_cv_splits,
    extract_patient_features,
    compute_dataset_hash,
)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    # Data
    coda_root: str = "ml/data/coda_tb"
    tbscreen_root: Optional[str] = None
    output_dir: str = "ml/artifacts"

    # Audio
    audio_recipe: str = "energy_v2"
    n_mels: int = 64
    target_frames: int = 64
    max_clips: int = 5

    # CV
    outer_folds: int = 5
    inner_folds: int = 3
    random_seed: int = 42
    val_size: float = 0.1

    # Training
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10  # Early stopping patience
    min_delta: float = 0.001  # Minimum improvement for early stopping

    # Model
    base_channels: int = 32
    audio_embedding_dim: int = 128
    clinical_hidden_dims: Tuple[int, ...] = (128, 64)
    clinical_embedding_dim: int = 32
    fused_dim: int = 128
    dropout: float = 0.3

    # Augmentation
    use_spec_augment: bool = True
    time_mask_param: int = 8
    freq_mask_param: int = 8

    # Regularisation / robustness (research-backed, near-free)
    label_smoothing: float = 0.0  # 0 disables; 0.05 recommended
    snapshot_topk: int = 0  # >0 averages the top-k epoch weights (SWA-style)
    # Seed-ensemble: >1 trains N independent models (seeds = random_seed..+N-1)
    # and averages their probabilities. Serving mirrors this by averaging the
    # exported ``backbone_seed*.onnx`` logits.
    ensemble_seeds: int = 1

    # Bootstrap CI
    n_bootstrap: int = 1000
    ci_level: float = 0.95

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("training", {}))

    def to_audio_config(self) -> AudioFeatureConfig:
        return AudioFeatureConfig(
            recipe=self.audio_recipe,
            n_mels=self.n_mels,
            target_frames=self.target_frames,
        )

    def to_audio_cnn_config(self) -> AudioCNNConfig:
        return AudioCNNConfig(
            base_channels=self.base_channels,
            n_mels=self.n_mels,
            target_frames=self.target_frames,
            dropout=self.dropout,
            embedding_dim=self.audio_embedding_dim,
        )

    def to_clinical_config(self) -> ClinicalEncoderConfig:
        return ClinicalEncoderConfig(
            input_dim=len(CLINICAL_FEATURE_ORDER),
            hidden_dims=self.clinical_hidden_dims,
            embedding_dim=self.clinical_embedding_dim,
            dropout=self.dropout,
        )

    def to_fusion_config(self) -> GatedFusionConfig:
        return GatedFusionConfig(
            audio_embedding_dim=self.audio_embedding_dim,
            clinical_embedding_dim=self.clinical_embedding_dim,
            fused_dim=self.fused_dim,
            dropout=self.dropout,
        )


# =============================================================================
# Deterministic Seeding
# =============================================================================

def set_seed(seed: int) -> None:
    """Seed every RNG and request deterministic kernels.

    CPU float reductions can vary slightly between runs when the intra-op
    thread count or scheduling differs, which made single-fold AUROC drift by
    up to ~0.04 in practice. ``use_deterministic_algorithms(True)`` plus a
    fixed thread count makes repeated runs reproducible, so reported CV
    numbers describe the model rather than the machine's thread scheduling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# =============================================================================
# Training Loop
# =============================================================================

def _class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute balanced class weights."""
    labels_int = np.asarray(labels).astype(np.int64).ravel()
    counts = np.bincount(labels_int, minlength=2).astype(np.float32)
    weights = len(labels_int) / (2.0 * counts.clip(min=1))
    return torch.tensor(weights, dtype=torch.float32)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    audio_batch: torch.Tensor,
    clinical_batch: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    device: torch.device,
    spec_aug_config: Optional[SpecAugmentConfig] = None,
    batch_size: int = 16,
    label_smoothing: float = 0.0,
) -> float:
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_patients = audio_batch.shape[0]

    # Shuffle patients
    perm = torch.randperm(n_patients)
    audio_batch = audio_batch[perm]
    clinical_batch = clinical_batch[perm]
    labels = labels[perm]

    # Mini-batch training
    batch_size = min(int(batch_size), n_patients)
    n_batches = (n_patients + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_patients)

        audio_b = audio_batch[start:end].to(device)
        clinical_b = clinical_batch[start:end].to(device)
        labels_b = labels[start:end].to(device)

        # Apply SpecAugment if configured
        if spec_aug_config is not None:
            audio_np = audio_b.cpu().numpy()
            for j in range(audio_np.shape[0]):
                for k in range(audio_np.shape[1]):
                    if audio_np[j, k].sum() != 0:  # Skip padding
                        audio_np[j, k] = spec_augment(audio_np[j, k], spec_aug_config, seed=config.random_seed)
            audio_b = torch.from_numpy(audio_np).to(device)

        optimizer.zero_grad()

        if isinstance(model, ResidualFusionClassifier):
            log_probs = model(audio_b, clinical_b)
        elif isinstance(model, AudioOnlyClassifier):
            log_probs = model(audio_b)
        elif isinstance(model, ClinicalOnlyClassifier):
            log_probs = model(clinical_b)
        else:
            raise TypeError(f"Unknown model type: {type(model)}")

        if label_smoothing > 0.0:
            # Smoothed one-hot target; weighted per-sample like nll_loss.
            tgt = torch.full_like(log_probs, label_smoothing / 2.0)
            tgt.scatter_(1, labels_b.unsqueeze(1), 1.0 - label_smoothing / 2.0)
            per_sample = -(tgt * log_probs).sum(dim=1)
            loss = (per_sample * class_weights.to(device)[labels_b]).mean()
        else:
            loss = F.nll_loss(log_probs, labels_b, weight=class_weights.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * (end - start)

    return total_loss / n_patients


def evaluate_model(
    model: nn.Module,
    audio_batch: torch.Tensor,
    clinical_batch: torch.Tensor,
    labels: np.ndarray,
    device: torch.device,
    use_tta: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (auroc, labels, probabilities).

    When ``use_tta`` is True and the model consumes audio, logits are averaged
    over the deterministic TTA views from ``ml.src.features.tta``. Serving
    mirrors this exactly (see ``apps/api/tbcare_api/inference/tta.py``), so the
    reported OOF metric matches online behaviour.
    """
    model.eval()
    with torch.no_grad():
        audio_b = audio_batch.to(device)
        clinical_b = clinical_batch.to(device)

        if isinstance(model, ClinicalOnlyClassifier):
            logits = model(clinical_b)
        elif use_tta and isinstance(model, AudioOnlyClassifier):
            arr = audio_batch.cpu().numpy().astype(np.float32)
            if arr.ndim == 4:
                arr = arr[:, None, :, :]

            def _run(batch_view: np.ndarray) -> np.ndarray:
                out = model(torch.from_numpy(batch_view).float().to(device))
                return out.cpu().numpy()

            logits = torch.from_numpy(tta_average_logits(_run, arr)).to(device)
        elif isinstance(model, ResidualFusionClassifier):
            logits = model(audio_b, clinical_b)
        elif isinstance(model, AudioOnlyClassifier):
            logits = model(audio_b)
        else:
            raise TypeError(f"Unknown model type: {type(model)}")

        probs = logits.softmax(dim=-1).cpu().numpy()  # (n, 2)
        prob_tb = probs[:, 1]

    try:
        auroc = roc_auc_score(labels, prob_tb)
    except ValueError:
        auroc = 0.5  # All same class

    return auroc, labels, prob_tb


def _recalibrate_batchnorm(
    model: nn.Module,
    audio: torch.Tensor,
    clinical: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_batches: int = 64,
) -> None:
    """Recompute BatchNorm running statistics for the given (averaged) weights.

    Standard SWA requires this step: after averaging the learnable weights, the
    BN ``running_mean``/``running_var`` buffers must be re-estimated by running
    forward passes over the training data with BN in training mode (so the
    running buffers update) but without tracking gradients.

    We first reset every BN buffer so the estimate does not depend on whatever
    state the last snapshot left behind, then run up to ``max_batches`` forward
    passes. ``momentum=None`` makes BN use a cumulative moving average.
    """
    n = audio.shape[0]
    if n == 0:
        return

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None  # cumulative moving average

    was_training = model.training
    model.train()
    with torch.no_grad():
        order = torch.randperm(n)
        batches = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            a = audio[idx].to(device)
            c = clinical[idx].to(device)
            if isinstance(model, ClinicalOnlyClassifier):
                model(c)
            elif isinstance(model, ResidualFusionClassifier):
                model(a, c)
            else:
                model(a)
            batches += 1
            if batches >= max_batches:
                break
    if not was_training:
        model.eval()


def train_model(
    model: nn.Module,
    train_audio: np.ndarray,
    train_clinical: np.ndarray,
    train_labels: np.ndarray,
    val_audio: np.ndarray,
    val_clinical: np.ndarray,
    val_labels: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], int, float]:
    """
    Train model with early stopping on validation AUROC.

    ``seed`` (default: ``config.random_seed``) controls weight init, batch
    shuffling and SpecAugment randomness, enabling reproducible seed-ensembles.

    Returns:
        (best_state_dict, best_epoch, best_auroc)
    """
    if seed is not None:
        set_seed(int(seed))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    class_weights = _class_weights(train_labels)

    # Convert to tensors
    train_audio_t = torch.from_numpy(train_audio).float()
    train_clinical_t = torch.from_numpy(train_clinical).float()
    train_labels_t = torch.from_numpy(train_labels).long()
    val_audio_t = torch.from_numpy(val_audio).float()
    val_clinical_t = torch.from_numpy(val_clinical).float()

    spec_aug = SpecAugmentConfig(
        time_mask_param=config.time_mask_param,
        freq_mask_param=config.freq_mask_param,
    ) if config.use_spec_augment else None

    best_auroc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    snapshots: list[tuple[float, Dict[str, torch.Tensor]]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_audio_t, train_clinical_t, train_labels_t,
            class_weights, device, spec_aug, batch_size=config.batch_size,
            label_smoothing=config.label_smoothing,
        )
        scheduler.step()

        val_auroc, _, _ = evaluate_model(model, val_audio_t, val_clinical_t, val_labels, device)
        epoch_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        snapshots.append((val_auroc, epoch_state))

        if val_auroc > best_auroc + config.min_delta:
            best_auroc = val_auroc
            best_state = epoch_state
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch}, AUROC: {best_auroc:.4f})")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = config.epochs

    # SWA-style snapshot averaging: average the weights of the top-k epochs.
    # Averaging weights (not predictions) yields one model, so the ONNX export
    # stays a single graph with no extra serving cost.
    #
    # BatchNorm caveat: averaged weights are NOT compatible with the per-epoch
    # running statistics. Averaging BN ``running_mean``/``running_var`` across
    # epochs (or leaving them as-is) collapses the classifier output toward 0.5.
    # The correct SWA recipe is: (1) average only the learnable/float parameters,
    # (2) reset BN buffers and recompute them with forward passes over the
    # training data. Without step (2) the model is near-degenerate.
    k = int(getattr(config, "snapshot_topk", 0) or 0)
    if k > 1 and len(snapshots) >= k:
        top = sorted(snapshots, key=lambda s: -s[0])[:k]
        keys = top[0][1].keys()
        averaged: Dict[str, torch.Tensor] = {}
        for key in keys:
            ref = top[0][1][key]
            if torch.is_floating_point(ref):
                averaged[key] = torch.stack([t[1][key].float() for t in top]).mean(dim=0)
            else:
                # Integer buffers (e.g. ``num_batches_tracked``) are not averaged;
                # keep the latest snapshot's value.
                averaged[key] = ref.clone()
        best_state = averaged

        # Step (2): recalibrate BatchNorm running stats for the averaged weights.
        model.load_state_dict(best_state)
        _recalibrate_batchnorm(model, train_audio_t, train_clinical_t, device, config.batch_size)
        best_state = {kk: vv.cpu().clone() for kk, vv in model.state_dict().items()}

    return best_state, best_epoch, best_auroc


# =============================================================================
# Bootstrap CI
# =============================================================================

def bootstrap_auroc_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: List[str],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap CI for AUROC, resampling at patient level.

    Returns:
        {"auroc": float, "ci_lower": float, "ci_upper": float}
    """
    rng = np.random.RandomState(seed)

    # Group by patient
    unique_patients = sorted(set(patient_ids))
    patient_to_idx = {}
    for i, pid in enumerate(patient_ids):
        patient_to_idx.setdefault(pid, []).append(i)

    bootstrap_aurocs = []
    for _ in range(n_bootstrap):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sampled_idx = []
        for pid in sampled_patients:
            sampled_idx.extend(patient_to_idx[pid])

        sampled_labels = labels[sampled_idx]
        sampled_probs = probabilities[sampled_idx]

        # Skip if all same class
        if len(np.unique(sampled_labels)) < 2:
            continue

        try:
            auroc = roc_auc_score(sampled_labels, sampled_probs)
            bootstrap_aurocs.append(auroc)
        except ValueError:
            continue

    if not bootstrap_aurocs:
        return {"auroc": 0.5, "ci_lower": 0.0, "ci_upper": 1.0}

    alpha = 1.0 - ci_level
    lower = np.percentile(bootstrap_aurocs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_aurocs, 100 * (1 - alpha / 2))

    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


# =============================================================================
# Nested CV Runner
# =============================================================================

def run_nested_cv(
    patients: List[PatientExample],
    config: TrainConfig,
) -> Dict[str, Any]:
    """
    Run nested cross-validation for audio, clinical, and fusion models.

    Returns complete CV results dict.
    """
    set_seed(config.random_seed)
    device = _resolve_device()

    audio_config = config.to_audio_config()
    clinical_preprocessor = ClinicalPreprocessor()

    # Create nested splits
    print(f"Creating {config.outer_folds}x{config.inner_folds} nested CV splits...")
    nested_folds = create_nested_cv_splits(
        patients, config.outer_folds, config.inner_folds, config.random_seed
    )

    # Verify no leakage
    _verify_no_leakage(nested_folds)

    all_results = {
        "audio_only": {"fold_aurocs": [], "oof_labels": [], "oof_probs": [], "oof_ids": []},
        "clinical_only": {"fold_aurocs": [], "oof_labels": [], "oof_probs": [], "oof_ids": []},
        "fusion": {"fold_aurocs": [], "oof_labels": [], "oof_probs": [], "oof_ids": []},
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for outer in nested_folds:
        fold_idx = outer["outer_fold"]
        test_patients = outer["test"]
        print(f"\n=== Outer Fold {fold_idx + 1}/{config.outer_folds} ===")
        print(f"  Test: {len(test_patients)} patients ({sum(p.label for p in test_patients)} TB+)")

        # Use best inner fold for training
        best_inner_auroc = 0.0
        best_inner = None

        for inner in outer["inner_folds"]:
            train_patients = inner["train"]
            val_patients = inner["val"]

            # Fit clinical preprocessor on train
            clinical_preprocessor.fit([p.clinical_features for p in train_patients])

            # Extract features
            train_audio, train_clinical, train_labels, _ = _extract_features_batch(
                train_patients, audio_config, clinical_preprocessor
            )
            val_audio, val_clinical, val_labels, _ = _extract_features_batch(
                val_patients, audio_config, clinical_preprocessor
            )

            # Train fusion model (quick inner loop for hyperparameter selection)
            inner_model = build_fusion_model(
                config.to_audio_cnn_config(),
                config.to_clinical_config(),
                config.to_fusion_config(),
            )

            _, _, inner_auroc = train_model(
                inner_model, train_audio, train_clinical, train_labels,
                val_audio, val_clinical, val_labels, config, device,
            )

            if inner_auroc > best_inner_auroc:
                best_inner_auroc = inner_auroc
                best_inner = inner

        # Retrain on best inner fold's train+val, evaluate on outer test
        all_train_patients = best_inner["train"] + best_inner["val"]
        clinical_preprocessor.fit([p.clinical_features for p in all_train_patients])

        train_audio, train_clinical, train_labels, train_ids = _extract_features_batch(
            all_train_patients, audio_config, clinical_preprocessor
        )
        test_audio, test_clinical, test_labels, test_ids = _extract_features_batch(
            test_patients, audio_config, clinical_preprocessor
        )

        # Split a validation fold for early stopping. Use a stratified shuffle
        # over labels, not the tail slice: patients are ordered by dataset
        # (CODA then TBscreen), so train[-n:] concentrated one dataset and one
        # class into validation and biased model selection.
        from sklearn.model_selection import train_test_split as _stratified_split

        indices = np.arange(len(all_train_patients))
        try:
            _, val_idx = _stratified_split(
                indices,
                test_size=config.val_size,
                random_state=config.random_seed,
                stratify=train_labels,
            )
        except ValueError:
            # Too few samples per class to stratify; fall back to a shuffle.
            _, val_idx = _stratified_split(
                indices,
                test_size=config.val_size,
                random_state=config.random_seed,
            )
        val_idx = np.sort(np.asarray(val_idx))
        train_mask = np.ones(len(indices), dtype=bool)
        train_mask[val_idx] = False

        val_audio = train_audio[val_idx]
        val_clinical = train_clinical[val_idx]
        val_labels = train_labels[val_idx]
        train_audio = train_audio[train_mask]
        train_clinical = train_clinical[train_mask]
        train_labels = train_labels[train_mask]

        # Train all three model types
        for model_name in ("audio_only", "clinical_only", "fusion"):
            print(f"  Training {model_name}...")

            if model_name == "audio_only":
                model = build_audio_model(config.to_audio_cnn_config())
            elif model_name == "clinical_only":
                model = build_clinical_model(config.to_clinical_config())
            else:
                model = build_fusion_model(
                    config.to_audio_cnn_config(),
                    config.to_clinical_config(),
                    config.to_fusion_config(),
                )

            best_state, best_epoch, best_auroc = train_model(
                model, train_audio, train_clinical, train_labels,
                val_audio, val_clinical, val_labels, config, device,
            )

            # Load best state and evaluate on test
            model.load_state_dict(best_state)
            test_audio_t = torch.from_numpy(test_audio).float()
            test_clinical_t = torch.from_numpy(test_clinical).float()
            test_auroc, _, test_probs = evaluate_model(
                model, test_audio_t, test_clinical_t, test_labels, device,
            )

            print(f"    Best epoch: {best_epoch}, Val AUROC: {best_auroc:.4f}, Test AUROC: {test_auroc:.4f}")

            all_results[model_name]["fold_aurocs"].append(test_auroc)
            all_results[model_name]["oof_labels"].extend(test_labels.tolist())
            all_results[model_name]["oof_probs"].extend(test_probs.tolist())
            all_results[model_name]["oof_ids"].extend(test_ids)

            # Save best fusion model
            if model_name == "fusion":
                fold_dir = output_dir / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, fold_dir / "fusion_model.pt")
                clinical_preprocessor.save(fold_dir / "clinical_preprocessor.joblib")

    # Compute overall metrics with bootstrap CI
    print("\n=== Final Results ===")
    final_metrics = {}

    for model_name, results in all_results.items():
        labels = np.array(results["oof_labels"])
        probs = np.array(results["oof_probs"])
        patient_ids = results["oof_ids"]

        ci_result = bootstrap_auroc_ci(
            labels, probs, patient_ids,
            n_bootstrap=config.n_bootstrap,
            ci_level=config.ci_level,
            seed=config.random_seed,
        )

        mean_fold_auroc = float(np.mean(results["fold_aurocs"]))
        std_fold_auroc = float(np.std(results["fold_aurocs"]))

        final_metrics[model_name] = {
            "pooled_auroc": ci_result["auroc"],
            "ci_lower": ci_result["ci_lower"],
            "ci_upper": ci_result["ci_upper"],
            "mean_fold_auroc": mean_fold_auroc,
            "std_fold_auroc": std_fold_auroc,
            "fold_aurocs": results["fold_aurocs"],
            "n_patients": len(set(patient_ids)),
            "n_tb_positive": int(labels.sum()),
            "n_tb_negative": int((1 - labels).sum()),
        }

        print(f"  {model_name}:")
        print(f"    Pooled AUROC: {ci_result['auroc']:.4f} [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")
        print(f"    Mean Fold AUROC: {mean_fold_auroc:.4f} +/- {std_fold_auroc:.4f}")

    # Save results
    results_path = output_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return final_metrics


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_features_batch(
    patients: List[PatientExample],
    audio_config: AudioFeatureConfig,
    preprocessor: ClinicalPreprocessor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract features for a batch of patients, with padding."""
    max_clips = max(len(p.audio_paths) for p in patients) if patients else 1

    audio_list, clinical_list, labels, ids = [], [], [], []
    skipped: list[tuple[str, str]] = []

    for p in patients:
        try:
            audio_feats, clinical_feats, label = extract_patient_features(
                p, audio_config, preprocessor
            )
        except Exception as e:  # noqa: BLE001 - report, do not abort the run
            # Never skip silently: a swallowed failure shrinks the cohort and
            # biases every metric without leaving a trace.
            skipped.append((str(p.patient_id), f"{type(e).__name__}: {e}"))
            continue

        # Pad clips
        n_clips = audio_feats.shape[0]
        if n_clips < max_clips:
            pad = np.zeros((max_clips - n_clips,) + audio_feats.shape[1:], dtype=np.float32)
            audio_feats = np.vstack([audio_feats, pad])

        audio_list.append(audio_feats)
        clinical_list.append(clinical_feats)
        labels.append(label)
        ids.append(p.patient_id)

    if skipped:
        print(
            f"  WARNING: skipped {len(skipped)}/{len(patients)} patients during "
            f"feature extraction:"
        )
        for patient_id, reason in skipped[:20]:
            print(f"    - {patient_id}: {reason}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")

    if not audio_list:
        return np.empty((0,)), np.empty((0,)), np.empty((0,)), []

    return (
        np.stack(audio_list),
        np.stack(clinical_list),
        np.array(labels, dtype=np.int64),
        ids,
    )


def _verify_no_leakage(nested_folds: List[Dict]) -> None:
    """Verify no patient appears in both train and test of same outer fold."""
    for outer in nested_folds:
        test_ids = {p.patient_id for p in outer["test"]}
        for inner in outer["inner_folds"]:
            train_ids = {p.patient_id for p in inner["train"]}
            val_ids = {p.patient_id for p in inner["val"]}
            overlap_train = test_ids & train_ids
            overlap_val = test_ids & val_ids
            if overlap_train or overlap_val:
                raise RuntimeError(
                    f"Patient leakage in fold {outer['outer_fold']}: "
                    f"train_overlap={overlap_train}, val_overlap={overlap_val}"
                )
    print("  No patient leakage detected.")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TBCare nested CV training pipeline")
    parser.add_argument("--config", type=str, help="YAML config path")
    parser.add_argument("--coda-root", type=str, default="ml/data/coda_tb")
    parser.add_argument("--tbscreen-root", type=str, default=None)
    parser.add_argument("--output", type=str, default="ml/artifacts")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.config:
        config = TrainConfig.from_yaml(args.config)
    else:
        config = TrainConfig(
            coda_root=args.coda_root,
            tbscreen_root=args.tbscreen_root,
            output_dir=args.output,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            epochs=args.epochs,
            random_seed=args.seed,
        )

    print("=== TBCare Training Pipeline ===")
    print(f"Config: {config}")

    # Load data
    print("\nLoading CODA TB data...")
    patients = load_coda_tb_patients(
        Path(config.coda_root),
        config.to_audio_config(),
        max_clips=config.max_clips,
    )
    print(f"  CODA TB: {len(patients)} patients ({sum(p.label for p in patients)} TB+)")

    if config.tbscreen_root:
        print("Loading TBscreen data...")
        tbscreen_patients = load_tbscreen_patients(
            Path(config.tbscreen_root),
            config.to_audio_config(),
            max_clips=config.max_clips,
        )
        print(f"  TBscreen: {len(tbscreen_patients)} patients ({sum(p.label for p in tbscreen_patients)} TB+)")
        patients.extend(tbscreen_patients)

    print(f"\nTotal: {len(patients)} patients ({sum(p.label for p in patients)} TB+)")

    # Compute data hash
    data_hash = compute_dataset_hash(patients)
    print(f"Data hash: {data_hash}")

    # Run nested CV
    start_time = time.time()
    results = run_nested_cv(patients, config)
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()