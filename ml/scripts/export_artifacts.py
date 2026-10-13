#!/usr/bin/env python3
"""
Train the TBCare audio + clinical + supervisor ensemble on CODA TB + TBscreen
data and export serving artifacts in the exact layout consumed by
``apps/api/tbcare_api/inference/ensemble.py``:

    ml/artifacts/
        backbone.onnx              AudioOnlyClassifier (encoder+head, no softmax)
                                   -> outputs 2 logits
        head.npz                   identity 2x2 layer (weight0/bias0)
        clinical_model.joblib      calibrated LogisticRegression on 26 features
        supervisor_model.joblib    calibrated LR on [p_audio, p_clinical, 26]
        clinical_preprocessor.joblib  dict from ClinicalPreprocessor.save()
        model_manifest.json        ModelManifest
        model_card.json            metrics for web + API thresholds
        cv_results.json            full nested-CV metrics

The audio ONNX model emits 2 logits; the serving ``ONNXAudioModel`` applies the
``head.npz`` weights to them (identity here) and then softmax, so the pipeline is
mathematically equivalent to the trained torch classifier.

Usage:
    PYTHONPATH=. python3 ml/scripts/export_artifacts.py \
        --config ml/configs/train.yaml \
        --coda-root ml/data/coda_tb \
        --tbscreen-root /path/to/TBscreen_Dataset \
        --output ml/artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.features.clinical import ClinicalPreprocessor, CLINICAL_FEATURE_ORDER
from ml.src.models.residual_fusion import (
    AudioOnlyClassifier,
    AudioCNNConfig,
)
from ml.scripts.data_loader import (
    load_coda_tb_patients,
    load_tbscreen_patients,
    extract_patient_features,
)
from ml.scripts.train import (
    TrainConfig,
    _extract_features_batch,
    train_model,
    evaluate_model,
    bootstrap_auroc_ci,
    set_seed,
    _resolve_device,
)


# =============================================================================
# Audio branch
# =============================================================================

class _ExportWrapper(nn.Module):
    """AudioOnlyClassifier without the final log-softmax: emits 2 logits."""

    def __init__(self, model: AudioOnlyClassifier):
        super().__init__()
        self.audio_encoder = model.audio_encoder
        self.classifier = model.classifier

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        emb = self.audio_encoder(audio)
        return self.classifier(emb)

# =============================================================================
# Cached feature extraction (avoids re-reading WAVs across CV folds)
# =============================================================================

_CACHE_DIR = Path(os.environ.get("TBCARE_FEAT_CACHE", "/tmp/tbcare_feat_cache"))


def _cache_key(patient, audio_config) -> str:
    """Hash a cache key over the audio recipe *and* the on-disk clip identity.

    The previous key used only paths and the recipe, so editing a WAV in place
    (same path, new content) silently served the old cached features. Including
    size+mtime invalidates the entry whenever the file actually changes.
    """
    import hashlib

    parts = [
        str(patient.patient_id),
        str(audio_config.recipe),
        str(audio_config.n_mels),
        str(audio_config.target_frames),
    ]
    for clip in patient.audio_paths:
        clip_path = Path(clip)
        try:
            stat = clip_path.stat()
            parts.append(f"{clip}:{stat.st_size}:{int(stat.st_mtime)}")
        except OSError:
            parts.append(str(clip))
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def extract_with_cache(patients, audio_config, pre):
    """Like train._extract_features_batch but caches per-patient logmel tensors
    on disk keyed by patient id, clip identity, and the audio recipe."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    max_clips = max(len(p.audio_paths) for p in patients) if patients else 1
    audio_list, clinical_list, labels, ids = [], [], [], []
    skipped: list[tuple[str, str]] = []

    for p in patients:
        cache_file = _CACHE_DIR / f"{_cache_key(p, audio_config)}.npz"

        if cache_file.exists():
            audio_feats = np.load(cache_file)["arr"]
        else:
            try:
                from ml.scripts.data_loader import extract_patient_features
                audio_feats, _, _ = extract_patient_features(p, audio_config, pre)
            except Exception as e:  # noqa: BLE001 - report, do not abort the run
                skipped.append((str(p.patient_id), f"{type(e).__name__}: {e}"))
                continue
            np.savez(cache_file, arr=audio_feats)

        n_clips = audio_feats.shape[0]
        if n_clips < max_clips:
            pad = np.zeros((max_clips - n_clips,) + audio_feats.shape[1:], dtype=np.float32)
            audio_feats = np.vstack([audio_feats, pad])

        audio_list.append(audio_feats)
        clinical_list.append(pre.transform([p.clinical_features])[0])
        labels.append(p.label)
        ids.append(p.patient_id)

    if skipped:
        # Never skip silently: a swallowed per-patient failure shrinks the
        # cohort and biases every downstream metric without any trace.
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


def train_audio_branch(patients, audio_config, device, config):
    """Train AudioOnlyClassifier with GroupKFold; return OOF probs + fitted models.

    When ``config.ensemble_seeds > 1`` a seed-ensemble is trained: each seed is
    an independent model, and predictions are the mean of per-seed TTA
    probabilities (both for the OOF estimate and for the exported final
    members). Seeds are derived from ``config.random_seed`` so the whole run
    stays reproducible.
    """
    n_splits = min(config.outer_folds, 5)
    gkf = GroupKFold(n_splits=n_splits)
    ids = [p.patient_id for p in patients]
    y = np.array([p.label for p in patients])

    n_seeds = max(1, int(getattr(config, "ensemble_seeds", 1)))
    seeds = [int(config.random_seed) + k for k in range(n_seeds)]

    pre = ClinicalPreprocessor()
    pre.fit([p.clinical_features for p in patients])

    oof_probs = np.full(len(patients), np.nan)
    oof_labels = np.array([p.label for p in patients])

    fold_aurocs = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(ids, y, ids)):
        tr = [patients[i] for i in tr_idx]
        va = [patients[i] for i in va_idx]

        tr_audio, tr_clin, tr_lab, _ = extract_with_cache(tr, audio_config, pre)
        va_audio, va_clin, va_lab, _ = extract_with_cache(va, audio_config, pre)
        if len(tr_lab) == 0 or len(va_lab) == 0:
            continue

        seed_probs = []
        for seed in seeds:
            model = AudioOnlyClassifier(config.to_audio_cnn_config())
            best_state, best_epoch, _ = train_model(
                model, tr_audio, tr_clin, tr_lab, va_audio, va_clin, va_lab,
                config, device, seed=seed,
            )
            model.load_state_dict(best_state)
            # TTA at evaluation time mirrors serving exactly (shared
            # deterministic views), so this OOF metric is what the deployed
            # model actually does.
            _, _, probs = evaluate_model(
                model,
                torch.from_numpy(va_audio).float(),
                torch.from_numpy(va_clin).float(),
                va_lab,
                device,
                use_tta=True,
            )
            seed_probs.append(probs)
            print(f"  [audio] fold {fold+1} seed {seed}: best epoch {best_epoch}")

        # Mean probability across seeds (and TTA views) = ensemble OOF.
        probs = np.mean(seed_probs, axis=0)
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(va_lab, probs))
        fold_aurocs.append(auroc)
        for k, i in enumerate(va_idx):
            oof_probs[i] = probs[k]
        print(
            f"  [audio] fold {fold+1}: val AUROC={auroc:.4f} "
            f"(seed-ensemble n={n_seeds})"
        )

    # Final members trained on all data (for export). The internal holdout used
    # only for early stopping must be representative: patients are ordered
    # CODA-then-TBscreen, so taking the last N (the previous behaviour) put
    # 100% of the holdout in a single dataset (TBscreen, ~71% TB+) while the
    # cohort is ~32% TB+. Split stratified by label instead.
    from sklearn.model_selection import train_test_split

    all_audio, all_clin, all_lab, _ = extract_with_cache(patients, audio_config, pre)
    final_models = []
    for seed in seeds:
        # Use the same stratified holdout for every seed so members differ only
        # by weight initialisation / sampling order, not by which patients they
        # saw for early stopping.
        n_val = max(2, int(round(len(patients) * config.val_size)))
        idx = np.arange(len(patients))
        tr_idx, va_idx = train_test_split(
            idx, test_size=n_val, random_state=config.random_seed, stratify=all_lab,
        )
        final_model = AudioOnlyClassifier(config.to_audio_cnn_config())
        best_state, _, _ = train_model(
            final_model, all_audio[tr_idx], all_clin[tr_idx], all_lab[tr_idx],
            all_audio[va_idx], all_clin[va_idx], all_lab[va_idx], config, device,
            seed=seed,
        )
        final_model.load_state_dict(best_state)
        final_models.append(final_model)

    return final_models, oof_probs, oof_labels, fold_aurocs, pre


def export_audio_onnx(model: AudioOnlyClassifier, n_mels: int, target_frames: int, out_path: Path):
    """Export audio branch to ONNX (2 logits output) + identity head.npz."""
    model.eval()
    wrapper = _ExportWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, 1, n_mels, target_frames)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=["logmel"],
            output_names=["logits"],
            dynamic_axes={"logmel": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
        )

    head_path = out_path.parent / "head.npz"
    np.savez(
        head_path,
        weight0=np.eye(2, dtype=np.float32),
        bias0=np.zeros(2, dtype=np.float32),
    )
    return head_path


# =============================================================================
# Clinical + supervisor branches
# =============================================================================

def train_clinical_branch(patients, pre, seed=42):
    """Calibrated LogisticRegression on 26 clinical features; OOF probs."""
    X = pre.transform([p.clinical_features for p in patients])
    y = np.array([p.label for p in patients])
    groups = [p.patient_id for p in patients]

    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(patients), np.nan)
    for tr_idx, va_idx in gkf.split(X, y, groups):
        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            method="sigmoid", cv=3,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]

    final = CalibratedClassifierCV(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        method="sigmoid", cv=5,
    )
    final.fit(X, y)
    return final, oof


def train_supervisor(prob_audio, prob_clinical, X_clinical, y, groups, seed=42):
    """Calibrated LR on [p_audio, p_clinical, clinical_features]."""
    X = np.hstack([
        np.nan_to_num(prob_audio, nan=0.5).reshape(-1, 1),
        np.nan_to_num(prob_clinical, nan=0.5).reshape(-1, 1),
        X_clinical,
    ]).astype(np.float32)

    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in gkf.split(X, y, groups):
        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            method="sigmoid", cv=3,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]

    final = CalibratedClassifierCV(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        method="sigmoid", cv=5,
    )
    final.fit(X, y)
    return final, oof


def thresholds_from_labels(labels, probs):
    """τ_B = max{τ : Se(τ) >= 0.90}; τ_A = min{τ : Sp(τ) >= 0.90}."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    fpr, tpr, thr = roc_curve(labels, probs)
    # roc_curve thr includes inf at index 0
    se = tpr
    sp = 1 - fpr

    tau_b = 0.40
    mask_b = se >= 0.90
    if mask_b.any():
        # max threshold (excluding inf) satisfying sensitivity
        valid = [(t, s) for t, s in zip(thr, se) if np.isfinite(t) and s >= 0.90]
        if valid:
            tau_b = float(max(t for t, _ in valid))

    tau_a = 0.70
    valid_a = [t for t, s in zip(thr, sp) if np.isfinite(t) and s >= 0.90]
    if valid_a:
        tau_a = float(min(valid_a))

    return round(tau_b, 4), round(tau_a, 4)


def metrics_at(labels, probs, threshold):
    """Sensitivity/specificity/accuracy at a single decision threshold."""
    labels = np.asarray(labels)
    pred = (np.asarray(probs) >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    se = tp / (tp + fn) if (tp + fn) else None
    sp = tn / (tn + fp) if (tn + fp) else None
    acc = (tp + tn) / len(labels) if len(labels) else None
    return se, sp, acc


def pauc_at_sens(labels, probs, target_sens: float = 0.90) -> Optional[float]:
    """Standard partial AUROC over the high-sensitivity region.

    Restricts the ROC to the FPR interval ``[0, 1 - target_sens]`` (i.e. where
    sensitivity >= target_sens) and normalizes by that interval's width, so the
    score is 1.0 for a perfect model and ~0.5 for a model with no signal in the
    high-sensitivity region.

    For ``target_sens = 0.90`` this is the pAUC@90%Sens used in TB triage
    literature.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return None
    max_fpr = 1.0 - target_sens
    if max_fpr <= 0:
        return None
    fpr, tpr, _ = roc_curve(labels, probs)
    # Interpolate TPR at the FPR cut, then integrate the ROC up to that point.
    tpr_at_cut = float(np.interp(max_fpr, fpr, tpr))
    sel = fpr <= max_fpr
    fp = np.concatenate([fpr[sel], [max_fpr]])
    tp = np.concatenate([tpr[sel], [tpr_at_cut]])
    # numpy renamed trapz -> trapezoid in 2.0; the API pins numpy 1.26.4, so
    # call the stable name and avoid AttributeError on the pinned version.
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    area = float(_trapz(tp, fp))
    return round(float(min(max(area / max_fpr, 0.0), 1.0)), 4)


def expected_calibration_error(labels, probs, n_bins: int = 10) -> Optional[float]:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(labels) == 0:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if not m.any():
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return round(float(ece), 4)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/train.yaml")
    parser.add_argument("--coda-root", default="ml/data/coda_tb")
    parser.add_argument("--tbscreen-root", default=None)
    parser.add_argument("--output", default="ml/artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="limit patients (debug)")
    parser.add_argument("--outer-folds", type=int, default=5, help="CV folds for audio branch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--ensemble-seeds", type=int, default=None,
        help="train N seed-ensemble members (default: config.ensemble_seeds)",
    )
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config) if Path(args.config).exists() else TrainConfig()
    config.output_dir = args.output
    config.epochs = args.epochs
    config.max_clips = args.max_clips
    config.outer_folds = args.outer_folds
    config.batch_size = args.batch_size
    if args.ensemble_seeds is not None:
        config.ensemble_seeds = max(1, int(args.ensemble_seeds))
    if args.tbscreen_root:
        config.tbscreen_root = args.tbscreen_root

    set_seed(config.random_seed)
    device = _resolve_device()
    audio_config = config.to_audio_config()
    print(f"Device: {device} | recipe: {audio_config.recipe} | top_db={audio_config.top_db}")

    # ---- Load data ----
    print("\nLoading CODA TB...")
    patients = load_coda_tb_patients(Path(config.coda_root), audio_config, max_clips=config.max_clips)
    print(f"  CODA TB: {len(patients)} patients ({sum(p.label for p in patients)} TB+)")
    if config.tbscreen_root:
        print("Loading TBscreen...")
        ts = load_tbscreen_patients(Path(config.tbscreen_root), audio_config, max_clips=config.max_clips)
        print(f"  TBscreen: {len(ts)} patients ({sum(p.label for p in ts)} TB+)")
        patients.extend(ts)
    if args.limit:
        rng = np.random.RandomState(config.random_seed)
        idx = rng.permutation(len(patients))[: args.limit]
        patients = [patients[i] for i in idx]

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Audio branch ----
    print("\n=== Audio branch (GroupKFold) ===")
    t0 = time.time()
    audio_models, oof_audio, y, audio_folds, pre = train_audio_branch(
        patients, audio_config, device, config
    )
    print(f"Audio done in {time.time()-t0:.1f}s")
    audio_auroc = roc_auc_score(y, np.nan_to_num(oof_audio, nan=0.5))

    # ---- Clinical branch ----
    print("\n=== Clinical branch ===")
    clinical_model, oof_clin = train_clinical_branch(patients, pre, config.random_seed)
    clin_auroc = roc_auc_score(y, np.nan_to_num(oof_clin, nan=0.5))

    # ---- Supervisor ----
    print("\n=== Supervisor ===")
    X_clinical = pre.transform([p.clinical_features for p in patients])
    groups = [p.patient_id for p in patients]
    supervisor, oof_fus = train_supervisor(
        oof_audio, oof_clin, X_clinical, y, groups, config.random_seed
    )
    fus_auroc = roc_auc_score(y, np.nan_to_num(oof_fus, nan=0.5))

    # Persist out-of-fold predictions so metrics can be recomputed / audited
    # without retraining.
    np.savez(
        out_dir / "oof_predictions.npz",
        labels=y,
        oof_audio=oof_audio,
        oof_clinical=oof_clin,
        oof_fusion=oof_fus,
        groups=np.array(groups),
    )
    print(f"  Saved oof_predictions.npz (n={len(y)})")

    # ---- Subgroup metrics by country (honest reporting) ----
    # Motivated by ESL-EPFL "Why ML-based cough models do not generalize":
    # pooled AUROC hides per-site behaviour, and CODA is dominated by a few
    # single-device countries. Analysis-only: it does not affect the exported
    # model or its thresholds.
    countries = [p.country for p in patients]
    oof_fus_clean = np.nan_to_num(oof_fus, nan=0.5)
    subgroup_metrics = {}
    for country in sorted(set(countries)):
        mask = np.array([c == country for c in countries])
        n = int(mask.sum())
        n_pos = int(y[mask].sum())
        sub = {"n": n, "n_positive": n_pos, "n_negative": n - n_pos}
        if n_pos > 0 and (n - n_pos) > 0:
            sub["auroc"] = round(float(roc_auc_score(y[mask], oof_fus_clean[mask])), 4)
        else:
            sub["auroc"] = None  # single-class subgroup: AUROC undefined
        subgroup_metrics[country] = sub
    (out_dir / "subgroup_metrics.json").write_text(json.dumps({
        "grouping": "country",
        "note": "OOF fusion AUROC per country. Single-class countries report null.",
        "subgroups": subgroup_metrics,
    }, indent=2))
    print("  Saved subgroup_metrics.json")
    for c, s in subgroup_metrics.items():
        print(f"    {c}: n={s['n']} TB+={s['n_positive']} AUROC={s['auroc']}")

    # ---- Thresholds + metrics ----
    tau_b, tau_a = thresholds_from_labels(y, oof_fus)
    se, sp, acc = metrics_at(y, oof_fus, tau_a)   # metrics reported at Tier A
    ci = bootstrap_auroc_ci(y, np.nan_to_num(oof_fus, nan=0.5), groups, config.n_bootstrap, config.ci_level, config.random_seed)

    print(f"\n=== Results ===")
    print(f"  AUROC audio={audio_auroc:.4f} clinical={clin_auroc:.4f} fusion={fus_auroc:.4f}")
    print(f"  Fusion AUROC CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  Thresholds: A={tau_a}, B={tau_b}")
    print(f"  Se={se}, Sp={sp}, Acc={acc}")

    # ---- Export sanity: torch vs onnx parity ----
    # With a seed-ensemble we export one backbone per member; serving averages
    # their logits, so parity is asserted on the ensemble mean.
    print("\nExporting ONNX...")
    import onnxruntime as ort

    backbone_path = out_dir / "backbone.onnx"
    head_path = None
    for k, member in enumerate(audio_models):
        if len(audio_models) == 1:
            member_path = backbone_path
        else:
            member_path = out_dir / f"backbone_seed{k}.onnx"
        head_path = export_audio_onnx(member, config.n_mels, config.target_frames, member_path)
        print(f"  {member_path.name} ({member_path.stat().st_size/1e6:.1f} MB)")
    print(f"  head.npz written ({head_path})")

    # ---- Parity check: torch logits (ensemble mean) vs ONNX (ensemble mean) ----
    member_paths = (
        [backbone_path] if len(audio_models) == 1
        else [out_dir / f"backbone_seed{k}.onnx" for k in range(len(audio_models))]
    )
    sessions = [
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        for p in member_paths
    ]
    in_name = sessions[0].get_inputs()[0].name
    rng = np.random.RandomState(0)
    x = rng.randn(4, 1, config.n_mels, config.target_frames).astype(np.float32)
    onnx_logits = np.mean([s.run(None, {in_name: x})[0] for s in sessions], axis=0)
    torch_logits = []
    for member in audio_models:
        member.eval()
        with torch.no_grad():
            torch_logits.append(_ExportWrapper(member)(torch.from_numpy(x)).numpy())
    torch_logits = np.mean(torch_logits, axis=0)
    max_abs = float(np.max(np.abs(torch_logits - onnx_logits)))
    print(f"  Parity torch-vs-ONNX max|Δlogit| = {max_abs:.2e} (n_members={len(audio_models)})")
    assert max_abs < 1e-3, f"ONNX parity failed: {max_abs}"
    clinical_model_dir = out_dir / "clinical_model.joblib"
    joblib.dump(clinical_model, clinical_model_dir)
    joblib.dump(supervisor, out_dir / "supervisor_model.joblib")
    pre.save(out_dir / "clinical_preprocessor.joblib")
    print("  Saved clinical_model.joblib, supervisor_model.joblib, clinical_preprocessor.joblib")

    # ---- cv_results ----
    cv_results = {
        "audio_only": {"pooled_auroc": float(audio_auroc), "fold_aurocs": audio_folds},
        "clinical_only": {"pooled_auroc": float(clin_auroc)},
        "fusion": {
            "pooled_auroc": float(fus_auroc),
            "ci_lower": float(ci["ci_lower"]),
            "ci_upper": float(ci["ci_upper"]),
        },
        "n_patients": int(len(patients)),
        "n_tb_positive": int(y.sum()),
        "n_tb_negative": int((1 - y).sum()),
    }
    (out_dir / "cv_results.json").write_text(json.dumps(cv_results, indent=2))

    # ---- model_card ----
    oof_fus_clean = np.nan_to_num(oof_fus, nan=0.5)
    pauc90 = pauc_at_sens(y, oof_fus_clean, 0.90)
    ece = expected_calibration_error(y, oof_fus_clean)
    model_card = {
        "model_version": "tbcare-ensemble-0.2",
        "trained_on": (
            "CODA TB (India, Madagascar, Philippines, South Africa, Tanzania, "
            "Uganda, Vietnam) + TBscreen/Nairobi; "
            f"{len(patients)} participants"
        ),
        "disclaimer": (
            "Research prototype. NOT a diagnostic device. Metrics are cross-validated "
            "(out-of-fold) and thresholds are derived from them."
        ),
        "metrics": {
            # keys consumed by apps/web (copy.howItWorks.metricsTable)
            "roc_auc": round(float(fus_auroc), 4),
            "pauc_90": pauc90,
            "sens_at_A": round(float(se), 4) if se is not None else None,
            "spec_at_A": round(float(sp), 4) if sp is not None else None,
            "ece": ece,
            "threshold_A": tau_a,
            "threshold_B": tau_b,
            # supplementary metrics
            "auc": round(float(fus_auroc), 4),
            "ci_95": [round(float(ci["ci_lower"]), 4), round(float(ci["ci_upper"]), 4)],
            "sensitivity": round(float(se), 4) if se is not None else None,
            "specificity": round(float(sp), 4) if sp is not None else None,
            "accuracy": round(float(acc), 4) if acc is not None else None,
            "audio_only_auc": round(float(audio_auroc), 4),
            "clinical_only_auc": round(float(clin_auroc), 4),
        },
        "thresholds": {"A": tau_a, "B": tau_b},
        "tier_thresholds": {"A": tau_a, "B": tau_b, "C": 0.0},
        "ablation": {
            "audio_only": round(float(audio_auroc), 4),
            "symptoms_only": round(float(clin_auroc), 4),
            "fusion": round(float(fus_auroc), 4),
        },
        "calibration_temperature": 1.0,
        "audio_ensemble_members": int(len(audio_models)),
        "ensemble_seeds": int(getattr(config, "ensemble_seeds", 1)),
        "n_participants_train": int(len(patients)),
        "subgroup_metrics": subgroup_metrics,
        "limitations": [
            "Tidak ada data Indonesia" if "ID" not in subgroup_metrics else "Termasuk data Indonesia",
            "Populasi: dewasa dengan batuk 2 minggu atau lebih",
            "Bukan alat diagnosis",
            "AUROC per negara tersedia di subgroup_metrics (bisa rendah di lokasi tertentu)",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))

    # ---- model_manifest ----
    from tbcare_api.models.manifest import ModelManifest, ModelStatus

    manifest = ModelManifest(
        model_version=model_card["model_version"],
        architecture_version="residual_spectrogram_cnn_v3+stacked_lr",
        status=ModelStatus.CANDIDATE,
        audio_config=audio_config.to_dict(),
        clinical_feature_order=list(CLINICAL_FEATURE_ORDER),
        metrics=model_card["metrics"],
    )
    manifest.to_json(out_dir / "model_manifest.json")
    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
