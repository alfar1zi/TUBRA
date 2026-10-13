from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, List, Mapping, Sequence
import numpy as np
import joblib
import onnxruntime as ort
from tbcare_api.inference.tta import tta_average_logits

from tbcare_api.models.manifest import ModelManifest, ModelStatus, ModelNotReadyError


# =============================================================================
# Country Normalization (serving-side, inlined from ml.src.features.clinical)
# =============================================================================

_VALID_COUNTRIES = frozenset({"IN", "MG", "PH", "ZA", "TZ", "UG", "VN"})

_COUNTRY_MAP = {
    "IND": "IN", "INDIA": "IN",
    "MDG": "MG", "MADAGASCAR": "MG",
    "PHL": "PH", "PHILIPPINES": "PH",
    "ZAF": "ZA", "SOUTH AFRICA": "ZA", "SA": "ZA",
    "TZA": "TZ", "TANZANIA": "TZ",
    "UGA": "UG", "UGANDA": "UG",
    "VNM": "VN", "VIETNAM": "VN", "VIE": "VN",
}


def _normalise_country(value: Any) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().upper()
    return _COUNTRY_MAP.get(v, v if v in _VALID_COUNTRIES else None)


# =============================================================================
# Serving-side Clinical Preprocessor
# =============================================================================

class ServingPreprocessor:
    """Lightweight preprocessor for inference. Loads from the dict saved by
    ClinicalPreprocessor.save() — no dependency on the ML training module.
    """

    def __init__(
        self,
        feature_order: tuple[str, ...],
        numeric_fields: tuple[str, ...],
        binary_fields: tuple[str, ...],
        numeric_scaler: Any,  # fitted RobustScaler
        binary_scaler: Any,   # fitted StandardScaler
        numeric_medians: dict[str, float],
        binary_modes: dict[str, float],
    ):
        self.feature_order = feature_order
        self.numeric_fields = set(numeric_fields)
        self.binary_fields = set(binary_fields)
        self.numeric_scaler = numeric_scaler
        self.binary_scaler = binary_scaler
        self.numeric_medians = numeric_medians or {}
        self.binary_modes = binary_modes or {}

        # Precompute indices
        self._numeric_idx = [i for i, f in enumerate(feature_order) if f in self.numeric_fields]
        self._binary_idx = [i for i, f in enumerate(feature_order) if f in self.binary_fields]

    @classmethod
    def load(cls, path: str | Path) -> "ServingPreprocessor":
        """Load from joblib dict produced by ClinicalPreprocessor.save()."""
        data = joblib.load(path)
        return cls(
            feature_order=data["feature_order"],
            numeric_fields=data["numeric_fields"],
            binary_fields=data["binary_fields"],
            numeric_scaler=data["numeric_scaler"],
            binary_scaler=data["binary_scaler"],
            numeric_medians=data["numeric_medians"],
            binary_modes=data["binary_modes"],
        )

    def transform_record(self, record: Mapping[str, Any]) -> np.ndarray:
        """Transform a single clinical record dict to (1, n_features) array."""
        feat = self._extract_features(record)
        X = np.array(
            [[feat.get(f, 0.0) for f in self.feature_order]],
            dtype=np.float32,
        )

        # Impute NaN
        for j, fname in enumerate(self.feature_order):
            if np.isnan(X[0, j]):
                if fname in self.numeric_fields:
                    X[0, j] = self.numeric_medians.get(fname, 0.0)
                else:
                    X[0, j] = self.binary_modes.get(fname, 0.0)

        # Scale
        if self.numeric_scaler is not None and self._numeric_idx:
            X[:, self._numeric_idx] = self.numeric_scaler.transform(X[:, self._numeric_idx])
        if self.binary_scaler is not None and self._binary_idx:
            X[:, self._binary_idx] = self.binary_scaler.transform(X[:, self._binary_idx])

        return X

    def transform_batch(self, records: List[Mapping[str, Any]]) -> np.ndarray:
        """Transform multiple records to (n, n_features) array."""
        if not records:
            return np.empty((0, len(self.feature_order)), dtype=np.float32)
        return np.vstack([self.transform_record(r) for r in records])

    # -------------------------------------------------------------------------

    @staticmethod
    def _to_float(val: Any) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    def _extract_features(self, record: Mapping[str, Any]) -> dict[str, float]:
        """Extract feature dict from a single clinical record.

        Mirrors ``ml.src.features.clinical.ClinicalPreprocessor._extract_features``
        exactly, including the missing-indicator (``miss_*``) features. About 70%
        of the CODA TB symptom fields are missing; encoding those as ``0.0``
        ("no") would teach/feed the model the wrong distribution. NaN + a firing
        indicator lets ``transform_record`` impute with training-set medians/modes
        while flagging the value as imputed.
        """
        feat: dict[str, float] = {}

        # Missing indicators default to 0.0 (explicitly reported) and are set to
        # 1.0 below when the source field is absent.
        for ind in ("miss_hemoptysis", "miss_fever", "miss_night_sweats",
                    "miss_weight_loss", "miss_smoke_lweek", "miss_tb_prior"):
            feat[ind] = 0.0

        # Demographics
        sex = str(record.get("sex", "")).strip().lower()
        feat["sex"] = 1.0 if sex in {"f", "female", "1", "wanita", "perempuan"} else 0.0

        for field in ("age", "height", "weight"):
            val = record.get(field)
            feat[field] = self._to_float(val) if val is not None else np.nan

        # Cough duration
        cough_dur = record.get("reported_cough_dur")
        feat["reported_cough_dur"] = self._to_float(cough_dur) if cough_dur is not None else np.nan

        # TB history (one-hot). A missing history is not "no prior TB".
        raw_prior = record.get("tb_prior", None)
        if raw_prior is None or (isinstance(raw_prior, float) and np.isnan(raw_prior)):
            feat["tb_prior"] = np.nan
            feat["miss_tb_prior"] = 1.0
        else:
            tb_prior = str(raw_prior).strip().lower()
            feat["tb_prior"] = 1.0 if tb_prior in {"yes", "y", "true", "1", "ya"} else 0.0

        tb_prior_type = str(record.get("tb_prior_type", record.get("tb_prior_detail", ""))).strip().lower()
        feat["tb_prior_Pul"] = 1.0 if "pul" in tb_prior_type else 0.0
        feat["tb_prior_Extrapul"] = 1.0 if "extra" in tb_prior_type else 0.0
        feat["tb_prior_Unknown"] = 1.0 if (feat["tb_prior"] == 1.0 and feat["tb_prior_Pul"] == 0.0 and feat["tb_prior_Extrapul"] == 0.0) else 0.0

        # Symptoms (binary): distinguish explicit "no" (0.0) from not recorded
        # (NaN + miss indicator). Mirrors the training preprocessor.
        for field in ("hemoptysis", "weight_loss", "smoke_lweek", "fever", "night_sweats"):
            raw = record.get(field, None)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                feat[field] = np.nan
                feat[f"miss_{field}"] = 1.0
            else:
                val = str(raw).strip().lower()
                if val in {"yes", "y", "true", "1", "ya"}:
                    feat[field] = 1.0
                elif val in {"no", "n", "false", "0", "tidak"}:
                    feat[field] = 0.0
                else:
                    feat[field] = np.nan
                    feat[f"miss_{field}"] = 1.0

        # Numeric vitals
        for field in ("heart_rate", "temperature"):
            val = record.get(field)
            feat[field] = self._to_float(val) if val is not None else np.nan

        # Cough count
        cough_count = record.get("Numberofcoughsoundscollected", record.get("cough_count", 5))
        feat["Numberofcoughsoundscollected"] = self._to_float(cough_count) if cough_count is not None else 5.0

        # Country one-hot (ZA reference)
        country = _normalise_country(record.get("country"))
        for c in ("IN", "MG", "PH", "TZ", "UG", "VN"):
            feat[f"country_{c}"] = 1.0 if country == c else 0.0

        # HIV status one-hot
        hiv = str(record.get("hiv", record.get("hiv_status", ""))).strip().lower()
        feat["hiv_Negative"] = 1.0 if hiv in {"neg", "negative", "0", "non-reactive", "negatif"} else 0.0
        feat["hiv_Positive"] = 1.0 if hiv in {"pos", "positive", "1", "reactive", "positif"} else 0.0
        feat["hiv_Unknown"] = 1.0 if (feat["hiv_Negative"] == 0.0 and feat["hiv_Positive"] == 0.0) else 0.0

        return feat


# =============================================================================
# ONNX Audio Model Wrapper
# =============================================================================

class ONNXAudioModel:
    """Wrapper for ONNX audio encoder + numpy classifier head.
    Accepts pre-computed log-mel spectrograms (numpy arrays).
    """

    def __init__(
        self,
        encoder_path: str | Path | Sequence[str | Path],
        head_path: str | Path,
        providers: Optional[List[str]] = None,
    ):
        self.providers = providers or ["CPUExecutionProvider"]

        # Load encoder(s). A sequence of paths enables seed-ensembling: each
        # backbone runs independently and their logits are averaged. A single
        # path preserves the classic single-model behaviour.
        if isinstance(encoder_path, (str, Path)):
            encoder_paths: List[str | Path] = [encoder_path]
        else:
            encoder_paths = list(encoder_path)
        if not encoder_paths:
            raise ValueError("ONNXAudioModel requires at least one encoder path")

        self.encoder_sessions: List[ort.InferenceSession] = []
        self.encoder_input_name = None
        self.encoder_output_name = None
        for path in encoder_paths:
            session = ort.InferenceSession(str(path), providers=self.providers)
            self.encoder_sessions.append(session)
            # All members share the same graph signature (exported identically),
            # so one set of io names is sufficient.
            if self.encoder_input_name is None:
                self.encoder_input_name = session.get_inputs()[0].name
                self.encoder_output_name = session.get_outputs()[0].name

        # Backwards-compatible alias for the first session.
        self.encoder_session = self.encoder_sessions[0]
        self.n_encoders = len(self.encoder_sessions)

        # Load head weights (npz)
        head_data = np.load(head_path)
        self.head_weights = {k: head_data[k] for k in head_data.files}

    def predict_proba(self, logmel: np.ndarray, use_tta: bool = True) -> np.ndarray:
        """Predict TB probability for one or more log-mel spectrograms.

        Accepted shapes:
            (n_mels, target_frames)               -> single clip
            (1, n_mels, target_frames)            -> single clip with channel dim
            (N, n_mels, target_frames)            -> N clips (N != 1)
            (B, 1, n_mels, target_frames)         -> B clips, channel dim present

        When ``use_tta`` is True the encoder output is averaged over the
        deterministic TTA views (``tbcare_api.inference.tta``), mirroring the
        training-time OOF evaluation so deployed behaviour matches reported
        metrics. The identity view is always included, so TTA can only help.

        Returns:
            probs: (B, 2) - probabilities for [non-TB, TB]
        """
        logmel = np.asarray(logmel, dtype=np.float32)
        if logmel.ndim == 2:
            logmel = logmel[None, None, ...]          # (1, 1, n_mels, T)
        elif logmel.ndim == 3:
            # (1, n_mels, T) is a single clip with a channel dim; anything else
            # with axis 0 != 1 is a batch of clips.
            if logmel.shape[0] == 1 and logmel.shape[1] != 1:
                logmel = logmel[None, ...]            # (1, 1, n_mels, T)
            else:
                logmel = logmel[:, None, ...]         # (N, 1, n_mels, T)
        elif logmel.ndim == 4 and logmel.shape[1] == 1:
            pass                                      # already (B, 1, n_mels, T)
        else:
            raise ValueError(f"Unexpected logmel shape: {logmel.shape}")

        # Run every encoder (optionally averaged over TTA views) and average the
        # resulting logits. With a single backbone this reduces to the classic
        # behaviour; with N seed-ensemble members it is the mean logit, which is
        # the standard (and calibration-preserving) way to combine members that
        # share an identical classifier head.
        def _run_for(session: ort.InferenceSession):
            def _run(batch_view: np.ndarray) -> np.ndarray:
                return session.run(
                    [self.encoder_output_name],
                    {self.encoder_input_name: batch_view.astype(np.float32)},
                )[0]

            return _run

        embeddings = []
        for session in self.encoder_sessions:
            _run = _run_for(session)
            if use_tta:
                embeddings.append(tta_average_logits(_run, logmel))
            else:
                embeddings.append(_run(logmel))
        embedding = np.mean(embeddings, axis=0)  # (B, encoder_out_dim)

        # Apply classifier head (manual forward pass)
        x = embedding
        for i in range(0, len(self.head_weights), 2):
            weight_key = f"weight{i//2}"
            bias_key = f"bias{i//2}"
            if weight_key not in self.head_weights or bias_key not in self.head_weights:
                # Do not fall back to positional dict order: a renamed or
                # reordered head would silently multiply by the wrong matrix
                # and produce plausible-but-wrong probabilities. Fail loudly.
                raise KeyError(
                    f"Classifier head is missing expected keys {weight_key!r}/"
                    f"{bias_key!r}; found {sorted(self.head_weights)}"
                )

            W = self.head_weights[weight_key]
            b = self.head_weights[bias_key]

            x = x @ W.T + b
            if i + 2 < len(self.head_weights):  # Not final layer
                x = np.maximum(x, 0)  # ReLU

        # Softmax (numerically stable). Subtracting the per-row max keeps
        # exp() from overflowing to inf for large logits, which would otherwise
        # yield nan probabilities and a 500 at serving time.
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        probs = exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)
        return probs


# =============================================================================
# Sklearn Clinical Model Wrapper
# =============================================================================

class SklearnClinicalModel:
    """Wrapper for sklearn clinical model (LogisticRegression + Calibration)."""

    def __init__(
        self,
        model_path: str | Path,
        preprocessor: ServingPreprocessor,
    ):
        self.preprocessor = preprocessor
        self.model = joblib.load(model_path)

    def predict_proba(self, record: Dict[str, Any]) -> np.ndarray:
        """Predict probability from clinical record.

        Args:
            record: Clinical feature dict
        Returns:
            probs: (1, 2) - probabilities for [non-TB, TB]
        """
        X = self.preprocessor.transform_record(record)  # (1, 26)
        return self.model.predict_proba(X)

    def predict_proba_batch(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Batch prediction."""
        X = self.preprocessor.transform_batch(records)
        return self.model.predict_proba(X)


# =============================================================================
# Supervisor (Fusion) Model
# =============================================================================

class SupervisorModel:
    """Supervisor model: Calibrated logistic regression on
    [prob_audio, prob_clinical, clinical_features].
    """

    def __init__(
        self,
        model_path: str | Path,
        preprocessor: ServingPreprocessor,
    ):
        self.preprocessor = preprocessor
        self.model = joblib.load(model_path)

    def predict_proba(
        self,
        prob_audio: float,
        prob_clinical: float,
        clinical_record: Dict[str, Any],
    ) -> np.ndarray:
        """Predict final fused probability.

        Args:
            prob_audio: P(TB) from audio model
            prob_clinical: P(TB) from clinical model
            clinical_record: Clinical features dict
        Returns:
            probs: (1, 2) - probabilities for [non-TB, TB]
        """
        clinical_features = self.preprocessor.transform_record(clinical_record).flatten()  # (26,)
        # Stack: [prob_audio, prob_clinical, clinical_features] -> (1, 28)
        X = np.hstack([[prob_audio, prob_clinical], clinical_features]).reshape(1, -1)
        return self.model.predict_proba(X)


# =============================================================================
# Stacked Ensemble (Main Serving Class)
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuration for stacked ensemble."""
    audio_encoder_path: str
    audio_head_path: str
    clinical_model_path: str
    supervisor_model_path: str
    clinical_preprocessor_path: str
    model_manifest_path: Optional[str] = None
    # Optional seed-ensemble members. When provided (len >= 1) they replace the
    # single ``audio_encoder_path`` backbone and their logits are averaged.
    audio_encoder_paths: Optional[List[str]] = None


class StackedEnsemble:
    """
    Complete stacked ensemble for TB screening:
    1. Audio branch: ONNX Residual CNN -> P(TB|audio)
    2. Clinical branch: Sklearn LogisticRegression -> P(TB|clinical)
    3. Supervisor: Calibrated LR on [P_audio, P_clinical, clinical_features] -> P(TB|fusion)

    All models must be calibrated for meaningful probability fusion.
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config

        # Load clinical preprocessor (dict format from ClinicalPreprocessor.save())
        self.preprocessor = ServingPreprocessor.load(config.clinical_preprocessor_path)

        # Load models. Prefer the seed-ensemble list when present.
        encoder_paths = config.audio_encoder_paths or [config.audio_encoder_path]
        self.audio_model = ONNXAudioModel(
            encoder_paths,
            config.audio_head_path,
        )
        self.clinical_model = SklearnClinicalModel(
            config.clinical_model_path,
            self.preprocessor,
        )
        self.supervisor = SupervisorModel(
            config.supervisor_model_path,
            self.preprocessor,
        )

        # Load manifest if provided
        self.manifest = None
        if config.model_manifest_path:
            self.manifest = ModelManifest.from_json(config.model_manifest_path)

    def predict(
        self,
        logmel: np.ndarray,
        clinical_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run full ensemble inference.

        Args:
            logmel: Pre-computed log-mel spectrogram array
            clinical_record: Clinical feature dict

        Returns:
            {
                "prob_audio": float,
                "prob_clinical": float,
                "prob_fusion": float,
                "risk_tier": "A" | "B" | "C",
                "model_version": str,
                "model_status": str,
            }
        """
        # Check model status
        model_status = ModelStatus.UNAVAILABLE
        model_version = "unknown"
        if self.manifest:
            model_status = self.manifest.status
            model_version = self.manifest.model_version
            if not self.manifest.is_available_for_inference():
                raise ModelNotReadyError(f"Model not available: status={model_status}")

        # 1. Audio prediction.
        # Accept a single clip (n_mels, T) / (1, n_mels, T) / (1, 1, n_mels, T),
        # or a batch of clips (N, n_mels, T) / (N, 1, n_mels, T). For multiple
        # cough clips we average P(TB) across clips, matching how the audio
        # expert was aggregated during training.
        arr = np.asarray(logmel, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, None, ...]          # (1, 1, n_mels, T)
        elif arr.ndim == 3:
            arr = arr[:, None, ...]             # (N, 1, n_mels, T)
        elif arr.ndim != 4:
            raise ValueError(f"Unexpected logmel shape: {arr.shape}")
        clip_probs = self.audio_model.predict_proba(arr)  # (N, 2)
        # Average P(TB) only over non-empty clips, matching training aggregation
        # (empty clips are all-zero padding and must not dilute the score).
        clip_present = arr.reshape(arr.shape[0], -1).any(axis=1)
        if not clip_present.any():
            clip_present = np.ones(arr.shape[0], dtype=bool)
        prob_audio = float(np.mean(clip_probs[clip_present, 1]))

        # 2. Clinical prediction
        clinical_probs = self.clinical_model.predict_proba(clinical_record)
        prob_clinical = float(clinical_probs[0, 1])

        # 3. Supervisor fusion
        fusion_probs = self.supervisor.predict_proba(prob_audio, prob_clinical, clinical_record)
        prob_fusion = float(fusion_probs[0, 1])

        # 4. Determine tier (per PRD thresholds)
        risk_tier = self._tier_from_prob(prob_fusion)

        return {
            "prob_audio": round(prob_audio, 4),
            "prob_clinical": round(prob_clinical, 4),
            "prob_fusion": round(prob_fusion, 4),
            "risk_tier": risk_tier,
            "model_version": model_version,
            "model_status": model_status,
        }

    def predict_proba_only(
        self,
        prob_audio: float,
        prob_clinical: float,
        clinical_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Predict using pre-computed audio/clinical probabilities."""
        fusion_probs = self.supervisor.predict_proba(prob_audio, prob_clinical, clinical_record)
        prob_fusion = float(fusion_probs[0, 1])
        risk_tier = self._tier_from_prob(prob_fusion)

        return {
            "prob_audio": round(prob_audio, 4),
            "prob_clinical": round(prob_clinical, 4),
            "prob_fusion": round(prob_fusion, 4),
            "risk_tier": risk_tier,
            "model_version": self.manifest.model_version if self.manifest else "unknown",
            "model_status": self.manifest.status if self.manifest else ModelStatus.UNAVAILABLE,
        }

    def _tier_from_prob(self, prob: float) -> str:
        """Map probability to tier.

        Thresholds come from the model manifest when available
        (``metrics.threshold_A`` / ``metrics.threshold_B``); otherwise the
        PRD interim defaults (A >= 0.70, B >= 0.40) are used.
        """
        tau_a, tau_b = self._tier_thresholds()
        if prob >= tau_a:
            return "A"
        if prob >= tau_b:
            return "B"
        return "C"

    def _tier_thresholds(self) -> tuple[float, float]:
        """Return (tau_A, tau_B), preferring values from the manifest metrics."""
        tau_a, tau_b = 0.70, 0.40
        if self.manifest and isinstance(self.manifest.metrics, dict):
            metrics = self.manifest.metrics
            try:
                if metrics.get("threshold_A") is not None:
                    tau_a = float(metrics["threshold_A"])
                if metrics.get("threshold_B") is not None:
                    tau_b = float(metrics["threshold_B"])
            except (TypeError, ValueError):
                tau_a, tau_b = 0.70, 0.40
        return tau_a, tau_b

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata for /health endpoint."""
        if self.manifest:
            return {
                "model_version": self.manifest.model_version,
                "architecture_version": self.manifest.architecture_version,
                "status": self.manifest.status,
                "metrics": self.manifest.metrics,
                "created_at": self.manifest.created_at,
                "git_commit": self.manifest.git_commit,
                "available_for_inference": self.manifest.is_available_for_inference(),
            }
        return {
            "model_version": "unknown",
            "status": ModelStatus.UNAVAILABLE,
            "available_for_inference": False,
        }


# =============================================================================
# Default Ensemble Factory
# =============================================================================

def create_ensemble_from_artifacts(artifacts_dir: str | Path) -> StackedEnsemble:
    """Create ensemble from standard artifact directory layout.

    If seed-ensemble backbones (``backbone_seed*.onnx``) are present, their
    logits are averaged at inference; otherwise the single ``backbone.onnx``
    is used. Both layouts are supported so a retrain may opt in by simply
    emitting extra members.
    """
    artifacts_dir = Path(artifacts_dir)

    seed_paths = sorted(artifacts_dir.glob("backbone_seed*.onnx"))
    encoder_paths = [str(p) for p in seed_paths] or [str(artifacts_dir / "backbone.onnx")]

    config = EnsembleConfig(
        audio_encoder_path=str(artifacts_dir / "backbone.onnx"),
        audio_encoder_paths=encoder_paths,
        audio_head_path=str(artifacts_dir / "head.npz"),
        clinical_model_path=str(artifacts_dir / "clinical_model.joblib"),
        supervisor_model_path=str(artifacts_dir / "supervisor_model.joblib"),
        clinical_preprocessor_path=str(artifacts_dir / "clinical_preprocessor.joblib"),
        model_manifest_path=str(artifacts_dir / "model_manifest.json"),
    )

    return StackedEnsemble(config)