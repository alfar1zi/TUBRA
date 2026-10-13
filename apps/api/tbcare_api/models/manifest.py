from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


# =============================================================================
# Model Status Enum
# =============================================================================

class ModelStatus:
    UNAVAILABLE = "unavailable"
    CANDIDATE = "candidate"
    VALIDATED = "validated"


class ModelNotReadyError(RuntimeError):
    """Raised when the loaded model may not serve inference yet.

    Distinct from a generic RuntimeError so the API layer can map it to a
    503 Service Unavailable (retryable) instead of a 500 server error.
    """


# =============================================================================
# Model Manifest
# =============================================================================

@dataclass
class ModelManifest:
    """
    Model manifest for governance, versioning, and deployment gating.
    Stored as model_manifest.json alongside model artifacts.
    """
    model_version: str
    architecture_version: str
    status: str = ModelStatus.UNAVAILABLE

    # Model configuration
    audio_config: Dict[str, Any] = field(default_factory=dict)
    clinical_feature_order: List[str] = field(default_factory=list)

    # Metrics (from nested CV)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Provenance
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    git_commit: Optional[str] = None
    data_hash: Optional[str] = None
    training_config_hash: Optional[str] = None

    # Deployment
    deployed_at: Optional[str] = None
    deployed_by: Optional[str] = None

    def __post_init__(self):
        if self.status not in {ModelStatus.UNAVAILABLE, ModelStatus.CANDIDATE, ModelStatus.VALIDATED}:
            raise ValueError(f"Invalid status: {self.status}")

    def to_json(self, path: str | Path) -> None:
        """Write manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelManifest":
        """Load manifest from JSON file."""
        return cls(**json.loads(Path(path).read_text()))

    def is_available_for_inference(self) -> bool:
        """Check if model can be used for inference requests."""
        if self.status == ModelStatus.VALIDATED:
            return True
        if self.status == ModelStatus.CANDIDATE:
            # Allow candidate only in staging with explicit flag
            return (
                os.getenv("DEPLOYMENT_ENV", "production") == "staging"
                and os.getenv("ALLOW_BLOCKED_CANDIDATE", "false").lower() == "true"
            )
        return False

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get info for /health endpoint."""
        return {
            "model_version": self.model_version,
            "architecture_version": self.architecture_version,
            "status": self.status,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "available_for_inference": self.is_available_for_inference(),
        }


# =============================================================================
# Manifest Builder (for training pipeline)
# =============================================================================

def build_manifest(
    model_version: str,
    architecture_version: str,
    audio_config: Dict[str, Any],
    clinical_feature_order: List[str],
    cv_metrics: Dict[str, Any],
    git_commit: Optional[str] = None,
    data_hash: Optional[str] = None,
    training_config_hash: Optional[str] = None,
    status: str = ModelStatus.CANDIDATE,
) -> ModelManifest:
    """Build manifest from training outputs."""
    return ModelManifest(
        model_version=model_version,
        architecture_version=architecture_version,
        status=status,
        audio_config=audio_config,
        clinical_feature_order=clinical_feature_order,
        metrics=cv_metrics,
        git_commit=git_commit or _get_git_commit(),
        data_hash=data_hash,
        training_config_hash=training_config_hash,
    )


def promote_to_validated(manifest_path: str | Path, deployed_by: str) -> ModelManifest:
    """Promote candidate to validated (run during deployment)."""
    manifest = ModelManifest.from_json(manifest_path)
    if manifest.status != ModelStatus.CANDIDATE:
        raise ValueError(f"Cannot promote from status: {manifest.status}")
    manifest.status = ModelStatus.VALIDATED
    manifest.deployed_at = datetime.utcnow().isoformat() + "Z"
    manifest.deployed_by = deployed_by
    manifest.to_json(manifest_path)
    return manifest


def block_candidate(manifest_path: str | Path) -> ModelManifest:
    """Block a candidate (set to unavailable)."""
    manifest = ModelManifest.from_json(manifest_path)
    manifest.status = ModelStatus.UNAVAILABLE
    manifest.to_json(manifest_path)
    return manifest


# =============================================================================
# Data Hashing (for reproducibility)
# =============================================================================

def compute_data_hash(data_paths: List[str | Path]) -> str:
    """Compute SHA256 hash of data files for reproducibility tracking."""
    hasher = hashlib.sha256()
    for path in sorted(data_paths):
        path = Path(path)
        if path.is_file():
            hasher.update(path.read_bytes())
        elif path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    hasher.update(file_path.read_bytes())
    return hasher.hexdigest()[:16]


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of training configuration."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# =============================================================================
# Git Integration
# =============================================================================

def _get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# =============================================================================
# Model Card Generator (for docs/model_card.json)
# =============================================================================

def generate_model_card(
    manifest: ModelManifest,
    cv_results: Dict[str, Any],
    dataset_info: Dict[str, Any],
    limitations: List[str],
) -> Dict[str, Any]:
    """Generate model_card.json for frontend display."""
    return {
        "model_version": manifest.model_version,
        "architecture": manifest.architecture_version,
        "status": manifest.status,
        "created_at": manifest.created_at,
        "git_commit": manifest.git_commit,
        "data_hash": manifest.data_hash,
        "dataset": dataset_info,
        "metrics": cv_results,
        "limitations": limitations,
        "disclaimer": (
            "Model ini adalah prototipe skrining dan BUKAN diagnosis. "
            "Hanya pemeriksaan dahak yang dapat memastikan TBC. "
            "Hasil tidak boleh digunakan untuk keputusan medis tanpa konfirmasi laboratorium."
        ),
        "intended_use": (
            "Skrining batuk dewasa ≥2 minggu untuk prioritas pemeriksaan dahak. "
            "Tier A: 1-2 hari, Tier B: minggu ini, Tier C: tetap periksa."
        ),
        "not_intended_for": [
            "Diagnosis TBC",
            "Penggantian pemeriksaan dahak",
            "Anak-anak di bawah 18 tahun",
            "Pasien sudah pernah periksa dahak",
        ],
    }


# =============================================================================
# Default Manifest Path
# =============================================================================

DEFAULT_MANIFEST_PATH = Path("ml/artifacts/model_manifest.json")
DEFAULT_MODEL_CARD_PATH = Path("apps/web/public/model_card.json")