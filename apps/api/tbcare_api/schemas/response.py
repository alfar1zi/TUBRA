from dataclasses import dataclass, asdict
from typing import Optional, List
import json


@dataclass(frozen=True)
class ScreenResponse:
    """Response for POST /api/v1/screen."""
    tier: str  # "A" | "B" | "C"
    explanation: str
    next_steps: List[str]
    metadata: dict
    gradcam_url: Optional[str] = None
    tier_label: Optional[str] = None
    actions: Optional[List[str]] = None
    disclaimer: Optional[str] = None

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ErrorResponse:
    """Canonical error envelope for all JSON endpoints.

    Shape (see docs/ARCHITECTURE.md 5.3):
        {"error": {"code": str, "retryable": bool,
                   "message": str, "details": dict|None}}
    ``message`` is a short, non-sensitive human hint; machine consumers must
    branch on ``code``, never on the message.
    """
    error: str  # error code
    message: str
    retryable: bool = False
    details: Optional[dict] = None

    def to_json(self) -> dict:
        err = {
            "code": self.error,
            "retryable": self.retryable,
            "message": self.message,
        }
        if self.details:
            err["details"] = self.details
        return {"error": err}


# Error codes (see docs/ARCHITECTURE.md 5.3). `retryable` marks transient
# failures the client may retry without changing its input.
ERROR_CODES = {
    "invalid_audio": "Berkas audio tidak valid",
    "payload_invalid": "Data gejala tidak valid",
    "quality_failed": "Rekaman tidak memenuhi syarat kualitas",
    "too_large": "Berkas terlalu besar",
    "model_unavailable": "Model belum siap melayani permintaan",
    "inference_error": "Galat saat menjalankan model",
    "server_error": "Galat server",
    "timeout": "Waktu pemrosesan habis",
}

# Codes whose failures are transient (retrying the same request may succeed).
RETRYABLE_CODES = {"model_unavailable", "server_error", "timeout"}

# Tier messages (from UI_SPEC.md)
TIER_MESSAGES = {
    "A": {
        "title": "Sebaiknya periksa dahak dalam 1 sampai 2 hari.",
        "explanation": "Pola suara batuk dan gejala yang Anda sebutkan membuat pemeriksaan ini jadi prioritas.",
    },
    "B": {
        "title": "Sebaiknya periksa dahak minggu ini.",
        "explanation": "Ada sinyal yang membuat pemeriksaan ini penting dalam waktu dekat.",
    },
    "C": {
        "title": "Tetap periksa dahak.",
        "explanation": "Suara batuk Anda tidak menunjukkan sinyal yang kuat, tetapi batuk lebih dari 2 minggu tetap perlu diperiksa. Alat ini tidak bisa menyatakan Anda sehat.",
    },
}

# Next steps per tier (from UI_SPEC.md)
NEXT_STEPS = [
    "Cari puskesmas terdekat",
    "Apa yang perlu saya siapkan?",
    "Mengapa hasilnya begini?",
]


def create_tier_response(tier: str, gradcam_url: Optional[str] = None,
                          metadata: Optional[dict] = None,
                          actions: Optional[List[str]] = None,
                          disclaimer: Optional[str] = None,
                          tier_label: Optional[str] = None) -> ScreenResponse:
    """Create a ScreenResponse for a given tier."""
    msg = TIER_MESSAGES[tier]
    return ScreenResponse(
        tier=tier,
        explanation=msg["explanation"],
        next_steps=NEXT_STEPS,
        gradcam_url=gradcam_url,
        metadata=metadata or {},
        tier_label=tier_label or msg["title"],
        actions=actions,
        disclaimer=disclaimer,
    )


def create_error_response(error_code: str, details: Optional[dict] = None) -> ErrorResponse:
    """Create an ErrorResponse for a given error code."""
    return ErrorResponse(
        error=error_code,
        message=ERROR_CODES.get(error_code, "Galat tidak dikenal"),
        retryable=error_code in RETRYABLE_CODES,
        details=details,
    )