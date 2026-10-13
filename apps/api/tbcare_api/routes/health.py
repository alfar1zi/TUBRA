"""GET /api/v1/health endpoint.

Reports service liveness plus the currently loaded model status, so operators
can tell whether inference is available (VALIDATED) or gated (CANDIDATE).
"""

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/v1/health", methods=["GET"])
def health():
    model_info = {}
    try:
        from .screen import get_ensemble

        model_info = get_ensemble().get_model_info()
    except Exception:  # noqa: BLE001 - health must never raise
        model_info = {"status": "UNAVAILABLE"}

    return jsonify({
        "status": "ok",
        "model_version": model_info.get("model_version", "unknown"),
        "model_status": model_info.get("status", "UNAVAILABLE"),
        "available_for_inference": model_info.get("available_for_inference", False),
    })
