from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_version": "2026.10.dev"
    })