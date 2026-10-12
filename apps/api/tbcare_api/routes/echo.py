from flask import Blueprint, request, jsonify, current_app
from ..audio.wav_io import validate_wav, compute_audio_metrics, WAVValidationError
from ..config import config

echo_bp = Blueprint("echo", __name__)


@echo_bp.route("/api/v1/echo-test", methods=["POST"])
def echo_test():
    if "audio" not in request.files:
        return jsonify({
            "error": {"code": "invalid_audio", "retryable": False}
        }), 400

    file = request.files["audio"]
    data = file.read()

    max_bytes = config.MAX_BODY_MB * 1024 * 1024
    if len(data) > max_bytes:
        return jsonify({
            "error": {"code": "too_large", "retryable": False}
        }), 413

    try:
        audio_int16, sample_rate = validate_wav(data)
    except WAVValidationError as e:
        return jsonify({
            "error": {"code": "invalid_audio", "retryable": False}
        }), 400

    metrics = compute_audio_metrics(audio_int16, sample_rate)
    metrics["applied_constraints"] = {
        "max_body_mb": config.MAX_BODY_MB,
        "required_format": "WAV PCM 16-bit mono"
    }

    return jsonify(metrics)