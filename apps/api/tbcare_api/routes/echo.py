from flask import Blueprint, request, jsonify
from ..audio.wav_io import validate_wav, compute_audio_metrics, WAVValidationError
from ..config import config
from ..schemas.response import create_error_response

echo_bp = Blueprint("echo", __name__)


@echo_bp.route("/api/v1/echo-test", methods=["POST"])
def echo_test():
    if "audio" not in request.files:
        return jsonify(create_error_response("invalid_audio", {"reason": "audio file missing"}).to_json()), 400

    file = request.files["audio"]
    data = file.read()

    max_bytes = config.MAX_BODY_MB * 1024 * 1024
    if len(data) > max_bytes:
        return jsonify(create_error_response("too_large").to_json()), 413

    try:
        audio_int16, sample_rate = validate_wav(data)
    except WAVValidationError as e:
        # Echo test is diagnostics-only; surface the reason (already curated by
        # the validator) but never a raw traceback.
        return jsonify(create_error_response("invalid_audio", {"reason": e.message}).to_json()), 400

    metrics = compute_audio_metrics(audio_int16, sample_rate)
    metrics["applied_constraints"] = {
        "max_body_mb": config.MAX_BODY_MB,
        "required_format": "WAV PCM 16-bit mono"
    }

    return jsonify(metrics)