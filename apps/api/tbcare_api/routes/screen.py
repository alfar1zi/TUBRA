"""POST /api/v1/screen endpoint.
Integrates: audio validation -> quality gate -> logmel -> stacked ensemble -> actions -> response.
"""
import time
import json
import logging
import numpy as np
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app

from tbcare_api.audio.wav_io import validate_wav, WAVValidationError, compute_audio_metrics
from tbcare_api.audio.quality import run_quality_pipeline, extract_cough_segments
from tbcare_api.audio.logmel import process_audio_to_logmel
from tbcare_api.inference.ensemble import StackedEnsemble, create_ensemble_from_artifacts
from tbcare_api.inference.actions import RuleEngine, Tier, build_tier_result
from tbcare_api.schemas.payload import ScreenPayload
from tbcare_api.schemas.response import (
    create_tier_response,
    create_error_response,
    ScreenResponse,
    ErrorResponse,
)
from tbcare_api.models.manifest import ModelNotReadyError

logger = logging.getLogger(__name__)

from tbcare_api.audio.wav_io import validate_wav, WAVValidationError, compute_audio_metrics
from tbcare_api.audio.quality import run_quality_pipeline, extract_cough_segments
from tbcare_api.audio.logmel import process_audio_to_logmel
from tbcare_api.inference.ensemble import StackedEnsemble, create_ensemble_from_artifacts
from tbcare_api.inference.actions import RuleEngine, Tier, build_tier_result
from tbcare_api.schemas.payload import ScreenPayload
from tbcare_api.schemas.response import (
    create_tier_response,
    create_error_response,
    ScreenResponse,
    ErrorResponse,
)

screen_bp = Blueprint("screen", __name__)

# Lazy-loaded ensemble
_ensemble: StackedEnsemble | None = None
_rule_engine: RuleEngine | None = None


def _artifacts_candidates() -> list[Path]:
    """Ordered candidate directories for model artifacts.

    Resolution is deliberately plural: the same code runs from a repo checkout
    (ml/artifacts 5 levels up), from the container image (artifacts COPYed to
    /app/ml/artifacts), and from the compose mount. Hardcoding one layout broke
    at least two of those.
    """
    file_path = Path(__file__).resolve()
    candidates: list[Path] = []

    configured = current_app.config.get("TBCARE_ARTIFACTS")
    if configured:
        candidates.append(Path(configured))

    # Repo checkout: .../apps/api/tbcare_api/routes/screen.py -> repo root.
    # The container image places the package at /app/tbcare_api instead, which
    # is two levels shallower, so walk up only when the depth allows it.
    if len(file_path.parents) > 4:
        candidates.append(file_path.parents[4] / "ml" / "artifacts")
    # Container image / compose mount.
    candidates.append(Path("/app/ml/artifacts"))
    candidates.append(file_path.parents[2] / "artifacts")

    return candidates


def get_ensemble() -> StackedEnsemble:
    global _ensemble
    if _ensemble is None:
        for candidate in _artifacts_candidates():
            if (candidate / "model_manifest.json").exists():
                _ensemble = create_ensemble_from_artifacts(candidate)
                break
        else:
            searched = ", ".join(str(c) for c in _artifacts_candidates())
            raise FileNotFoundError(f"no model artifacts found; searched: {searched}")
    return _ensemble


def get_rule_engine() -> RuleEngine:
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = RuleEngine()
    return _rule_engine


@screen_bp.route("/api/v1/screen", methods=["POST"])
def screen():
    start_time = time.time()

    # 1. Validate multipart form data
    if "audio" not in request.files:
        return jsonify(create_error_response("invalid_audio", {"reason": "audio file missing"}).to_json()), 400

    if "payload" not in request.form:
        return jsonify(create_error_response("payload_invalid", {"reason": "payload missing"}).to_json()), 400

    audio_file = request.files["audio"]
    payload_json = request.form["payload"]

    # 2. Parse and validate payload
    try:
        payload_data = json.loads(payload_json)
        payload = ScreenPayload.from_json(payload_data)
    except json.JSONDecodeError:
        return jsonify(create_error_response("payload_invalid").to_json()), 400
    except ValueError:
        return jsonify(create_error_response("payload_invalid").to_json()), 400

    # 3. Read audio bytes
    audio_bytes = audio_file.read()

    # 4. Validate WAV format
    try:
        audio_int16, orig_sr = validate_wav(audio_bytes)
    except WAVValidationError as e:
        return jsonify(create_error_response("invalid_audio", {"reason": e.message}).to_json()), 400

    audio_metrics = compute_audio_metrics(audio_int16, orig_sr)

    # Check duration limit (30s)
    if audio_metrics["duration_s"] > 30.0:
        return jsonify(create_error_response("invalid_audio", {"reason": "duration exceeds 30 seconds"}).to_json()), 400

    # 5. Run quality pipeline
    try:
        passed, quality_details, audio_norm, sr = run_quality_pipeline(audio_bytes)
    except Exception:
        logger.exception("quality pipeline failed")
        return jsonify(create_error_response("server_error").to_json()), 500

    if not passed:
        # Quality failed - return specific reason
        reason = quality_details.get("reason", "quality check failed")
        cough_count = quality_details.get("num_coughs", 0)
        return jsonify(create_error_response("quality_failed", {
            "reason": reason,
            "reasons": quality_details.get("reasons", []),
            "cough_count": cough_count,
            "snr_db": quality_details.get("snr_db"),
            "clipping_ratio": quality_details.get("clipping_ratio"),
        }).to_json()), 400

    # 6. Extract cough segments (max 5). This is used for the quality gate /
    #    user-facing cough count only: at least 3 coughs must be present.
    segments = extract_cough_segments(audio_norm, sr, max_segments=5)

    if len(segments) < 3:
        return jsonify(create_error_response("quality_failed", {
            "reason": f"only {len(segments)} cough segments extracted, need at least 3",
            "cough_count": len(segments),
        }).to_json()), 400

    # 7. Build the model input from the WHOLE recording, not the segments.
    #    Training (ml/scripts/data_loader.py::extract_patient_features) feeds
    #    each full clip through the energy_v2 recipe, which selects the
    #    highest-energy 1.5 s window. Serving must do the same, otherwise the
    #    model sees a different input distribution than it was trained on
    #    (see docs/proposal/training_improvement.md, T1). The detected segments
    #    above are still used for the quality decision and cough count.
    try:
        target_frames = 64  # from audio.yaml
        logmel = process_audio_to_logmel(audio_norm, sr, target_frames=target_frames)
        logmel_batch = logmel[None, ...]  # (1, n_mels, n_frames)
    except Exception:
        logger.exception("log-Mel extraction failed")
        return jsonify(create_error_response("server_error").to_json()), 500

    # 8. Map payload to clinical record for ensemble
    clinical_record = payload.to_clinical_record()

    # 9. Run stacked ensemble inference on the whole-clip log-Mel
    try:
        ensemble = get_ensemble()
        ensemble_result = ensemble.predict(logmel_batch, clinical_record)
    except ModelNotReadyError as e:
        logger.warning("model not ready: %s", e)
        return jsonify(create_error_response("model_unavailable").to_json()), 503
    except Exception:
        logger.exception("ensemble inference failed")
        return jsonify(create_error_response("server_error").to_json()), 500

    # 10-13. Everything below reads fields produced by the ensemble. A malformed
    # or partial result (missing key, unexpected tier label) must not escape as
    # an unhandled exception and a 500 HTML page; wrap it and return JSON.
    try:
        probability = ensemble_result["prob_fusion"]
        tier = Tier(ensemble_result["risk_tier"])

        # 10. Build tier result (explanation + ordered actions + disclaimer).
        #     Grad-CAM is intentionally not wired here: the ONNX audio branch
        #     does not expose intermediate feature maps. The clinical
        #     explanation is provided via ordered, auditable actions instead.
        tier_result = build_tier_result(
            tier=tier,
            prob_fusion=probability,
            clinical_record=clinical_record,
        )

        # 12. Build metadata
        metadata = {
            "model_version": ensemble_result.get("model_version", "dev"),
            "probability": round(probability, 4),
            "prob_audio": ensemble_result.get("prob_audio"),
            "prob_clinical": ensemble_result.get("prob_clinical"),
            "cough_segments_used": len(segments),
            "audio_metrics": audio_metrics,
            "quality_details": quality_details,
            "processing_time_ms": round((time.time() - start_time) * 1000, 1),
        }

        # 13. Return tier response with actions
        response = create_tier_response(
            tier=tier.value,
            gradcam_url=None,
            metadata=metadata,
            actions=tier_result.actions,
            disclaimer=tier_result.disclaimer,
            tier_label=tier_result.tier_label,
        )
    except (KeyError, ValueError) as e:
        logger.exception("malformed ensemble result: %s", e)
        return jsonify(create_error_response("server_error").to_json()), 500

    return jsonify(response.to_json())