"""
POST /api/v1/screen endpoint.
Integrates: audio validation -> quality gate -> logmel -> backbone -> head -> calibrate -> tiers -> explain -> response.
"""

import io
import time
import json
from flask import Blueprint, request, jsonify, current_app

from tbcare_api.audio.wav_io import validate_wav, WAVValidationError, compute_audio_metrics
from tbcare_api.audio.quality import run_quality_pipeline, extract_cough_segments
from tbcare_api.audio.logmel import process_audio_to_logmel
from tbcare_api.inference.backbone import get_backbone
from tbcare_api.inference.head import get_head
from tbcare_api.inference.tiers import get_tier_thresholds, probability_to_tier
from tbcare_api.explain.gradcam import compute_gradcam, aggregate_heatmaps, resize_heatmap
from tbcare_api.explain.occlusion import compute_occlusion, format_symptom_explanation
from tbcare_api.schemas.payload import ScreenPayload
from tbcare_api.schemas.response import (
    create_tier_response,
    create_error_response,
    ScreenResponse,
    ErrorResponse,
)

screen_bp = Blueprint("screen", __name__)

# Load model_card.json for thresholds
def load_model_card():
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "artifacts"
    card_path = base / "model_card.json"
    if card_path.exists():
        with open(card_path) as f:
            return json.load(f)
    return {"tier_thresholds": {"A": 0.7, "B": 0.4, "C": 0.0}}


MODEL_CARD = load_model_card()
TIER_THRESHOLDS = get_tier_thresholds(MODEL_CARD)


@screen_bp.route("/api/v1/screen", methods=["POST"])
def screen():
    start_time = time.time()

    # 1. Validate multipart form data
    if "audio" not in request.files:
        return jsonify(create_error_response("invalid_audio", {"reason": "audio file missing"}).to_json()), 400

    if "payload" not in request.form:
        return jsonify(create_error_response("invalid_payload", {"reason": "payload missing"}).to_json()), 400

    audio_file = request.files["audio"]
    payload_json = request.form["payload"]

    # 2. Parse and validate payload
    try:
        payload_data = json.loads(payload_json)
        payload = ScreenPayload.from_json(payload_data)
    except json.JSONDecodeError as e:
        return jsonify(create_error_response("invalid_payload", {"reason": f"invalid JSON: {e}"}).to_json()), 400
    except ValueError as e:
        return jsonify(create_error_response("invalid_payload", {"reason": str(e)}).to_json()), 400

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
    except Exception as e:
        return jsonify(create_error_response("quality_failed", {"reason": str(e)}).to_json()), 500

    if not passed:
        # Quality failed - return specific reason
        reason = quality_details.get("reason", "quality check failed")
        cough_count = quality_details.get("cough_count", 0)
        return jsonify(create_error_response("quality_failed", {
            "reason": reason,
            "cough_count": cough_count,
            "snr_db": quality_details.get("snr_db"),
            "clipping_ratio": quality_details.get("clipping_ratio"),
        }).to_json()), 400

    # 6. Extract cough segments (max 5)
    segments = extract_cough_segments(audio_norm, sr, max_segments=5)

    if len(segments) < 3:
        return jsonify(create_error_response("quality_failed", {
            "reason": f"only {len(segments)} cough segments extracted, need at least 3",
            "cough_count": len(segments),
        }).to_json()), 400

    # 7. Convert segments to log-Mel spectrograms
    try:
        logmel_batch = []
        # Model expects 5s worth of frames at 16kHz with 160 hop = 157 frames
        target_frames = 157
        for seg in segments:
            logmel = process_audio_to_logmel(seg, sr, target_frames=target_frames)
            logmel_batch.append(logmel)
        logmel_batch = np.stack(logmel_batch, axis=0)  # (N, n_mels, n_frames)
    except Exception as e:
        return jsonify(create_error_response("inference_error", {"reason": f"logmel failed: {e}"}).to_json()), 500

    # 8. Run backbone inference
    try:
        backbone = get_backbone()
        embeddings, feature_maps = backbone.run(logmel_batch)
    except Exception as e:
        return jsonify(create_error_response("inference_error", {"reason": f"backbone failed: {e}"}).to_json()), 500

    # 9. Prepare symptom vector
    symptoms, availability = payload.to_symptom_vector()
    symptoms = np.array(symptoms, dtype=np.float32)
    availability = np.array(availability, dtype=np.float32)

    # 10. Run head inference
    try:
        head = get_head()
        head_result = head.forward(embeddings, symptoms, availability)
    except Exception as e:
        return jsonify(create_error_response("inference_error", {"reason": f"head failed: {e}"}).to_json()), 500

    probability = head_result["probability"]
    tier = probability_to_tier(probability, TIER_THRESHOLDS)

    # 11. Compute Grad-CAM
    gradcam_url = None
    gradcam_data = None
    try:
        head_npz = np.load(
            current_app.config.get("HEAD_PATH")
            or (Path(__file__).resolve().parent.parent.parent.parent / "ml" / "artifacts" / "head.npz")
        )
        heatmaps = compute_gradcam(
            feature_maps=feature_maps,
            head_W1=head_npz["W1"],
            head_w2=head_npz["w2"],
            head_b1=head_npz["b1"],
            symptoms=symptoms,
            availability=availability,
            embeddings=embeddings,
        )
        # Aggregate to single heatmap
        agg_heatmap = aggregate_heatmaps(heatmaps, method="mean")
        # Resize to match original spectrogram size (64x64)
        agg_heatmap = resize_heatmap(agg_heatmap, (64, 64))
        gradcam_data = agg_heatmap.tolist()

        # For now, embed as base64 data URL (could also serve as separate endpoint)
        import base64
        heatmap_bytes = (agg_heatmap * 255).astype(np.uint8).tobytes()
        # Simple PNG encoding would need PIL - skip for now, return array
        # gradcam_url = f"data:image/png;base64,{base64.b64encode(heatmap_bytes).decode()}"
    except Exception:
        # Grad-CAM failure shouldn't block the response
        pass

    # 12. Compute symptom occlusion
    symptom_explanations = []
    try:
        head = get_head()
        occlusion_results = compute_occlusion(head, embeddings, symptoms, availability)
        symptom_explanations = format_symptom_explanation(occlusion_results)
    except Exception:
        pass

    # 13. Build metadata
    metadata = {
        "model_version": MODEL_CARD.get("model_version", "dev"),
        "probability": round(probability, 4),
        "logit": round(head_result["logit"], 4),
        "thresholds": TIER_THRESHOLDS,
        "cough_segments_used": len(segments),
        "audio_metrics": audio_metrics,
        "quality_details": quality_details,
        "processing_time_ms": round((time.time() - start_time) * 1000, 1),
        "symptom_explanations": symptom_explanations,
        "gradcam": gradcam_data,
    }

    # 14. Return tier response
    response = create_tier_response(tier, gradcam_url=gradcam_url, metadata=metadata)
    return jsonify(response.to_json())


# Import at module level to avoid circular import
import numpy as np
from pathlib import Path