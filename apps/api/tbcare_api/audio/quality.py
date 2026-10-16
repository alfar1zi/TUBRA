import numpy as np
from .wav_io import validate_wav, WAVValidationError
from .resample import ensure_sample_rate
from .events import quality_gate, detect_cough_events
from .cough_segments import extract_segments, segment_coughs
from .config import AudioConfig, AUDIO_CONFIG


def preprocess_audio(
    wav_bytes: bytes,
    target_sr: int = 16000,
    config: AudioConfig | None = None,
) -> tuple[np.ndarray, int, dict]:
    """
    Full preprocessing pipeline: validate WAV, resample, peak normalize.
    Returns (audio_float32, sample_rate, metadata).
    """
    if config is None:
        config = AUDIO_CONFIG

    # Validate and load WAV
    audio_int16, orig_sr = validate_wav(wav_bytes)

    # Convert to float32 [-1, 1]
    audio_float = audio_int16.astype(np.float32) / 32768.0

    # Resample to target sample rate
    audio_float, sr = ensure_sample_rate(audio_float, orig_sr, target_sr)

    # Peak normalization (METHOD.md 3.1)
    peak = np.max(np.abs(audio_float)) + 1e-12
    audio_norm = audio_float / peak

    metadata = {
        "original_sample_rate": orig_sr,
        "target_sample_rate": sr,
        "peak_normalization_factor": float(peak),
        "duration_s": len(audio_norm) / sr,
        "num_samples": len(audio_norm),
    }

    return audio_norm, sr, metadata


def run_quality_pipeline(
    wav_bytes: bytes,
    target_sr: int = 16000,
    config: AudioConfig | None = None,
) -> tuple[bool, dict, np.ndarray, int]:
    """
    Run full quality check pipeline on raw WAV bytes.
    Returns (passed, quality_details, audio_norm, sample_rate).
    """
    if config is None:
        config = AUDIO_CONFIG

    # Preprocess
    audio_norm, sr, preprocess_meta = preprocess_audio(wav_bytes, target_sr, config)

    # Run quality gate
    passed, quality_details = quality_gate(audio_norm, sr, config)

    # Merge metadata
    quality_details.update(preprocess_meta)

    return passed, quality_details, audio_norm, sr


def extract_cough_segments(
    audio: np.ndarray,
    sample_rate: int,
    config: AudioConfig | None = None,
    max_segments: int = 5,
) -> list[np.ndarray]:
    """Cut up to ``max_segments`` coughs, using YOHO when the model is present.

    The energy gate runs first: it is cheap and its events are what YOHO falls
    back to. When YOHO finds nothing there is nothing to prefer it for, so the
    energy events stand.
    """
    if config is None:
        config = AUDIO_CONFIG

    energy_events, _, _ = detect_cough_events(audio, sample_rate, config)
    events, _backend = segment_coughs(audio, sample_rate, energy_events)
    return extract_segments(audio, sample_rate, events, max_segments)
