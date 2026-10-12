import io
import wave
import numpy as np


class WAVValidationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def validate_wav(data: bytes) -> tuple[np.ndarray, int]:
    """
    Validate WAV file is PCM 16-bit mono.
    Returns (audio_samples_int16, sample_rate).
    Raises WAVValidationError if invalid.
    """
    if len(data) < 44:
        raise WAVValidationError("File too small to be a valid WAV")

    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            if wf.getnchannels() != 1:
                raise WAVValidationError("Audio must be mono (1 channel)")
            if wf.getsampwidth() != 2:
                raise WAVValidationError("Audio must be 16-bit PCM")
            if wf.getcomptype() != "NONE":
                raise WAVValidationError("Audio must be uncompressed PCM")

            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)

            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_int16, sample_rate

    except wave.Error as e:
        raise WAVValidationError(f"Invalid WAV format: {e}")


def compute_audio_metrics(audio_int16: np.ndarray, sample_rate: int) -> dict:
    """Compute basic audio metrics from int16 samples."""
    if len(audio_int16) == 0:
        return {
            "sample_rate": sample_rate,
            "duration_s": 0.0,
            "peak": 0.0,
            "rms": 0.0,
            "num_samples": 0,
        }

    audio_float = audio_int16.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(audio_float)))
    rms = float(np.sqrt(np.mean(audio_float ** 2)))
    duration = len(audio_int16) / sample_rate

    return {
        "sample_rate": sample_rate,
        "duration_s": round(duration, 3),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "num_samples": int(len(audio_int16)),
    }