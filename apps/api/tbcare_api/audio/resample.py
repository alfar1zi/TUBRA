import numpy as np
from scipy.signal import resample_poly


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio from orig_sr to target_sr using polyphase resampling.
    Args:
        audio: 1D float32 array, normalized to [-1, 1]
        orig_sr: original sample rate
        target_sr: target sample rate
    Returns:
        Resampled audio as float32 array
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    # Use resample_poly for high-quality rational resampling
    # Calculate up/down factors
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    # Limit up/down to reasonable values to avoid memory issues
    max_factor = 1000
    if up > max_factor or down > max_factor:
        # Fall back to simpler ratio
        scale = max(up, down) / max_factor
        up = int(up / scale)
        down = int(down / scale)
        g = gcd(up, down)
        up = up // g
        down = down // g

    resampled = resample_poly(audio, up, down).astype(np.float32)
    return resampled


def ensure_sample_rate(audio: np.ndarray, sample_rate: int, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Ensure audio is at target sample rate. Resample if needed.
    Returns (audio_at_target_sr, target_sr).
    """
    if sample_rate != target_sr:
        audio = resample_audio(audio, sample_rate, target_sr)
        sample_rate = target_sr
    return audio, sample_rate