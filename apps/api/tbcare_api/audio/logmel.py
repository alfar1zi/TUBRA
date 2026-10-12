import numpy as np
from .config import AUDIO_CONFIG


def hz_to_mel(hz: float) -> float:
    """Convert Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert Mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """
    Create triangular Mel filterbank.
    Returns filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    # Mel frequency bins
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        if center > left:
            filterbank[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filterbank[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    return filterbank


def compute_stft(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform magnitude squared.
    Returns power spectrogram of shape (n_frames, n_fft // 2 + 1).
    """
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames <= 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)

    stft = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        frame = audio[start:end]

        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))

        # Apply window
        frame = frame * window

        # FFT
        fft_result = np.fft.rfft(frame)
        power = np.abs(fft_result) ** 2
        stft[i] = power

    return stft


def compute_logmel(
    audio: np.ndarray,
    sample_rate: int,
    config=None,
) -> np.ndarray:
    """
    Compute log-Mel spectrogram from audio.
    Args:
        audio: 1D float32 array, normalized to [-1, 1], at sample_rate
        sample_rate: sample rate in Hz
        config: AudioConfig (optional)
    Returns:
        Log-Mel spectrogram of shape (n_frames, n_mels), float32
    """
    if config is None:
        config = AUDIO_CONFIG

    n_fft = config.frame_length
    hop_length = config.hop_length
    n_mels = config.n_mels
    f_min = config.f_min
    f_max = min(config.f_max, sample_rate / 2)

    # Hann window
    window = np.hanning(n_fft).astype(np.float32)

    # STFT power spectrogram
    power_spec = compute_stft(audio, n_fft, hop_length, window)

    if power_spec.shape[0] == 0:
        return np.zeros((0, n_mels), dtype=np.float32)

    # Mel filterbank
    mel_fb = create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)

    # Apply filterbank
    mel_spec = power_spec @ mel_fb.T  # (n_frames, n_mels)

    # Log compression
    log_mel = np.log(mel_spec + 1e-10).astype(np.float32)

    return log_mel


def normalize_logmel(log_mel: np.ndarray) -> np.ndarray:
    """
    Normalize log-Mel spectrogram per spectrogram (zero mean, unit std).
    Args:
        log_mel: (n_frames, n_mels)
    Returns:
        Normalized log-Mel of same shape
    """
    if log_mel.size == 0:
        return log_mel

    mean = np.mean(log_mel, axis=0, keepdims=True)
    std = np.std(log_mel, axis=0, keepdims=True) + 1e-10
    normalized = (log_mel - mean) / std
    return normalized.astype(np.float32)


def process_audio_to_logmel(
    audio: np.ndarray,
    sample_rate: int,
    config=None,
    target_frames: int | None = None,
) -> np.ndarray:
    """
    Full pipeline: resample (if needed) -> log-Mel -> normalize.
    Returns normalized log-Mel spectrogram of shape (n_mels, n_frames).
    """
    if config is None:
        config = AUDIO_CONFIG

    target_sr = config.sample_rate

    # Resample if needed (should already be done in quality pipeline)
    from .resample import ensure_sample_rate
    audio, sr = ensure_sample_rate(audio, sample_rate, target_sr)

    # Compute log-Mel
    log_mel = compute_logmel(audio, sr, config)

    # Pad or truncate to target_frames
    if target_frames is not None:
        n_frames = log_mel.shape[0]
        if n_frames < target_frames:
            # Pad with zeros
            pad_width = ((0, target_frames - n_frames), (0, 0))
            log_mel = np.pad(log_mel, pad_width, mode='constant', constant_values=-80)
        elif n_frames > target_frames:
            # Truncate
            log_mel = log_mel[:target_frames]

    # Normalize
    log_mel_norm = normalize_logmel(log_mel)

    # Transpose to (n_mels, n_frames) for consistency
    return log_mel_norm.T