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


# Feature-extraction constants. These MUST match the training recipe exported
# into ml/artifacts/model_manifest.json (``audio_config``), otherwise the ONNX
# model receives out-of-distribution features at serving time.
#
# Ground truth (model_manifest.json, written by export_artifacts.py from
# train.yaml -> AudioFeatureConfig): recipe=energy_v2, n_fft=1024,
# hop_length=256, n_mels=64, f_min=0.0, f_max=None (=> Nyquist), top_db=80.
# ml/configs/audio.yaml still lists the older 512/160/50/8000 values and is NOT
# what the exported model was trained with; do not copy from it.
# AGENTS.md section 4.5 requires a log-Mel parity test with max diff <= 1e-4.
_FEATURE_SAMPLE_RATE = 16000
_FEATURE_N_FFT = 1024
_FEATURE_HOP_LENGTH = 256
_FEATURE_N_MELS = 64
_FEATURE_F_MIN = 0.0
_FEATURE_F_MAX = None  # None => sample_rate / 2
_FEATURE_TOP_DB = 80.0
_FEATURE_ENERGY_WINDOW_S = 1.5
_FEATURE_TARGET_FRAMES = 64


def _select_energy_window(
    samples: np.ndarray,
    sample_rate: int,
    window_s: float,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Select the highest-energy window of fixed duration.

    Mirrors ``ml/src/features/audio.py::_select_energy_window`` exactly.
    """
    target_len = int(window_s * sample_rate)
    if len(samples) <= target_len:
        return np.pad(samples, (0, target_len - len(samples)))

    n_frames = 1 + (len(samples) - n_fft) // hop_length
    if n_frames <= 0:
        return samples[:target_len]

    energies = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = samples[start:start + n_fft]
        energies[i] = np.mean(frame ** 2)

    best_frame = int(np.argmax(energies))
    start_sample = best_frame * hop_length
    end_sample = start_sample + target_len

    if end_sample > len(samples):
        end_sample = len(samples)
        start_sample = max(0, end_sample - target_len)

    return samples[start_sample:end_sample]


def _normalize_and_resize(values: np.ndarray, target_frames: int) -> np.ndarray:
    """Min-max normalize to [0, 1] and resize to target frames.

    Mirrors ``ml/src/features/audio.py::_normalize_and_resize``.
    ``values`` is (n_mels, n_frames); returns (n_mels, target_frames).
    """
    min_val = float(values.min())
    max_val = float(values.max())

    if max_val > min_val:
        normalized = (values - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(values, dtype=np.float32)

    if normalized.shape[1] != target_frames:
        from scipy.ndimage import zoom
        zoom_factor = target_frames / normalized.shape[1]
        normalized = zoom(normalized, (1, zoom_factor), order=1)

    return normalized.astype(np.float32)


def process_audio_to_logmel(
    audio: np.ndarray,
    sample_rate: int,
    config=None,
    target_frames: int | None = None,
) -> np.ndarray:
    """Training-identical log-Mel pipeline (recipe ``energy_v2``).

    Steps (must stay in lockstep with ``ml/src/features/audio.py``):
      1. resample to 16 kHz
      2. select the highest-energy 1.5 s window
      3. STFT with scipy (Hann window, boundary=None, padded=False)
      4. HTK mel filterbank
      5. 10*log10 with a -80 dB floor, then map to [0, 1] via (x+top_db)/top_db
      6. min-max normalize to [0, 1] and resize to ``target_frames``

    Returns (n_mels, target_frames), matching the ONNX ``logmel`` input.
    """
    if config is None:
        config = AUDIO_CONFIG
    if target_frames is None:
        target_frames = _FEATURE_TARGET_FRAMES

    from .resample import ensure_sample_rate

    audio, sr = ensure_sample_rate(audio, sample_rate, _FEATURE_SAMPLE_RATE)
    audio = np.asarray(audio, dtype=np.float32)

    # 2. Highest-energy window (training does this too).
    audio = _select_energy_window(
        audio, sr, _FEATURE_ENERGY_WINDOW_S, _FEATURE_N_FFT, _FEATURE_HOP_LENGTH
    )

    # 3. STFT, matching scipy.signal.stft(boundary=None, padded=False).
    window = np.hanning(_FEATURE_N_FFT).astype(np.float32)
    n_frames = 1 + (len(audio) - _FEATURE_N_FFT) // _FEATURE_HOP_LENGTH
    if n_frames <= 0:
        return np.zeros((_FEATURE_N_MELS, target_frames), dtype=np.float32)

    from scipy import signal as _signal
    stft = _signal.stft(
        audio,
        fs=sr,
        window=window,
        nperseg=_FEATURE_N_FFT,
        noverlap=_FEATURE_N_FFT - _FEATURE_HOP_LENGTH,
        nfft=_FEATURE_N_FFT,
        boundary=None,
        padded=False,
    )[2]
    magnitude = np.abs(stft) ** 2.0

    # 4. HTK mel filterbank (identical formula to training). f_max=None means
    #    Nyquist, matching AudioFeatureConfig's default.
    mel_f_max = _FEATURE_F_MAX if _FEATURE_F_MAX is not None else sr / 2.0
    mel_fb = create_mel_filterbank(
        sr, _FEATURE_N_FFT, _FEATURE_N_MELS, _FEATURE_F_MIN, mel_f_max
    )
    mel_spec = mel_fb @ magnitude  # (n_mels, n_frames)

    # 5. dB + [0, 1] normalization.
    log_mel = 10.0 * np.log10(
        np.maximum(mel_spec, 10 ** (-_FEATURE_TOP_DB / 10.0))
    )
    log_mel = (log_mel + _FEATURE_TOP_DB) / _FEATURE_TOP_DB

    # 6. Min-max normalize + resize, identical to training.
    return _normalize_and_resize(log_mel.astype(np.float32), target_frames)