from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal
from scipy import signal
from scipy.io import wavfile
import io

# =============================================================================
# Audio Feature Recipes
# =============================================================================

RecipeName = Literal["legacy_v1", "energy_v2", "tbscreen_v1"]

SUPPORTED_RECIPES = frozenset({"legacy_v1", "energy_v2", "tbscreen_v1"})


@dataclass(frozen=True)
class AudioFeatureConfig:
    """Configuration for log-mel spectrogram extraction.

    Recipes:
    - legacy_v1: Original TBCare implementation (librosa-style)
    - energy_v2: SuaraNafas ENERGY_LOG_MEL_V2 — energy window + HTK mel + anti-alias resample
    - tbscreen_v1: SuaraNafas TBSCREEN_LOG_MEL_V1 — centered STFT + reflect padding
    """
    recipe: RecipeName = "energy_v2"
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None
    target_frames: int = 91
    center: bool = False
    pad_mode: str = "constant"
    power: float = 2.0
    top_db: float | None = None
    energy_window_s: float = 1.5  # For energy_v2: select highest-energy window
    duration_seconds: float = 1.5

    def __post_init__(self) -> None:
        if self.recipe not in SUPPORTED_RECIPES:
            raise ValueError(f"Unsupported recipe: {self.recipe}. Choose from {SUPPORTED_RECIPES}")
        if self.recipe == "tbscreen_v1" and not (self.center and self.pad_mode == "reflect"):
            raise ValueError("tbscreen_v1 requires center=True and pad_mode='reflect'")
        # energy_v2 / legacy_v1 rely on a dB floor + normalization; default it.
        if self.top_db is None and self.recipe in {"energy_v2", "legacy_v1"}:
            object.__setattr__(self, "top_db", 80.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "AudioFeatureConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("audio", {}))


# =============================================================================
# Core DSP Functions
# =============================================================================

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _htk_mel_filterbank(config: AudioFeatureConfig) -> np.ndarray:
    """HTK-style mel filterbank (standard in speech processing)."""
    max_freq = config.f_max or config.sample_rate / 2.0
    mel_min = _hz_to_mel(config.f_min)
    mel_max = _hz_to_mel(max_freq)
    mel_points = np.linspace(mel_min, mel_max, config.n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((config.n_fft + 1) * hz_points / config.sample_rate).astype(int)

    filterbank = np.zeros((config.n_mels, config.n_fft // 2 + 1), dtype=np.float32)

    for m in range(1, config.n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        if center > left:
            filterbank[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filterbank[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    return filterbank.astype(np.float32)


def _legacy_mel_filterbank(config: AudioFeatureConfig) -> np.ndarray:
    """Original librosa-style linear mel filterbank."""
    frequencies = np.linspace(0.0, config.sample_rate / 2.0, config.n_fft // 2 + 1)
    mel_f = _hz_to_mel(frequencies)
    mel_min = _hz_to_mel(config.f_min)
    mel_max = _hz_to_mel(config.f_max if config.f_max is not None else config.sample_rate / 2.0)
    mel_bins = np.linspace(mel_min, mel_max, config.n_mels + 2)

    filterbank = np.zeros((config.n_mels, config.n_fft // 2 + 1), dtype=np.float32)

    for m in range(1, config.n_mels + 1):
        left_mel = mel_bins[m - 1]
        center_mel = mel_bins[m]
        right_mel = mel_bins[m + 1]

        left_slope = (mel_f - left_mel) / (center_mel - left_mel)
        right_slope = (right_mel - mel_f) / (right_mel - center_mel)
        filterbank[m - 1] = np.maximum(0, np.minimum(left_slope, right_slope))

    return filterbank.astype(np.float32)


def _anti_aliased_resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Anti-aliased resampling via polyphase filtering.

    Delegates to :func:`scipy.signal.resample_poly`, the same primitive the
    serving path uses (``apps/api/tbcare_api/audio/resample.py``). Keeping both
    paths on one implementation guarantees the training and serving feature
    pipelines see identical audio.

    The previous hand-rolled Kaiser kernel silently attenuated the signal by
    roughly 50x (the convolution was not scaled by the upsampling factor), so
    every 44.1 kHz training clip was mangled relative to serving. Do not
    reimplement this by hand.
    """
    if source_rate == target_rate:
        return samples.astype(np.float32)

    from math import gcd

    g = gcd(source_rate, target_rate)
    up = target_rate // g
    down = source_rate // g
    resampled = signal.resample_poly(samples.astype(np.float64), up, down)
    return resampled.astype(np.float32)


def _linear_resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Simple linear interpolation resampling."""
    if source_rate == target_rate:
        return samples.astype(np.float32)
    duration = len(samples) / source_rate
    target_len = int(duration * target_rate)
    source_pos = np.linspace(0, len(samples) - 1, len(samples))
    target_pos = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(target_pos, source_pos, samples).astype(np.float32)


def _select_energy_window(samples: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    """Select highest-energy window of fixed duration."""
    target_len = int(config.energy_window_s * config.sample_rate)
    if len(samples) <= target_len:
        return np.pad(samples, (0, target_len - len(samples)))

    frame_len = config.n_fft
    hop = config.hop_length
    n_frames = 1 + (len(samples) - frame_len) // hop

    energies = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = samples[start:start + frame_len]
        energies[i] = np.mean(frame ** 2)

    best_frame = np.argmax(energies)
    start_sample = best_frame * hop
    end_sample = start_sample + target_len

    if end_sample > len(samples):
        end_sample = len(samples)
        start_sample = max(0, end_sample - target_len)

    return samples[start_sample:end_sample]


def _normalize_and_resize(values: np.ndarray, target_frames: int) -> np.ndarray:
    """Normalize to [0, 1] and resize to target frames."""
    min_val = float(values.min())
    max_val = float(values.max())

    if max_val > min_val:
        normalized = (values - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(values, dtype=np.float32)

    # Resize via interpolation
    if normalized.shape[1] != target_frames:
        from scipy.ndimage import zoom
        zoom_factor = target_frames / normalized.shape[1]
        normalized = zoom(normalized, (1, zoom_factor), order=1)

    return normalized.astype(np.float32)[None, :, :]  # (1, n_mels, target_frames)


# =============================================================================
# Recipe Implementations
# =============================================================================

def _extract_legacy_v1(samples: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    """Original TBCare recipe: librosa-style STFT + linear mel."""
    window = np.hanning(config.n_fft).astype(np.float32)
    stft = signal.stft(
        samples,
        fs=config.sample_rate,
        window=window,
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length,
        nfft=config.n_fft,
        boundary=None,
        padded=False,
    )[2]
    magnitude = np.abs(stft) ** config.power
    mel_fb = _legacy_mel_filterbank(config)
    mel_spec = mel_fb @ magnitude
    log_mel = np.log10(mel_spec + 1e-10)
    return _normalize_and_resize(log_mel, config.target_frames)


def _extract_energy_v2(samples: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    """SuaraNafas ENERGY_LOG_MEL_V2: energy window + HTK mel + anti-alias resample."""
    # Select highest-energy window
    samples = _select_energy_window(samples, config)

    # STFT
    window = np.hanning(config.n_fft).astype(np.float32)
    stft = signal.stft(
        samples,
        fs=config.sample_rate,
        window=window,
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length,
        nfft=config.n_fft,
        boundary=None,
        padded=False,
    )[2]
    magnitude = np.abs(stft) ** config.power
    mel_fb = _htk_mel_filterbank(config)
    mel_spec = mel_fb @ magnitude

    # Convert to dB
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, 10 ** (-config.top_db / 10.0)))
    log_mel = (log_mel + config.top_db) / config.top_db  # Normalize to [0, 1]

    return _normalize_and_resize(log_mel, config.target_frames)


def _extract_tbscreen_v1(samples: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    """SuaraNafas TBSCREEN_LOG_MEL_V1: centered STFT + reflect padding."""
    if not config.center or config.pad_mode != "reflect":
        raise ValueError("tbscreen_v1 requires center=True and pad_mode='reflect'")

    window = np.hanning(config.n_fft).astype(np.float32)
    stft = signal.stft(
        samples,
        fs=config.sample_rate,
        window=window,
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length,
        nfft=config.n_fft,
        boundary=config.pad_mode,
        padded=True,
    )[2]
    magnitude = np.abs(stft) ** config.power
    mel_fb = _htk_mel_filterbank(config)
    mel_spec = mel_fb @ magnitude

    # Clip to avoid log(0), then convert to dB
    mel_spec = np.maximum(mel_spec, 1e-10)
    decibels = 10.0 * np.log10(mel_spec)

    # Normalize per utterance
    mean = float(decibels.mean())
    std = float(decibels.std()) + 1e-8
    normalized = (decibels - mean) / std

    return _normalize_and_resize(normalized, config.target_frames)


# =============================================================================
# Public API
# =============================================================================

def extract_logmel(
    samples: np.ndarray,
    sample_rate: int,
    config: AudioFeatureConfig | None = None,
) -> np.ndarray:
    """Extract log-mel spectrogram from audio samples.

    Args:
        samples: Mono audio as float32 in [-1, 1]
        sample_rate: Input sample rate
        config: Feature configuration (uses default if None)

    Returns:
        Log-mel spectrogram of shape (1, n_mels, target_frames)
    """
    if config is None:
        config = AudioFeatureConfig()

    # Resample if needed
    if sample_rate != config.sample_rate:
        if config.recipe == "energy_v2":
            samples = _anti_aliased_resample(samples, sample_rate, config.sample_rate)
        else:
            samples = _linear_resample(samples, sample_rate, config.sample_rate)

    # Peak normalization, matching the serving pipeline
    # (`apps/api/tbcare_api/audio/quality.py::preprocess_audio`) so training and
    # serving see identical audio. Done after resampling, as serving does.
    peak = float(np.max(np.abs(samples))) + 1e-12
    samples = samples / peak

    # Dispatch to recipe
    if config.recipe == "legacy_v1":
        return _extract_legacy_v1(samples, config)
    elif config.recipe == "energy_v2":
        return _extract_energy_v2(samples, config)
    elif config.recipe == "tbscreen_v1":
        return _extract_tbscreen_v1(samples, config)
    else:
        raise ValueError(f"Unknown recipe: {config.recipe}")


def extract_logmel_from_wav(
    wav_bytes: bytes,
    config: AudioFeatureConfig | None = None,
) -> np.ndarray:
    """Extract log-mel directly from WAV bytes."""
    sample_rate, samples = wavfile.read(io.BytesIO(wav_bytes))
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    elif samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    # Handle stereo: take first channel
    if samples.ndim > 1:
        samples = samples[:, 0]

    return extract_logmel(samples, sample_rate, config)


# =============================================================================
# SpecAugment (Training-only Augmentation)
# =============================================================================

@dataclass(frozen=True)
class SpecAugmentConfig:
    """SpecAugment configuration (Park et al., 2019)."""
    time_mask_param: int = 8
    freq_mask_param: int = 8
    num_time_masks: int = 2
    num_freq_masks: int = 2
    fill_value: float = 0.0


def spec_augment(
    log_mel: np.ndarray,
    config: SpecAugmentConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Apply SpecAugment to log-mel tensor.

    Args:
        log_mel: Shape (1, n_mels, n_frames) or (n_mels, n_frames)
        config: Augmentation config
        seed: Optional seed for the mask RNG. Pass one (or rely on
            ``np.random.seed`` via ``set_seed``) for reproducible runs; without
            a seed this draws from OS entropy and breaks determinism.

    Returns:
        Augmented log-mel of same shape
    """
    if config is None:
        config = SpecAugmentConfig()

    # Ensure (n_mels, n_frames)
    if log_mel.ndim == 3 and log_mel.shape[0] == 1:
        log_mel = log_mel[0]

    n_mels, n_frames = log_mel.shape
    out = log_mel.copy()

    # Seeded when given; otherwise inherit numpy's global seed, which
    # set_seed() fixes, so a training run stays reproducible.
    rng = np.random.default_rng(seed) if seed is not None else np.random

    # Frequency masks
    for _ in range(config.num_freq_masks):
        f = rng.integers(0, min(config.freq_mask_param, n_mels))
        f0 = rng.integers(0, n_mels - f + 1)
        out[f0:f0 + f, :] = config.fill_value

    # Time masks
    for _ in range(config.num_time_masks):
        t = rng.integers(0, min(config.time_mask_param, n_frames))
        t0 = rng.integers(0, n_frames - t + 1)
        out[:, t0:t0 + t] = config.fill_value

    # Restore batch dim if input had it
    if log_mel.ndim == 2:
        return out[None, :, :]
    return out

# =============================================================================
# Parity Check (Training vs Serving)
# =============================================================================

def check_parity(
    wav_bytes: bytes,
    serving_config: AudioFeatureConfig,
    training_config: AudioFeatureConfig,
    tolerance: float = 1e-4,
) -> tuple[bool, float]:
    """Verify training and serving produce identical features.

    Returns:
        (passed, max_abs_diff)
    """
    serving_feat = extract_logmel_from_wav(wav_bytes, serving_config)
    training_feat = extract_logmel_from_wav(wav_bytes, training_config)

    max_diff = float(np.max(np.abs(serving_feat - training_feat)))
    return max_diff <= tolerance, max_diff