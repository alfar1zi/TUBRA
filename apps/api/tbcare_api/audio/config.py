from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_length: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 8000.0
    target_cough_duration_s: float = 0.5
    min_cough_duration_s: float = 0.1
    max_cough_duration_s: float = 2.0
    noise_percentile: float = 10.0
    noise_margin_db: float = 6.0
    hysteresis_db: float = 3.0
    snr_threshold_db: float = 10.0
    clipping_threshold_ratio: float = 0.01
    min_coughs_required: int = 3
    max_coughs_expected: int = 10


AUDIO_CONFIG = AudioConfig()