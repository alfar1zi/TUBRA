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
    # --- Stage 2: spectral gate (cough vs noise/music/speech) ---
    # A cough is a sharp, broadband, high-frequency burst. Tones, music and
    # voiced speech concentrate their energy in a few low/mid bands, so a
    # spectral-centroid floor plus a zero-crossing-rate window separate them
    # without rejecting real coughs. Defaults were calibrated on the CODA-TB
    # recordings (>90% of real cough loud-frames pass) and reject tones,
    # music, speech and low-frequency thuds.
    use_spectral_gate: bool = True
    min_spectral_centroid_hz: float = 350.0
    min_zero_crossing_rate: float = 0.03
    max_zero_crossing_rate: float = 0.34


AUDIO_CONFIG = AudioConfig()