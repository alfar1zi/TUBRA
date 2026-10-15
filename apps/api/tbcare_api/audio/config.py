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
    snr_threshold_db: float = 8.0
    clipping_threshold_ratio: float = 0.01
    # The web flow records one 15 s take and asks the user for three to five
    # coughs. A take that yields fewer than three segmented bursts is not the
    # input the model was validated on, so it is sent back for re-recording
    # instead of being accepted on a single marginal burst.
    min_coughs_required: int = 3
    max_coughs_expected: int = 10
    # A single burst this far above the room noise is unambiguous on its own,
    # so one detected cough is accepted instead of demanding more. CODA-TB
    # real coughs sit at a median margin of ~25 dB; 22 dB keeps the fallback
    # for a genuinely loud isolated cough while rejecting weak transients.
    strong_burst_margin_db: float = 22.0
    # --- Stage 2: spectral gate (cough vs noise/music/speech) ---
    # A cough is a sharp, broadband, high-frequency burst. Tones, music and
    # voiced speech concentrate their energy in a few low/mid bands, so a
    # spectral-centroid floor plus a zero-crossing-rate window separate them
    # without rejecting real coughs. Defaults were calibrated on the CODA-TB
    # recordings (>90% of real cough loud-frames pass) and reject tones,
    # music, speech and low-frequency thuds.
    use_spectral_gate: bool = True
    min_spectral_centroid_hz: float = 350.0
    max_zero_crossing_rate: float = 0.7
    # Longest burst accepted as one cough. On the 160 CODA-TB coughs every
    # burst ran 12-17 frames, while speech, a held note and a snore run 39-49
    # and a hand clap 2. Burst length is what separates them: a third of real
    # coughs have a zero-crossing rate as high as a clap, so the shape checks
    # alone (centroid floor, loose ZCR cap) only catch lows and transients.
    max_burst_frames: int = 28


AUDIO_CONFIG = AudioConfig()