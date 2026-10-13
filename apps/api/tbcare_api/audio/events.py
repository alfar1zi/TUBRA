import numpy as np
from .config import AUDIO_CONFIG


def rms_energy_frames(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    Compute RMS energy per frame.
    Args:
        audio: 1D float32 array, normalized to [-1, 1]
        frame_length: frame size in samples
        hop_length: hop size in samples
    Returns:
        1D array of RMS energy per frame
    """
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    if n_frames <= 0:
        return np.array([], dtype=np.float32)

    energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        energies[i] = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
    return energies


def spectral_frame_features(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame spectral features used to tell a cough apart from noise.

    A cough is a broadband, sharp, high-frequency burst. Phone-microphone
    noise, music and sustained tones are either narrowband or low-frequency,
    so an energy-only gate counts them as coughs. These three features
    separate them:

    - ``centroid_hz``: spectral centroid (Hz). Cough energy sits high
      (a few kHz); hums/thuds sit low.
    - ``flatness``: geometric/arithmetic mean of the power spectrum in
      [0, 1]. Near 1 for broadband noise (cough), near 0 for tones/music.
    - ``zcr``: zero-crossing rate in [0, 1]; high for broadband bursts.

    Returns three arrays aligned with :func:`rms_energy_frames`.
    """
    win = np.hanning(frame_length).astype(np.float32)
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    if n_frames <= 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    freqs = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
    centroid = np.zeros(n_frames, dtype=np.float32)
    flatness = np.zeros(n_frames, dtype=np.float32)
    zcr = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        windowed = frame * win
        power = np.abs(np.fft.rfft(windowed)) ** 2 + 1e-12
        total = float(np.sum(power))
        if total <= 0:
            continue
        prob = power / total
        centroid[i] = float(np.sum(freqs * prob))
        flatness[i] = float(np.exp(np.mean(np.log(power))) / np.mean(power))
        zcr[i] = float(np.mean(np.abs(np.diff(np.sign(windowed)))) / 2.0)

    return centroid, flatness, zcr


def _segment_mean(values: np.ndarray, start_frame: int, end_frame: int) -> float:
    """Mean of a per-frame feature over a cough candidate's frame span."""
    if len(values) == 0:
        return 0.0
    lo = min(max(start_frame, 0), len(values) - 1)
    hi = min(max(end_frame, lo + 1), len(values))
    return float(np.mean(values[lo:hi]))


def detect_cough_events(
    audio: np.ndarray,
    sample_rate: int,
    config=None,
) -> tuple[list[tuple[float, float]], np.ndarray, float]:
    """

    Stage 1 (energy): candidate bursts that rise above the noise floor.
    Stage 2 (spectral): keep only candidates whose spectrum looks like a
    cough (broadband, high centroid, high ZCR). Music, tones, speech and
    low-frequency thuds are rejected here even when they are loud.

    Set ``config.use_spectral_gate = False`` to fall back to energy-only.

    Args:
        audio: 1D float32 array, normalized to [-1, 1]
        sample_rate: sample rate in Hz
        config: AudioConfig (optional, uses default if None)
    Returns:
        - List of (start_time_s, end_time_s) for each detected cough
        - Frame-level energy in dB
        - Estimated noise floor in dB
    """
    if config is None:
        config = AUDIO_CONFIG

    frame_length = config.frame_length
    hop_length = config.hop_length

    energies = rms_energy_frames(audio, frame_length, hop_length)
    if len(energies) == 0:
        return [], np.array([]), -120.0

    energy_db = 20.0 * np.log10(energies + 1e-12)

    noise_floor_db = float(np.percentile(energy_db, config.noise_percentile))
    tau_hi = noise_floor_db + config.noise_margin_db
    tau_lo = tau_hi - config.hysteresis_db

    min_frames = int(config.min_cough_duration_s * sample_rate / hop_length)
    max_frames = int(config.max_cough_duration_s * sample_rate / hop_length)

    candidates = []
    in_cough = False
    start_frame = 0

    for i, e_db in enumerate(energy_db):
        if not in_cough and e_db > tau_hi:
            in_cough = True
            start_frame = i
        elif in_cough and e_db < tau_lo:
            duration_frames = i - start_frame
            if min_frames <= duration_frames <= max_frames:
                start_time = start_frame * hop_length / sample_rate
                end_time = i * hop_length / sample_rate
                candidates.append((start_frame, i, start_time, end_time))
            in_cough = False

    if in_cough:
        duration_frames = len(energy_db) - start_frame
        if min_frames <= duration_frames <= max_frames:
            start_time = start_frame * hop_length / sample_rate
            end_time = len(audio) / sample_rate
            candidates.append((start_frame, len(energy_db), start_time, end_time))

    if not getattr(config, "use_spectral_gate", True) or not candidates:
        return (
            [(s, e) for _, _, s, e in candidates],
            energy_db,
            noise_floor_db,
        )

    centroid, flatness, zcr = spectral_frame_features(
        audio, frame_length, hop_length, sample_rate
    )
    events = []
    for cand_start, cand_end, start_time, end_time in candidates:
        mean_centroid = _segment_mean(centroid, cand_start, cand_end)
        mean_zcr = _segment_mean(zcr, cand_start, cand_end)
        if (
            mean_centroid >= config.min_spectral_centroid_hz
            and config.min_zero_crossing_rate
            <= mean_zcr
            <= config.max_zero_crossing_rate
        ):
            events.append((start_time, end_time))

    return events, energy_db, noise_floor_db


def estimate_snr(audio: np.ndarray, events: list[tuple[float, float]], sample_rate: int, noise_floor_db: float) -> float:
    """
    Estimate SNR in dB: 10 * log10(P_cough / P_noise_floor)
    """
    if not events:
        return -120.0

    total_cough_power = 0.0
    total_cough_samples = 0

    for start_s, end_s in events:
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        if end_idx > start_idx and end_idx <= len(audio):
            segment = audio[start_idx:end_idx]
            total_cough_power += np.sum(segment ** 2)
            total_cough_samples += len(segment)

    if total_cough_samples == 0:
        return -120.0

    cough_power = total_cough_power / total_cough_samples
    noise_power = 10 ** (noise_floor_db / 10)

    if noise_power <= 0:
        return 120.0

    snr_db = 10.0 * np.log10(cough_power / noise_power)
    return float(snr_db)


def estimate_clipping_ratio(audio: np.ndarray, threshold: float = 0.99) -> float:
    """
    Estimate ratio of samples near clipping (|x| > threshold).
    """
    return float(np.mean(np.abs(audio) > threshold))


def quality_gate(
    audio: np.ndarray,
    sample_rate: int,
    config=None,
) -> tuple[bool, dict]:
    """
    Run quality checks on audio.
    Returns (passed, details_dict).
    """
    if config is None:
        config = AUDIO_CONFIG

    events, energy_db, noise_floor_db = detect_cough_events(audio, sample_rate, config)

    snr_db = estimate_snr(audio, events, sample_rate, noise_floor_db)
    clipping_ratio = estimate_clipping_ratio(audio)
    num_coughs = len(events)

    passed = True
    reasons = []

    if num_coughs < config.min_coughs_required:
        passed = False
        reasons.append(f"too_few_coughs: {num_coughs} < {config.min_coughs_required}")

    if snr_db < config.snr_threshold_db:
        passed = False
        reasons.append(f"too_noisy: SNR {snr_db:.1f} dB < {config.snr_threshold_db} dB")

    if clipping_ratio > config.clipping_threshold_ratio:
        passed = False
        reasons.append(f"clipped: ratio {clipping_ratio:.4f} > {config.clipping_threshold_ratio}")

    details = {
        "passed": passed,
        "num_coughs": num_coughs,
        "cough_segments_ms": [[int(s * 1000), int(e * 1000)] for s, e in events],
        "snr_db": round(snr_db, 1),
        "clipping_ratio": round(clipping_ratio, 4),
        "noise_floor_db": round(noise_floor_db, 1),
        "reasons": reasons,
    }

    return passed, details