"""Build overlapping 8 s windows for YOHO training.

A CODA-TB recording is one cough inside 0.5 s of audio, and the cough itself is
about 0.2 s. The target the network regresses is the event boundary, so a crop
is first trimmed to its cough (:func:`trim_to_event`) and only then placed in
an 8 s window. Paste the whole crop instead and every label says "the event
starts where the file starts", which is a property of the dataset and not of a
cough.

Windows are built from one participant's crops at a time, so a window never
mixes two voices, and the evaluation side is run under two conditions: pasted
windows, and crops padded on their own without a seam (``paste-free``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .frontend import SAMPLE_RATE

WINDOW_SECONDS = 8.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)


@dataclass
class Window:
    audio: np.ndarray
    events: list[tuple[float, float]]
    kind: str = ""
    meta: dict = field(default_factory=dict)


def load_mono_16k(path: Path | str) -> np.ndarray:
    """Read a clip as 16 kHz mono float32."""
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        import scipy.signal

        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def peak_normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current > 0:
        audio = audio * (peak / current)
    return audio.astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9))


class BackgroundPool:
    """Random 8 s backgrounds from a folder of non-cough recordings."""

    def __init__(
        self,
        root: Path,
        categories: tuple[str, ...] = ("speech", "music", "noise", "bursts"),
    ) -> None:
        self.clips: list[tuple[Path, str]] = []
        for category in categories:
            for path in sorted((root / category).glob("*.wav")):
                self.clips.append((path, category))
        if not self.clips:
            # An empty pool only fails later inside random.choice with a
            # message that says nothing about the folder. Name the path now.
            raise ValueError(f"no .wav clips under {root} for categories {categories}")
        self.cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.clips)

    def _load(self, path: Path) -> np.ndarray:
        if path not in self.cache:
            audio = load_mono_16k(path)
            if audio.shape[0] < WINDOW_SAMPLES:
                repeats = int(np.ceil(WINDOW_SAMPLES / audio.shape[0]))
                audio = np.tile(audio, repeats)
            self.cache[path] = audio
        return self.cache[path]

    def sample(self, rng: random.Random) -> tuple[np.ndarray, str]:
        path, category = rng.choice(self.clips)
        audio = self._load(path)
        start = rng.randrange(0, audio.shape[0] - WINDOW_SAMPLES + 1)
        window = audio[start : start + WINDOW_SAMPLES].copy()
        return peak_normalize(window), category


def _mix_crop_at(
    canvas: np.ndarray,
    crop: np.ndarray,
    start: int,
    *,
    target_snr_db: float | None,
    rng: random.Random,
) -> tuple[int, int]:
    """Add ``crop`` to ``canvas`` at ``start`` with a 5 ms crossfade."""
    length = crop.shape[0]
    if target_snr_db is not None:
        background = canvas[start : start + length]
        gain_db = target_snr_db + (rms_db(background) - rms_db(crop))
        crop = crop * (10.0 ** (gain_db / 20.0))
    else:
        gain_db = rng.uniform(-9.0, 6.0)
        crop = crop * (10.0 ** (gain_db / 20.0))

    fade = min(int(0.005 * SAMPLE_RATE), length // 2)
    envelope = np.ones(length, dtype=np.float32)
    if fade > 0:
        envelope[:fade] = np.linspace(0.0, 1.0, fade)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    canvas[start : start + length] += crop * envelope
    return start, start + length


# The span within 20 dB of the clip's loudest frame, plus 30 ms either side.
TRIM_RELATIVE_DB = 20.0
TRIM_CONTEXT_S = 0.03
_TRIM_FRAME = 256
_TRIM_HOP = 128


def trim_to_event(crop: np.ndarray) -> np.ndarray:
    """The cough part of ``crop``: loud frames plus a little context."""
    if crop.shape[0] < _TRIM_FRAME + _TRIM_HOP:
        return crop
    n_frames = 1 + (crop.shape[0] - _TRIM_FRAME) // _TRIM_HOP
    energy = np.empty(n_frames, dtype=np.float32)
    for index in range(n_frames):
        frame = crop[index * _TRIM_HOP : index * _TRIM_HOP + _TRIM_FRAME]
        energy[index] = 20.0 * np.log10(np.sqrt(np.mean(frame**2)) + 1e-9)
    loud = np.nonzero(energy >= energy.max() - TRIM_RELATIVE_DB)[0]
    if loud.size == 0:
        return crop
    context = int(TRIM_CONTEXT_S * SAMPLE_RATE)
    start = max(0, int(loud[0]) * _TRIM_HOP - context)
    end = min(crop.shape[0], int(loud[-1]) * _TRIM_HOP + _TRIM_FRAME + context)
    return crop[start:end]


def compose_cough_window(
    crops: list[np.ndarray],
    rng: random.Random,
    background: np.ndarray | None = None,
    category: str = "silence",
    snr_range_db: tuple[float, float] = (-5.0, 15.0),
) -> Window:
    """Lay trimmed coughs out left to right with gaps inside one 8 s window."""
    canvas = (
        np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        if background is None
        else background.copy()
    )
    target_snr = rng.uniform(*snr_range_db) if background is not None else None

    cursor = rng.randrange(0, int(0.4 * SAMPLE_RATE))
    events: list[tuple[float, float]] = []
    for crop in crops:
        event = trim_to_event(crop)
        if cursor + event.shape[0] > WINDOW_SAMPLES:
            break
        start, end = _mix_crop_at(canvas, event, cursor, target_snr_db=target_snr, rng=rng)
        events.append((start / SAMPLE_RATE, end / SAMPLE_RATE))
        # Gaps of 0.25-1.4 s keep several coughs per window without fusing them.
        cursor = end + int(rng.uniform(0.25, 1.4) * SAMPLE_RATE)
        target_snr = rng.uniform(*snr_range_db) if background is not None else None
    return Window(audio=peak_normalize(canvas), events=events, kind=f"cough+{category}")


def silence_window(rng: random.Random) -> Window:
    noise = rng.choice((0.0, 1e-4, 3e-4))
    return Window(
        audio=np.random.default_rng(rng.randrange(2**32)).normal(0.0, noise, WINDOW_SAMPLES).astype(np.float32),
        events=[],
        kind="silence",
    )


def background_window(pool: BackgroundPool, rng: random.Random) -> Window:
    audio, category = pool.sample(rng)
    return Window(audio=audio, events=[], kind=f"bg:{category}")


def build_training_windows(
    crops_per_participant: dict[str, list[np.ndarray]],
    participant_ids: list[str],
    rng: random.Random,
    n_windows: int,
    background_pool: BackgroundPool | None,
    cough_probability: float = 0.6,
    background_probability: float = 0.25,
) -> list[Window]:
    """Draw ``n_windows`` composed windows for one split.

    ``crops_per_participant`` maps a participant to preloaded 16 kHz crops, so a
    window never mixes two voices-turned-cough and keeps speaker identity.
    """
    windows: list[Window] = []
    for _ in range(n_windows):
        draw = rng.random()
        if draw < background_probability and background_pool is not None and len(background_pool):
            windows.append(background_window(background_pool, rng))
            continue
        if draw >= cough_probability:
            windows.append(silence_window(rng))
            continue
        participant = rng.choice(participant_ids)
        crops = crops_per_participant[participant]
        count = rng.choice((1, 2, 2, 3, 3, 4))
        chosen = [crops[rng.randrange(len(crops))] for _ in range(count)]
        if background_pool is not None and len(background_pool) and rng.random() < 0.5:
            background, category = background_pool.sample(rng)
            windows.append(compose_cough_window(chosen, rng, background, category))
        else:
            windows.append(compose_cough_window(chosen, rng))
    return windows
