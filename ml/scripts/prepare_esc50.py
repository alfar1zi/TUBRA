"""Extract cough/sneeze samples and background clips from ESC-50.

ESC-50 (Piczak 2015, CC BY-NC 3.0) is the only external audio this project
uses. It is downloaded from the public GitHub mirror, never from a private
collection:

    curl -L -o /tmp/esc50.zip \
        https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
    unzip -q /tmp/esc50.zip -d /tmp/esc50x

Then:

    python3 ml/scripts/prepare_esc50.py

It writes `ml/data/cough_v2` (coughing + sneezing) and `ml/data/negatives`
(speech, music, noise, bursts, hard). Everything is resampled to 16 kHz mono
and the manifest records the source file and licence for each clip.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 16000
LICENCE = "CC BY-NC 3.0"

COUGH_CLIPS = ("coughing", "sneezing")

# Folders the training script consumes (see BackgroundPool).
BACKGROUNDS = {
    "speech": ("laughing",),
    "music": ("church_bells",),
    "noise": ("rain",),
    "bursts": ("clapping",),
}

# Categories the energy baseline flagged most often; kept apart so evaluation
# can report false positives on the sounds hardest to separate from a cough.
HARD = (
    "frog",
    "clock_tick",
    "hand_saw",
    "pig",
    "keyboard_typing",
    "brushing_teeth",
    "hen",
    "crying_baby",
)


def load_mono(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        import scipy.signal

        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE, rate).astype("float32")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio * (0.9 / peak)
    return audio.astype("float32")


def write_clip(src: Path, dst: Path) -> None:
    audio = load_mono(src)
    sf.write(dst, audio, SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50", type=Path, default=Path("/tmp/esc50x/ESC-50-master"))
    parser.add_argument("--out", type=Path, default=ROOT / "ml/data")
    args = parser.parse_args()

    meta = args.esc50 / "meta" / "esc50.csv"
    audio_dir = args.esc50 / "audio"
    if not meta.exists() or not audio_dir.is_dir():
        raise SystemExit(f"ESC-50 not found at {args.esc50}; see the docstring")

    rows = list(csv.DictReader(meta.open()))
    by_category: dict[str, list[str]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["filename"])

    targets: list[tuple[Path, str, str]] = []
    for folder, categories in BACKGROUNDS.items():
        for category in categories:
            for name in by_category[category]:
                targets.append((args.out / "negatives" / folder, name, category))
    for category in COUGH_CLIPS:
        for name in by_category[category]:
            targets.append((args.out / "cough_v2", name, category))
    for category in HARD:
        for name in by_category[category]:
            targets.append((args.out / "negatives" / "hard", name, category))

    manifest_path = args.out / "cough_v2" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "category", "source", "license"])
        for folder, name, category in targets:
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / name
            if not dst.exists():
                write_clip(audio_dir / name, dst)
                copied += 1
            writer.writerow([dst.relative_to(args.out), category, f"esc50:{name}", LICENCE])

    print(f"wrote {len(targets)} clips ({copied} new) under {args.out}")


if __name__ == "__main__":
    main()
