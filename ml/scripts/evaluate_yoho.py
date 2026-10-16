"""Score a trained YOHO checkpoint on held-out clips and on non-cough audio.

Training reports segment F1 on composed 8 s windows, which answers "does the
model place boundaries inside a window built from CODA-TB crops". It does not
answer "how often does the deployed path fire on a sneeze, a clap or a kitchen
noise". This script does: every clip runs through the same nuclei -> YOHO ->
vote path the API uses, clips that contain a cough are scored against the
cough span, and clips that must stay silent are counted as false alarms.

    python3 ml/scripts/evaluate_yoho.py --checkpoint ml/artifacts/yoho/yoho_fold0.pt

Writes ``ml/artifacts/yoho/eval_fold0.json`` and prints a table.
"""

from __future__ import annotations

import os

_THREADS = os.environ.setdefault("TBCARE_TORCH_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.src.yoho.dataset import Window, load_mono_16k, peak_normalize  # noqa: E402
from ml.src.yoho.detector import CONTEXT_SAMPLES, detect_cough_segments, steps_by_planes  # noqa: E402
from ml.src.yoho.metrics import match_events  # noqa: E402
from ml.src.yoho.model import MobileNetYoho  # noqa: E402
from ml.scripts.train_yoho import energy_predictions  # noqa: E402


def detect(model, audio: np.ndarray) -> list[tuple[float, float]]:
    """The deployed path on one clip."""

    def predict(features: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            raw = model(torch.from_numpy(features)).numpy()
        return steps_by_planes(np.asarray(raw[0]))

    return detect_cough_segments(audio, predict=predict)


def padded(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a clip to the CODA-TB level and put it in one 8 s window.

    The clips come from different corpora and several sit far below the level
    of a cough recording. Scored as they are, the nuclei would fire on the
    quiet background of the padded window instead of on the sound under test,
    so every clip is brought to the peak the CODA-TB clips have.
    """
    audio = peak_normalize(audio, 0.9)
    if audio.shape[0] >= CONTEXT_SAMPLES:
        return audio[:CONTEXT_SAMPLES], 0.0
    offset = (CONTEXT_SAMPLES - audio.shape[0]) // 2
    window = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
    window[offset : offset + audio.shape[0]] = audio
    return window, offset / 16000

def score_clips(model, clips: dict[str, list[np.ndarray]], expect_silence: bool) -> dict:
    totals = {"clips": 0, "hits": 0, "false_alarms": 0, "misses": 0, "seconds": 0.0, "elapsed": 0.0}
    per_category: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    for category, audios in clips.items():
        stats = per_category.setdefault(category, {"clips": 0, "with_detection": 0, "events": 0})
        for audio in audios:
            window, offset = padded(audio)
            events = detect(model, window)
            stats["clips"] += 1
            stats["events"] += len(events)
            stats["with_detection"] += int(bool(events))
            totals["clips"] += 1
            totals["seconds"] += window.shape[0] / 16000
            if expect_silence:
                totals["false_alarms"] += len(events)
            else:
                truth = [(offset, offset + audio.shape[0] / 16000)]
                hits, extra = match_events(truth, events)
                totals["hits"] += hits
                totals["misses"] += 1 - hits
                totals["false_alarms"] += extra
    totals["elapsed"] = time.perf_counter() - started
    totals["per_category"] = per_category
    return totals


def coda_clips(fold: int, limit: int) -> list[np.ndarray]:
    manifest = pd.read_csv(ROOT / "ml/data/processed/manifest_folds.csv")
    manifest = manifest[manifest["cough_type"].isin(["solicited", "passive", "forced"])]
    manifest = manifest[manifest["fold"] == fold].head(limit)
    clips = []
    for path in manifest["audio_path"]:
        try:
            clips.append(load_mono_16k(ROOT / path))
        except Exception:
            continue
    return clips


def folder_clips(folder: Path, pattern: str = "*.wav") -> list[np.ndarray]:
    clips = []
    for path in sorted(folder.glob(pattern)):
        try:
            clips.append(load_mono_16k(path))
        except Exception:
            continue
    return clips


def split_cough_v2(folder: Path) -> dict[str, list[np.ndarray]]:
    manifest = pd.read_csv(folder / "manifest.csv")
    out: dict[str, list[np.ndarray]] = {}
    missing: dict[str, int] = {}
    for path, category in zip(manifest["path"], manifest["category"]):
        try:
            audio = load_mono_16k(ROOT / "ml/data" / path)
        except Exception:
            missing[category] = missing.get(category, 0) + 1
            continue
        out.setdefault(category, []).append(audio)
    # The manifest is shared between machines. Folder counts that do not match
    # it would quietly change what the false-alarm numbers mean.
    print(f"cough_v2 loaded {dict((k, len(v)) for k, v in out.items())} missing {missing}", file=sys.stderr)
    return out


def summarize(name: str, result: dict, expect_silence: bool) -> dict:
    seconds = result["seconds"]
    hours = seconds / 3600 if seconds else 0.0
    events_per_hour = result["false_alarms"] / hours if hours else 0.0
    if expect_silence:
        detection_rate = result["clips"] and sum(c["with_detection"] for c in result["per_category"].values()) / result["clips"]
        return {
            "clips": result["clips"],
            "events": result["false_alarms"],
            "clips_with_detection": round(detection_rate, 3),
            "false_alarms_per_hour": round(events_per_hour, 1),
            "real_time_factor": round(result["elapsed"] / seconds, 4) if seconds else 0.0,
            "per_category": result["per_category"],
        }
    precision = result["hits"] / (result["hits"] + result["false_alarms"]) if result["hits"] + result["false_alarms"] else 0.0
    recall = result["hits"] / (result["hits"] + result["misses"]) if result["hits"] + result["misses"] else 0.0
    return {
        "clips": result["clips"],
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "false_alarms": result["false_alarms"],
        "false_alarms_per_hour": round(events_per_hour, 1),
        "real_time_factor": round(result["elapsed"] / seconds, 4) if seconds else 0.0,
        "per_category": result["per_category"],
    }


def energy_baseline(clips: list[np.ndarray], expect_cough: bool) -> dict:
    """The energy gate on the same clips, run through the same matching."""
    hits = false_alarms = 0
    for audio in clips:
        # The energy gate is scored without padding: a zero tail pushes its
        # floor down and the gate changes character on clips that already have
        # silence of their own.
        events = energy_predictions(None, Window(audio=audio, events=[(0.0, audio.shape[0] / 16000)]))
        matched, extra = match_events([(0.0, audio.shape[0] / 16000)] if expect_cough else [], events)
        hits += matched
        false_alarms += extra
    return {
        "clips": len(clips),
        "recall": round(hits / len(clips), 3) if clips else 0.0,
        "false_alarms": false_alarms,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ml/artifacts/yoho/yoho_fold0.pt")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--coda-clips", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "ml/artifacts/yoho")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(int(_THREADS))
    model = MobileNetYoho()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    report: dict[str, dict] = {}
    coda = coda_clips(args.fold, args.coda_clips)
    report["coda_holdout"] = summarize("coda", score_clips(model, {"coda": coda}, False), False)
    v2 = split_cough_v2(ROOT / "ml/data/cough_v2")
    report["esc50_cough"] = summarize("esc50_cough", score_clips(model, {"coughing": v2.get("coughing", [])}, False), False)
    report["esc50_sneeze"] = summarize("esc50_sneeze", score_clips(model, {"sneezing": v2.get("sneezing", [])}, True), True)
    hard = folder_clips(ROOT / "ml/data/negatives/hard")
    report["esc50_hard"] = summarize("esc50_hard", score_clips(model, {"hard": hard}, True), True)
    for extra in ("bursts", "speech", "music", "noise"):
        clips = folder_clips(ROOT / "ml/data/negatives" / extra)
        if clips:
            report[f"esc50_{extra}"] = summarize(extra, score_clips(model, {extra: clips}, True), True)
    # The energy gate on the same clips is the number this work has to beat:
    # without it the recall above reads as an improvement over nothing.
    report["energy_baseline_coda"] = energy_baseline(coda, expect_cough=True)
    report["energy_baseline_negatives"] = energy_baseline(
        hard + [clip for category, clips in v2.items() if category != "coughing" for clip in clips],
        expect_cough=False,
    )

    out_path = args.out / f"eval_fold{args.fold}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
