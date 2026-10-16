"""Train MobileNet-YOHO for cough detection on CODA-TB crops.

Usage
-----
    python3 ml/scripts/train_yoho.py --fold 0 --epochs 12 --steps-per-epoch 400

Design notes
------------
* Every window is composed on the fly from 0.5 s CODA crops (positives) laid
  out inside 8 s of near silence, with each crop trimmed to its cough. Non-cough
  audio is a small minority: mostly silence, plus the hard negatives in
  ``ml/data/negatives/hard`` that the energy baseline fires on. Nothing is
  pre-extracted, so every epoch sees a different arrangement.
* Splits come from ``manifest_folds.csv`` (participant-grouped, seed 42);
  validation participants never appear in training windows.
* Evaluation has two conditions, and both are reported: ``compose`` (the same
  pasted windows used in training) and ``paste-free`` (each crop padded on its
  own). If ``compose`` is much better than ``paste-free``, the model learned the
  paste seam instead of the cough.
* The energy baseline from ``apps/api/tbcare_api/audio/events.py`` is scored on
  the same validation windows for comparison.
"""

from __future__ import annotations

import os

# This laptop has four physical cores; torch defaults to the eight hardware
# threads, and the measured step time at eight is worse than at four because
# the hyperthreads share execution units. Set before torch is imported.
_THREADS = os.environ.setdefault("TBCARE_TORCH_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "api"))

torch.set_num_threads(int(_THREADS))

from ml.src.yoho.dataset import (  # noqa: E402
    WINDOW_SAMPLES,
    BackgroundPool,
    Window,
    background_window,
    compose_cough_window,
    load_mono_16k,
    peak_normalize,
    silence_window,
    trim_to_event,
)
from ml.src.yoho.detector import (  # noqa: E402
    CONTEXT_STEPS,
    WINDOW_MEL_FRAMES,
    detect_cough_segments,
)
from ml.src.yoho.frontend import log_mel, pad_to_window  # noqa: E402
from ml.src.yoho.metrics import segment_f1  # noqa: E402
from ml.src.yoho.model import MobileNetYoho, yoho_loss  # noqa: E402
from ml.src.yoho.postprocess import decode, encode_labels  # noqa: E402

STEP_S = 8.0 / CONTEXT_STEPS


def load_crops(manifest: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    crops: dict[str, list[np.ndarray]] = {}
    for participant, path in zip(manifest["participant"], manifest["audio_path"]):
        try:
            audio = load_mono_16k(ROOT / path)
        except Exception:
            continue
        crops.setdefault(participant, []).append(peak_normalize(audio, 0.8))
    return crops


def window_batch(windows: list[Window]) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack(
        [pad_to_window(log_mel(w.audio), WINDOW_MEL_FRAMES).T for w in windows]
    )[:, None, :, :]
    # Pad to a fixed batch: on this CPU a 16-row batch costs 8x the time of a
    # 32-row batch, so small batches are measured on padding instead of data.
    remainder = len(windows) % 32
    if remainder:
        tail = np.repeat(features[-1:], 32 - remainder, axis=0)
        features = np.concatenate([features, tail], axis=0)
    targets = np.stack(
        [encode_labels(w.events, CONTEXT_STEPS, 8.0) for w in windows]
    )
    if targets.shape[0] != features.shape[0]:
        tail = np.repeat(targets[-1:], features.shape[0] - targets.shape[0], axis=0)
        targets = np.concatenate([targets, tail], axis=0)
    return features.astype(np.float32), targets.astype(np.float32)


def sample_windows(
    crops: dict[str, list[np.ndarray]],
    participants: list[str],
    pool: BackgroundPool | None,
    rng: random.Random,
    count: int,
) -> list[Window]:
    """Draw training windows: mostly coughs, a little silence, few hard negatives.

    Only cough is a class. Backgrounds exist to stop the network from equating
    "cough" with "the only thing audible", so they stay rare: silence covers
    quiet rooms, the hard pool covers the few non-cough sounds that the energy
    baseline mistakes for coughs. Every other sound on earth is left to the
    detector stage instead of being enumerated here.
    """
    windows: list[Window] = []
    for _ in range(count):
        draw = rng.random()
        if draw < 0.06:
            windows.append(silence_window(rng))
        elif pool is not None and draw < 0.14:
            windows.append(background_window(pool, rng))
        else:
            participant = rng.choice(participants)
            count_coughs = rng.choice((1, 2, 2, 3))
            coughs = [crops[participant][rng.randrange(len(crops[participant]))] for _ in range(count_coughs)]
            windows.append(compose_cough_window(coughs, rng))
    return windows


def evaluation_windows(
    crops: dict[str, list[np.ndarray]],
    participants: list[str],
    pool: BackgroundPool | None,
    rng: random.Random,
    per_participant: int = 2,
) -> list[Window]:
    """Composed windows plus crop-only windows for the paste-free condition."""
    composed = sample_windows(
        crops, participants, pool, rng, len(participants) * per_participant
    )
    plain: list[Window] = []
    for participant in participants:
        for crop in crops[participant][:per_participant]:
            audio = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            offset = int(0.15 * 16000)
            audio[offset : offset + crop.shape[0]] = crop
            event = trim_to_event(peak_normalize(crop, 0.8))
            plain.append(
                Window(
                    audio=audio,
                    events=[(offset / 16000, (offset + event.shape[0]) / 16000)],
                    kind="paste-free",
                )
            )
    return composed + plain


def yoho_predictions(model: MobileNetYoho, window: Window) -> list[tuple[float, float]]:
    features, _ = window_batch([window])
    with torch.no_grad():
        output = model(torch.from_numpy(features))[0].numpy()
    # The head is class-major: step k of class 0 sits at column 3 * k.
    steps = output.reshape(3, -1).T
    return decode(steps[:, :3], 8.0)


def streaming_predictions(model: MobileNetYoho, window: Window) -> list[tuple[float, float]]:
    """The deployed path: nuclei, per-nucleus YOHO windows, boundary votes."""

    def predict(features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(features)).numpy()

    return detect_cough_segments(window.audio, predict=predict)


def energy_predictions(model: MobileNetYoho, window: Window) -> list[tuple[float, float]]:
    del model
    from tbcare_api.audio.config import AUDIO_CONFIG
    from tbcare_api.audio.events import detect_cough_events

    events, _, _ = detect_cough_events(window.audio, 16000, AUDIO_CONFIG)
    return [(float(start), float(end)) for start, end in events]


def evaluate(model: MobileNetYoho, windows: list[Window]) -> dict[str, dict]:
    groups = {
        "yoho-compose": [w for w in windows if w.kind != "paste-free"],
        "yoho-pastefree": [w for w in windows if w.kind == "paste-free"],
        "streaming-compose": [w for w in windows if w.kind != "paste-free"],
        "streaming-pastefree": [w for w in windows if w.kind == "paste-free"],
        "energy-compose": [w for w in windows if w.kind != "paste-free"],
        "energy-pastefree": [w for w in windows if w.kind == "paste-free"],
    }
    predictors = {
        "yoho-compose": yoho_predictions,
        "yoho-pastefree": yoho_predictions,
        "streaming-compose": streaming_predictions,
        "streaming-pastefree": streaming_predictions,
        "energy-compose": energy_predictions,
        "energy-pastefree": energy_predictions,
    }
    results = {}
    for name, subset in groups.items():
        truths = [w.events for w in subset]
        preds = [predictors[name](model, w) for w in subset]
        seconds = 8.0 * len(subset)
        started = time.perf_counter()
        for window in subset:
            predictors[name](model, window)
        elapsed = time.perf_counter() - started
        results[name] = {
            **segment_f1(truths, preds),
            "real_time_factor": elapsed / seconds if seconds else 0.0,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-participants", type=int, default=120)
    parser.add_argument("--negatives", type=Path, default=ROOT / "ml/data/negatives")
    parser.add_argument("--out", type=Path, default=ROOT / "ml/artifacts/yoho")
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="warm start from this checkpoint instead of random weights",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    rng = random.Random(42)
    manifest = pd.read_csv(ROOT / "ml/data/processed/manifest_folds.csv")
    manifest = manifest[manifest["cough_type"].isin(["solicited", "passive", "forced"])]
    crops = load_crops(manifest)
    participants = sorted(crops)
    # Participant-disjoint: a participant in fold k is never seen in training
    # when scoring fold k, whatever dataset their recordings came from.
    pooled = manifest.groupby("participant")["fold"].first()
    train_ids = [p for p in participants if p in pooled.index and pooled[p] != args.fold]
    val_ids = [p for p in participants if p in pooled.index and pooled[p] == args.fold]
    rng.shuffle(val_ids)
    val_ids = val_ids[: args.val_participants]

    def optional_pool(categories: tuple[str, ...]) -> BackgroundPool | None:
        # Check for clips, not just for the folder: an empty folder used to
        # build a pool that blew up on the first sample.
        if not any((args.negatives / c).glob("*.wav") for c in categories):
            return None
        return BackgroundPool(args.negatives, categories)

    # Only two kinds of non-cough enter training: silence, and a small pool of
    # the non-cough sounds the energy baseline fires on most often. Cough is the
    # single class; everything else would just teach the network to catalogue
    # noise it will never be scored on.
    pool = optional_pool(("hard",))
    print(
        f"participants: train {len(train_ids)}, val {len(val_ids)}; "
        f"hard negatives: {len(pool) if pool else 0}"
    )

    model = MobileNetYoho()
    if args.init is not None:
        # Warm start: continue from an earlier run with a lower learning rate,
        # instead of paying for the first epochs again from random weights.
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"warm start from {args.init}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"parameters: {param_count / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    val_windows = evaluation_windows(crops, val_ids, pool, random.Random(7))
    # Presence is a regression target on [0, 1]: for every positive step there
    # are ~8-15 silent ones, so an unweighted MSE settles on "never cough"
    # before it starts fitting events. Reported next to the loss.
    positives = sum(1 for w in val_windows if w.events)
    print(f"validation windows: {len(val_windows)} ({positives} with a cough)")
    history = []
    baseline = evaluate(model, val_windows)
    print("epoch  0 loss  ---", json.dumps({k: round(v["f1"], 3) for k, v in baseline.items()}))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        started = time.perf_counter()
        for _ in range(args.steps_per_epoch):
            windows = sample_windows(crops, train_ids, pool, rng, args.batch_size)
            features, targets = window_batch(windows)
            output = model(torch.from_numpy(features))
            if output.shape[-1] != CONTEXT_STEPS:
                raise RuntimeError(f"model returned {tuple(output.shape)} for {features.shape}")
            # Keep each target plane 3-D ``(B, T, 1)``; the loss reads one
            # example at a time and expects that shape.
            loss = yoho_loss(
                output,
                torch.from_numpy(targets[:, :, 0:1]),
                torch.from_numpy(targets[:, :, 1:2]),
                torch.from_numpy(targets[:, :, 2:3]),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        scores = evaluate(model, val_windows)
        history.append({"epoch": epoch, "loss": total / args.steps_per_epoch, "scores": scores})
        summary = {k: round(v["f1"], 3) for k, v in scores.items()}
        compose = scores["yoho-compose"]
        print(
            f"epoch {epoch:2d} loss {total / args.steps_per_epoch:.4f} "
            f"{time.perf_counter() - started:5.0f}s {json.dumps(summary)} "
            f"tp {compose['tp']} fp {compose['fp']} fn {compose['fn']}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / f"yoho_fold{args.fold}.pt")
    (args.out / f"yoho_fold{args.fold}.json").write_text(
        json.dumps(
            {
                "fold": args.fold,
                "parameters": param_count,
                "scores": history[-1]["scores"],
                "history": history,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
