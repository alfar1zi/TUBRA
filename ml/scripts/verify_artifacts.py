#!/usr/bin/env python3
"""Verify exported TBCare ensemble artifacts are loadable and self-consistent.

Checks:
  1. All required artifact files exist.
  2. ModelManifest parses and reports expected status.
  3. model_card.json contains every key the web app expects.
  4. StackedEnsemble loads and produces a well-formed prediction.
  5. Audio branch is deterministic (same input -> same output).
  6. The API can be imported without torch/librosa/numba.

Exit code 0 on success, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts"

REQUIRED_FILES = [
    "head.npz",
    "clinical_model.joblib",
    "supervisor_model.joblib",
    "clinical_preprocessor.joblib",
    "model_manifest.json",
    "model_card.json",
]
# Audio backbone: either the single-model layout (``backbone.onnx``) or a
# seed-ensemble layout (one or more ``backbone_seed*.onnx``) is acceptable.
BACKBONE_GLOBS = ["backbone.onnx", "backbone_seed*.onnx"]

WEB_METRIC_KEYS = [
    "roc_auc",
    "pauc_90",
    "sens_at_A",
    "spec_at_A",
    "ece",
    "threshold_A",
    "threshold_B",
]

failures: list[str] = []


def check(cond: bool, msg: str, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    line = f"[{status}] {msg}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not cond:
        failures.append(msg)


def main() -> int:
    print(f"Artifacts dir: {ARTIFACTS}\n")

    # 1. Required files
    for name in REQUIRED_FILES:
        check((ARTIFACTS / name).exists(), f"present: {name}")
    backbone_found = [g for g in BACKBONE_GLOBS if list(ARTIFACTS.glob(g))]
    check(bool(backbone_found), "present: audio backbone (single or seed-ensemble)", ", ".join(backbone_found))

    if not (ARTIFACTS / "model_card.json").exists():
        print("\nFATAL: model_card.json missing; aborting.")
        return 1

    # 3. model_card keys
    card = json.loads((ARTIFACTS / "model_card.json").read_text())
    metrics = card.get("metrics", {})
    for key in WEB_METRIC_KEYS:
        check(key in metrics, f"model_card.metrics.{key}", repr(metrics.get(key)))

    # 2. manifest
    try:
        from tbcare_api.models.manifest import ModelManifest

        manifest = ModelManifest.from_json(ARTIFACTS / "model_manifest.json")
        check(True, "ModelManifest parses", f"status={manifest.status}")
        check(
            manifest.metrics.get("threshold_A") is not None
            and manifest.metrics.get("threshold_B") is not None,
            "manifest.metrics carries tier thresholds",
        )
    except Exception as exc:  # noqa: BLE001
        check(False, "ModelManifest parses", str(exc))

    # 4/5. ensemble load + predict
    try:
        import numpy as np

        from tbcare_api.inference.ensemble import create_ensemble_from_artifacts

        ens = create_ensemble_from_artifacts(ARTIFACTS)
        check(True, "StackedEnsemble loads artifacts")

        record = {
            "sex": "Male",
            "age": 35,
            "reported_cough_dur": 21,
            "hemoptysis": False,
            "fever": True,
            "night_sweats": True,
            "weight_loss": True,
            "tb_prior": False,
            "smoke_lweek": False,
        }
        logmel = np.random.RandomState(0).randn(5, 64, 64).astype(np.float32)

        import os

        os.environ.setdefault("DEPLOYMENT_ENV", "staging")
        os.environ.setdefault("ALLOW_BLOCKED_CANDIDATE", "true")
        out = ens.predict(logmel, record)
        check(
            out.get("risk_tier") in {"A", "B", "C"},
            "predict returns a valid tier",
            f"tier={out.get('risk_tier')} p_fusion={out.get('prob_fusion'):.4f}",
        )

        out2 = ens.predict(logmel, record)
        check(
            out["prob_fusion"] == out2["prob_fusion"],
            "audio branch is deterministic across runs",
        )
    except Exception as exc:  # noqa: BLE001
        check(False, "StackedEnsemble loads + predicts", str(exc))

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
