"""Export YOHO to ONNX and check that ONNX Runtime matches torch.

The API loads ``yoho.onnx`` through ONNX Runtime; training stays in torch. A
drift between the two would only show up in production, so the parity check
runs here and in ``ml/tests/test_yoho_onnx.py`` with a tight tolerance.

Also writes ``yoho.json`` next to the model: input/output shapes, the time
stride and the post-processing constants the server uses after the network,
so the metadata describes the deployed path and not a different decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.src.yoho.detector import (  # noqa: E402
    CONTEXT_STEPS,
    MAX_EVENT_DURATION_S,
    MERGE_GAP_S,
    NUCLEUS_TOLERANCE_S,
    WINDOW_MEL_FRAMES,
)
from ml.src.yoho.frontend import N_MELS  # noqa: E402
from ml.src.yoho.model import MobileNetYoho  # noqa: E402


def export(checkpoint: Path, out_dir: Path) -> dict:
    model = MobileNetYoho(n_mels=N_MELS)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "yoho.onnx"
    # Batch 1 and a fixed 640-frame window: the API scores one 8 s context at a
    # time, and a fixed time axis keeps the step axis unambiguous.
    sample = torch.zeros(1, 1, N_MELS, WINDOW_MEL_FRAMES, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            str(onnx_path),
            input_names=["logmel"],
            output_names=["yoho"],
            opset_version=17,
            do_constant_folding=True,
        )

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    # Four windows, one at a time, so the parity check covers more than one draw.
    max_diff = 0.0
    for _ in range(4):
        inputs = rng.normal(-40.0, 8.0, (1, 1, N_MELS, WINDOW_MEL_FRAMES)).astype(np.float32)
        with torch.no_grad():
            torch_output = model(torch.from_numpy(inputs)).numpy()
        onnx_output = session.run(None, {"logmel": inputs})[0]
        if tuple(torch_output.shape) != (1, 3, CONTEXT_STEPS):
            raise SystemExit(f"unexpected torch output {torch_output.shape}")
        max_diff = max(max_diff, float(np.max(np.abs(torch_output - onnx_output))))


    metadata = {
        "input_name": "logmel",
        "output_name": "yoho",
        "input_shape": [1, 1, N_MELS, WINDOW_MEL_FRAMES],
        "output_shape": [1, 3, CONTEXT_STEPS],
        "time_stride": model.time_stride,
        "n_steps": CONTEXT_STEPS,
        "max_diff_torch_onnx": max_diff,
        "segment": {
            "presence_threshold": 0.5,
            "merge_gap_s": MERGE_GAP_S,
            "max_event_duration_s": MAX_EVENT_DURATION_S,
            "nucleus_tolerance_s": NUCLEUS_TOLERANCE_S,
        },
    }
    (out_dir / "yoho.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ml/artifacts/yoho/yoho_fold0.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "ml/artifacts/yoho")
    args = parser.parse_args()
    metadata = export(args.checkpoint, args.out)
    print(json.dumps(metadata, indent=2))
    if metadata["max_diff_torch_onnx"] > 1e-4:
        raise SystemExit(f"ONNX drift too large: {metadata['max_diff_torch_onnx']}")


if __name__ == "__main__":
    main()
