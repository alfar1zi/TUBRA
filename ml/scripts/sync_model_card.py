#!/usr/bin/env python3
"""Copy the freshly exported model card to the web app and verify integrity.

Run AFTER `ml/scripts/export_artifacts.py`. The card is the single source of
truth for every number shown in the UI (AGENTS.md section 4.1), so the copies
must stay byte-identical.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ml" / "artifacts" / "model_card.json"
DESTS = [
    ROOT / "apps" / "web" / "public" / "model_card.json",
    ROOT / "apps" / "web" / "dist" / "model_card.json",
]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found; run export_artifacts.py first")
        return 1

    src_hash = md5(SRC)
    print(f"source {SRC} md5={src_hash}")

    for dest in DESTS:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC, dest)
        if md5(dest) != src_hash:
            print(f"ERROR: {dest} does not match source")
            return 1
        print(f"copied {dest} md5={md5(dest)} OK")

    print("all copies byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
