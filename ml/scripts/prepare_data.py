#!/usr/bin/env python3
"""
Prepare data for TB cough screening model training.
Combines CODA TB and TBscreen datasets.
Creates participant-level manifest and GroupKFold splits.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

# Add project root to path (prepare_data.py lives in ml/scripts/, so root is 3 levels up)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.configs import AudioConfig


def _audio_path_str(audio_file: Path) -> str:
    """Repo-relative path when the file lives under PROJECT_ROOT, else absolute.

    TBscreen audio lives outside the repo, so it is stored as an absolute path.
    """
    resolved = audio_file.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_coda_tb_data(coda_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CODA TB clinical and cough metadata.
    Returns (clinical_df, cough_df) with participant as key.
    """
    clinical_path = coda_root / "meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv"
    cough_path = coda_root / "meta_data/Cough Metadata/CODA_TB_Solicited_Meta_Info.csv"
    additional_path = coda_root / "meta_data/Clinical/CODA_TB_additional_variables_train.csv"
    
    clinical_df = pd.read_csv(clinical_path)
    cough_df = pd.read_csv(cough_path)
    additional_df = pd.read_csv(additional_path)
    
    # Merge clinical with additional (train/test split)
    clinical_df = clinical_df.merge(
        additional_df[["participant", "type", "Country", "HIVstatus", 
                       "Microbiologicreferencestandard", "Sputumxpertreferencestandard"]],
        on="participant", how="left"
    )
    
    # Add dataset source
    clinical_df["dataset"] = "CODA_TB"
    cough_df["dataset"] = "CODA_TB"
    
    print(f"CODA TB: {len(clinical_df)} participants, {len(cough_df)} cough recordings")
    return clinical_df, cough_df


def load_tbscreen_data(tbscreen_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load TBscreen (Nairobi) passive + forced cough metadata.

    Layout:
      Passive_coughs/Passive_coughs.csv  (+ Audio_files/)
      Forced_coughs/Forced_coughs.csv    (+ Audio_files/)
    Labels are 'TB' / 'NTB'. ``path`` holds the filename stem; audio is
    ``<section>/Audio_files/<path>.wav`` and only exists where consent == yes.

    Returns (clinical_df, cough_df) keyed by ``participant``.
    """
    passive_meta = tbscreen_root / "Passive_coughs/Passive_coughs.csv"
    forced_meta = tbscreen_root / "Forced_coughs/Forced_coughs.csv"

    frames = []
    for meta_path, cough_type, subdir in (
        (passive_meta, "passive", "Passive_coughs"),
        (forced_meta, "forced", "Forced_coughs"),
    ):
        if not meta_path.exists():
            continue
        df = pd.read_csv(meta_path, low_memory=False)
        df = df.rename(columns={"subject": "participant", "Label": "tb_status"})
        df["dataset"] = "TBscreen"
        df["cough_type"] = cough_type
        df["audio_dir"] = str(meta_path.parent / "Audio_files")
        frames.append(df)

    if not frames:
        empty = pd.DataFrame(columns=["participant", "tb_status", "dataset"])
        return empty, empty

    cough_df = pd.concat(frames, ignore_index=True)

    # TB (1) vs NTB (0); drop rows without a usable label.
    cough_df["tb_status"] = cough_df["tb_status"].map({"TB": 1, "NTB": 0})
    cough_df = cough_df[cough_df["tb_status"].notna()].copy()
    cough_df["tb_status"] = cough_df["tb_status"].astype(int)

    # One clinical row per participant.
    clinical_cols = [
        c for c in [
            "participant", "age", "gndr_a", "hiv_infected_a", "tb_status",
            "fever", "un_wght_loss", "nght_swts", "cough_bld", "prev_tb",
            "smkr_a", "cough_drtn", "dataset",
        ] if c in cough_df.columns
    ]
    clinical_df = cough_df[clinical_cols].drop_duplicates(subset=["participant"]).copy()

    print(f"TBscreen: {len(clinical_df)} participants, {len(cough_df)} cough recordings")
    return clinical_df, cough_df


def build_manifest(
    clinical_df: pd.DataFrame,
    cough_df: pd.DataFrame,
    coda_audio_root: Path,
    tbscreen_audio_root: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Build unified manifest with audio file paths and labels.
    One row per cough recording.
    """
    rows = []
    
    # CODA TB audio files
    coda_cough = cough_df[cough_df["dataset"] == "CODA_TB"]
    coda_clinical = clinical_df[clinical_df["dataset"] == "CODA_TB"]
    
    for _, row in coda_cough.iterrows():
        audio_file = coda_audio_root / row["filename"]
        if audio_file.exists():
            clin = coda_clinical[coda_clinical["participant"] == row["participant"]]
            if len(clin) == 0:
                continue
            clin = clin.iloc[0]
            rows.append({
                "participant": row["participant"],
                "audio_path": _audio_path_str(audio_file),
                "dataset": "CODA_TB",
                "cough_type": "solicited",
                "tb_status": clin["tb_status"],
                "age": clin["age"],
                "sex": clin["sex"],
                "hiv_status": clin["HIVstatus"],
                "country": clin["Country"],
                "split": clin["type"],
            })
    
    # TBscreen audio files
    tbscreen_clinical = clinical_df[clinical_df["dataset"] == "TBscreen"]
    tbscreen_cough = cough_df[cough_df["dataset"] == "TBscreen"]

    def _yn(val) -> Optional[str]:
        """Map TBscreen 0/1 to the Yes/No string used elsewhere in the manifest."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip().lower()
        if s in {"1", "1.0", "yes", "y", "true"}:
            return "Yes"
        if s in {"0", "0.0", "no", "n", "false"}:
            return "No"
        return None

    for _, row in tbscreen_cough.iterrows():
        # Determine subfolder based on cough_type
        if row["cough_type"] == "passive":
            audio_dir = tbscreen_audio_root / "Passive_coughs/Audio_files"
        else:
            audio_dir = tbscreen_audio_root / "Forced_coughs/Audio_files"

        stem = str(row["path"]).strip()
        audio_file = audio_dir / stem
        if not audio_file.exists():
            audio_file = audio_dir / f"{stem}.wav"
        if not audio_file.exists():
            continue

        clin = tbscreen_clinical[tbscreen_clinical["participant"] == row["participant"]]
        clin = clin.iloc[0] if len(clin) else row

        rows.append({
            "participant": f"TBSCREEN_{row['participant']}",
            "audio_path": _audio_path_str(audio_file),
            "dataset": "TBscreen",
            "cough_type": row["cough_type"],
            "tb_status": int(row["tb_status"]),
            "age": clin.get("age"),
            "sex": "Female" if str(clin.get("gndr_a", "")).strip() in {"1", "1.0"} else "Male",
            "hiv_status": _yn(clin.get("hiv_infected_a")),
            "country": "KE",  # Nairobi
            "hemoptysis": _yn(clin.get("cough_bld")),
            "fever": _yn(clin.get("fever")),
            "night_sweats": _yn(clin.get("nght_swts")),
            "weight_loss": _yn(clin.get("un_wght_loss")),
            "prior_tb": _yn(clin.get("prev_tb")),
            "smoking": _yn(clin.get("smkr_a")),
            "cough_duration_weeks": clin.get("cough_drtn"),
            "split": "train",  # TBscreen has no predefined split
        })
    
    manifest = pd.DataFrame(rows)
    
    # Map sex to numeric
    manifest["female"] = manifest["sex"].map({"Female": 1.0, "Male": 0.0, 0: 0.0, 1: 1.0}).fillna(0.0)
    
    print(f"Manifest: {len(manifest)} total recordings from {manifest['participant'].nunique()} participants")
    print(f"  TB+: {(manifest['tb_status'] == 1).sum()}, TB-: {(manifest['tb_status'] == 0).sum()}")
    print(f"  Datasets: {manifest['dataset'].value_counts().to_dict()}")
    print(f"  Cough types: {manifest['cough_type'].value_counts().to_dict()}")
    
    manifest.to_csv(output_path, index=False)
    return manifest


def create_groupkfold_splits(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create GroupKFold splits based on participant ID.
    Ensures no participant appears in both train and val.
    """
    # Use only train portion from CODA TB if split exists
    train_manifest = manifest[manifest["split"] == "train"].copy().reset_index(drop=True)
    
    # If no split column or all train, use all
    if len(train_manifest) == 0:
        train_manifest = manifest.copy().reset_index(drop=True)
    
    groups = train_manifest["participant"].values
    y = train_manifest["tb_status"].values
    
    gkf = GroupKFold(n_splits=n_splits)
    fold_col = np.full(len(train_manifest), -1, dtype=int)
    
    for fold_idx, (_, val_idx) in enumerate(gkf.split(train_manifest, y, groups)):
        fold_col[val_idx] = fold_idx
    
    train_manifest["fold"] = fold_col
    
    # For test portion from CODA TB, assign fold=-1 (held out)
    test_manifest = manifest[manifest["split"] == "test"].copy()
    if len(test_manifest) > 0:
        test_manifest["fold"] = -1
        combined = pd.concat([train_manifest, test_manifest], ignore_index=True)
    else:
        combined = train_manifest
    
    print(f"GroupKFold: {n_splits} folds created")
    for f in range(n_splits):
        train_count = (combined["fold"] != f).sum()
        val_count = (combined["fold"] == f).sum()
        test_count = (combined["fold"] == -1).sum()
        print(f"  Fold {f}: train={train_count}, val={val_count}, test={test_count}")
    
    return combined


def test_no_participant_leak(manifest: pd.DataFrame, n_folds: int = 5) -> bool:
    """Fail if any participant appears in more than one validation group.

    Folds are the GroupKFold validation splits; fold ``-1`` marks the held-out
    test set (if any). A participant must live in exactly one fold and must
    never appear in both the test set and any validation fold.
    """
    # Each participant must belong to exactly one fold (including -1).
    per_participant = manifest.groupby("participant")["fold"].nunique()
    multi = per_participant[per_participant > 1]
    if len(multi) > 0:
        print(f"LEAK: participants in multiple folds: {multi.to_dict()}")
        return False

    # Held-out test participants must not appear in any validation fold.
    test_parts = set(manifest.loc[manifest["fold"] == -1, "participant"])
    train_parts = set(manifest.loc[manifest["fold"] >= 0, "participant"])
    overlap = test_parts & train_parts
    if overlap:
        print(f"TEST-TRAIN LEAK: {overlap}")
        return False

    print(f"No participant leak test: PASSED ({n_folds} folds)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare TB cough data manifest and folds")
    parser.add_argument("--coda-root", type=Path, required=True, help="CODA TB extracted root")
    parser.add_argument("--tbscreen-root", type=Path, required=True, help="TBscreen dataset root")
    parser.add_argument("--output-dir", type=Path, default=Path("ml/data/processed"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    coda_clinical, coda_cough = load_coda_tb_data(args.coda_root)
    tbscreen_clinical, tbscreen_cough = load_tbscreen_data(args.tbscreen_root)
    
    # Combine clinical
    clinical = pd.concat([coda_clinical, tbscreen_clinical], ignore_index=True)
    cough = pd.concat([coda_cough, tbscreen_cough], ignore_index=True)
    
    # Build manifest
    manifest_path = args.output_dir / "manifest.csv"
    manifest = build_manifest(
        clinical, cough,
        args.coda_root / "raw_data/solicited_data",
        args.tbscreen_root,
        manifest_path
    )
    
    # Create folds
    manifest_with_folds = create_groupkfold_splits(manifest, args.n_splits, args.seed)
    folds_path = args.output_dir / "manifest_folds.csv"
    manifest_with_folds.to_csv(folds_path, index=False)
    
    # Test no leak
    assert test_no_participant_leak(
        manifest_with_folds, n_folds=args.n_splits
    ), "Participant leak detected!"
    
    # Save split indices for reproducibility. tb_status is per-clip in the
    # manifest, so participant counts must deduplicate to one row per
    # participant first; otherwise the totals mix clip counts with patient
    # counts and overstate both the cohort size and the TB-positive count.
    _participants = manifest_with_folds.drop_duplicates("participant")
    splits_info = {
        "n_splits": args.n_splits,
        "seed": args.seed,
        "total_participants": int(_participants["participant"].nunique()),
        "total_recordings": int(len(manifest_with_folds)),
        "tb_positive_participants": int((_participants["tb_status"] == 1).sum()),
        "tb_negative_participants": int((_participants["tb_status"] == 0).sum()),
        "folds": {}
    }
    
    for fold in range(args.n_splits):
        train_idx = manifest_with_folds[manifest_with_folds["fold"] != fold].index.tolist()
        val_idx = manifest_with_folds[manifest_with_folds["fold"] == fold].index.tolist()
        test_idx = manifest_with_folds[manifest_with_folds["fold"] == -1].index.tolist()
        
        splits_info["folds"][fold] = {
            "train_participants": int(manifest_with_folds.loc[train_idx, "participant"].nunique()),
            "val_participants": int(manifest_with_folds.loc[val_idx, "participant"].nunique()),
            "test_participants": int(manifest_with_folds.loc[test_idx, "participant"].nunique()),
            "train_recordings": len(train_idx),
            "val_recordings": len(val_idx),
            "test_recordings": len(test_idx),
        }
    
    splits_path = args.output_dir / "splits_info.json"
    with open(splits_path, "w") as f:
        json.dump(splits_info, f, indent=2)
    
    print(f"\nOutput files:")
    print(f"  Manifest: {manifest_path}")
    print(f"  Manifest with folds: {folds_path}")
    print(f"  Splits info: {splits_path}")


if __name__ == "__main__":
    main()