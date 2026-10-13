from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ml.src.features.audio import AudioFeatureConfig, extract_logmel_from_wav
from ml.src.features.clinical import ClinicalPreprocessor, CLINICAL_FEATURE_ORDER


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PatientExample:
    """Single patient with all cough clips and clinical features."""
    patient_id: str
    label: int  # 0=non-TB, 1=TB
    audio_paths: Tuple[Path, ...]
    clinical_features: Dict[str, Any]
    country: str
    split: str  # "train", "val", "test"


@dataclass
class DataConfig:
    """Configuration for data loading."""
    coda_tb_root: Path
    tbscreen_root: Optional[Path] = None
    audio_config: AudioFeatureConfig = field(default_factory=AudioFeatureConfig)
    max_clips_per_patient: int = 5
    min_clips_per_patient: int = 1
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42


# =============================================================================
# CODA TB Data Loading
# =============================================================================

def load_coda_tb_manifest(coda_root: Path) -> pd.DataFrame:
    """Load and merge CODA TB metadata files."""
    coda_root = Path(coda_root)

    # Clinical metadata
    clinical_path = coda_root / "extracted" / "dataset" / "meta_data" / "Clinical" / "CODA_TB_additional_variables_train.csv"
    if not clinical_path.exists():
        # Try alternative locations
        clinical_path = coda_root / "meta_data" / "Clinical" / "CODA_TB_additional_variables_train.csv"
    clinical = pd.read_csv(clinical_path, low_memory=False)

    # Solicited cough metadata
    cough_path = coda_root / "extracted" / "dataset" / "meta_data" / "Cough Metadata" / "CODA_TB_Solicited_Meta_Info.csv"
    if not cough_path.exists():
        cough_path = coda_root / "meta_data" / "Cough Metadata" / "CODA_TB_Solicited_Meta_Info.csv"
    cough = pd.read_csv(cough_path, low_memory=False)

    # Merge on participant + filename
    # Clinical has 'participant' column, cough has 'participant' and 'filename'
    merged = cough.merge(clinical, on="participant", how="left", suffixes=("_cough", "_clin"))

    return merged


def load_coda_tb_patients(
    coda_root: Path,
    audio_config: AudioFeatureConfig,
    max_clips: int = 5,
) -> List[PatientExample]:
    """Load CODA TB patients with audio paths and clinical features."""
    coda_root = Path(coda_root)
    manifest = load_coda_tb_manifest(coda_root)

    # Clinical meta (tb_status numeric) + additional variables (Country, HIV).
    clinical_path = coda_root / "extracted/dataset/meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv"
    if not clinical_path.exists():
        clinical_path = coda_root / "meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv"
    clinical = pd.read_csv(clinical_path, low_memory=False)

    additional_path = coda_root / "extracted/dataset/meta_data/Clinical/CODA_TB_additional_variables_train.csv"
    if not additional_path.exists():
        additional_path = coda_root / "meta_data/Clinical/CODA_TB_additional_variables_train.csv"
    additional = pd.read_csv(additional_path, low_memory=False)
    extra_cols = [c for c in ["participant", "Country", "HIVstatus"] if c in additional.columns]

    clinical = clinical.merge(additional[extra_cols], on="participant", how="left")
    clinical = clinical.set_index("participant")

    patients = []

    for participant_id, group in manifest.groupby("participant"):
        label = None
        if participant_id in clinical.index:
            raw = clinical.loc[participant_id, "tb_status"]
            if pd.notna(raw):
                label = int(raw)
        if label is None:
            continue

        country = "UNK"
        if participant_id in clinical.index and pd.notna(clinical.loc[participant_id].get("Country")):
            country = str(clinical.loc[participant_id]["Country"]).strip().upper()

        clinical_row = group.iloc[0]
        clinical_features = _extract_clinical_from_row(clinical_row)
        if participant_id in clinical.index:
            clin = clinical.loc[participant_id]
            for src, dst in (
                ("sex", "sex"), ("age", "age"), ("height", "height"),
                ("weight", "weight"), ("reported_cough_dur", "reported_cough_dur"),
                ("tb_prior", "tb_prior"), ("hemoptysis", "hemoptysis"),
                ("heart_rate", "heart_rate"), ("temperature", "temperature"),
                ("weight_loss", "weight_loss"), ("smoke_lweek", "smoke_lweek"),
                ("fever", "fever"), ("night_sweats", "night_sweats"),
            ):
                if src in clin.index and pd.notna(clin[src]):
                    clinical_features[dst] = clin[src]
            if "HIVstatus" in clin.index and pd.notna(clin["HIVstatus"]):
                clinical_features["hiv"] = clin["HIVstatus"]
        clinical_features.setdefault("country", country)

        audio_dir = coda_root / "extracted/dataset/raw_data/solicited_data"
        if not audio_dir.exists():
            audio_dir = coda_root / "raw_data/solicited_data"

        audio_paths = []
        for _, row in group.iterrows():
            filename = str(row.get("filename", row.get("Filename", ""))).strip()
            if filename:
                audio_path = audio_dir / filename
                if audio_path.exists():
                    audio_paths.append(audio_path)

        if len(audio_paths) < 1:
            continue

        audio_paths = tuple(sorted(audio_paths)[:max_clips])

        patients.append(PatientExample(
            patient_id=str(participant_id),
            label=label,
            audio_paths=audio_paths,
            clinical_features=clinical_features,
            country=country,
            split="train",
        ))

    return patients


def _extract_clinical_from_row(row: pd.Series) -> Dict[str, Any]:
    """Extract clinical features from a manifest row."""
    features = {}

    # Direct mappings
    mappings = {
        "sex": "sex",
        "age": "age",
        "height": "height",
        "weight": "weight",
        "reported_cough_dur": "reported_cough_dur",
        "tb_prior": "tb_prior",
        "tb_prior_type": "tb_prior_type",
        "hemoptysis": "hemoptysis",
        "heart_rate": "heart_rate",
        "temperature": "temperature",
        "weight_loss": "weight_loss",
        "smoke_lweek": "smoke_lweek",
        "fever": "fever",
        "night_sweats": "night_sweats",
        "Numberofcoughsoundscollected": "Numberofcoughsoundscollected",
        "country": "Country",
        "hiv": "hiv",
    }

    for feat_name, col_name in mappings.items():
        if col_name in row.index:
            val = row[col_name]
            if pd.notna(val):
                features[feat_name] = val

    return features


# =============================================================================
# TBscreen Data Loading
# =============================================================================

def _tbscreen_yes_no(val: Any) -> Optional[bool]:
    """Map TBscreen 0/1 (and string) flags to bool; blank -> None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().lower()
    if s in {"1", "1.0", "yes", "y", "true"}:
        return True
    if s in {"0", "0.0", "no", "n", "false"}:
        return False
    return None


def _tbscreen_label(val: Any) -> Optional[int]:
    """Map TBscreen ``Label`` to binary TB status. TB -> 1, NTB -> 0, else None."""
    s = str(val).strip().upper()
    if s == "TB":
        return 1
    if s == "NTB":
        return 0
    return None



def _tbscreen_audio_candidates(audio_dir: Path, csv_path: str) -> List[Path]:
    """Map a TBscreen CSV ``path`` value to candidate on-disk filenames.

    Returns candidates in priority order; the caller takes the first that
    exists. Handles three conventions observed across the dataset dumps:

    1. exact name: ``PID_1A1_yeti_0`` -> ``PID_1A1_yeti_0.wav``
    2. published ZIP name: ``PID_1A_1_pixel_0`` -> ``PID_1A1_pixel.wav``
       (session digits joined, trailing ``_<n>`` dropped, device last)
    3. index-stripped: ``PID_1A_1_pixel_0`` -> ``PID_1A_1_pixel.wav``
    """
    name = str(csv_path).strip()
    candidates: List[Path] = []
    if not name:
        return candidates

    candidates.append(audio_dir / f"{name}.wav")
    candidates.append(audio_dir / name)

    # Match on the original case (IGNORECASE) and rebuild the candidate in that
    # case: on case-sensitive filesystems the published files keep upper-case
    # ``PID``, so a lower-cased candidate would not resolve.
    m = re.match(r"^(pid_\w+?)_(\d+)_(\w+)_\d+$", name, flags=re.IGNORECASE)
    if m:
        candidates.append(
            audio_dir / f"{m.group(1)}{m.group(2)}_{m.group(3)}.wav"
        )
    stripped = re.sub(r"_\d+$", "", name)
    if stripped != name:
        candidates.append(audio_dir / f"{stripped}.wav")

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for c in candidates:
        key = c.name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

def load_tbscreen_patients(
    tbscreen_root: Path,
    audio_config: AudioFeatureConfig,
    max_clips: int = 5,
) -> List[PatientExample]:
    """Load TBscreen patients (passive cough recordings, participant-level labels).

    Real layout (TBscreen_Dataset/):
      Passive_coughs/Passive_coughs.csv   -> one row per recording
      Passive_coughs/Audio_files/*.wav    -> filename == ``path`` column + ".wav"

    Participants with cohort A/C are longitudinal single-device subjects; the
    ``Label`` column (TB / NTB) is the ground truth. Only consented recordings
    (``Permission_sound == Yes``) are used.
    """
    tbscreen_root = Path(tbscreen_root)

    csv_candidates = [
        tbscreen_root / "Passive_coughs" / "Passive_coughs.csv",
        tbscreen_root / "Passive_coughs.csv",
        tbscreen_root / "Metadata.csv",
    ]
    meta_file = next((p for p in csv_candidates if p.exists()), None)
    if meta_file is None:
        return []

    df = pd.read_csv(meta_file, low_memory=False)
    if "subject" not in df.columns or "Label" not in df.columns:
        return []

    # Consent filter (keep rows where consent is explicitly Yes, or unknown).
    if "Permission_sound" in df.columns:
        consent = df["Permission_sound"].astype(str).str.strip().str.lower()
        df = df[consent.isin({"yes", "ya", "1", "true"})].copy()
    if df.empty:
        return []

    audio_dir = meta_file.parent / "Audio_files"

    patients: List[PatientExample] = []
    for subject_id, group in df.groupby("subject"):
        label = _tbscreen_label(group["Label"].iloc[0])
        if label is None:
            continue

        first = group.iloc[0]
        clinical_features: Dict[str, Any] = {}
        if "age" in group.columns and pd.notna(first.get("age")):
            clinical_features["age"] = first["age"]
        if "gndr_a" in group.columns and pd.notna(first.get("gndr_a")):
            # TBscreen gndr_a: 1 == female, 0 == male (see Metadata.csv)
            clinical_features["sex"] = "Female" if float(first["gndr_a"]) == 1 else "Male"
        if "hiv_infected_a" in group.columns:
            hiv = _tbscreen_yes_no(first.get("hiv_infected_a"))
            if hiv is not None:
                clinical_features["hiv"] = "Positive" if hiv else "Negative"
        for src, dst in (
            ("fever", "fever"),
            ("un_wght_loss", "weight_loss"),
            ("nght_swts", "night_sweats"),
            ("cough_bld", "hemoptysis"),
            ("prev_tb", "tb_prior"),
            ("smkr_a", "smoke_lweek"),
        ):
            if src in group.columns:
                v = _tbscreen_yes_no(first.get(src))
                if v is not None:
                    clinical_features[dst] = v
        if "cough_drtn" in group.columns and pd.notna(first.get("cough_drtn")):
            clinical_features["reported_cough_dur"] = first["cough_drtn"]

        # Resolve each CSV `path` to an on-disk file. TBscreen's published
        # filenames differ from the CSV: `PID_1A_1_pixel_0` (CSV) is stored as
        # `PID_1A1_pixel.wav` (session digits joined, trailing index dropped,
        # device last). The plain `path + ".wav"` lookup silently missed most
        # rows, which is why earlier runs saw only a subset of participants.
        # `_tbscreen_candidates` also handles the plain and index-stripped
        # forms so this stays correct if a future dump uses either convention.
        audio_paths = []
        for _, row in group.iterrows():
            name = str(row.get("path", "")).strip()
            if not name:
                continue
            for candidate in _tbscreen_audio_candidates(audio_dir, name):
                if candidate.exists():
                    audio_paths.append(candidate)
                    break


        if len(audio_paths) < 1:
            continue

        audio_paths = tuple(sorted(audio_paths, key=lambda p: p.name)[:max_clips])
        clinical_features.setdefault("Numberofcoughsoundscollected", len(audio_paths))

        patients.append(PatientExample(
            patient_id=f"TBSCREEN_{subject_id}",
            label=label,
            audio_paths=audio_paths,
            clinical_features=clinical_features,
            country="KE",  # TBscreen cohort (Kenya)
            split="train",
        ))

    return patients


# =============================================================================
# GroupKFold Splits
# =============================================================================

def create_group_kfold_splits(
    patients: List[PatientExample],
    n_splits: int = 5,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> List[Dict[str, List[PatientExample]]]:
    """
    Create GroupKFold splits ensuring no patient leakage.

    Returns list of fold dicts with 'train', 'val', 'test' patient lists.
    """
    np.random.seed(random_seed)

    patient_ids = [p.patient_id for p in patients]
    labels = [p.label for p in patients]
    groups = patient_ids  # Group by patient_id

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(patient_ids, labels, groups))

    fold_data = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        # Further split train_val into train/val
        train_val_patients = [patients[i] for i in train_val_idx]
        test_patients = [patients[i] for i in test_idx]

        # Shuffle train_val
        np.random.shuffle(train_val_patients)
        val_count = max(1, int(len(train_val_patients) * val_size))
        val_patients = train_val_patients[:val_count]
        train_patients = train_val_patients[val_count:]

        # Assign split labels
        for p in train_patients:
            p.split = "train"
        for p in val_patients:
            p.split = "val"
        for p in test_patients:
            p.split = "test"

        fold_data.append({
            "fold": fold_idx,
            "train": train_patients,
            "val": val_patients,
            "test": test_patients,
        })

    return fold_data


def create_nested_cv_splits(
    patients: List[PatientExample],
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Create nested CV splits: outer for evaluation, inner for hyperparameter tuning.

    Returns list of outer folds, each containing inner folds.
    """
    np.random.seed(random_seed)

    patient_ids = [p.patient_id for p in patients]
    labels = [p.label for p in patients]
    groups = patient_ids

    outer_gkf = GroupKFold(n_splits=outer_splits)
    outer_splits_list = list(outer_gkf.split(patient_ids, labels, groups))

    nested_folds = []
    for outer_idx, (train_val_idx, test_idx) in enumerate(outer_splits_list):
        train_val_patients = [patients[i] for i in train_val_idx]
        test_patients = [patients[i] for i in test_idx]

        # Inner CV on train_val
        inner_patient_ids = [p.patient_id for p in train_val_patients]
        inner_labels = [p.label for p in train_val_patients]
        inner_groups = inner_patient_ids

        inner_gkf = GroupKFold(n_splits=inner_splits)
        inner_splits_list = list(inner_gkf.split(inner_patient_ids, inner_labels, inner_groups))

        inner_folds = []
        for inner_idx, (train_idx, val_idx) in enumerate(inner_splits_list):
            train_patients = [train_val_patients[i] for i in train_idx]
            val_patients = [train_val_patients[i] for i in val_idx]

            for p in train_patients:
                p.split = "train"
            for p in val_patients:
                p.split = "val"

            inner_folds.append({
                "inner_fold": inner_idx,
                "train": train_patients,
                "val": val_patients,
            })

        for p in test_patients:
            p.split = "test"

        nested_folds.append({
            "outer_fold": outer_idx,
            "test": test_patients,
            "inner_folds": inner_folds,
        })

    return nested_folds


# =============================================================================
# Feature Extraction for Training
# =============================================================================

def extract_patient_features(
    patient: PatientExample,
    audio_config: AudioFeatureConfig,
    clinical_preprocessor: ClinicalPreprocessor,
    fit_preprocessor: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract features for a single patient.

    Returns:
        audio_features: (n_clips, 1, n_mels, target_frames)
        clinical_features: (38,) - preprocessed
        label: int
    """
    # Audio: extract log-mel for each clip
    audio_feats = []
    for audio_path in patient.audio_paths:
        wav_bytes = audio_path.read_bytes()
        logmel = extract_logmel_from_wav(wav_bytes, audio_config)  # (1, n_mels, T)
        audio_feats.append(logmel)

    audio_array = np.stack(audio_feats, axis=0)  # (n_clips, 1, n_mels, T)

    # Clinical: preprocess
    clinical_array = clinical_preprocessor.transform([patient.clinical_features])[0]  # (38,)

    return audio_array, clinical_array, patient.label


def prepare_fold_data(
    patients: List[PatientExample],
    audio_config: AudioFeatureConfig,
    clinical_preprocessor: ClinicalPreprocessor,
    fit_preprocessor: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare batched data for a fold.

    Returns:
        audio_batch: (n_patients, max_clips, 1, n_mels, T) - padded
        clinical_batch: (n_patients, 38)
        labels: (n_patients,)
        patient_ids: List[str]
    """
    max_clips = max(len(p.audio_paths) for p in patients)

    audio_list = []
    clinical_list = []
    labels = []
    patient_ids = []

    for patient in patients:
        audio_feats, clinical_feats, label = extract_patient_features(
            patient, audio_config, clinical_preprocessor, fit_preprocessor
        )

        # Pad audio clips to max_clips
        n_clips = audio_feats.shape[0]
        if n_clips < max_clips:
            pad_shape = (max_clips - n_clips,) + audio_feats.shape[1:]
            padding = np.zeros(pad_shape, dtype=np.float32)
            audio_feats = np.vstack([audio_feats, padding])

        audio_list.append(audio_feats)
        clinical_list.append(clinical_feats)
        labels.append(label)
        patient_ids.append(patient.patient_id)

    return (
        np.stack(audio_list, axis=0),  # (n_patients, max_clips, 1, n_mels, T)
        np.stack(clinical_list, axis=0),  # (n_patients, 38)
        np.array(labels, dtype=np.int64),
        patient_ids,
    )


# =============================================================================
# Data Hash for Reproducibility
# =============================================================================

def compute_dataset_hash(patients: List[PatientExample]) -> str:
    """Compute hash of dataset for reproducibility tracking."""
    hasher = hashlib.sha256()
    for p in sorted(patients, key=lambda x: x.patient_id):
        hasher.update(p.patient_id.encode())
        hasher.update(str(p.label).encode())
        for ap in p.audio_paths:
            hasher.update(str(ap).encode())
            if ap.exists():
                hasher.update(ap.read_bytes()[:1024])  # First 1KB of audio
        # Clinical features
        for k in sorted(p.clinical_features.keys()):
            hasher.update(f"{k}:{p.clinical_features[k]}".encode())
    return hasher.hexdigest()[:16]