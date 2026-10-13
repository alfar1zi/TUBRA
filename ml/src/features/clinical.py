from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
from pathlib import Path


# =============================================================================
# CODA TB Clinical Feature Schema (26 features, exact order)
# =============================================================================

CLINICAL_FEATURE_ORDER = (
    # Demographics (4)
    "sex",                    # 0: Male, 1: Female
    "age",                    # years
    "height",                 # cm
    "weight",                 # kg
    # Symptom duration (1)
    "reported_cough_dur",     # weeks
    # TB history (4) - one-hot
    "tb_prior",
    "tb_prior_Pul",
    "tb_prior_Extrapul",
    "tb_prior_Unknown",
    # Symptoms (6) - binary
    "hemoptysis",
    "heart_rate",             # bpm
    "temperature",            # Celsius
    "weight_loss",
    "smoke_lweek",
    "fever",
    "night_sweats",
    # Cough count (1)
    "Numberofcoughsoundscollected",
    # Country one-hot (7) - reference: ZA (South Africa) dropped
    "country_IN",             # India
    "country_MG",             # Madagascar
    "country_PH",             # Philippines
    "country_TZ",             # Tanzania
    "country_UG",             # Uganda
    "country_VN",             # Vietnam
    # HIV status (3) - one-hot
    "hiv_Negative",
    "hiv_Positive",
    "hiv_Unknown",
    # Missing indicators (6) - 1.0 if the source field was not recorded.
    # Without these, mode imputation cannot be distinguished from an actual
    # report, which matters when ~70% of symptom fields are missing.
    "miss_hemoptysis",
    "miss_fever",
    "miss_night_sweats",
    "miss_weight_loss",
    "miss_smoke_lweek",
    "miss_tb_prior",
)

# Fields that are numeric (RobustScaler)
NUMERIC_FIELDS = (
    "age",
    "height",
    "weight",
    "reported_cough_dur",
    "heart_rate",
    "temperature",
    "Numberofcoughsoundscollected",
)

# Fields that are binary (StandardScaler with mean=0.5, scale=0.5 for [0,1] -> [-1,1])
BINARY_FIELDS = (
    "sex",
    "tb_prior",
    "tb_prior_Pul",
    "tb_prior_Extrapul",
    "tb_prior_Unknown",
    "hemoptysis",
    "weight_loss",
    "smoke_lweek",
    "fever",
    "night_sweats",
    "country_IN",
    "country_MG",
    "country_PH",
    "country_TZ",
    "country_UG",
    "country_VN",
    "hiv_Negative",
    "hiv_Positive",
    "hiv_Unknown",
    "miss_hemoptysis",
    "miss_fever",
    "miss_night_sweats",
    "miss_weight_loss",
    "miss_smoke_lweek",
    "miss_tb_prior",
)

# Valid country codes from CODA TB
VALID_COUNTRIES = frozenset({"IN", "MG", "PH", "ZA", "TZ", "UG", "VN"})


# =============================================================================
# Country Normalization
# =============================================================================

def normalise_country(value: Any) -> Optional[str]:
    """Normalize country code to CODA TB standard."""
    if value is None:
        return None
    v = str(value).strip().upper()
    # Map common variants
    mapping = {
        "IND": "IN", "INDIA": "IN",
        "MDG": "MG", "MADAGASCAR": "MG",
        "PHL": "PH", "PHILIPPINES": "PH",
        "ZAF": "ZA", "SOUTH AFRICA": "ZA", "SA": "ZA",
        "TZA": "TZ", "TANZANIA": "TZ",
        "UGA": "UG", "UGANDA": "UG",
        "VNM": "VN", "VIETNAM": "VN", "VIE": "VN",
    }
    return mapping.get(v, v if v in VALID_COUNTRIES else None)


# =============================================================================
# Clinical Preprocessor
# =============================================================================

@dataclass
class ClinicalPreprocessor:
    """Preprocessor for CODA TB clinical features.

    Handles:
    - Missing value imputation (median for numeric, mode for binary)
    - RobustScaler for numeric fields (robust to outliers)
    - StandardScaler for binary fields (maps [0,1] -> [-1,1])
    - Country one-hot encoding (7 countries, ZA as reference)
    - HIV status one-hot encoding (3 categories)
    - TB prior one-hot encoding (4 categories)
    """
    feature_order: tuple[str, ...] = CLINICAL_FEATURE_ORDER
    numeric_fields: tuple[str, ...] = NUMERIC_FIELDS
    binary_fields: tuple[str, ...] = BINARY_FIELDS

    # Fitted scalers (set after fit())
    numeric_scaler: Optional[RobustScaler] = None
    binary_scaler: Optional[StandardScaler] = None

    # Imputation values (set after fit())
    numeric_medians: dict[str, float] = None
    binary_modes: dict[str, float] = None

    def __post_init__(self):
        if self.numeric_medians is None:
            self.numeric_medians = {}
        if self.binary_modes is None:
            self.binary_modes = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit(self, records: list[Mapping[str, Any]]) -> "ClinicalPreprocessor":
        """Fit scalers and imputation values on training data."""
        if not records:
            raise ValueError("No records provided for fitting")

        # Convert to feature matrix
        X = self._records_to_matrix(records, fit_mode=True)

        # Fit numeric scaler (RobustScaler - robust to outliers)
        numeric_idx = [i for i, f in enumerate(self.feature_order) if f in self.numeric_fields]
        if numeric_idx:
            self.numeric_scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=True)
            self.numeric_scaler.fit(X[:, numeric_idx])

        # Fit binary scaler (StandardScaler - maps [0,1] to [-1,1])
        binary_idx = [i for i, f in enumerate(self.feature_order) if f in self.binary_fields]
        if binary_idx:
            self.binary_scaler = StandardScaler(with_mean=True, with_std=True)
            self.binary_scaler.fit(X[:, binary_idx])

        return self

    def transform(self, records: list[Mapping[str, Any]]) -> np.ndarray:
        """Transform records to feature matrix (n_samples, 38)."""
        if not records:
            return np.empty((0, len(self.feature_order)), dtype=np.float32)

        X = self._records_to_matrix(records, fit_mode=False)

        # Apply numeric scaler
        if self.numeric_scaler is not None:
            numeric_idx = [i for i, f in enumerate(self.feature_order) if f in self.numeric_fields]
            X[:, numeric_idx] = self.numeric_scaler.transform(X[:, numeric_idx])

        # Apply binary scaler
        if self.binary_scaler is not None:
            binary_idx = [i for i, f in enumerate(self.feature_order) if f in self.binary_fields]
            X[:, binary_idx] = self.binary_scaler.transform(X[:, binary_idx])

        return X.astype(np.float32)

    def fit_transform(self, records: list[Mapping[str, Any]]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(records).transform(records)

    def save(self, path: str | Path) -> None:
        """Save preprocessor to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "feature_order": self.feature_order,
            "numeric_fields": self.numeric_fields,
            "binary_fields": self.binary_fields,
            "numeric_scaler": self.numeric_scaler,
            "binary_scaler": self.binary_scaler,
            "numeric_medians": self.numeric_medians,
            "binary_modes": self.binary_modes,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "ClinicalPreprocessor":
        """Load preprocessor from disk."""
        data = joblib.load(path)
        preprocessor = cls(
            feature_order=data["feature_order"],
            numeric_fields=data["numeric_fields"],
            binary_fields=data["binary_fields"],
        )
        preprocessor.numeric_scaler = data["numeric_scaler"]
        preprocessor.binary_scaler = data["binary_scaler"]
        preprocessor.numeric_medians = data["numeric_medians"]
        preprocessor.binary_modes = data["binary_modes"]
        return preprocessor

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _records_to_matrix(self, records: list[Mapping[str, Any]], fit_mode: bool) -> np.ndarray:
        """Convert list of record dicts to feature matrix."""
        n_samples = len(records)
        n_features = len(self.feature_order)
        X = np.zeros((n_samples, n_features), dtype=np.float32)

        for i, record in enumerate(records):
            feat = self._extract_features(record, fit_mode)
            for j, fname in enumerate(self.feature_order):
                X[i, j] = feat.get(fname, 0.0)

        # Compute imputation values if fit_mode
        if fit_mode:
            self._compute_imputation_values(X)

        # Impute missing (NaN) values
        X = self._impute_missing(X, fit_mode)

        return X

    def _extract_features(self, record: Mapping[str, Any], fit_mode: bool) -> dict[str, float]:
        """Extract features from a single record (see CLINICAL_FEATURE_ORDER)."""
        feat = {}
        # Missing indicators default to 0.0 (present); set to 1.0 below when a
        # source field is absent so the model can tell imputation apart from a
        # real report.
        for ind in ("miss_hemoptysis", "miss_fever", "miss_night_sweats",
                    "miss_weight_loss", "miss_smoke_lweek", "miss_tb_prior"):
            feat[ind] = 0.0

        # --- Demographics ---
        # sex: Male=0, Female=1
        sex = str(record.get("sex", "")).strip().lower()
        feat["sex"] = 1.0 if sex in {"f", "female", "1", "wanita", "perempuan"} else 0.0

        # age, height, weight - numeric
        for field in ("age", "height", "weight"):
            val = record.get(field)
            feat[field] = self._to_float(val) if val is not None else np.nan

        # --- Cough duration ---
        cough_dur = record.get("reported_cough_dur")
        feat["reported_cough_dur"] = self._to_float(cough_dur) if cough_dur is not None else np.nan

        # --- TB History (one-hot) ---
        # A missing history is not the same as "no prior TB"; use NaN so it is
        # imputed rather than asserted as negative.
        raw_prior = record.get("tb_prior", None)
        if raw_prior is None or (isinstance(raw_prior, float) and np.isnan(raw_prior)):
            feat["tb_prior"] = np.nan
            feat["miss_tb_prior"] = 1.0
        else:
            tb_prior = str(raw_prior).strip().lower()
            feat["tb_prior"] = 1.0 if tb_prior in {"yes", "y", "true", "1", "ya"} else 0.0

        tb_prior_type = str(record.get("tb_prior_type", record.get("tb_prior_detail", ""))).strip().lower()
        feat["tb_prior_Pul"] = 1.0 if "pul" in tb_prior_type else 0.0
        feat["tb_prior_Extrapul"] = 1.0 if "extra" in tb_prior_type else 0.0
        feat["tb_prior_Unknown"] = 1.0 if (feat["tb_prior"] == 1.0 and feat["tb_prior_Pul"] == 0.0 and feat["tb_prior_Extrapul"] == 0.0) else 0.0

        # --- Symptoms (binary) ---
        # Distinguish "explicitly no" (0.0) from "not recorded" (NaN). About
        # 70% of CODA TB symptom fields are missing; silently mapping them to
        # 0.0 would teach the model that most patients have no symptoms. NaN
        # instead triggers mode imputation in _impute_missing.
        for field in ("hemoptysis", "weight_loss", "smoke_lweek", "fever", "night_sweats"):
            raw = record.get(field, None)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                feat[field] = np.nan
                feat[f"miss_{field}"] = 1.0
            else:
                val = str(raw).strip().lower()
                if val in {"yes", "y", "true", "1", "ya"}:
                    feat[field] = 1.0
                elif val in {"no", "n", "false", "0", "tidak"}:
                    feat[field] = 0.0
                else:
                    feat[field] = np.nan
                    feat[f"miss_{field}"] = 1.0

        # heart_rate, temperature - numeric
        for field in ("heart_rate", "temperature"):
            val = record.get(field)
            feat[field] = self._to_float(val) if val is not None else np.nan

        # --- Cough count ---
        cough_count = record.get("Numberofcoughsoundscollected", record.get("cough_count", 5))
        feat["Numberofcoughsoundscollected"] = self._to_float(cough_count) if cough_count is not None else 5.0

        # --- Country (one-hot, ZA as reference) ---
        country = normalise_country(record.get("country"))
        for c in ("IN", "MG", "PH", "TZ", "UG", "VN"):
            feat[f"country_{c}"] = 1.0 if country == c else 0.0
        # ZA is reference (all zeros)

        # --- HIV Status (one-hot) ---
        hiv = str(record.get("hiv", record.get("hiv_status", ""))).strip().lower()
        feat["hiv_Negative"] = 1.0 if hiv in {"neg", "negative", "0", "non-reactive", "negatif"} else 0.0
        feat["hiv_Positive"] = 1.0 if hiv in {"pos", "positive", "1", "reactive", "positif"} else 0.0
        feat["hiv_Unknown"] = 1.0 if (feat["hiv_Negative"] == 0.0 and feat["hiv_Positive"] == 0.0) else 0.0

        return feat

    @staticmethod
    def _to_float(val: Any) -> float:
        """Convert value to float, return NaN on failure."""
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    def _compute_imputation_values(self, X: np.ndarray) -> None:
        """Compute median (numeric) and mode (binary) for imputation."""
        for j, fname in enumerate(self.feature_order):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                if fname in self.numeric_fields:
                    self.numeric_medians[fname] = float(np.median(valid))
                else:
                    # Mode for binary: most common value
                    unique, counts = np.unique(valid, return_counts=True)
                    self.binary_modes[fname] = float(unique[np.argmax(counts)])

    def _impute_missing(self, X: np.ndarray, fit_mode: bool) -> np.ndarray:
        """Impute NaN values with median (numeric) or mode (binary)."""
        for j, fname in enumerate(self.feature_order):
            col = X[:, j]
            nan_mask = np.isnan(col)
            if np.any(nan_mask):
                if fname in self.numeric_fields:
                    fill_value = self.numeric_medians.get(fname, 0.0)
                else:
                    fill_value = self.binary_modes.get(fname, 0.0)
                col[nan_mask] = fill_value
        return X


# =============================================================================
# Convenience Function for Single Record (Serving)
# =============================================================================

def encode_single_record(
    record: Mapping[str, Any],
    preprocessor: ClinicalPreprocessor,
) -> np.ndarray:
    """Encode a single record for inference. Returns (1, 38) array."""
    return preprocessor.transform([record])[0:1]


# =============================================================================
# Default Preprocessor Instance
# =============================================================================

DEFAULT_PREPROCESSOR = ClinicalPreprocessor()