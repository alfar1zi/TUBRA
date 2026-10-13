"""Clinical preprocessing tests.

Guards the fix in docs/proposal/training_improvement.md (P3): a *missing*
symptom must not be encoded as "symptom absent" (0.0). About 70% of the CODA TB
symptom fields are missing, so mapping them to 0.0 would inject a large,
systematic bias into the clinical branch.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def pre():
    from ml.src.features.clinical import ClinicalPreprocessor

    # Fit on a small synthetic cohort that has some known positives/negatives.
    records = []
    for i in range(50):
        rec = {
            "sex": "Male" if i % 2 else "Female",
            "age": 30 + i,
            "height": 170,
            "weight": 60,
            "reported_cough_dur": 3,
            "hemoptysis": "yes" if i % 5 == 0 else "no",
            "fever": "yes" if i % 3 == 0 else "no",
            "night_sweats": "yes" if i % 4 == 0 else "no",
            "weight_loss": "yes" if i % 6 == 0 else "no",
            "smoke_lweek": "no",
            "tb_prior": "no",
            "heart_rate": 80,
            "temperature": 37.0,
            "country": "KE",
            "hiv": "Negative",
        }
        records.append(rec)
    return ClinicalPreprocessor().fit(records)


def test_missing_symptom_sets_indicator(pre):
    """A missing symptom must fire its missing-indicator, not silently be 'no'."""
    base = {
        "sex": "Male", "age": 40, "height": 170, "weight": 60,
        "reported_cough_dur": 3, "country": "KE", "hiv": "Negative",
    }
    order = list(pre.feature_order)
    missing = pre.transform([dict(base)])[0]
    explicit_no = pre.transform([{**base, "hemoptysis": "no", "fever": "no",
                                  "night_sweats": "no", "weight_loss": "no",
                                  "smoke_lweek": "no", "tb_prior": "no"}])[0]

    for f in ("hemoptysis", "fever", "night_sweats", "weight_loss",
              "smoke_lweek", "tb_prior"):
        j = order.index(f"miss_{f}")
        assert missing[j] > explicit_no[j], (
            f"missing indicator for '{f}' did not fire"
        )


def test_explicit_yes_and_no_differ(pre):
    base = {"sex": "Male", "age": 40, "height": 170, "weight": 60,
            "reported_cough_dur": 3, "country": "KE", "hiv": "Negative"}
    yes = pre.transform([{**base, "hemoptysis": "yes"}])[0]
    no = pre.transform([{**base, "hemoptysis": "no"}])[0]
    j = list(pre.feature_order).index("hemoptysis")
    assert yes[j] > no[j]


def test_transform_shape_is_feature_order(pre):
    rec = {"sex": "Male", "age": 40, "country": "KE"}
    out = pre.transform([rec])
    assert out.shape == (1, len(pre.feature_order))


def test_serving_and_training_encode_identically(pre):
    """The API's preprocessor and the training preprocessor must agree."""
    rec = {
        "sex": "Female", "age": 45, "height": 160, "weight": 55,
        "reported_cough_dur": 4, "hemoptysis": "yes", "fever": "yes",
        "night_sweats": "no", "weight_loss": "yes", "smoke_lweek": "no",
        "tb_prior": "yes", "heart_rate": 90, "temperature": 38.0,
        "country": "ID", "hiv": "Positive",
    }
    training = pre.transform([rec])[0]
    # encode_single_record uses the same fitted preprocessor object.
    from ml.src.features.clinical import encode_single_record

    serving = encode_single_record(rec, pre)[0]
    np.testing.assert_allclose(training, serving, rtol=1e-6, atol=1e-6)


def _api_serving_preprocessor(fitted_pre):
    """Build the API's ServingPreprocessor from a fitted training preprocessor.

    The API preprocessor is a separate, torch-free re-implementation; this
    constructs it from the same fitted state so we can compare the two.
    """
    from apps.api.tbcare_api.inference.ensemble import ServingPreprocessor

    saved = fitted_pre.to_dict() if hasattr(fitted_pre, "to_dict") else None
    if saved is None:
        saved = {
            "feature_order": tuple(fitted_pre.feature_order),
            "numeric_fields": tuple(fitted_pre.numeric_fields),
            "binary_fields": tuple(fitted_pre.binary_fields),
            "numeric_scaler": fitted_pre.numeric_scaler,
            "binary_scaler": fitted_pre.binary_scaler,
            "numeric_medians": fitted_pre.numeric_medians,
            "binary_modes": fitted_pre.binary_modes,
        }
    return ServingPreprocessor(
        feature_order=saved["feature_order"],
        numeric_fields=saved["numeric_fields"],
        binary_fields=saved["binary_fields"],
        numeric_scaler=saved["numeric_scaler"],
        binary_scaler=saved["binary_scaler"],
        numeric_medians=saved["numeric_medians"],
        binary_modes=saved["binary_modes"],
    )


@pytest.mark.parametrize(
    "record",
    [
        # Every symptom missing ("don't know" / skipped).
        {
            "sex": "male", "age": 40, "reported_cough_dur": 3.0,
            "hemoptysis": None, "fever": None, "night_sweats": None,
            "weight_loss": None, "tb_prior": None, "smoke_lweek": None,
        },
        # All symptoms explicit.
        {
            "sex": "female", "age": 25, "reported_cough_dur": 6.0,
            "hemoptysis": True, "fever": False, "night_sweats": True,
            "weight_loss": False, "tb_prior": True, "smoke_lweek": False,
        },
        # Mixed: some explicit, some missing.
        {
            "sex": "male", "age": 60, "reported_cough_dur": 10.0,
            "hemoptysis": False, "fever": None, "night_sweats": True,
            "weight_loss": None, "tb_prior": None, "smoke_lweek": True,
        },
    ],
)
def test_api_serving_matches_training_with_missing_and_present_symptoms(pre, record):
    """The API's ServingPreprocessor must encode exactly like training.

    Guards the P3 regression on the *serving* side: the API used to map a
    missing symptom to 0.0 ("absent") and never fire the ``miss_*`` indicator,
    silently diverging from training (which uses NaN + indicator).
    """
    import sys

    sys.path.insert(0, "apps/api")
    serving_pre = _api_serving_preprocessor(pre)

    training = pre.transform([dict(record)])[0]
    api_serving = serving_pre.transform_record(dict(record))[0]
    np.testing.assert_allclose(
        training, api_serving, rtol=1e-6, atol=1e-6,
        err_msg="API serving preprocessor diverged from training",
    )


def test_api_serving_fires_missing_indicators(pre):
    """A missing symptom must fire ``miss_*`` in the API preprocessor too."""
    import sys

    sys.path.insert(0, "apps/api")
    serving_pre = _api_serving_preprocessor(pre)

    base = {"sex": "male", "age": 40, "reported_cough_dur": 3.0}
    missing = serving_pre.transform_record(dict(base))[0]
    explicit_no = serving_pre.transform_record(
        {
            **base,
            "hemoptysis": False, "fever": False, "night_sweats": False,
            "weight_loss": False, "tb_prior": False, "smoke_lweek": False,
        }
    )[0]

    order = list(pre.feature_order)
    for f in ("hemoptysis", "fever", "night_sweats", "weight_loss",
              "smoke_lweek", "tb_prior"):
        j = order.index(f"miss_{f}")
        assert missing[j] > explicit_no[j], (
            f"API serving did not fire miss_{f} for a missing symptom"
        )
