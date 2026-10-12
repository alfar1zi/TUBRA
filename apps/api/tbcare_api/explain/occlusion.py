"""
Symptom occlusion analysis.
Replace each symptom with "unknown" (availability=0) and measure logit change.
"""

import numpy as np
from typing import List, Tuple

from tbcare_api.inference.head import HeadModel


SYMPTOM_NAMES = [
    "hemoptysis",
    "fever",
    "night_sweats",
    "weight_loss",
    "prior_tb",
    "smoking_never",
    "smoking_former",
    "smoking_current",
    "age",
]


def compute_occlusion(
    head: HeadModel,
    embeddings: np.ndarray,
    symptoms: np.ndarray,
    availability: np.ndarray,
) -> List[Tuple[str, float]]:
    """
    Compute occlusion importance for each symptom.

    Args:
        head: HeadModel instance
        embeddings: (N, 512)
        symptoms: (9,) symptom values
        availability: (9,) availability mask

    Returns:
        List of (symptom_name, logit_change) sorted by importance (descending)
    """
    # Baseline logit
    baseline = head.forward(embeddings, symptoms, availability)
    baseline_logit = baseline["logit"]

    results = []

    # For each symptom group
    # Boolean symptoms (indices 0-4): hemoptysis, fever, night_sweats, weight_loss, prior_tb
    for i in range(5):
        if availability[i] == 0:
            # Already unknown - no change
            results.append((SYMPTOM_NAMES[i], 0.0))
            continue

        # Occlude: set to 0.5 (unknown), availability to 0
        occ_symptoms = symptoms.copy()
        occ_availability = availability.copy()
        occ_symptoms[i] = 0.5
        occ_availability[i] = 0

        occ = head.forward(embeddings, occ_symptoms, occ_availability)
        change = baseline_logit - occ["logit"]  # positive = symptom increased logit
        results.append((SYMPTOM_NAMES[i], float(change)))

    # Smoking (indices 5-7): one-hot, occlude all three together
    if any(availability[5:8]):
        occ_symptoms = symptoms.copy()
        occ_availability = availability.copy()
        occ_symptoms[5:8] = [1/3, 1/3, 1/3]
        occ_availability[5:8] = [0, 0, 0]

        occ = head.forward(embeddings, occ_symptoms, occ_availability)
        change = baseline_logit - occ["logit"]
        # Assign same change to all three, but we'll report as single "smoking"
        results.append(("smoking", float(change)))
    else:
        results.append(("smoking", 0.0))

    # Age (index 8)
    if availability[8] == 0:
        results.append((SYMPTOM_NAMES[8], 0.0))
    else:
        occ_symptoms = symptoms.copy()
        occ_availability = availability.copy()
        occ_symptoms[8] = 0.5  # middle age
        occ_availability[8] = 0

        occ = head.forward(embeddings, occ_symptoms, occ_availability)
        change = baseline_logit - occ["logit"]
        results.append((SYMPTOM_NAMES[8], float(change)))

    # Filter out smoking_* individual entries, keep only "smoking"
    filtered = [(name, val) for name, val in results if not name.startswith("smoking_")]

    # Sort by absolute change (descending)
    filtered.sort(key=lambda x: abs(x[1]), reverse=True)

    return filtered


def format_symptom_explanation(occlusion_results: List[Tuple[str, float]], top_k: int = 3) -> List[str]:
    """
    Convert occlusion results to human-readable explanations.
    """
    name_map = {
        "hemoptysis": "Batuk berdarah",
        "fever": "Demam",
        "night_sweats": "Keringat malam",
        "weight_loss": "Berat badan turun",
        "prior_tb": "Riwayat TBC",
        "smoking": "Merokok",
        "age": "Usia",
    }

    explanations = []
    for name, change in occlusion_results[:top_k]:
        if change <= 0:
            continue
        label = name_map.get(name, name)
        if name == "smoking":
            explanations.append(f"{label}: menaikkan prioritas")
        elif name in ("hemoptysis", "fever", "night_sweats", "weight_loss", "prior_tb"):
            explanations.append(f"{label}: menaikkan prioritas")
        elif name == "age":
            explanations.append(f"{label}: memengaruhi prioritas")

    if not explanations:
        return ["Jawaban gejala Anda tidak banyak memengaruhi hasil."]

    return explanations