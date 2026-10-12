"""
Temperature calibration utilities.
The temperature T is stored in head.npz and applied in head.py.
This module provides calibration helpers for future use.
"""

import numpy as np


def apply_temperature(logit: float, temperature: float) -> float:
    """Apply temperature scaling to logit and return probability."""
    return 1.0 / (1.0 + np.exp(-logit / temperature))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    For future use when evaluating calibration on test set.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probabilities[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
    return float(ece)