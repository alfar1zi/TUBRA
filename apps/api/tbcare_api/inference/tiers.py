"""
Tier threshold mapping.
Thresholds come from model_card.json (set during training).
"""


def get_tier_thresholds(model_card: dict) -> dict:
    """Extract tier thresholds from model card."""
    return model_card.get("tier_thresholds", {"A": 0.7, "B": 0.4, "C": 0.0})


def probability_to_tier(probability: float, thresholds: dict) -> str:
    """
    Map probability to tier.

    Args:
        probability: Model output probability (0-1)
        thresholds: Dict with keys "A", "B", "C" (C is always 0.0)

    Returns:
        "A" | "B" | "C"
    """
    if probability >= thresholds["A"]:
        return "A"
    elif probability >= thresholds["B"]:
        return "B"
    else:
        return "C"


def get_all_tiers(thresholds: dict) -> list[str]:
    """Get ordered list of tiers (highest priority first)."""
    return ["A", "B", "C"]