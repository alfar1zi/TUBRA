from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


# =============================================================================
# Tier Definitions (per PRD)
# =============================================================================

class Tier(str, Enum):
    A = "A"  # Periksa dahak 1-2 hari
    B = "B"  # Periksa dahak minggu ini
    C = "C"  # Tetap periksa dahak


TIER_LABELS = {
    Tier.A: "Segera periksa dahak (1-2 hari)",
    Tier.B: "Periksa dahak minggu ini",
    Tier.C: "Tetap periksa dahak",
}

TIER_EXPLANATIONS = {
    Tier.A: (
        "Berdasarkan gejala dan suara batuk Anda, risiko TBC tergolong tinggi. "
        "Segera periksa dahak ke fasilitas kesehatan terdekat dalam 1-2 hari. "
        "Pemeriksaan dahak (tes mikroskopis atau TCM) adalah satu-satunya cara memastikan TBC."
    ),
    Tier.B: (
        "Berdasarkan gejala dan suara batuk Anda, risiko TBC tergolong sedang. "
        "Periksa dahak ke fasilitas kesehatan terdekat dalam minggu ini. "
        "Pemeriksaan dahak (tes mikroskopis atau TCM) adalah satu-satunya cara memastikan TBC."
    ),
    Tier.C: (
        "Berdasarkan gejala dan suara batuk Anda, risiko TBC tergolong rendah. "
        "Namun, karena Anda batuk 2 minggu atau lebih, Anda tetap perlu periksa dahak. "
        "Pemeriksaan dahak (tes mikroskopis atau TCM) adalah satu-satunya cara memastikan TBC."
    ),
}

# =============================================================================
# Action Definitions
# =============================================================================

@dataclass(frozen=True)
class Action:
    """Single actionable step for the user."""
    id: str
    label: str
    description: str
    priority: int  # Lower = higher priority
    category: str  # "immediate", "this_week", "followup", "info"


# Pre-defined action catalog
ACTION_CATALOG = {
    # Immediate actions (Tier A)
    "sputum_test_asap": Action(
        id="sputum_test_asap",
        label="Periksa dahak segera (1-2 hari)",
        description="Datang ke puskesmas/rumah sakit dengan laboratorium TBC untuk pemeriksaan dahak (mikroskopis/TCM). Bawa hasil skrining ini.",
        priority=1,
        category="immediate",
    ),
    "bring_screening_result": Action(
        id="bring_screening_result",
        label="Bawa hasil skrining ke fasilitas kesehatan",
        description="Hasil skrining ini membantu tenaga medis memprioritaskan pemeriksaan Anda.",
        priority=2,
        category="immediate",
    ),
    "mask_and_distance": Action(
        id="mask_and_distance",
        label="Gunakan masker & jaga jarak",
        description="Hingga hasil dahak keluar, gunakan masker dan hindari kontak dekat dengan orang lain, terutama anak-anak dan lansia.",
        priority=3,
        category="immediate",
    ),

    # This week actions (Tier B)
    "sputum_test_this_week": Action(
        id="sputum_test_this_week",
        label="Periksa dahak minggu ini",
        description="Jadwalkan pemeriksaan dahak di puskesmas/rumah sakit dengan laboratorium TBC dalam minggu ini.",
        priority=1,
        category="this_week",
    ),
    "monitor_symptoms": Action(
        id="monitor_symptoms",
        label="Pantau gejala",
        description="Catat apakah batuk memburuk, muncul demam, keringat malam, atau sesak napas. Jika memburuk, periksa lebih cepat.",
        priority=2,
        category="this_week",
    ),
    "mask_precaution": Action(
        id="mask_precaution",
        label="Gunakan masker di tempat umum",
        description="Sebagai langkah preventif, gunakan masker saat di tempat umum hingga pemeriksaan dahak selesai.",
        priority=3,
        category="this_week",
    ),

    # Follow-up actions (Tier C)
    "sputum_test_still_needed": Action(
        id="sputum_test_still_needed",
        label="Tetap periksa dahak",
        description="Meskipun risiko rendah, batuk ≥2 minggu wajib diperiksa dahak. Janjikan pemeriksaan di fasilitas kesehatan.",
        priority=1,
        category="followup",
    ),
    "followup_if_worsen": Action(
        id="followup_if_worsen",
        label="Periksa lebih cepat jika gejala memburuk",
        description="Jika batuk berdahak berwarna, demam, keringat malam, penurunan berat badan, atau sesak napas — periksa segera.",
        priority=2,
        category="followup",
    ),
    "general_health": Action(
        id="general_health",
        label="Jaga kesehatan umum",
        description="Istirahat cukup, nutrisi seimbang, hindari merokok, dan minum obat batuk sesuai anjuran dokter/apoteker.",
        priority=3,
        category="followup",
    ),

    # Info actions (all tiers)
    "not_diagnosis": Action(
        id="not_diagnosis",
        label="Ini bukan diagnosis",
        description="TBCCare adalah alat skrining, bukan diagnosis. Hanya pemeriksaan dahak yang dapat memastikan TBC.",
        priority=1,
        category="info",
    ),
    "find_facility": Action(
        id="find_facility",
        label="Cari fasilitas pemeriksaan dahak terdekat",
        description="Puskesmas dan rumah sakit rujukan TBC menyediakan pemeriksaan dahak gratis (Jaminan Kesehatan Nasional).",
        priority=2,
        category="info",
    ),
    "tb_info": Action(
        id="tb_info",
        label="Pelajari lebih lanjut tentang TBC",
        description="TBC menyembuhkan jika diobati lengkap. Obat gratis tersedia di program NTP (National TB Program).",
        priority=3,
        category="info",
    ),
}


# =============================================================================
# Rule Engine
# =============================================================================

class RuleEngine:
    """
    Deterministic rule-based action generator.
    Maps (tier, clinical_features) -> ordered list of actions.
    No LLM, no randomness, fully auditable.
    """

    def __init__(self):
        self.catalog = ACTION_CATALOG

    def generate_actions(
        self,
        tier: Tier,
        prob_fusion: float,
        clinical_record: Dict[str, Any],
    ) -> List[Action]:
        """Generate ordered actions for a given tier and clinical context."""
        actions = []

        # Tier-specific primary actions
        if tier == Tier.A:
            actions.extend([
                self.catalog["sputum_test_asap"],
                self.catalog["bring_screening_result"],
                self.catalog["mask_and_distance"],
            ])
        elif tier == Tier.B:
            actions.extend([
                self.catalog["sputum_test_this_week"],
                self.catalog["monitor_symptoms"],
                self.catalog["mask_precaution"],
            ])
        else:  # Tier.C
            actions.extend([
                self.catalog["sputum_test_still_needed"],
                self.catalog["followup_if_worsen"],
                self.catalog["general_health"],
            ])

        # Conditional actions based on clinical features
        actions.extend(self._conditional_actions(clinical_record))

        # Universal info actions (always last)
        actions.extend([
            self.catalog["not_diagnosis"],
            self.catalog["find_facility"],
            self.catalog["tb_info"],
        ])

        # Sort by priority
        actions.sort(key=lambda a: a.priority)
        return actions

    def _conditional_actions(self, clinical_record: Dict[str, Any]) -> List[Action]:
        """Add conditional actions based on clinical features."""
        actions = []

        # Hemoptysis (batuk darah) -> urgency upgrade
        if self._is_yes(clinical_record.get("hemoptysis")):
            actions.append(Action(
                id="hemoptysis_urgent",
                label="Batuk darah: periksa segera hari ini",
                description="Batuk darah (hemoptysis) adalah tanda bahaya. Periksa ke darurat/UGD terdekat hari ini.",
                priority=0,  # Highest priority
                category="immediate",
            ))

        # HIV positive -> higher urgency
        hiv = str(clinical_record.get("hiv", "")).strip().lower()
        if hiv in {"pos", "positive", "1", "reactive", "positif"}:
            actions.append(Action(
                id="hiv_positive_urgent",
                label="HIV positif: prioritas pemeriksaan TBC",
                description="Orang dengan HIV memiliki risiko TBC lebih tinggi. Segera periksa dahak.",
                priority=0,
                category="immediate",
            ))

        # Prior TB -> recurrence risk
        if self._is_yes(clinical_record.get("tb_prior")):
            actions.append(Action(
                id="prior_tb_recurrence",
                label="Riwayat TBC: waspadai reaktivasi",
                description="Anda pernah TBC. Gejala baru bisa berarti reaktivasi. Wajib periksa dahak.",
                priority=1,
                category="this_week",
            ))

        # Weight loss + night sweats + fever = classic TB triad
        if (self._is_yes(clinical_record.get("weight_loss")) and
            self._is_yes(clinical_record.get("night_sweats")) and
            self._is_yes(clinical_record.get("fever"))):
            actions.append(Action(
                id="classic_triad",
                label="Gejala klasik TBC (penurunan BB, keringat malam, demam)",
                description="Kombinasi tiga gejala ini meningkatkan kecurigaan TBC. Periksa dahak prioritas.",
                priority=1,
                category="this_week",
            ))

        # Smoking
        if self._is_yes(clinical_record.get("smoke_lweek")):
            actions.append(Action(
                id="smoking_cessation",
                label="Merokok memperburuk batuk & risiko TBC",
                description="Merokok merusak paru dan menyembunyikan gejala TBC. Pertimbangkan berhenti merokok.",
                priority=2,
                category="followup",
            ))

        return actions

    @staticmethod
    def _is_yes(value: Any) -> bool:
        return str(value).strip().lower() in {"yes", "y", "true", "1", "ya"}

    def generate_actions_text(
        self,
        tier: Tier,
        prob_fusion: float,
        clinical_record: Dict[str, Any],
        language: str = "id",
    ) -> List[str]:
        """Generate human-readable action strings."""
        actions = self.generate_actions(tier, prob_fusion, clinical_record)
        return [f"{a.label}: {a.description}" for a in actions]


# =============================================================================
# Tier-Specific Output Builder
# =============================================================================

@dataclass(frozen=True)
class TierResult:
    """Complete result for a tier."""
    tier: Tier
    tier_label: str
    probability: float
    explanation: str
    actions: List[str]
    disclaimer: str


DISCLAIMER = (
    "PENTING: TBCCare adalah alat skrining, BUKAN diagnosis. "
    "Hanya pemeriksaan dahak (mikroskopis/TCM) yang dapat memastikan TBC. "
    "Hasil ini tidak menggantikan anjuran dokter."
)


def build_tier_result(
    tier: Tier,
    prob_fusion: float,
    clinical_record: Dict[str, Any],
    language: str = "id",
) -> TierResult:
    """Build complete tier result with explanation and actions."""
    engine = RuleEngine()
    actions = engine.generate_actions_text(tier, prob_fusion, clinical_record, language)

    return TierResult(
        tier=tier,
        tier_label=TIER_LABELS[tier],
        probability=prob_fusion,
        explanation=TIER_EXPLANATIONS[tier],
        actions=actions,
        disclaimer=DISCLAIMER,
    )


# =============================================================================
# Convenience Functions (for API)
# =============================================================================

def get_tier_label(tier: Tier) -> str:
    return TIER_LABELS[tier]


def get_tier_explanation(tier: Tier) -> str:
    return TIER_EXPLANATIONS[tier]


def get_disclaimer() -> str:
    return DISCLAIMER
