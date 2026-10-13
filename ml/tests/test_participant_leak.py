"""Participant-leak tests.

AGENTS.md section 4.5: data separation is per participant, never per cough file.
ARCHITECTURE.md section 6.2: an automated `test_no_participant_leak` must fail
the build if a participant appears in more than one fold.

These tests are pure and synthetic (no real data needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_manifest(n_participants: int = 40, clips_per: int = 3) -> pd.DataFrame:
    """One row per cough clip, several clips per participant."""
    rows = []
    rng = np.random.RandomState(0)
    for p in range(n_participants):
        pid = f"P{p:03d}"
        label = int(rng.rand() < 0.4)
        for c in range(clips_per):
            rows.append(
                {
                    "participant": pid,
                    "file": f"{pid}/clip{c}.wav",
                    "tb_status": label,
                    "split": "train",
                }
            )
    return pd.DataFrame(rows)


def test_groupkfold_keeps_participant_in_one_fold():
    """Every participant's clips must land in exactly one fold."""
    from ml.scripts.prepare_data import create_groupkfold_splits

    manifest = _make_manifest(40)
    folded = create_groupkfold_splits(manifest, n_splits=5, seed=42)

    per_participant_folds = folded.groupby("participant")["fold"].nunique()
    assert (per_participant_folds == 1).all(), (
        "a participant was split across folds: "
        f"{per_participant_folds[per_participant_folds > 1].to_dict()}"
    )


def test_no_participant_leak_passes_on_clean_splits():
    from ml.scripts.prepare_data import (
        create_groupkfold_splits,
        test_no_participant_leak,
    )

    folded = create_groupkfold_splits(_make_manifest(40), n_splits=5, seed=42)
    assert test_no_participant_leak(folded, n_folds=5) is True


def test_no_participant_leak_detects_injected_leak():
    """A deliberately leaked participant must make the check fail."""
    from ml.scripts.prepare_data import (
        create_groupkfold_splits,
        test_no_participant_leak,
    )

    folded = create_groupkfold_splits(_make_manifest(40), n_splits=5, seed=42)
    victim = folded["participant"].iloc[0]
    victim_fold = int(folded.loc[folded["participant"] == victim, "fold"].iloc[0])

    # Duplicate one of the victim's rows into a different fold. This is the
    # exact mistake the check exists to catch: one person in two folds.
    dup = folded[folded["participant"] == victim].iloc[[0]].copy()
    dup["fold"] = (victim_fold + 1) % 5
    leaked = pd.concat([folded, dup], ignore_index=True)

    assert leaked[leaked["participant"] == victim]["fold"].nunique() > 1
    assert test_no_participant_leak(leaked, n_folds=5) is False


def test_fold_count_is_parameterized_not_hardcoded():
    """The leak check must honour the fold count actually used for training."""
    from ml.scripts.prepare_data import (
        create_groupkfold_splits,
        test_no_participant_leak,
    )

    folded = create_groupkfold_splits(_make_manifest(40), n_splits=3, seed=42)
    assert set(folded["fold"].unique()) <= {-1, 0, 1, 2}
    # With 3 folds, fold index 3 and 4 do not exist and must not be probed.
    assert test_no_participant_leak(folded, n_folds=3) is True
