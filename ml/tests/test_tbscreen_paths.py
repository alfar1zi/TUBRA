"""TBscreen filename resolution tests.

The published TBscreen ZIP stores coughs under a different name scheme than
the ``Passive_coughs.csv`` ``path`` column (``PID_1A_1_pixel_0`` ->
``PID_1A1_pixel.wav``). The naive ``path + ".wav"`` lookup silently missed most
rows and made earlier runs train on a fraction of the cohort. These tests pin
the resolution rules from ``_tbscreen_audio_candidates``.
"""

from __future__ import annotations

from pathlib import Path

from ml.scripts.data_loader import _tbscreen_audio_candidates


def _names(cands):
    return [c.name for c in cands]


def test_exact_name_is_first_candidate():
    cands = _tbscreen_audio_candidates(Path("/tmp"), "PID_1A1_yeti_0")
    assert _names(cands)[0] == "PID_1A1_yeti_0.wav"


def test_joined_zip_name_is_generated():
    cands = _tbscreen_audio_candidates(Path("/tmp"), "PID_1A_1_pixel_0")
    assert "PID_1A1_pixel.wav" in _names(cands)


def test_index_stripped_form_is_generated():
    cands = _tbscreen_audio_candidates(Path("/tmp"), "PID_1A_1_pixel_0")
    assert "PID_1A_1_pixel.wav" in _names(cands)


def test_empty_path_yields_no_candidates():
    assert _tbscreen_audio_candidates(Path("/tmp"), "") == []
    assert _tbscreen_audio_candidates(Path("/tmp"), "   ") == []


def test_candidates_are_unique_case_insensitively():
    cands = _tbscreen_audio_candidates(Path("/tmp"), "PID_108A0_yeti_0")
    keys = [c.name.lower() for c in cands]
    assert len(keys) == len(set(keys))
