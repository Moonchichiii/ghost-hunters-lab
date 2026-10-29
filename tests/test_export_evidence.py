"""Tests for the scientific evidence export contract."""

from pathlib import Path

import pytest

from ghost_lab.export_evidence import (
    HELDOUT_DATASET_HASH,
    HELDOUT_HASH,
    side_effect_summary,
    verified_report,
)


def test_reject_modified_evidence(tmp_path: Path) -> None:
    path = tmp_path / "changed.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verified_report(path, HELDOUT_HASH, HELDOUT_DATASET_HASH)


def test_reject_incorrect_dataset_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "report.json"
    path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "ghost_lab.export_evidence.sha256_file",
        lambda _path: HELDOUT_HASH,
    )

    with pytest.raises((ValueError, KeyError)):
        verified_report(path, HELDOUT_HASH, HELDOUT_DATASET_HASH)


def test_side_effect_summary_rejects_incomplete_data() -> None:
    report = {
        "experiment": "ghost_side_effects_v1",
        "expected_measurements": 36,
        "conditions": {
            "baseline": {},
            "ghost_positive": {},
            "random_2026": {},
        },
        "records": [],
    }

    with pytest.raises(ValueError, match="12 baseline prompts"):
        side_effect_summary(report)
