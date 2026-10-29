"""Tests for the frozen held-out evaluation contract."""

from pathlib import Path

import pytest

from ghost_lab.evaluate_heldout import (
    DATASET,
    EXPECTED_HASH,
    compact_record,
    conditions,
    load_dataset,
)
from ghost_lab.provenance import sha256_file


def test_frozen_dataset_is_unchanged() -> None:
    assert sha256_file(DATASET) == EXPECTED_HASH

    prompts = load_dataset(DATASET)
    assert len(prompts) == 20

    categories = [item["category"] for item in prompts]
    assert categories.count("neutral") == 8
    assert categories.count("ambiguous") == 6
    assert categories.count("supernatural") == 6


def test_dataset_rejects_changed_content(tmp_path: Path) -> None:
    modified = tmp_path / "heldout.json"
    modified.write_text('{"prompts": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(modified)


def test_seven_conditions_are_defined() -> None:
    variants = conditions()

    assert len(variants) == 7
    assert variants["baseline"] == (None, 0)
    assert variants["ghost_positive"][1] == 16

    random_names = [name for name in variants if name.startswith("random_")]
    assert len(random_names) == 5

    assert all(variants[name][1] == 16 for name in random_names)


def test_compact_record_preserves_missing_scores() -> None:
    prompt = {
        "id": "h01",
        "category": "neutral",
        "text": "Test prompt",
    }
    measurement = {
        "condition": "baseline",
        "target_logprob": None,
        "comparator_logprob": -2.5,
        "preference_score": None,
        "top_logprobs": [{"id": 880, "logprob": -2.5}],
    }

    result = compact_record(prompt, measurement)

    assert result["prompt_id"] == "h01"
    assert result["category"] == "neutral"
    assert result["target_logprob"] is None
    assert result["preference_score"] is None
    assert "top_logprobs" not in result
