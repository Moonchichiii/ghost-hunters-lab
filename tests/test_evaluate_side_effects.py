"""Tests for the frozen side-effect evaluation."""

from pathlib import Path
from unittest.mock import patch

import pytest

from ghost_lab.evaluate_side_effects import (
    CONDITIONS,
    EXPECTED_HASH,
    load_dataset,
    measure,
    verify_tokens,
)
from ghost_lab.provenance import sha256_file


def test_frozen_side_effect_dataset() -> None:
    assert sha256_file(Path("eval/side_effect_prompts.json")) == EXPECTED_HASH
    prompts = load_dataset(Path("eval/side_effect_prompts.json"))

    assert len(prompts) == 12
    assert len(CONDITIONS) == 3


def test_modified_dataset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "changed.json"
    path.write_text('{"prompts": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset(path)


def test_tokenizer_mismatch_is_rejected() -> None:
    prompts = [{"target": " Paris", "target_id": 9079}]

    with (
        patch(
            "ghost_lab.evaluate_side_effects.request_json",
            return_value={"tokens": [9999]},
        ),
        pytest.raises(ValueError, match="Token mismatch"),
    ):
        verify_tokens(prompts)


def test_missing_target_remains_missing() -> None:
    prompt = {
        "id": "s01",
        "category": "geography",
        "text": "The capital city of France is",
        "target": " Paris",
        "target_id": 9079,
    }

    response = {
        "content": " London",
        "completion_probabilities": [{"top_logprobs": [{"id": 100, "logprob": -0.2}]}],
    }

    with patch(
        "ghost_lab.evaluate_side_effects.request_json",
        return_value=response,
    ):
        result = measure(prompt, "baseline")

    assert result["target_logprob"] is None
    assert result["target_probability"] is None
    assert result["target_in_top_n"] is False
