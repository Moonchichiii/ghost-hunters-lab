"""Unit tests for paired held-out evaluation analysis."""

import pytest

from ghost_lab.analyze_heldout import (
    paired_deltas,
    summarize,
    validate_preference_scores,
)


def test_paired_deltas_use_matching_prompt() -> None:
    records = [
        {
            "prompt_id": "p01",
            "condition": "baseline",
            "target_logprob": -3.0,
        },
        {
            "prompt_id": "p02",
            "condition": "baseline",
            "target_logprob": -4.0,
        },
        {
            "prompt_id": "p02",
            "condition": "ghost_positive",
            "target_logprob": -3.5,
        },
        {
            "prompt_id": "p01",
            "condition": "ghost_positive",
            "target_logprob": -2.8,
        },
    ]

    result = paired_deltas(records, "target_logprob")

    assert result["ghost_positive"]["p01"] == pytest.approx(0.2)
    assert result["ghost_positive"]["p02"] == pytest.approx(0.5)


def test_missing_scores_are_not_fabricated() -> None:
    records = [
        {
            "prompt_id": "p01",
            "condition": "baseline",
            "target_logprob": -3.0,
        },
        {
            "prompt_id": "p01",
            "condition": "ghost_positive",
            "target_logprob": None,
        },
    ]

    assert paired_deltas(records, "target_logprob") == {}


def test_summary_statistics() -> None:
    result = summarize([0.1, -0.2, 0.3, 0.0])

    assert result["n"] == 4
    assert result["mean"] == pytest.approx(0.05)
    assert result["positive"] == 2
    assert result["negative"] == 1
    assert result["zero"] == 1


def test_empty_summary_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize([])


def test_preference_score_matches_recorded_logprobs() -> None:
    records = [
        {
            "target_logprob": -2.4,
            "comparator_logprob": -2.7,
            "preference_score": 0.3,
        }
    ]

    validate_preference_scores(records)


def test_incorrect_preference_score_is_rejected() -> None:
    records = [
        {
            "target_logprob": -2.4,
            "comparator_logprob": -2.7,
            "preference_score": -0.3,
        }
    ]

    with pytest.raises(ValueError, match="does not match"):
        validate_preference_scores(records)
