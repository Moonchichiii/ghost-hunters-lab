"""Unit tests for interpreting llama.cpp evaluation responses."""

import math
from unittest.mock import patch

import pytest

from ghost_lab.evaluate_probabilities import measure
from ghost_lab.evaluate_steering import evaluate


def test_probability_evaluator_uses_token_ids() -> None:
    """Check exact token lookup and preference calculations."""
    response = {
        "content": " man",
        "tokens_evaluated": 6,
        "timings": {"predicted_per_second": 12.0},
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"id": 880, "token": " man", "logprob": -2.5463206768},
                    {"id": 25556, "token": " ghost", "logprob": -2.8365495205},
                ]
            }
        ],
    }

    with patch(
        "ghost_lab.evaluate_probabilities.request_json",
        return_value=response,
    ) as mock_request:
        result = measure(
            "The mysterious figure was a",
            condition="baseline",
            strength=0,
        )

    assert result["target_token_id"] == 25556
    assert result["comparator_token_id"] == 880
    assert result["preference_score"] == pytest.approx(-0.2902288437)
    assert result["target_probability"] == pytest.approx(math.exp(-2.8365495205))

    mock_request.assert_called_once()
    assert mock_request.call_args.args[:2] == (
        "http://127.0.0.1:8081",
        "/completion",
    )


def test_probability_evaluator_handles_missing_target() -> None:
    """A missing top-N token must not be treated as zero probability."""
    response = {
        "content": " large",
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"id": 880, "token": " man", "logprob": -2.5},
                ]
            }
        ],
    }

    with patch(
        "ghost_lab.evaluate_probabilities.request_json",
        return_value=response,
    ):
        result = measure(
            "The figure visible through the window was a",
            condition="baseline",
            strength=0,
        )

    assert result["target_logprob"] is None
    assert result["target_probability"] is None
    assert result["preference_score"] is None
    assert result["target_in_top_n"] is False


def test_text_evaluator_extracts_timings() -> None:
    """Text evaluation must preserve output and real inference speed."""
    response = {
        "content": "The library was quiet.",
        "tokens_predicted": 10,
        "timings": {"predicted_per_second": 12.25},
        "stop_type": "eos",
        "truncated": False,
    }

    with patch(
        "ghost_lab.evaluate_steering.request_json",
        return_value=response,
    ):
        result = evaluate("Describe a library.")

    assert result["response"] == "The library was quiet."
    assert result["tokens"] == 10
    assert result["tokens_per_second"] == pytest.approx(12.25)
    assert result["stop_type"] == "eos"


def test_text_evaluator_rejects_invalid_content() -> None:
    """Invalid HTTP data must not silently become an experiment result."""
    with (
        patch(
            "ghost_lab.evaluate_steering.request_json",
            return_value={"content": None, "timings": {}},
        ),
        pytest.raises(TypeError, match="content must be a string"),
    ):
        evaluate("Describe a library.")
