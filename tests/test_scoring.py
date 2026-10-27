"""Tests for probability and preference calculations."""

import math

import pytest

from ghost_lab.scoring import (
    lookup_preference,
    preference_score,
    probability_from_logprob,
)


def test_preference_score() -> None:
    result = preference_score(-2.8365, -2.5463)

    assert result == pytest.approx(-0.2902)


def test_positive_preference() -> None:
    result = preference_score(-1.0, -3.0)

    assert result == pytest.approx(2.0)


def test_probability_from_logprob() -> None:
    result = probability_from_logprob(math.log(0.25))

    assert result == pytest.approx(0.25)


def test_missing_token_returns_none() -> None:
    scores = {" ghost": -2.0}

    result = lookup_preference(scores, " ghost", " man")

    assert result is None


def test_present_tokens_return_preference() -> None:
    scores = {" ghost": -2.0, " man": -3.0}

    result = lookup_preference(scores, " ghost", " man")

    assert result == pytest.approx(1.0)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_logprob_is_rejected(invalid: float) -> None:
    with pytest.raises(ValueError):
        preference_score(invalid, -2.0)


from ghost_lab.scoring import lookup_preference_by_id


def test_token_id_preference() -> None:
    scores = {25556: -2.0, 880: -3.0}

    result = lookup_preference_by_id(scores, 25556, 880)

    assert result == pytest.approx(1.0)


def test_missing_token_id_returns_none() -> None:
    scores = {880: -2.0}

    result = lookup_preference_by_id(scores, 25556, 880)

    assert result is None
