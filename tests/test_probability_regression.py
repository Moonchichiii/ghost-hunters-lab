"""Regression tests using previously observed Gemma measurements."""

import pytest

from ghost_lab.scoring import (
    lookup_preference_by_id,
    preference_score,
)

GHOST_TOKEN_ID = 25556
MAN_TOKEN_ID = 880


@pytest.mark.parametrize(
    ("baseline", "steered", "expected_change"),
    [
        (-0.2902, -0.1798, 0.1104),
        (1.0182, 1.1581, 0.1399),
        (-4.3220, -4.1096, 0.2124),
        (-5.5256, -5.4479, 0.0777),
        (-0.7928, -0.6307, 0.1621),
        (-4.5920, -4.6357, -0.0437),
    ],
)
def test_observed_ghost_preference_changes(
    baseline: float,
    steered: float,
    expected_change: float,
) -> None:
    """Preserve the arithmetic used in the six-prompt pilot."""
    assert steered - baseline == pytest.approx(
        expected_change,
        abs=0.0001,
    )


def test_observed_baseline_preference() -> None:
    """Check the first observed ghost-versus-man comparison."""
    scores = {
        GHOST_TOKEN_ID: -2.8365495204925537,
        MAN_TOKEN_ID: -2.546320676803589,
    }

    result = lookup_preference_by_id(
        scores,
        GHOST_TOKEN_ID,
        MAN_TOKEN_ID,
    )

    assert result == pytest.approx(-0.2902288437)


def test_scoring_formula_remains_unchanged() -> None:
    assert preference_score(-2.0, -3.0) == pytest.approx(1.0)
