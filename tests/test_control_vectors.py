"""Tests for deterministic steering-control generation."""

import numpy as np
import pytest

from ghost_lab.make_control_vector import (
    DIMENSIONS,
    make_random_direction,
)


def test_random_direction_is_reproducible() -> None:
    first = make_random_direction(2026, 1.0)
    second = make_random_direction(2026, 1.0)

    np.testing.assert_array_equal(first, second)


def test_random_directions_differ_between_seeds() -> None:
    first = make_random_direction(2026, 1.0)
    second = make_random_direction(2027, 1.0)

    assert not np.array_equal(first, second)


def test_random_direction_matches_reference_norm() -> None:
    vector = make_random_direction(2026, 1.0)

    assert vector.shape == (DIMENSIONS,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("norm", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_reference_norm_is_rejected(norm: float) -> None:
    with pytest.raises(ValueError, match="Reference norm"):
        make_random_direction(2026, norm)
