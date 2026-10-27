"""Pure functions for evaluating token preference."""

import math
from collections.abc import Mapping


def preference_score(
    target_logprob: float,
    comparator_logprob: float,
) -> float:
    """Calculate the log-probability preference for one token over another."""
    if not math.isfinite(target_logprob):
        raise ValueError("Target log probability must be finite")

    if not math.isfinite(comparator_logprob):
        raise ValueError("Comparator log probability must be finite")

    return target_logprob - comparator_logprob


def probability_from_logprob(logprob: float) -> float:
    """Convert a log probability into an ordinary probability."""
    if not math.isfinite(logprob) or logprob > 0:
        raise ValueError("Expected a finite log probability <= 0")

    return math.exp(logprob)


def lookup_preference(
    scores: Mapping[str, float],
    target: str,
    comparator: str,
) -> float | None:
    """Return a preference score, or None if either token is missing."""
    target_lp = scores.get(target)
    comparator_lp = scores.get(comparator)

    if target_lp is None or comparator_lp is None:
        return None

    return preference_score(target_lp, comparator_lp)


def lookup_preference_by_id(
    scores: Mapping[int, float],
    target_id: int,
    comparator_id: int,
) -> float | None:
    """Calculate preference using exact vocabulary token IDs."""
    target_lp = scores.get(target_id)
    comparator_lp = scores.get(comparator_id)

    if target_lp is None or comparator_lp is None:
        return None

    return preference_score(target_lp, comparator_lp)
