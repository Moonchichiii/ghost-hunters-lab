"""Analyze held-out steering results against matched random controls."""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

RANDOM_PREFIX = "random_"
GHOST = "ghost_positive"
BASELINE = "baseline"


def load_report(path: Path) -> dict[str, Any]:
    """Load completed evaluation evidence without modifying it."""
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    if report.get("status") != "completed":
        raise ValueError("Evaluation is not completed")

    records = report["records"]
    expected = report["expected_measurements"]

    if len(records) != expected:
        raise ValueError("Unexpected measurement count")

    pairs = [(item["condition"], item["prompt_id"]) for item in records]

    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate condition/prompt pairs")

    return report


def validate_preference_scores(records: list[dict[str, Any]]) -> None:
    """Verify preference scores against their recorded log probabilities."""
    for record in records:
        target = record["target_logprob"]
        comparator = record["comparator_logprob"]
        preference = record["preference_score"]

        if target is None or comparator is None:
            if preference is not None:
                raise ValueError("Preference exists despite a missing token score")
            continue

        expected = float(target) - float(comparator)

        if not all(
            math.isfinite(value)
            for value in (float(target), float(comparator), expected)
        ):
            raise ValueError("Non-finite token log probability")

        if preference is None or not math.isfinite(float(preference)):
            raise ValueError("Missing or non-finite preference score")

        if not math.isclose(
            float(preference),
            expected,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise ValueError("Preference score does not match log probabilities")


def paired_deltas(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float]]:
    """Compare each intervention with its matching prompt baseline."""
    baseline = {
        item["prompt_id"]: item[field]
        for item in records
        if item["condition"] == BASELINE
    }

    changes: dict[str, dict[str, float]] = defaultdict(dict)

    for item in records:
        condition = item["condition"]

        if condition == BASELINE:
            continue

        prompt_id = item["prompt_id"]
        value = item[field]
        reference = baseline.get(prompt_id)

        if value is None or reference is None:
            continue

        value = float(value)
        reference = float(reference)

        if not math.isfinite(value) or not math.isfinite(reference):
            raise ValueError("Non-finite measurement")

        changes[condition][prompt_id] = value - reference

    return dict(changes)


def summarize(values: list[float]) -> dict[str, float | int]:
    """Summarize observed paired differences."""
    if not values:
        raise ValueError("Cannot summarize empty measurements")

    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "positive": sum(value > 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
    }


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    """Analyze primary and secondary metrics for the frozen experiment."""
    records = report["records"]

    validate_preference_scores(records)

    changes = paired_deltas(records, "target_logprob")
    preference_changes = paired_deltas(records, "preference_score")

    if GHOST not in changes:
        raise ValueError("Missing ghost intervention")

    controls = sorted(
        condition for condition in changes if condition.startswith(RANDOM_PREFIX)
    )

    if len(controls) != 5:
        raise ValueError("Expected five random controls")

    ghost_values = changes[GHOST]
    all_prompt_ids = set(ghost_values)

    if any(set(changes[control]) != all_prompt_ids for control in controls):
        raise ValueError("Incomplete paired control measurements")

    if len(all_prompt_ids) != 20:
        raise ValueError("Expected 20 paired prompts")

    summaries = {
        condition: summarize(list(values.values()))
        for condition, values in changes.items()
    }

    preference_summaries = {
        condition: summarize(list(values.values()))
        for condition, values in preference_changes.items()
    }

    if set(preference_summaries) != set(summaries):
        raise ValueError("Incomplete preference measurements")

    if any(
        set(preference_changes[name]) != all_prompt_ids for name in preference_summaries
    ):
        raise ValueError("Incomplete paired preference measurements")

    random_means = [float(summaries[name]["mean"]) for name in controls]

    category_by_prompt = {
        item["prompt_id"]: item["category"]
        for item in records
        if item["condition"] == BASELINE
    }

    if set(category_by_prompt) != all_prompt_ids:
        raise ValueError("Incomplete baseline category information")

    categories = sorted(set(category_by_prompt.values()))

    category_summaries = {
        category: summarize(
            [
                delta
                for prompt_id, delta in ghost_values.items()
                if category_by_prompt[prompt_id] == category
            ]
        )
        for category in categories
    }

    outperform_all = sum(
        ghost_delta > max(changes[control][prompt_id] for control in controls)
        for prompt_id, ghost_delta in ghost_values.items()
    )

    preference_ghost = preference_changes[GHOST]

    preference_outperform_all = sum(
        ghost_delta
        > max(preference_changes[control][prompt_id] for control in controls)
        for prompt_id, ghost_delta in preference_ghost.items()
    )

    preference_random_means = [
        float(preference_summaries[name]["mean"]) for name in controls
    ]

    return {
        "experiment": report["experiment"],
        "dataset_sha256": report["dataset_sha256"],
        "primary_measure": "delta_log_probability_of_exact_ghost_token",
        "secondary_measure": "delta_ghost_vs_man_log_preference",
        "summaries": summaries,
        "preference_summaries": preference_summaries,
        "ghost_mean_exceeds_random_means": sum(
            float(summaries[GHOST]["mean"]) > random_mean
            for random_mean in random_means
        ),
        "ghost_exceeds_all_randoms_per_prompt": outperform_all,
        "preference_ghost_mean_exceeds_random_means": sum(
            float(preference_summaries[GHOST]["mean"]) > random_mean
            for random_mean in preference_random_means
        ),
        "preference_ghost_exceeds_all_randoms_per_prompt": (preference_outperform_all),
        "category_summaries": category_summaries,
        "limitations": [
            "Twenty frozen prompts, not a representative language sample",
            "Only five sampled norm-matched random directions",
            "Exact next-token metric, not concept-level ghost understanding",
            "Ghost-versus-man preference is a two-token comparison",
            "No general-capability side-effect evaluation",
        ],
    }


def main() -> None:
    """Analyze the original full-precision held-out evidence."""
    path = Path("eval/results/heldout_20261029T100108969558Z.json")

    report = load_report(path)
    analysis = analyze(report)

    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
