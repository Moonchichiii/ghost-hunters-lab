"""Export verified, compact evidence for the lab website and article."""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

from ghost_lab.analyze_heldout import (
    analyze,
    load_report,
    paired_deltas,
    summarize,
)
from ghost_lab.provenance import sha256_file

HELDOUT_HASH = "a1c7fce2d532e444236930adf0d2c738bea85dc476eec1f57cf506c56fe28910"
SIDE_EFFECT_HASH = "dd9f08c28b3a06a975aec5a1f48a72a9606e02ae58fb17574110004e01e8b6ec"

HELDOUT_DATASET_HASH = (
    "3fdc774a6274f77512694bb983ea30b7445179a4e31617e88fd5e7d448e2de73"
)
SIDE_EFFECT_DATASET_HASH = (
    "0098d695adf1532ca0c4575420d3b845c26f4e02e7f6b74f30cbce874ca0791d"
)

DEFAULT_HELDOUT = Path("eval/results/heldout_20261029T100108969558Z.json")
DEFAULT_SIDE_EFFECTS = Path("eval/results/side_effects_20261029T105547813287Z.json")
DEFAULT_OUTPUT = Path("eval/summary.json")


def verified_report(
    path: Path,
    expected_hash: str,
    dataset_hash: str,
) -> dict[str, Any]:
    """Reject altered, incomplete, or misidentified evidence."""
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Evidence SHA-256 mismatch: {path}")

    report = load_report(path)

    if report["dataset_sha256"] != dataset_hash:
        raise ValueError(f"Dataset SHA-256 mismatch: {path}")

    return report


def validate_side_effects(report: dict[str, Any]) -> None:
    """Verify the exact frozen side-effect experiment structure."""
    if report["experiment"] != "ghost_side_effects_v1":
        raise ValueError("Unexpected side-effect experiment")

    conditions = {"baseline", "ghost_positive", "random_2026"}
    records = report["records"]

    if report["expected_measurements"] != 36:
        raise ValueError("Expected 36 side-effect measurements")

    if set(report["conditions"]) != conditions:
        raise ValueError("Unexpected side-effect conditions")

    for record in records:
        value = record["target_logprob"]

        if value is None:
            raise ValueError("Missing side-effect target score")

        if not math.isfinite(float(value)) or float(value) > 0:
            raise ValueError("Invalid side-effect target log probability")

    ids = {
        record["prompt_id"] for record in records if record["condition"] == "baseline"
    }

    if len(ids) != 12:
        raise ValueError("Expected 12 baseline prompts")

    expected_pairs = {
        (condition, prompt_id) for condition in conditions for prompt_id in ids
    }
    actual_pairs = {(record["condition"], record["prompt_id"]) for record in records}

    if actual_pairs != expected_pairs:
        raise ValueError("Incomplete side-effect condition coverage")


def side_effect_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Report the magnitude and direction of unrelated token changes."""
    validate_side_effects(report)

    records = report["records"]
    changes = paired_deltas(records, "target_logprob")

    categories = {
        record["prompt_id"]: record["category"]
        for record in records
        if record["condition"] == "baseline"
    }

    condition_summaries: dict[str, Any] = {}

    for condition in ("ghost_positive", "random_2026"):
        values = changes[condition]

        condition_summaries[condition] = {
            **summarize(list(values.values())),
            "mean_absolute_change": mean(abs(value) for value in values.values()),
            "category_mean_changes": {
                category: mean(
                    value
                    for prompt_id, value in values.items()
                    if categories[prompt_id] == category
                )
                for category in sorted(set(categories.values()))
            },
        }

    per_prompt = [
        {
            "prompt_id": prompt_id,
            "category": categories[prompt_id],
            "ghost_delta": changes["ghost_positive"][prompt_id],
            "random_delta": changes["random_2026"][prompt_id],
        }
        for prompt_id in sorted(categories)
    ]

    return {
        "measurement": "change_in_expected_exact_token_log_probability",
        "condition_summaries": condition_summaries,
        "per_prompt": per_prompt,
        "interpretation_limit": (
            "These are exact-token side-effect probes, "
            "not measurements of general model accuracy."
        ),
    }


def heldout_prompt_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Include all prompts so the UI can show successes and failures."""
    records = report["records"]
    primary = paired_deltas(records, "target_logprob")
    secondary = paired_deltas(records, "preference_score")

    baselines = {
        record["prompt_id"]: record
        for record in records
        if record["condition"] == "baseline"
    }

    interventions = sorted(primary)

    return [
        {
            "prompt_id": prompt_id,
            "category": baseline["category"],
            "prompt": baseline["prompt"],
            "primary_deltas": {
                condition: primary[condition][prompt_id] for condition in interventions
            },
            "preference_deltas": {
                condition: secondary[condition][prompt_id]
                for condition in interventions
            },
        }
        for prompt_id, baseline in sorted(baselines.items())
    ]


def build_summary(
    heldout: dict[str, Any],
    side_effects: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic publication artifact."""
    if heldout["experiment"] != "ghost_feature_heldout_v1":
        raise ValueError("Unexpected held-out experiment")

    heldout_analysis = analyze(heldout)

    if len(heldout["records"]) != 140:
        raise ValueError("Expected 140 held-out measurements")

    if len(heldout["conditions"]) != 7:
        raise ValueError("Expected seven held-out conditions")

    if heldout["model_sha256"] != side_effects["model_sha256"]:
        raise ValueError("Experiments used different model files")

    if heldout["llama_cpp_revision"] != side_effects["llama_cpp_revision"]:
        raise ValueError("Experiments used different llama.cpp revisions")

    for condition in ("ghost_positive", "random_2026"):
        if (
            heldout["conditions"][condition]["vector_sha256"]
            != side_effects["conditions"][condition]["vector_sha256"]
        ):
            raise ValueError(f"Different steering vector: {condition}")

    return {
        "schema_version": 1,
        "title": "The Ghost Hunter's Lab — Experimental Evidence",
        "model": "Gemma 3 4B IT Q4_K_M",
        "feature_id": 23871,
        "layer": 22,
        "strength": 16,
        "provenance": {
            "model_sha256": heldout["model_sha256"],
            "llama_cpp_revision": heldout["llama_cpp_revision"],
            "source_files": {
                "heldout_sha256": HELDOUT_HASH,
                "side_effects_sha256": SIDE_EFFECT_HASH,
            },
            "dataset_sha256": {
                "heldout": HELDOUT_DATASET_HASH,
                "side_effects": SIDE_EFFECT_DATASET_HASH,
            },
            "heldout_vector_sha256": {
                name: config["vector_sha256"]
                for name, config in heldout["conditions"].items()
            },
        },
        "heldout": {
            "observations": 140,
            "prompts": 20,
            "analysis": heldout_analysis,
            "per_prompt": heldout_prompt_results(heldout),
        },
        "side_effects": {
            "observations": 36,
            "prompts": 12,
            **side_effect_summary(side_effects),
        },
        "scientific_status": {
            "activation_steering": "measured",
            "weight_level_exorcism": "not_attempted",
            "general_capability_preservation": "not_established",
            "selective_concept_control": "not_established",
        },
    }


def export(
    heldout_path: Path,
    side_effect_path: Path,
    output_path: Path,
) -> None:
    """Verify evidence and export a stable JSON snapshot."""
    heldout = verified_report(heldout_path, HELDOUT_HASH, HELDOUT_DATASET_HASH)
    side_effects = verified_report(
        side_effect_path, SIDE_EFFECT_HASH, SIDE_EFFECT_DATASET_HASH
    )

    summary = build_summary(heldout, side_effects)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)


def main() -> None:
    """Export the two recorded experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--side-effects", type=Path, default=DEFAULT_SIDE_EFFECTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()

    export(args.heldout, args.side_effects, args.output)
    print(f"Evidence exported: {args.output}")


if __name__ == "__main__":
    main()
