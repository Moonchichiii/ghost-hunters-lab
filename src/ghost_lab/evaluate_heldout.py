"""Evaluate frozen prompts against ghost and matched random directions."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ghost_lab.evaluate_probabilities import (
    COMPARATOR_ID,
    LAYER,
    MODEL,
    MODEL_DIR,
    N_PROBS,
    RESULTS_DIR,
    SEED,
    SERVER,
    TARGET_ID,
    measure,
)
from ghost_lab.provenance import git_revision, runtime_metadata, sha256_file
from ghost_lab.server import ServerConfig, running_server

DATASET = Path("eval/heldout_prompts.json")
EXPECTED_HASH = "3fdc774a6274f77512694bb983ea30b7445179a4e31617e88fd5e7d448e2de73"
STRENGTH = 16
SEEDS = (2026, 2027, 2028, 2029, 2030)


def load_dataset(path: Path) -> list[dict[str, str]]:
    """Load the exact, committed held-out dataset."""
    if sha256_file(path) != EXPECTED_HASH:
        raise ValueError("Held-out dataset hash mismatch")

    data = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[dict[str, str]] = data["prompts"]

    if data["target_token_id"] != TARGET_ID:
        raise ValueError("Target token ID mismatch")

    if data["comparator_token_id"] != COMPARATOR_ID:
        raise ValueError("Comparator token ID mismatch")

    ids = [item["id"] for item in prompts]
    texts = [item["text"] for item in prompts]

    if len(prompts) != 20:
        raise ValueError("Expected exactly 20 prompts")

    if len(set(ids)) != 20 or len(set(texts)) != 20:
        raise ValueError("Prompts and IDs must be unique")

    return prompts


def conditions() -> dict[str, tuple[Path | None, int]]:
    """Define baseline, SAE intervention and five matched controls."""
    result: dict[str, tuple[Path | None, int]] = {
        "baseline": (None, 0),
        "ghost_positive": (MODEL_DIR / "ghost-23871.gguf", STRENGTH),
    }

    for seed in SEEDS:
        result[f"random_{seed}"] = (
            MODEL_DIR / f"random-control-{seed}.gguf",
            STRENGTH,
        )

    return result


def compact_record(
    prompt: dict[str, str],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    """Keep observed scores without the large top-1000 token list."""
    record = {key: value for key, value in measurement.items() if key != "top_logprobs"}

    record["prompt_id"] = prompt["id"]
    record["category"] = prompt["category"]
    return record


def save_report(path: Path, report: dict[str, Any]) -> None:
    """Checkpoint via atomic replacement, avoiding partial JSON writes."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    """Run seven isolated conditions over the frozen prompt set."""
    prompts = load_dataset(DATASET)
    variants = conditions()

    for path in (
        SERVER,
        MODEL,
        *(vector for vector, _ in variants.values() if vector is not None),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = RESULTS_DIR / f"heldout_{timestamp}.json"

    report: dict[str, Any] = {
        "experiment": "ghost_feature_heldout_v1",
        "status": "running",
        "timestamp": timestamp,
        "dataset_sha256": EXPECTED_HASH,
        "model_sha256": sha256_file(MODEL),
        "llama_cpp_revision": git_revision(Path.home() / "src/llama.cpp"),
        "runtime": runtime_metadata(),
        "feature_id": 23871,
        "layer": LAYER,
        "inference": {
            "temperature": 0,
            "seed": SEED,
            "n_predict": 1,
            "n_probs": N_PROBS,
            "cache_prompt": False,
            "post_sampling_probs": False,
        },
        "target_token_id": TARGET_ID,
        "comparator_token_id": COMPARATOR_ID,
        "conditions": {
            name: {
                "vector": str(vector) if vector else None,
                "strength": strength,
                "vector_sha256": (sha256_file(vector) if vector else None),
            }
            for name, (vector, strength) in variants.items()
        },
        "expected_measurements": len(prompts) * len(variants),
        "records": [],
    }

    records: list[dict[str, Any]] = report["records"]
    save_report(output, report)

    try:
        for name, (vector, strength) in variants.items():
            print(f"\n=== {name.upper()} ===", flush=True)

            config = ServerConfig(
                executable=SERVER,
                model=MODEL,
                port=8081,
                layer=LAYER,
                vector=vector,
                strength=float(strength),
            )

            log_path = RESULTS_DIR / f"server_{name}_{timestamp}.log"

            with running_server(config, log_path):
                for prompt in prompts:
                    measurement = measure(prompt["text"], name, strength)
                    record = compact_record(prompt, measurement)
                    records.append(record)

                    save_report(output, report)

                    target = record["target_logprob"]
                    display = f"{target:.4f}" if target is not None else "MISSING"

                    print(
                        f"{len(records):3d}/140 {prompt['id']} logP(ghost)={display}",
                        flush=True,
                    )

        if len(records) != report["expected_measurements"]:
            raise RuntimeError("Incomplete evaluation")

    except BaseException as exc:
        report["status"] = "interrupted"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_report(output, report)
        raise

    report["status"] = "completed"
    report["completed_at"] = datetime.now(UTC).isoformat()
    save_report(output, report)

    print(f"\nCompleted {len(records)} measurements.")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
