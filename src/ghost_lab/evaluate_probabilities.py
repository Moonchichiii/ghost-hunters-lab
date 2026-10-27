"""Measure ghost-token preference under controlled model interventions."""

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ghost_lab.inference import request_json
from ghost_lab.provenance import git_revision, runtime_metadata, sha256_file
from ghost_lab.scoring import lookup_preference_by_id
from ghost_lab.server import ServerConfig, running_server

MODEL_DIR = Path.home() / "models/ghost-hunters-lab"
SERVER = Path.home() / "src/llama.cpp/build/bin/llama-server"
MODEL = MODEL_DIR / "google_gemma-3-4b-it-Q4_K_M.gguf"

RESULTS_DIR = Path("eval/results")
URL = "http://127.0.0.1:8081"

SEED = 2026
LAYER = 22
N_PROBS = 1000

PROMPTS = [
    "The mysterious figure was a",
    "The shape in the abandoned hallway was a",
    "The figure standing beside the grave was a",
    "The person waiting outside the station was a",
    "The visitor who knocked at midnight was a",
    "The figure visible through the window was a",
]

CONDITIONS = {
    "baseline": (None, 0),
    "ghost_positive": (MODEL_DIR / "ghost-23871.gguf", 16),
    "ghost_negative": (MODEL_DIR / "ghost-23871.gguf", -16),
    "random_positive": (MODEL_DIR / "random-control.gguf", 16),
}

TARGET_ID = 25556
COMPARATOR_ID = 880

TARGET = " ghost"
COMPARATOR = " man"


def measure(
    prompt: str,
    condition: str,
    strength: int,
) -> dict[str, Any]:
    """Measure token probabilities for one prompt and intervention."""
    response = request_json(
        URL,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": N_PROBS,
            "temperature": 0,
            "seed": SEED,
            "cache_prompt": False,
            "post_sampling_probs": False,
        },
    )

    probabilities = response.get("completion_probabilities", [])
    if not probabilities:
        raise ValueError("Server returned no token probabilities")

    candidates = probabilities[0]["top_logprobs"]

    scores = {int(item["id"]): float(item["logprob"]) for item in candidates}

    target_lp = scores.get(TARGET_ID)
    comparator_lp = scores.get(COMPARATOR_ID)

    preference = lookup_preference_by_id(
        scores,
        TARGET_ID,
        COMPARATOR_ID,
    )

    return {
        "prompt": prompt,
        "condition": condition,
        "strength": strength,
        "generated": response["content"],
        "target_token_id": TARGET_ID,
        "comparator_token_id": COMPARATOR_ID,
        "target_logprob": target_lp,
        "comparator_logprob": comparator_lp,
        "target_probability": (math.exp(target_lp) if target_lp is not None else None),
        "preference_score": preference,
        "target_in_top_n": target_lp is not None,
        "comparator_in_top_n": comparator_lp is not None,
        "top_logprobs": candidates,
        "tokens_evaluated": response.get("tokens_evaluated"),
        "timings": response.get("timings"),
    }


def main() -> None:
    """Run the same prompts under each isolated model condition."""
    for path in (
        SERVER,
        MODEL,
        *(vector for vector, _ in CONDITIONS.values() if vector is not None),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Identify the exact model, vectors, runtime, and prompt set.
    # Compute these fingerprints once per experiment.
    llama_repository = Path.home() / "src/llama.cpp"

    provenance = {
        "model_sha256": sha256_file(MODEL),
        "llama_cpp_revision": git_revision(llama_repository),
        "runtime": runtime_metadata(),
        "vector_sha256": {
            name: sha256_file(vector)
            for name, (vector, _) in CONDITIONS.items()
            if vector is not None
        },
        # Hash a compact, deterministic JSON serialization of the prompt list.
        "prompt_sha256": hashlib.sha256(
            json.dumps(
                PROMPTS,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = RESULTS_DIR / f"probabilities_{timestamp}.json"

    report: dict[str, Any] = {
        "experiment": "feature_specificity_pilot",
        "timestamp": timestamp,
        "provenance": provenance,
        "model": MODEL.name,
        "feature_id": 23871,
        "layer": LAYER,
        "seed": SEED,
        "temperature": 0,
        "n_probs": N_PROBS,
        "target": TARGET,
        "target_id": TARGET_ID,
        "comparator": COMPARATOR,
        "comparator_id": COMPARATOR_ID,
        "conditions": {
            name: {
                "vector": str(vector) if vector is not None else None,
                "strength": strength,
            }
            for name, (vector, strength) in CONDITIONS.items()
        },
        "records": [],
    }

    records: list[dict[str, Any]] = report["records"]

    for condition, (vector, strength) in CONDITIONS.items():
        print(f"\n=== {condition.upper()} ===", flush=True)

        log_path = RESULTS_DIR / f"server_{condition}_{timestamp}.log"

        config = ServerConfig(
            executable=SERVER,
            model=MODEL,
            port=8081,
            layer=LAYER,
            vector=vector,
            strength=float(strength),
        )

        with running_server(config, log_path):
            for prompt in PROMPTS:
                record = measure(prompt, condition, strength)
                records.append(record)

                # Preserve completed measurements after every run.
                output.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

                score = record["preference_score"]
                display = f"{score:+.4f}" if score is not None else "N/A"

                print(f"{prompt} -> {display}", flush=True)

    print(f"\nCompleted {len(records)} measurements.")
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
