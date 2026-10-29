"""Measure unrelated next-token effects of ghost activation steering."""

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ghost_lab.evaluate_probabilities import (
    LAYER,
    MODEL,
    MODEL_DIR,
    N_PROBS,
    RESULTS_DIR,
    SEED,
    SERVER,
    URL,
)
from ghost_lab.inference import request_json
from ghost_lab.provenance import git_revision, runtime_metadata, sha256_file
from ghost_lab.server import ServerConfig, running_server

DATASET = Path("eval/side_effect_prompts.json")
EXPECTED_HASH = "0098d695adf1532ca0c4575420d3b845c26f4e02e7f6b74f30cbce874ca0791d"
STRENGTH = 16

CONDITIONS: dict[str, tuple[Path | None, int]] = {
    "baseline": (None, 0),
    "ghost_positive": (MODEL_DIR / "ghost-23871.gguf", STRENGTH),
    "random_2026": (MODEL_DIR / "random-control-2026.gguf", STRENGTH),
}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Validate the frozen probe set before model inference."""
    if sha256_file(path) != EXPECTED_HASH:
        raise ValueError("Side-effect dataset hash mismatch")

    data = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[dict[str, Any]] = data["prompts"]

    if len(prompts) != 12:
        raise ValueError("Expected 12 side-effect prompts")

    if data["conditions"] != list(CONDITIONS):
        raise ValueError("Unexpected experimental conditions")

    ids = [item["id"] for item in prompts]
    texts = [item["text"] for item in prompts]

    if len(set(ids)) != 12 or len(set(texts)) != 12:
        raise ValueError("Duplicate prompt ID or text")

    for item in prompts:
        if type(item["target_id"]) is not int:
            raise TypeError("Expected an integer target token ID")

    return prompts


def verify_tokens(prompts: list[dict[str, Any]]) -> None:
    """Confirm exact target IDs against the running model tokenizer."""
    for text, token_id in sorted(
        {(item["target"], item["target_id"]) for item in prompts}
    ):
        response = request_json(
            URL,
            "/tokenize",
            {"content": text, "add_special": False},
        )
        if response["tokens"] != [token_id]:
            raise ValueError(f"Token mismatch for {text!r}")


def measure(prompt: dict[str, Any], condition: str) -> dict[str, Any]:
    """Measure one expected token under one model condition."""
    response = request_json(
        URL,
        "/completion",
        {
            "prompt": prompt["text"],
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

    target_logprob = scores.get(prompt["target_id"])
    if target_logprob is not None and (
        not math.isfinite(target_logprob) or target_logprob > 0
    ):
        raise ValueError("Invalid target log probability")

    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt": prompt["text"],
        "condition": condition,
        "target": prompt["target"],
        "target_id": prompt["target_id"],
        "target_logprob": target_logprob,
        "target_probability": (
            math.exp(target_logprob) if target_logprob is not None else None
        ),
        "target_in_top_n": target_logprob is not None,
        "generated": response["content"],
        "timings": response.get("timings"),
    }


def save_report(path: Path, report: dict[str, Any]) -> None:
    """Write an atomic checkpoint without corrupting existing JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    """Evaluate all three conditions using the frozen probe set."""
    prompts = load_dataset(DATASET)

    for path in (
        SERVER,
        MODEL,
        *(vector for vector, _ in CONDITIONS.values() if vector is not None),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = RESULTS_DIR / f"side_effects_{timestamp}.json"

    report: dict[str, Any] = {
        "experiment": "ghost_side_effects_v1",
        "status": "running",
        "dataset_sha256": EXPECTED_HASH,
        "model_sha256": sha256_file(MODEL),
        "llama_cpp_revision": git_revision(Path.home() / "src/llama.cpp"),
        "runtime": runtime_metadata(),
        "feature_id": 23871,
        "layer": LAYER,
        "inference": {
            "n_predict": 1,
            "n_probs": N_PROBS,
            "temperature": 0,
            "seed": SEED,
            "cache_prompt": False,
            "post_sampling_probs": False,
        },
        "conditions": {
            name: {
                "strength": strength,
                "vector": str(vector) if vector else None,
                "vector_sha256": sha256_file(vector) if vector else None,
            }
            for name, (vector, strength) in CONDITIONS.items()
        },
        "expected_measurements": len(prompts) * len(CONDITIONS),
        "records": [],
    }

    records: list[dict[str, Any]] = report["records"]
    save_report(output, report)

    try:
        for name, (vector, strength) in CONDITIONS.items():
            print(f"\n=== {name.upper()} ===", flush=True)

            config = ServerConfig(
                executable=SERVER,
                model=MODEL,
                port=8081,
                layer=LAYER,
                vector=vector,
                strength=float(strength),
            )

            log_path = RESULTS_DIR / f"server_side_{name}_{timestamp}.log"

            with running_server(config, log_path):
                if name == "baseline":
                    verify_tokens(prompts)
                    print("Tokenizer verification passed", flush=True)

                for prompt in prompts:
                    record = measure(prompt, name)
                    records.append(record)
                    save_report(output, report)

                    value = record["target_logprob"]
                    display = f"{value:.4f}" if value is not None else "MISSING"

                    print(
                        f"{len(records):2d}/36 "
                        f"{prompt['id']} "
                        f"logP({prompt['target']!r})={display}",
                        flush=True,
                    )

        if len(records) != report["expected_measurements"]:
            raise RuntimeError("Incomplete side-effect evaluation")

    except Exception as exc:
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
