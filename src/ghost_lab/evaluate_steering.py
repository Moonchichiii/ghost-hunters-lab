"""Evaluate Gemma with baseline, ghost, and random control vectors."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ghost_lab.inference import request_json
from ghost_lab.server import ServerConfig, running_server

MODEL_DIR = Path.home() / "models" / "ghost-hunters-lab"
SERVER = Path.home() / "src/llama.cpp/build/bin/llama-server"
MODEL = MODEL_DIR / "google_gemma-3-4b-it-Q4_K_M.gguf"

URL = "http://127.0.0.1:8081"
RESULTS_DIR = Path("eval/results")

PROMPTS = [
    "Describe an ordinary library on a rainy afternoon.",
    "Describe a peaceful garden in the morning.",
]

CONDITIONS = {
    "baseline": None,
    "ghost": MODEL_DIR / "ghost-23871.gguf",
    "random": MODEL_DIR / "random-control.gguf",
}

STRENGTH = 16
SEED = 2026
LAYER = 22


def evaluate(prompt: str) -> dict[str, Any]:
    """Generate one response and return measurable output."""
    result = request_json(
        URL,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": 100,
            "temperature": 0,
            "seed": SEED,
            "cache_prompt": False,
        },
    )

    content = result.get("content")
    if not isinstance(content, str):
        raise TypeError("Model response content must be a string")

    timings = result.get("timings", {})

    return {
        "prompt": prompt,
        "response": content,
        "tokens": result.get("tokens_predicted"),
        "tokens_per_second": timings.get("predicted_per_second"),
        "stop_type": result.get("stop_type"),
        "truncated": result.get("truncated"),
    }


def save_results(records: list[dict[str, Any]]) -> None:
    """Preserve experimental evidence after every completed run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": MODEL.name,
        "feature_id": 23871,
        "layer": LAYER,
        "strength": STRENGTH,
        "seed": SEED,
        "temperature": 0,
        "records": records,
    }

    path = RESULTS_DIR / "possession_smoke.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run identical prompts under three isolated model configurations."""
    for path in (
        SERVER,
        MODEL,
        *(vector for vector in CONDITIONS.values() if vector is not None),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for condition, vector in CONDITIONS.items():
        print(f"\n=== {condition.upper()} ===", flush=True)

        log_path = RESULTS_DIR / f"server_{condition}.log"

        config = ServerConfig(
            executable=SERVER,
            model=MODEL,
            port=8081,
            layer=LAYER,
            vector=vector,
            strength=0.0 if vector is None else float(STRENGTH),
        )

        with running_server(config, log_path):
            for prompt in PROMPTS:
                print(f"Running: {prompt}", flush=True)

                measurement = evaluate(prompt)
                measurement["condition"] = condition
                measurement["strength"] = 0 if vector is None else STRENGTH

                records.append(measurement)
                save_results(records)

                print(measurement["response"].strip(), flush=True)

    print(f"\nCompleted {len(records)} evaluations.")
    print("Saved: eval/results/possession_smoke.json")


if __name__ == "__main__":
    main()
