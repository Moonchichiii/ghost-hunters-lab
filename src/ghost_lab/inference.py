"""Minimal HTTP client for local llama.cpp inference."""

import json
import urllib.request
from typing import Any

SERVER_URL = "http://127.0.0.1:8081/completion"


def request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 180,
) -> dict[str, Any]:
    """Send a JSON request and require a JSON-object response."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}

    request = urllib.request.Request(
        base_url.rstrip("/") + "/" + path.lstrip("/"),
        data=data,
        headers=headers,
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        result: Any = json.load(response)

    if not isinstance(result, dict):
        raise TypeError("Expected a JSON object from llama-server")

    return result


def generate(prompt: str) -> dict[str, Any]:
    """Generate text and return structured performance measurements."""

    payload = {
        "prompt": prompt,
        "n_predict": 80,
        "temperature": 0,
        "seed": 2026,
        "cache_prompt": False,
    }

    request = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)

    if not isinstance(result.get("content"), str):
        raise TypeError("Model response content must be a string")

    timings = result.get("timings", {})

    return {
        "response": result["content"],
        "tokens_predicted": result.get("tokens_predicted"),
        "tokens_evaluated": result.get("tokens_evaluated"),
        "tokens_per_second": timings.get("predicted_per_second"),
        "generation_ms": timings.get("predicted_ms"),
        "stop_type": result.get("stop_type"),
        "truncated": result.get("truncated"),
    }


if __name__ == "__main__":
    result = generate("Describe a quiet library on a rainy afternoon.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
