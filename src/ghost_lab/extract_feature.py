"""Extract one SAE decoder direction without loading the full checkpoint."""

import hashlib
import json
import math
import struct
from pathlib import Path

FEATURE_ID = 23871
MODEL_DIR = Path.home() / "models" / "ghost-hunters-lab"
CHECKPOINT = MODEL_DIR / "sae" / "params.safetensors"

OUTPUT = MODEL_DIR / "ghost-23871.f32"
METADATA = MODEL_DIR / "ghost-23871.json"


def extract_feature() -> None:
    """Read and validate one row of the SAE decoder matrix."""

    with CHECKPOINT.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))

        tensor = header["w_dec"]
        rows, dimensions = tensor["shape"]

        if tensor["dtype"] != "F32":
            raise ValueError("Expected float32 decoder weights")

        if not 0 <= FEATURE_ID < rows:
            raise ValueError("Feature index is out of range")

        row_size = dimensions * 4
        data_start = 8 + header_size + tensor["data_offsets"][0]

        file.seek(data_start + FEATURE_ID * row_size)
        raw = file.read(row_size)

    if len(raw) != row_size:
        raise ValueError("Incomplete feature vector")

    values = struct.unpack(f"<{dimensions}f", raw)

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Feature contains invalid values")

    norm = math.sqrt(sum(value * value for value in values))

    if norm == 0:
        raise ValueError("Feature vector has zero magnitude")

    OUTPUT.write_bytes(raw)

    metadata = {
        "model": "google/gemma-3-4b-it",
        "layer": 22,
        "sae": "layer_22_width_65k_l0_medium",
        "feature_id": FEATURE_ID,
        "dimensions": dimensions,
        "norm": norm,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }

    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2))
    print(f"Saved vector: {OUTPUT}")
    print(f"Vector size: {OUTPUT.stat().st_size} bytes")


if __name__ == "__main__":
    extract_feature()
