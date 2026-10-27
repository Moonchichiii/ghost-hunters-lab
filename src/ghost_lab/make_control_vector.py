"""Create GGUF control vectors for the possession experiment."""

import json
from pathlib import Path

import gguf
import numpy as np

MODEL_DIR = Path.home() / "models" / "ghost-hunters-lab"
SOURCE = MODEL_DIR / "ghost-23871.f32"

LAYER = 22
DIMENSIONS = 2560
RANDOM_SEED = 2026


def write_control_vector(path: Path, vector: np.ndarray) -> None:
    """Save one layer-specific direction in GGUF format."""
    writer = gguf.GGUFWriter(str(path), "controlvector")

    writer.add_tensor(f"direction.{LAYER}", vector)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    ghost = np.fromfile(SOURCE, dtype="<f4")

    if ghost.shape != (DIMENSIONS,):
        raise ValueError(f"Expected {DIMENSIONS} values")

    if not np.all(np.isfinite(ghost)):
        raise ValueError("Ghost vector contains invalid values")

    norm = float(np.linalg.norm(ghost))
    if not norm > 0:
        raise ValueError("Ghost direction has zero norm")

    rng = np.random.default_rng(RANDOM_SEED)
    random = rng.standard_normal(DIMENSIONS).astype(np.float32)
    random *= np.float32(norm / np.linalg.norm(random))

    ghost_path = MODEL_DIR / "ghost-23871.gguf"
    random_path = MODEL_DIR / "random-control.gguf"

    write_control_vector(ghost_path, ghost)
    write_control_vector(random_path, random)

    print(
        json.dumps(
            {
                "layer": LAYER,
                "dimensions": DIMENSIONS,
                "seed": RANDOM_SEED,
                "ghost_norm": norm,
                "random_norm": float(np.linalg.norm(random)),
                "ghost_file": str(ghost_path),
                "random_file": str(random_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
