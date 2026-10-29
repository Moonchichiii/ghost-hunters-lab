"""Create reproducible GGUF control vectors for steering experiments."""

import json
from pathlib import Path

import gguf
import numpy as np
from numpy.typing import NDArray

MODEL_DIR = Path.home() / "models" / "ghost-hunters-lab"
SOURCE = MODEL_DIR / "ghost-23871.f32"

LAYER = 22
DIMENSIONS = 2560
RANDOM_SEEDS = (2026, 2027, 2028, 2029, 2030)

type Vector = NDArray[np.float32]


def load_ghost_direction(path: Path) -> Vector:
    """Load and validate the extracted SAE direction."""
    vector = np.fromfile(path, dtype="<f4")

    if vector.shape != (DIMENSIONS,):
        raise ValueError(f"Expected {DIMENSIONS} float32 values")

    if not np.all(np.isfinite(vector)):
        raise ValueError("Ghost direction contains non-finite values")

    if np.linalg.norm(vector) <= 0:
        raise ValueError("Ghost direction has zero norm")

    return vector


def make_random_direction(seed: int, reference_norm: float) -> Vector:
    """Create a seeded random direction matching the reference norm."""
    if not np.isfinite(reference_norm) or reference_norm <= 0:
        raise ValueError("Reference norm must be finite and positive")

    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(DIMENSIONS).astype(np.float32)

    vector *= np.float32(reference_norm / np.linalg.norm(vector))
    return vector


def write_control_vector(path: Path, vector: Vector) -> None:
    """Write a single layer-specific float32 direction to GGUF."""
    if vector.shape != (DIMENSIONS,):
        raise ValueError("Invalid control-vector dimensions")

    if not np.all(np.isfinite(vector)):
        raise ValueError("Control vector contains non-finite values")

    writer = gguf.GGUFWriter(str(path), "controlvector")

    writer.add_tensor(f"direction.{LAYER}", vector)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    """Generate five independent controls without overwriting originals."""
    ghost = load_ghost_direction(SOURCE)
    reference_norm = float(np.linalg.norm(ghost))

    records: list[dict[str, int | float | str]] = []

    for seed in RANDOM_SEEDS:
        vector = make_random_direction(seed, reference_norm)
        path = MODEL_DIR / f"random-control-{seed}.gguf"

        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

        write_control_vector(path, vector)

        records.append(
            {
                "seed": seed,
                "dimensions": DIMENSIONS,
                "norm": float(np.linalg.norm(vector)),
                "path": str(path),
            }
        )

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
