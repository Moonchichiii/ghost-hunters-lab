"""Record reproducibility metadata for local model experiments."""

import hashlib
import platform
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 hash without loading it into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    """Return the exact Git commit used for an external dependency."""
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def runtime_metadata() -> dict[str, str]:
    """Identify the Python runtime and operating system."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
