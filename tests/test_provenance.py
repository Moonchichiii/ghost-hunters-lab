"""Tests for experiment reproducibility metadata."""

import hashlib
from pathlib import Path
from unittest.mock import patch

from ghost_lab.provenance import git_revision, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "example.bin"
    data = b"ghost-hunters-lab"
    path.write_bytes(data)

    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_empty_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


def test_git_revision() -> None:
    with patch("ghost_lab.provenance.subprocess.run") as mock_run:
        mock_run.return_value.stdout = "fae9ac561123456789\n"

        assert git_revision(Path("/tmp/llama.cpp")) == "fae9ac561123456789"
