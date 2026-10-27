"""Tests for model-server configuration."""

import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_lab.server import (
    ServerConfig,
    build_command,
    check_port,
    stop_server,
    wait_until_ready,
)


def test_default_server_configuration() -> None:
    config = ServerConfig(
        executable=Path("/tmp/llama-server"),
        model=Path("/tmp/model.gguf"),
    )

    assert config.port == 8081
    assert config.host == "127.0.0.1"
    assert config.layer == 22
    assert config.vector is None
    assert config.strength == 0.0


def test_server_configuration_is_immutable() -> None:
    config = ServerConfig(
        executable=Path("/tmp/llama-server"),
        model=Path("/tmp/model.gguf"),
    )

    with pytest.raises(FrozenInstanceError):
        config.port = 8082  # type: ignore[misc]


def test_baseline_command_has_no_control_vector() -> None:
    config = ServerConfig(
        executable=Path("/tmp/llama-server"),
        model=Path("/tmp/model.gguf"),
    )

    command = build_command(config)

    assert "--control-vector-scaled" not in command
    assert "--control-vector-layer-range" not in command
    assert "--port" in command


def test_ghost_command_contains_intervention() -> None:
    config = ServerConfig(
        executable=Path("/tmp/llama-server"),
        model=Path("/tmp/model.gguf"),
        vector=Path("/tmp/ghost.gguf"),
        strength=16.0,
        layer=22,
    )

    command = build_command(config)

    assert "--control-vector-scaled" in command
    assert "/tmp/ghost.gguf:16.0" in command
    assert command[-3:] == ["--control-vector-layer-range", "22", "22"]


def test_stop_server_terminates_running_process() -> None:
    process = MagicMock()
    process.poll.return_value = None

    stop_server(process)

    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=10)


def test_stop_server_kills_unresponsive_process() -> None:
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="llama-server", timeout=10),
        0,
    ]

    stop_server(process)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()


def test_stop_server_ignores_exited_process() -> None:
    process = MagicMock()
    process.poll.return_value = 0

    stop_server(process)

    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_wait_until_ready_detects_exited_process() -> None:
    process = MagicMock()
    process.poll.return_value = 1

    config = ServerConfig(
        executable=Path("/tmp/llama-server"),
        model=Path("/tmp/model.gguf"),
    )

    with pytest.raises(RuntimeError, match="exited during startup"):
        wait_until_ready(process, config)


def test_check_port_rejects_occupied_port() -> None:
    config = ServerConfig(
        executable=Path("/tmp/llama-server"),
        model=Path("/tmp/model.gguf"),
    )

    with patch("ghost_lab.server.socket.socket") as mock_socket:
        mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0

        with pytest.raises(RuntimeError, match="already occupied"):
            check_port(config)
