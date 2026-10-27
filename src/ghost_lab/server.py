import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager

"""Lifecycle management for a locally running llama.cpp server."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Configuration for one reproducible model-server condition."""

    executable: Path
    model: Path
    port: int = 8081
    host: str = "127.0.0.1"
    context_size: int = 1024
    threads: int = 4
    layer: int = 22
    vector: Path | None = None
    strength: float = 0.0


def build_command(config: ServerConfig) -> list[str]:
    """Build the llama-server command for one condition."""
    command = [
        str(config.executable),
        "-m",
        str(config.model),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "-c",
        str(config.context_size),
        "-t",
        str(config.threads),
        "--parallel",
        "1",
    ]

    if config.vector is not None:
        command.extend(
            [
                "--control-vector-scaled",
                f"{config.vector}:{config.strength}",
                "--control-vector-layer-range",
                str(config.layer),
                str(config.layer),
            ]
        )

    return command


def check_port(config: ServerConfig) -> None:
    """Refuse to use an already occupied evaluation port."""
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex((config.host, config.port)) == 0:
            raise RuntimeError(f"Port {config.port} is already occupied")


def wait_until_ready(
    process: subprocess.Popen[bytes],
    config: ServerConfig,
    timeout: float = 120,
) -> None:
    """Wait for the launched model server to become ready."""
    url = f"http://{config.host}:{config.port}/health"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited during startup")

        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                status = json.load(response)

            if status.get("status") == "ok":
                return

        except urllib.error.URLError, TimeoutError, ValueError:
            pass

        time.sleep(0.5)

    raise TimeoutError("llama-server did not become ready")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    """Terminate the server process started by this application."""
    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@contextmanager
def running_server(
    config: ServerConfig,
    log_path: Path,
) -> Iterator[None]:
    """Start, verify, and always stop a local llama-server."""
    for path in (config.executable, config.model):
        if not path.is_file():
            raise FileNotFoundError(path)

    if config.vector is not None and not config.vector.is_file():
        raise FileNotFoundError(config.vector)

    check_port(config)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            build_command(config),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        try:
            wait_until_ready(process, config)
            yield
        finally:
            stop_server(process)
