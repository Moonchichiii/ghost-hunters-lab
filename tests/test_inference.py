"""Tests for the shared llama.cpp HTTP client."""

import io
import json
from unittest.mock import patch

import pytest

from ghost_lab.inference import request_json


def test_request_json_returns_response() -> None:
    payload = {"content": "Hello"}

    with patch("ghost_lab.inference.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = io.BytesIO(
            json.dumps(payload).encode()
        )

        result = request_json("http://127.0.0.1:8081", "/completion")

    assert result == payload


def test_request_json_rejects_json_array() -> None:
    with patch("ghost_lab.inference.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = io.BytesIO(b"[]")

        with pytest.raises(TypeError, match="Expected a JSON object"):
            request_json("http://127.0.0.1:8081", "/completion")
