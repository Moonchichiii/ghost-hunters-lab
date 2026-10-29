from fastapi.testclient import TestClient

from ghost_lab.web.app import create_app


def test_home_renders_headline() -> None:
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "change one thought" in response.text
