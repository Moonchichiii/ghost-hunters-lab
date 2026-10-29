from fastapi.testclient import TestClient

from ghost_lab.web.app import create_app


def test_home_renders_headline() -> None:
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "change one thought" in response.text


def test_stylesheet_is_linked_and_served_locally() -> None:
    client = TestClient(create_app())
    page = client.get("/")
    assert page.status_code == 200
    assert "/static/css/site.css" in page.text
    stylesheet = client.get("/static/css/site.css")
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")


def test_favicon_is_linked_and_served_locally() -> None:
    client = TestClient(create_app())
    page = client.get("/")
    assert "/static/favicon.svg" in page.text
    icon = client.get("/static/favicon.svg")
    assert icon.status_code == 200
    assert icon.headers["content-type"].startswith("image/svg+xml")
