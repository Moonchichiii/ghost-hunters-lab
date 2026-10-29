"""FastAPI application factory for The Ghost Hunter's Lab."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ghost_lab.web.routes import pages

WEB_DIR = Path(__file__).parent


def create_app() -> FastAPI:
    app = FastAPI(title="The Ghost Hunter's Lab", docs_url=None, redoc_url=None)
    app.state.templates = Jinja2Templates(directory=WEB_DIR / "templates")
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    app.include_router(pages.router)
    return app


app = create_app()
