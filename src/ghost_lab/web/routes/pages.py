"""Render the editorial investigation using recorded scientific evidence."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

EVIDENCE_FILE = Path(__file__).resolve().parents[4] / "eval" / "summary.json"


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Render the homepage with verified, read-only experiment results."""
    evidence: dict[str, Any] = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))

    if evidence.get("schema_version") != 1:
        raise ValueError("Unsupported evidence schema")

    templates: Jinja2Templates = request.app.state.templates

    return templates.TemplateResponse(
        request,
        "index.html",
        {"evidence": evidence},
    )
