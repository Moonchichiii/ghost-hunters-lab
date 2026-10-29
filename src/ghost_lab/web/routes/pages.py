"""Page routes for The Ghost Hunter's Lab."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Render the editorial homepage."""
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(request, "index.html")
