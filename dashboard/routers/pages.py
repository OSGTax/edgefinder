"""Page routes — the trading desk (home) and the symbol chart page.

Greenfield rebuild: the old portfolio/strategies/screener/trades/picks/lab/ops
pages were removed in the cutover. ``/`` redirects to the desk.
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/")
async def home():
    """The trading desk is the home page now."""
    return RedirectResponse(url="/desk", status_code=307)


@router.get("/desk", response_class=HTMLResponse)
async def desk_page(request: Request):
    return templates.TemplateResponse(request=request, name="desk.html")


@router.get("/symbol", response_class=HTMLResponse)
async def symbol_page(request: Request):
    return templates.TemplateResponse(request=request, name="symbol.html")


@router.get("/symbol/{symbol}", response_class=HTMLResponse)
async def symbol_page_sym(request: Request, symbol: str):
    return templates.TemplateResponse(request=request, name="symbol.html")
