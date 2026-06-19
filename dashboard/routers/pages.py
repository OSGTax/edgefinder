"""Page routes — serves rendered HTML templates for each dashboard page."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


@router.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request):
    return templates.TemplateResponse(request=request, name="strategies.html")


@router.get("/screener", response_class=HTMLResponse)
async def screener_page(request: Request):
    return templates.TemplateResponse(request=request, name="screener.html")


@router.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request):
    return templates.TemplateResponse(request=request, name="trades.html")


@router.get("/symbol", response_class=HTMLResponse)
async def symbol_page(request: Request):
    return templates.TemplateResponse(request=request, name="symbol.html")


@router.get("/symbol/{symbol}", response_class=HTMLResponse)
async def symbol_page_sym(request: Request, symbol: str):
    return templates.TemplateResponse(request=request, name="symbol.html")


@router.get("/research")
async def research_redirect(request: Request):
    """Old Research page -> Symbol Workstation (deep links preserved)."""
    ticker = request.query_params.get("ticker")
    return RedirectResponse(
        url=f"/symbol/{ticker.upper()}" if ticker else "/symbol",
        status_code=307)


@router.get("/ops", response_class=HTMLResponse)
async def ops_page(request: Request):
    return templates.TemplateResponse(request=request, name="ops.html")


@router.get("/lab", response_class=HTMLResponse)
async def lab_page(request: Request):
    return templates.TemplateResponse(request=request, name="lab.html")


@router.get("/picks", response_class=HTMLResponse)
async def picks_page(request: Request):
    return templates.TemplateResponse(request=request, name="picks.html")


@router.get("/backtest")
async def backtest_redirect():
    """Old Backtest page -> Lab explorer (quick-backtest retired)."""
    return RedirectResponse(url="/lab", status_code=307)
