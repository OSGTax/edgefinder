"""Page routes — serves rendered HTML templates for each dashboard page."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
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


@router.get("/research", response_class=HTMLResponse)
async def research_page(request: Request):
    return templates.TemplateResponse(request=request, name="research.html")
