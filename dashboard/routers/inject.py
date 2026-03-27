"""Manual Injection API — add tickers to strategies for evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from dashboard.dependencies import get_db
from edgefinder.db.models import ManualInjection, Ticker

router = APIRouter()


class InjectRequest(BaseModel):
    symbol: str
    target_strategy: str | None = None  # None = all strategies
    notes: str = ""
    expires_hours: int | None = None


@router.post("")
def inject_ticker(req: InjectRequest, db: Session = Depends(get_db)):
    """Manually inject a ticker for strategy evaluation."""
    symbol = req.symbol.upper()

    # Ensure ticker exists in master registry
    ticker = db.query(Ticker).filter_by(symbol=symbol).first()
    if not ticker:
        ticker = Ticker(symbol=symbol, source="injected", is_active=True)
        db.add(ticker)
    else:
        ticker.is_active = True

    # Create injection record
    expires = None
    if req.expires_hours:
        expires = datetime.utcnow() + timedelta(hours=req.expires_hours)

    injection = ManualInjection(
        symbol=symbol,
        target_strategy=req.target_strategy,
        expires_at=expires,
        notes=req.notes,
    )
    db.add(injection)
    db.commit()

    return {
        "symbol": symbol,
        "target_strategy": req.target_strategy or "all",
        "expires_at": expires.isoformat() if expires else None,
        "message": f"{symbol} injected for evaluation",
    }


@router.get("")
def list_injections(db: Session = Depends(get_db)):
    """Get all active manual injections."""
    injections = db.query(ManualInjection).order_by(ManualInjection.created_at.desc()).all()
    now = datetime.utcnow()
    return [
        {
            "id": inj.id,
            "symbol": inj.symbol,
            "target_strategy": inj.target_strategy,
            "expires_at": inj.expires_at.isoformat() if inj.expires_at else None,
            "is_expired": inj.expires_at < now if inj.expires_at else False,
            "notes": inj.notes,
            "created_at": inj.created_at.isoformat() if inj.created_at else None,
        }
        for inj in injections
    ]


@router.delete("/{injection_id}")
def remove_injection(injection_id: int, db: Session = Depends(get_db)):
    """Remove a manual injection."""
    inj = db.query(ManualInjection).filter_by(id=injection_id).first()
    if not inj:
        return {"error": "Injection not found"}
    db.delete(inj)
    db.commit()
    return {"message": f"Injection {injection_id} removed"}
