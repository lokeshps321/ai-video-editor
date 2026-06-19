from __future__ import annotations

from typing import Any

from fastapi import Depends, Header, HTTPException
from sqlmodel import Session, select

from .database import get_session
from .models import Project


def get_project_or_404(
    project_id: str, session: Session = Depends(get_session)
) -> Project:
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def get_current_user(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    Extract and verify the Clerk Bearer token from the Authorization header.
    Returns the decoded JWT payload (includes 'sub' = Clerk user ID).
    """
    import jwt as pyjwt

    from .auth import verify_clerk_token

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = verify_clerk_token(token)
        return payload
    except pyjwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Auth error: {exc}") from exc
