from __future__ import annotations

from typing import Any

from fastapi import Depends, Header, HTTPException
from sqlmodel import Session, select

from .database import get_session
from .models import Project


def require_project_owner(
    session: Session,
    project_id: str,
    current_user: dict[str, Any],
) -> Project:
    """Return a project only when it belongs to the authenticated Clerk user."""
    project = session.exec(select(Project).where(Project.id == project_id)).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != current_user.get("sub"):
        # Do not expose legacy/unowned projects or another user's project data
        # to any signed-in user.
        raise HTTPException(status_code=403, detail="Access denied")
    return project


def get_current_user(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    Extract and verify the Clerk Bearer token from the Authorization header.
    Returns the decoded JWT payload (includes 'sub' = Clerk user ID).
    For development, allows a default test user if SKIP_AUTH_DEV is set.
    """
    import os
    import jwt as pyjwt

    from .auth import verify_clerk_token

    if not authorization or not authorization.startswith("Bearer "):
        if os.getenv("SKIP_AUTH_DEV") == "true":
            return {"sub": "dev-user"}
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


def get_project_or_404(
    project_id: str,
    session: Session = Depends(get_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> Project:
    return require_project_owner(session, project_id, current_user)
