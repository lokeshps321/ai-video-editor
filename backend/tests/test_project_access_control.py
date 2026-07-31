from __future__ import annotations

from fastapi.testclient import TestClient

from app.deps import get_current_user
from app.main import app


def _as_user(user_id: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: {"sub": user_id}


def test_projects_and_edits_are_isolated_to_the_clerk_owner() -> None:
    with TestClient(app) as client:
        _as_user("user-alice")
        created = client.post(
            "/api/v1/projects",
            json={"name": "Alice private edit", "fps": 30, "width": 1080, "height": 1920},
        )
        assert created.status_code == 200
        project_id = created.json()["id"]

        _as_user("user-bob")
        assert client.get(f"/api/v1/projects/{project_id}").status_code == 403
        assert client.get(f"/api/v1/media?project_id={project_id}").status_code == 403
        assert (
            client.post(
                f"/api/v1/timeline/operations?project_id={project_id}",
                json={"operations": []},
            ).status_code
            == 403
        )
        projects = client.get("/api/v1/projects")
        assert projects.status_code == 200
        assert all(project["id"] != project_id for project in projects.json())

        _as_user("user-alice")
        reopened = client.get(f"/api/v1/projects/{project_id}")
        assert reopened.status_code == 200
        assert reopened.json()["id"] == project_id
