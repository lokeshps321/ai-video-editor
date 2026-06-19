import os

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from fastapi.testclient import TestClient

from app.main import app


def _create_project(client: TestClient, name: str = "Project Route Test") -> str:
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "fps": 30, "width": 1080, "height": 1920},
    )
    assert response.status_code == 200
    return response.json()["id"]


def test_list_projects_returns_created_project() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Listed Project")

    response = client.get("/api/v1/projects")
    assert response.status_code == 200
    projects = response.json()
    assert any(item["id"] == project_id and item["name"] == "Listed Project" for item in projects)


def test_get_project_returns_single_project() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Single Project")

    response = client.get(f"/api/v1/projects/{project_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == project_id
    assert payload["name"] == "Single Project"
    assert payload["timeline"]["tracks"]


def test_rename_project_updates_name() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Before Rename")

    response = client.patch(
        f"/api/v1/projects/{project_id}",
        json={"name": "After Rename"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "After Rename"

    get_response = client.get(f"/api/v1/projects/{project_id}")
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "After Rename"


def test_rename_project_rejects_empty_name() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Named Project")

    response = client.patch(
        f"/api/v1/projects/{project_id}",
        json={"name": "   "},
    )
    assert response.status_code == 400


def test_delete_project_removes_project() -> None:
    client = TestClient(app)
    project_id = _create_project(client, "Delete Me")

    response = client.delete(f"/api/v1/projects/{project_id}")
    assert response.status_code == 200
    assert response.json()["detail"] == "Project deleted"

    get_response = client.get(f"/api/v1/projects/{project_id}")
    assert get_response.status_code == 404
