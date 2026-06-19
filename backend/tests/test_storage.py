from pathlib import Path

from app.storage import LocalStorage


def test_local_storage_resolves_relative_roots_from_backend_dir() -> None:
    storage = LocalStorage("uploads", "renders", "tmp")
    backend_root = Path(__file__).resolve().parents[1]

    assert storage.upload_root == backend_root / "uploads"
    assert storage.render_root == backend_root / "renders"
    assert storage.tmp_root == backend_root / "tmp"
