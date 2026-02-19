from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from .config import get_settings

settings = get_settings()


class LocalStorage:
    def __init__(self, upload_dir: str, render_dir: str, tmp_dir: str) -> None:
        self.upload_root = Path(upload_dir).resolve()
        self.render_root = Path(render_dir).resolve()
        self.tmp_root = Path(tmp_dir).resolve()
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.render_root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile, project_id: str) -> tuple[str, str]:
        project_dir = self.upload_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename or "").suffix.lower()
        safe_name = f"{uuid4()}{suffix}"
        destination = project_dir / safe_name

        with destination.open("wb") as stream:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        await file.close()
        return str(destination), str(destination.relative_to(self.upload_root))

    def output_path(self, project_id: str, ext: str) -> str:
        out_dir = self.render_root / project_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"{uuid4()}.{ext}")

    def to_public_upload_path(self, relative_path: str) -> str:
        return f"/static/uploads/{relative_path}"

    def to_public_render_path(self, absolute_path: str) -> str:
        rel = Path(absolute_path).resolve().relative_to(self.render_root)
        return f"/static/renders/{rel.as_posix()}"

    def resolve_upload_asset(self, storage_path: str) -> str:
        path = Path(storage_path)
        if path.is_absolute():
            return str(path)
        return str(self.upload_root / storage_path)


storage = LocalStorage(
    upload_dir=os.getenv("UPLOAD_DIR", settings.upload_dir),
    render_dir=os.getenv("RENDER_DIR", settings.render_dir),
    tmp_dir=os.getenv("TMP_DIR", settings.tmp_dir),
)

