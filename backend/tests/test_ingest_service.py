from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

pytest.importorskip("sqlmodel")

from app import ingest_service
from app.storage import storage


def test_url_download_returns_platform_title_without_using_it_as_storage_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage, "upload_root", tmp_path)
    monkeypatch.setattr(ingest_service.shutil, "which", lambda _: "/usr/bin/tool")

    def fake_run(command: list[str], **_kwargs) -> CompletedProcess[str]:
        output_template = Path(command[command.index("-o") + 1])
        output_path = Path(str(output_template).replace("%(ext)s", "mp4"))
        output_path.write_bytes(b"video")
        return CompletedProcess(
            command,
            0,
            stdout=(
                "__CLIPMIND_SOURCE_TITLE__Coolio - Gangsta's Paradise "
                "(Official Music Video)\n"
            ),
        )

    monkeypatch.setattr(ingest_service.subprocess, "run", fake_run)

    absolute_path, relative_path, source_title = (
        ingest_service.download_video_with_ytdlp(
            "https://example.com/video", "project-1"
        )
    )

    assert Path(absolute_path).name.endswith(".mp4")
    assert relative_path == f"project-1/{Path(absolute_path).name}"
    assert source_title == "Coolio - Gangsta's Paradise (Official Music Video)"
    assert ingest_service.source_title_to_filename(
        source_title, absolute_path
    ) == "Coolio - Gangsta's Paradise (Official Music Video).mp4"
