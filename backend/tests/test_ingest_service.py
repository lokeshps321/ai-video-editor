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


def test_download_video_from_url_prefers_apify_when_token_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_PROVIDER", "auto")
    monkeypatch.setenv("APIFY_API_TOKEN", "test-token")
    seen: list[str] = []

    def fake_apify(
        url: str, project_id: str, progress_callback=None
    ) -> tuple[str, str, str]:
        del progress_callback
        seen.append("apify")
        return ("/tmp/a.mp4", f"{project_id}/a.mp4", "Apify Title")

    def fake_ytdlp(
        url: str, project_id: str, progress_callback=None
    ) -> tuple[str, str, str]:
        del progress_callback
        seen.append("ytdlp")
        return ("/tmp/b.mp4", f"{project_id}/b.mp4", "Yt Title")

    monkeypatch.setattr(ingest_service, "download_video_with_apify", fake_apify)
    monkeypatch.setattr(ingest_service, "download_video_with_ytdlp", fake_ytdlp)

    result = ingest_service.download_video_from_url(
        "https://www.youtube.com/watch?v=abc", "project-1"
    )
    assert seen == ["apify"]
    assert result[2] == "Apify Title"


def test_download_video_from_url_falls_back_to_ytdlp_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_PROVIDER", "auto")
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    seen: list[str] = []

    def fake_apify(
        url: str, project_id: str, progress_callback=None
    ) -> tuple[str, str, str]:
        del progress_callback
        seen.append("apify")
        raise AssertionError("apify should not run without token")

    def fake_ytdlp(
        url: str, project_id: str, progress_callback=None
    ) -> tuple[str, str, str]:
        del progress_callback
        seen.append("ytdlp")
        return ("/tmp/b.mp4", f"{project_id}/b.mp4", "Yt Title")

    monkeypatch.setattr(ingest_service, "download_video_with_apify", fake_apify)
    monkeypatch.setattr(ingest_service, "download_video_with_ytdlp", fake_ytdlp)

    result = ingest_service.download_video_from_url(
        "https://www.youtube.com/watch?v=abc", "project-1"
    )
    assert seen == ["ytdlp"]
    assert result[2] == "Yt Title"


def test_extract_apify_download_url_from_nested_output() -> None:
    item = {
        "videoId": "BB49x_uMlGA",
        "status": "succeeded",
        "output": {
            "url": "https://api.apify.com/v2/key-value-stores/xxx/records/BB49x_uMlGA.mp4"
        },
    }
    assert ingest_service._extract_apify_download_url(item).endswith(".mp4")


def test_extract_apify_download_url_from_streamers_field() -> None:
    item = {
        "downloadedFileUrl": "https://api.apify.com/v2/key-value-stores/xxx/records/video.mp4"
    }
    assert ingest_service._extract_apify_download_url(item).endswith("video.mp4")


def test_apify_streamers_payload_uses_kv_store() -> None:
    payload = ingest_service._apify_actor_payload(
        "https://www.youtube.com/watch?v=abc",
        "streamers~youtube-video-downloader",
    )
    assert payload["storeInKVStore"] is True
    assert payload["videos"][0]["url"].endswith("abc")
