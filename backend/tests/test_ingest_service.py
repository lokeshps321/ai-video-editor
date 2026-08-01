from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

from app import ingest_service
from app.storage import storage


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class _FakePopen:
    def __init__(self, command: list[str], **_kwargs) -> None:
        self.command = command
        output_template = Path(command[command.index("-o") + 1])
        output_path = Path(str(output_template).replace("%(ext)s", "mp4"))
        output_path.write_bytes(b"video")
        self.stdout = _FakeStdout(
            [
                "[download]  12.5% of 10.00MiB at 1.00MiB/s ETA 00:08\n",
                "[download]  55.0% of 10.00MiB at 1.00MiB/s ETA 00:04\n",
                "[download] 100% of 10.00MiB in 00:00:10\n",
                "__CLIPMIND_SOURCE_TITLE__Coolio - Gangsta's Paradise "
                "(Official Music Video)\n",
            ]
        )
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self._returncode

    def kill(self) -> None:
        self._returncode = -9


def test_url_download_returns_platform_title_without_using_it_as_storage_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage, "upload_root", tmp_path)
    monkeypatch.setattr(ingest_service.shutil, "which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr(ingest_service.subprocess, "Popen", _FakePopen)

    progress_events: list[tuple[int, str]] = []

    absolute_path, relative_path, source_title = (
        ingest_service.download_video_with_ytdlp(
            "https://example.com/video",
            "project-1",
            progress_callback=lambda pct, msg: progress_events.append((pct, msg)),
        )
    )

    assert Path(absolute_path).name.endswith(".mp4")
    assert relative_path == f"project-1/{Path(absolute_path).name}"
    assert source_title == "Coolio - Gangsta's Paradise (Official Music Video)"
    assert ingest_service.source_title_to_filename(
        source_title, absolute_path
    ) == "Coolio - Gangsta's Paradise (Official Music Video).mp4"
    assert any(pct == 55 and "Downloading video" in msg for pct, msg in progress_events)


def test_ytdlp_cmd_includes_oauth2_and_newline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage, "upload_root", tmp_path)
    monkeypatch.setattr(ingest_service.shutil, "which", lambda _: "/usr/bin/tool")
    seen: list[list[str]] = []

    class CapturingPopen(_FakePopen):
        def __init__(self, command: list[str], **kwargs) -> None:
            seen.append(command)
            super().__init__(command, **kwargs)

    monkeypatch.setattr(ingest_service.subprocess, "Popen", CapturingPopen)
    ingest_service.download_video_with_ytdlp(
        "https://example.com/video", "project-1"
    )
    cmd = seen[0]
    assert "--newline" in cmd
    assert cmd[cmd.index("--username") + 1] == "oauth2"
    assert cmd[cmd.index("--password") + 1] == ""


def test_download_video_from_url_prefers_ytdlp_in_auto(
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
    assert seen == ["ytdlp"]
    assert result[2] == "Yt Title"


def test_download_video_from_url_falls_back_to_apify_when_ytdlp_fails(
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
        raise RuntimeError("bot check")

    monkeypatch.setattr(ingest_service, "download_video_with_apify", fake_apify)
    monkeypatch.setattr(ingest_service, "download_video_with_ytdlp", fake_ytdlp)

    result = ingest_service.download_video_from_url(
        "https://www.youtube.com/watch?v=abc", "project-1"
    )
    assert seen == ["ytdlp", "apify"]
    assert result[2] == "Apify Title"


def test_download_video_from_url_uses_apify_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_PROVIDER", "apify")
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
