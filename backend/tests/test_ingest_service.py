from __future__ import annotations

from dataclasses import replace
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


_TITLE = "Coolio - Gangsta's Paradise (Official Music Video)"


class _FakePopen:
    """Stands in for yt-dlp: writes the media file, the title side-file and progress."""

    write_title_file = True

    def __init__(self, command: list[str], **_kwargs) -> None:
        self.command = command
        output_template = Path(command[command.index("-o") + 1])
        output_path = Path(str(output_template).replace("%(ext)s", "mp4"))
        output_path.write_bytes(b"video")
        if self.write_title_file:
            Path(command[command.index("--print-to-file") + 2]).write_text(
                f"{_TITLE}\n", encoding="utf-8"
            )
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
    assert source_title == _TITLE
    assert ingest_service.source_title_to_filename(
        source_title, absolute_path
    ) == f"{_TITLE}.mp4"
    # 55% of the download maps into the 5-70 band this stage owns.
    assert any(pct == 40 and "Downloading video" in msg for pct, msg in progress_events)
    assert progress_events == sorted(progress_events, key=lambda ev: ev[0])


def test_url_download_falls_back_to_stdout_title(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Older/oddly-configured runs still emit the title sentinel on stdout."""
    monkeypatch.setattr(_FakePopen, "write_title_file", False)
    monkeypatch.setattr(storage, "upload_root", tmp_path)
    monkeypatch.setattr(ingest_service.shutil, "which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr(ingest_service.subprocess, "Popen", _FakePopen)

    _, _, source_title = ingest_service.download_video_with_ytdlp(
        "https://example.com/video", "project-1"
    )

    assert source_title == _TITLE


def test_url_download_removes_title_side_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Path] = []

    class CapturingPopen(_FakePopen):
        def __init__(self, command: list[str], **kwargs) -> None:
            seen.append(Path(command[command.index("--print-to-file") + 2]))
            super().__init__(command, **kwargs)

    monkeypatch.setattr(storage, "upload_root", tmp_path)
    monkeypatch.setattr(ingest_service.shutil, "which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr(ingest_service.subprocess, "Popen", CapturingPopen)
    ingest_service.download_video_with_ytdlp(
        "https://example.com/video", "project-1"
    )

    assert not seen[0].exists()
    # The side file must not sit next to the media, or it can be mistaken for it.
    assert seen[0].parent != tmp_path / "project-1"


def _capture_ytdlp_cmd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[str]:
    """Run download_video_with_ytdlp against a fake Popen and return the argv."""
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
    return seen[0]


def _with_settings(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    monkeypatch.setattr(
        ingest_service,
        "settings",
        replace(ingest_service.settings, **overrides),
    )


def test_ytdlp_cmd_has_no_login_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """yt-dlp removed OAuth login; --username oauth2 now hard-fails the download."""
    _with_settings(
        monkeypatch, yt_dlp_cookies_file="", yt_dlp_cookies_from_browser=""
    )
    cmd = _capture_ytdlp_cmd(monkeypatch, tmp_path)

    assert "--newline" in cmd
    assert "--username" not in cmd
    assert "--password" not in cmd
    assert "--cookies" not in cmd
    assert "--cookies-from-browser" not in cmd


def test_ytdlp_cmd_enables_js_challenge_solver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a JS runtime + EJS solver, YouTube media URLs 403."""
    _with_settings(monkeypatch, yt_dlp_remote_components="ejs:github")
    cmd = _capture_ytdlp_cmd(monkeypatch, tmp_path)

    assert cmd[cmd.index("--js-runtimes") + 1] == "node"
    assert cmd[cmd.index("--remote-components") + 1] == "ejs:github"


def test_ytdlp_cmd_omits_remote_components_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _with_settings(monkeypatch, yt_dlp_remote_components="")
    cmd = _capture_ytdlp_cmd(monkeypatch, tmp_path)

    assert "--remote-components" not in cmd


def test_ytdlp_cmd_caps_format_height(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _with_settings(monkeypatch, yt_dlp_max_height=720)
    cmd = _capture_ytdlp_cmd(monkeypatch, tmp_path)

    selector = cmd[cmd.index("-f") + 1]
    assert "height<=720" in selector
    assert "avc1" in selector


def test_ytdlp_cmd_uses_cookies_file_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    _with_settings(
        monkeypatch,
        yt_dlp_cookies_file=str(cookies),
        yt_dlp_cookies_from_browser="chrome",
    )
    cmd = _capture_ytdlp_cmd(monkeypatch, tmp_path)

    assert cmd[cmd.index("--cookies") + 1] == str(cookies)
    # The file wins over the browser when both are set.
    assert "--cookies-from-browser" not in cmd


def test_ytdlp_cmd_skips_missing_cookies_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _with_settings(
        monkeypatch,
        yt_dlp_cookies_file=str(tmp_path / "nope.txt"),
        yt_dlp_cookies_from_browser="",
    )
    cmd = _capture_ytdlp_cmd(monkeypatch, tmp_path)

    assert "--cookies" not in cmd


def test_ytdlp_cmd_uses_cookies_from_browser_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _with_settings(
        monkeypatch, yt_dlp_cookies_file="", yt_dlp_cookies_from_browser="chrome"
    )
    cmd = _capture_ytdlp_cmd(monkeypatch, tmp_path)

    assert cmd[cmd.index("--cookies-from-browser") + 1] == "chrome"


def test_ytdlp_bot_check_failure_gets_actionable_hint() -> None:
    message = ingest_service._ytdlp_failure_message(
        "ERROR: [youtube] abc: Sign in to confirm you're not a bot."
    )
    assert "YTDLP_COOKIES_FILE" in message
    assert "Sign in to confirm" in message


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
