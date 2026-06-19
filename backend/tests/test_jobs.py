from __future__ import annotations

import pytest

from app import jobs
from app.schemas import ExportSettings


def test_enqueue_render_job_local_mode_uses_memory_safe_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.jobs._use_rq_workers", lambda: False)
    captured: dict[str, object] = {}

    class _FakeThread:
        def __init__(self, *, target, args, name: str, daemon: bool) -> None:
            captured["target"] = target
            captured["args"] = args
            captured["name"] = name
            captured["daemon"] = daemon

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr("app.jobs.threading.Thread", _FakeThread)

    export_settings = ExportSettings(
        format="mp4",
        aspect_ratio="16:9",
        resolution="720p",
        fps=30,
        quality="low",
    )
    jobs.enqueue_render_job("job-12345678", export_settings)

    assert captured["target"] == jobs._run_render_job_with_local_limit
    assert captured["args"] == ("job-12345678", export_settings)
    assert captured["started"] is True


def test_run_render_job_with_local_limit_releases_slot_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSemaphore:
        def __init__(self) -> None:
            self.acquire_calls = 0
            self.release_calls = 0

        def acquire(self) -> None:
            self.acquire_calls += 1

        def release(self) -> None:
            self.release_calls += 1

    fake_semaphore = _FakeSemaphore()
    monkeypatch.setattr("app.jobs._LOCAL_RENDER_JOB_SLOTS", fake_semaphore)
    monkeypatch.setattr("app.jobs.process_render_job", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    export_settings = ExportSettings(
        format="mp4",
        aspect_ratio="16:9",
        resolution="720p",
        fps=30,
        quality="low",
    )
    with pytest.raises(RuntimeError):
        jobs._run_render_job_with_local_limit("job-fail", export_settings)

    assert fake_semaphore.acquire_calls == 1
    assert fake_semaphore.release_calls == 1
