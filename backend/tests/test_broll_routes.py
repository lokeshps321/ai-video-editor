import os
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from fastapi.testclient import TestClient

from app.broll_external_service import ExternalBrollCandidate
from app.main import app
from app.storage import storage
from app.transcription_service import TranscriptPayload, TranscriptWordPayload


def _create_project(client: TestClient, name: str = "Broll Test") -> str:
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "fps": 30, "width": 1080, "height": 1920},
    )
    assert response.status_code == 200
    return response.json()["id"]


def _upload_video(client: TestClient, project_id: str, filename: str) -> str:
    response = client.post(
        "/api/v1/media/upload",
        data={"project_id": project_id},
        files={"file": (filename, b"fake-video-bytes", "video/mp4")},
    )
    assert response.status_code == 200
    return response.json()["id"]


def _fake_transcript(_path: str, _duration_sec: float) -> TranscriptPayload:
    words = [
        TranscriptWordPayload(id="w1", text="building", start_sec=0.0, end_sec=0.2),
        TranscriptWordPayload(id="w2", text="a", start_sec=0.2, end_sec=0.35),
        TranscriptWordPayload(id="w3", text="great", start_sec=0.35, end_sec=0.55),
        TranscriptWordPayload(id="w4", text="product.", start_sec=0.55, end_sec=0.9),
        TranscriptWordPayload(id="w5", text="this", start_sec=1.0, end_sec=1.2),
        TranscriptWordPayload(id="w6", text="needs", start_sec=1.2, end_sec=1.4),
        TranscriptWordPayload(id="w7", text="strong", start_sec=1.4, end_sec=1.6),
        TranscriptWordPayload(id="w8", text="visuals.", start_sec=1.6, end_sec=1.9),
        TranscriptWordPayload(id="w9", text="keep", start_sec=2.0, end_sec=2.2),
        TranscriptWordPayload(id="w10", text="editing", start_sec=2.2, end_sec=2.4),
        TranscriptWordPayload(id="w11", text="fast", start_sec=2.4, end_sec=2.6),
        TranscriptWordPayload(id="w12", text="always.", start_sec=2.6, end_sec=2.9),
    ]
    return TranscriptPayload(
        source="test_provider",
        language="en",
        text=" ".join(word.text for word in words),
        words=words,
        is_mock=False,
    )


def _overlay_clip_count(timeline: dict[str, object]) -> int:
    tracks = timeline.get("tracks")
    if not isinstance(tracks, list):
        return 0
    for track in tracks:
        if not isinstance(track, dict):
            continue
        if track.get("kind") != "overlay":
            continue
        clips = track.get("clips")
        if isinstance(clips, list):
            return len(clips)
    return 0


def test_broll_suggest_choose_reject_and_transcript_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 10.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.routers.transcript.generate_transcript", _fake_transcript)

    with TestClient(app) as client:
        project_id = _create_project(client)
        transcript_asset_id = _upload_video(client, project_id, "talking-head.mp4")
        _upload_video(client, project_id, "factory-broll.mp4")

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": transcript_asset_id},
        )
        assert generate_res.status_code == 200
        transcript_id = generate_res.json()["transcript"]["id"]

        suggest_res = client.post(
            f"/api/v1/broll/suggest?project_id={project_id}",
            json={"max_slots": 3, "candidates_per_slot": 2, "include_external_sources": False, "ai_rerank": False},
        )
        assert suggest_res.status_code == 200
        suggest_payload = suggest_res.json()
        assert suggest_payload["transcript_id"] == transcript_id
        assert suggest_payload["created_slots"] >= 1
        assert len(suggest_payload["slots"]) == suggest_payload["created_slots"]

        first_slot = suggest_payload["slots"][0]
        assert len(first_slot["candidates"]) == 2
        assert isinstance(first_slot["candidates"][0]["confidence"], float)
        assert isinstance(first_slot["candidates"][0]["score_breakdown"], dict)
        assert isinstance(first_slot["candidates"][0]["entities"], list)
        first_candidate_id = first_slot["candidates"][0]["id"]

        list_res = client.get(f"/api/v1/broll/slots?project_id={project_id}")
        assert list_res.status_code == 200
        listed_slots = list_res.json()
        assert len(listed_slots) == suggest_payload["created_slots"]

        choose_res = client.post(
            f"/api/v1/broll/slots/{first_slot['id']}/choose?project_id={project_id}",
            json={"candidate_id": first_candidate_id},
        )
        assert choose_res.status_code == 200
        chosen_slot = choose_res.json()
        assert chosen_slot["status"] == "chosen"
        assert chosen_slot["chosen_candidate_id"] == first_candidate_id

        reject_res = client.post(
            f"/api/v1/broll/slots/{first_slot['id']}/reject?project_id={project_id}",
            json={"reason": "not relevant"},
        )
        assert reject_res.status_code == 200
        rejected_slot = reject_res.json()
        assert rejected_slot["status"] == "rejected"
        assert rejected_slot["chosen_candidate_id"] is None

        transcript_res = client.get(f"/api/v1/transcript?project_id={project_id}&transcript_id={transcript_id}")
        assert transcript_res.status_code == 200
        assert len(transcript_res.json()["words"]) == 12


def test_broll_suggest_replace_existing_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 10.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.routers.transcript.generate_transcript", _fake_transcript)

    with TestClient(app) as client:
        project_id = _create_project(client, "Broll Replace Existing")
        transcript_asset_id = _upload_video(client, project_id, "source.mp4")
        _upload_video(client, project_id, "cutaway.mp4")

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": transcript_asset_id},
        )
        assert generate_res.status_code == 200

        first_suggest = client.post(
            f"/api/v1/broll/suggest?project_id={project_id}",
            json={
                "max_slots": 2,
                "candidates_per_slot": 2,
                "replace_existing": True,
                "include_external_sources": False,
                "ai_rerank": False,
            },
        )
        assert first_suggest.status_code == 200
        first_count = first_suggest.json()["created_slots"]
        assert first_count == 2

        second_suggest = client.post(
            f"/api/v1/broll/suggest?project_id={project_id}",
            json={
                "max_slots": 1,
                "candidates_per_slot": 1,
                "replace_existing": False,
                "include_external_sources": False,
                "ai_rerank": False,
            },
        )
        assert second_suggest.status_code == 200
        assert second_suggest.json()["created_slots"] == 1

        list_res = client.get(f"/api/v1/broll/slots?project_id={project_id}")
        assert list_res.status_code == 200
        assert len(list_res.json()) == 3


def test_broll_suggest_requires_transcript() -> None:
    with TestClient(app) as client:
        project_id = _create_project(client, "Broll Missing Transcript")
        _upload_video(client, project_id, "source.mp4")

        suggest_res = client.post(
            f"/api/v1/broll/suggest?project_id={project_id}",
            json={"max_slots": 2, "candidates_per_slot": 2},
        )
        assert suggest_res.status_code == 404
        assert "Transcript not found" in suggest_res.text


def test_broll_external_candidate_can_be_materialized_on_choose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 10.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.routers.transcript.generate_transcript", _fake_transcript)

    monkeypatch.setattr(
        "app.routers.broll.search_external_broll_candidates",
        lambda **_: [
            ExternalBrollCandidate(
                source_type="pexels_video",
                source_url="https://example.com/demo.mp4",
                source_label="Pexels Demo Clip",
                score=0.91,
                reason={"provider": "pexels"},
            )
        ],
    )
    monkeypatch.setattr("app.routers.broll.probe_stream_flags", lambda _: {"has_video": True, "has_audio": False})
    monkeypatch.setattr("app.routers.broll.probe_duration_seconds", lambda _: 4.2)

    def _fake_download_external_video(project_id: str, _url: str) -> tuple[str, str, str]:
        project_dir = storage.upload_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        destination = project_dir / "external-demo.mp4"
        destination.write_bytes(b"fake-external-video")
        relative = str(destination.resolve().relative_to(storage.upload_root))
        return (str(destination.resolve()), relative, "video/mp4")

    monkeypatch.setattr("app.routers.broll._download_external_video", _fake_download_external_video)

    with TestClient(app) as client:
        project_id = _create_project(client, "Broll External Materialize")
        transcript_asset_id = _upload_video(client, project_id, "source.mp4")
        _upload_video(client, project_id, "local-cutaway.mp4")

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": transcript_asset_id},
        )
        assert generate_res.status_code == 200

        suggest_res = client.post(
            f"/api/v1/broll/suggest?project_id={project_id}",
            json={"max_slots": 1, "candidates_per_slot": 3},
        )
        assert suggest_res.status_code == 200
        slot = suggest_res.json()["slots"][0]
        external = next((cand for cand in slot["candidates"] if cand["source_type"] == "pexels_video"), None)
        assert external is not None
        assert external["asset_id"] is None

        media_before = client.get(f"/api/v1/media?project_id={project_id}")
        assert media_before.status_code == 200
        count_before = len(media_before.json())

        choose_res = client.post(
            f"/api/v1/broll/slots/{slot['id']}/choose?project_id={project_id}",
            json={"candidate_id": external["id"]},
        )
        assert choose_res.status_code == 200
        chosen_slot = choose_res.json()
        chosen = next(cand for cand in chosen_slot["candidates"] if cand["id"] == external["id"])
        assert chosen["asset_id"]
        assert chosen_slot["status"] == "chosen"

        media_after = client.get(f"/api/v1/media?project_id={project_id}")
        assert media_after.status_code == 200
        assets = media_after.json()
        assert len(assets) == count_before + 1
        assert any(asset["id"] == chosen["asset_id"] for asset in assets)

        materialized_path = storage.upload_root / project_id / "external-demo.mp4"
        assert materialized_path.exists()
        Path(materialized_path).unlink(missing_ok=True)


def test_broll_ai_rerank_toggle_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 10.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.routers.transcript.generate_transcript", _fake_transcript)

    def _raise_if_called(**_: object) -> list[tuple[str, str | None, str | None, str | None, float, dict[str, object]]]:
        raise RuntimeError("rerank should not be called when ai_rerank=false")

    monkeypatch.setattr("app.routers.broll.rerank_broll_candidates", _raise_if_called)

    with TestClient(app) as client:
        project_id = _create_project(client, "Broll AI Toggle")
        transcript_asset_id = _upload_video(client, project_id, "speaker.mp4")
        _upload_video(client, project_id, "cutaway.mp4")

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": transcript_asset_id},
        )
        assert generate_res.status_code == 200

        suggest_res = client.post(
            f"/api/v1/broll/suggest?project_id={project_id}",
            json={"max_slots": 1, "candidates_per_slot": 2, "include_external_sources": False, "ai_rerank": False},
        )
        assert suggest_res.status_code == 200

    monkeypatch.setattr(
        "app.routers.broll.rerank_broll_candidates",
        lambda **kwargs: [
            (
                kwargs["candidates"][0][0],
                kwargs["candidates"][0][1],
                kwargs["candidates"][0][2],
                kwargs["candidates"][0][3],
                0.93,
                {
                    **kwargs["candidates"][0][5],
                    "confidence": 0.88,
                    "score_breakdown": {"semantic": 0.9, "entity": 0.7},
                    "entities": ["Elon Musk"],
                },
            )
        ]
        + kwargs["candidates"][1:],
    )

    with TestClient(app) as client:
        project_id = _create_project(client, "Broll AI Metadata")
        transcript_asset_id = _upload_video(client, project_id, "speaker-2.mp4")
        _upload_video(client, project_id, "cutaway-2.mp4")

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": transcript_asset_id},
        )
        assert generate_res.status_code == 200

        suggest_res = client.post(
            f"/api/v1/broll/suggest?project_id={project_id}",
            json={"max_slots": 1, "candidates_per_slot": 2, "include_external_sources": False, "ai_rerank": True},
        )
        assert suggest_res.status_code == 200
        payload = suggest_res.json()
        candidate = payload["slots"][0]["candidates"][0]
        assert candidate["confidence"] == pytest.approx(0.88, rel=1e-3)
        assert candidate["score_breakdown"]["semantic"] == pytest.approx(0.9, rel=1e-3)
        assert candidate["entities"] == ["Elon Musk"]


def test_broll_auto_apply_syncs_and_can_undo_in_one_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 10.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.routers.transcript.generate_transcript", _fake_transcript)

    with TestClient(app) as client:
        project_id = _create_project(client, "Broll Auto Apply")
        transcript_asset_id = _upload_video(client, project_id, "speaker-auto.mp4")
        _upload_video(client, project_id, "cutaway-auto.mp4")

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": transcript_asset_id},
        )
        assert generate_res.status_code == 200

        auto_res = client.post(
            f"/api/v1/broll/auto-apply?project_id={project_id}",
            json={
                "max_slots": 2,
                "candidates_per_slot": 2,
                "include_external_sources": False,
                "ai_rerank": False,
                "replace_existing": True,
                "clear_existing_overlay": True,
                "fallback_to_top_candidate": True,
            },
        )
        assert auto_res.status_code == 200
        payload = auto_res.json()
        assert payload["created_slots"] >= 1
        assert payload["auto_chosen_slots"] >= 1
        assert payload["synced_clip_count"] >= 1
        assert _overlay_clip_count(payload["timeline"]) == payload["synced_clip_count"]

        undo_res = client.post(f"/api/v1/projects/{project_id}/undo")
        assert undo_res.status_code == 200
        assert _overlay_clip_count(undo_res.json()["timeline"]) == 0


def test_broll_slot_reroll_adds_variants_without_resetting_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.routers.media.probe_duration_seconds", lambda _: 10.0)
    monkeypatch.setattr("app.routers.media.probe_stream_flags", lambda _: {"has_video": True, "has_audio": True})
    monkeypatch.setattr("app.routers.transcript.generate_transcript", _fake_transcript)

    request_count = {"value": 0}

    def _fake_external_candidates(**_: object) -> list[ExternalBrollCandidate]:
        request_count["value"] += 1
        idx = request_count["value"]
        return [
            ExternalBrollCandidate(
                source_type="pexels_video",
                source_url=f"https://example.com/reroll-{idx}.mp4",
                source_label=f"Pexels Reroll {idx}",
                score=0.9 - (idx * 0.01),
                reason={"provider": "pexels", "reroll_index": idx},
            )
        ]

    monkeypatch.setattr("app.routers.broll.search_external_broll_candidates", _fake_external_candidates)
    monkeypatch.setattr("app.routers.broll.probe_stream_flags", lambda _: {"has_video": True, "has_audio": False})
    monkeypatch.setattr("app.routers.broll.probe_duration_seconds", lambda _: 4.2)

    def _fake_download_external_video(project_id: str, source_url: str) -> tuple[str, str, str]:
        stem = source_url.rsplit("/", 1)[-1].replace(".mp4", "")
        project_dir = storage.upload_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        destination = project_dir / f"{stem}.mp4"
        destination.write_bytes(b"fake-external-video")
        relative = str(destination.resolve().relative_to(storage.upload_root))
        return (str(destination.resolve()), relative, "video/mp4")

    monkeypatch.setattr("app.routers.broll._download_external_video", _fake_download_external_video)

    with TestClient(app) as client:
        project_id = _create_project(client, "Broll Slot Reroll")
        transcript_asset_id = _upload_video(client, project_id, "speaker-reroll.mp4")

        generate_res = client.post(
            f"/api/v1/transcript/generate?project_id={project_id}",
            json={"asset_id": transcript_asset_id},
        )
        assert generate_res.status_code == 200

        suggest_res = client.post(
            f"/api/v1/broll/suggest?project_id={project_id}",
            json={
                "max_slots": 1,
                "candidates_per_slot": 1,
                "include_project_assets": False,
                "include_external_sources": True,
                "ai_rerank": False,
            },
        )
        assert suggest_res.status_code == 200
        slot = suggest_res.json()["slots"][0]
        first_candidate = slot["candidates"][0]
        assert first_candidate["source_url"] == "https://example.com/reroll-1.mp4"

        choose_res = client.post(
            f"/api/v1/broll/slots/{slot['id']}/choose?project_id={project_id}",
            json={"candidate_id": first_candidate["id"]},
        )
        assert choose_res.status_code == 200
        chosen_slot = choose_res.json()
        assert chosen_slot["status"] == "chosen"
        assert chosen_slot["chosen_candidate_id"] == first_candidate["id"]

        media_after_choose = client.get(f"/api/v1/media?project_id={project_id}")
        assert media_after_choose.status_code == 200
        media_count_after_choose = len(media_after_choose.json())

        reroll_res = client.post(
            f"/api/v1/broll/slots/{slot['id']}/reroll?project_id={project_id}",
            json={
                "candidates_per_slot": 1,
                "include_project_assets": False,
                "include_external_sources": True,
                "ai_rerank": False,
            },
        )
        assert reroll_res.status_code == 200
        rerolled_slot = reroll_res.json()
        assert rerolled_slot["status"] == "chosen"
        assert rerolled_slot["chosen_candidate_id"] == first_candidate["id"]
        assert len(rerolled_slot["candidates"]) >= 2
        source_urls = {item["source_url"] for item in rerolled_slot["candidates"] if item["source_url"]}
        assert "https://example.com/reroll-1.mp4" in source_urls
        assert "https://example.com/reroll-2.mp4" in source_urls

        media_after_reroll = client.get(f"/api/v1/media?project_id={project_id}")
        assert media_after_reroll.status_code == 200
        assert len(media_after_reroll.json()) == media_count_after_choose
