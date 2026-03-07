import os

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")
os.environ.setdefault("BROLL_LLM_ENABLED", "false")

from app.broll_ai_service import rerank_broll_candidates
from app.models import MediaAsset
from app.routers.broll import _empty_sequence_state, _rank_candidates, _remember_sequence_candidate, _sequence_diversify_candidates


def _asset(
    asset_id: str,
    filename: str,
    *,
    duration_sec: float,
    metadata_json: str,
) -> MediaAsset:
    return MediaAsset(
        id=asset_id,
        project_id="project-1",
        media_type="video",
        filename=filename,
        storage_path=f"{asset_id}.mp4",
        mime_type="video/mp4",
        duration_sec=duration_sec,
        metadata_json=metadata_json,
    )


def test_rerank_broll_candidates_prefers_process_aligned_specific_clip() -> None:
    assets_by_id = {
        "asset-office": _asset(
            "asset-office",
            "office-team-meeting.mp4",
            duration_sec=6.0,
            metadata_json='{"tags":["office","meeting","team","speaker"]}',
        ),
        "asset-dashboard": _asset(
            "asset-dashboard",
            "dashboard-workflow-closeup.mp4",
            duration_sec=3.0,
            metadata_json='{"tags":["dashboard","workflow","screen","hands","close up"]}',
        ),
    }
    candidates = [
        (
            "project_asset",
            "asset-office",
            None,
            "office-team-meeting.mp4",
            0.84,
            {
                "query_mode": "environment",
                "keyword_hits": [],
                "crop_score": 0.88,
            },
        ),
        (
            "project_asset",
            "asset-dashboard",
            None,
            "dashboard-workflow-closeup.mp4",
            0.74,
            {
                "query_mode": "process",
                "keyword_hits": ["dashboard", "workflow", "screen"],
                "crop_score": 0.91,
            },
        ),
    ]

    ranked = rerank_broll_candidates(
        chunk_text="Watch the dashboard workflow as we process each order on screen.",
        concept_text="dashboard workflow screen",
        concept_tokens=["dashboard", "workflow", "screen"],
        slot_duration_sec=2.4,
        candidates=candidates,
        assets_by_id=assets_by_id,
        visual_intent="process_step",
    )

    assert ranked[0][1] == "asset-dashboard"
    assert ranked[0][5]["score_breakdown"]["alignment"] > ranked[1][5]["score_breakdown"]["alignment"]
    assert "specificity_low" in ranked[1][5]["weak_reason_codes"]


def test_rank_candidates_prefers_metadata_rich_non_primary_asset() -> None:
    assets = [
        _asset(
            "transcript-asset",
            "speaker-talking-head.mp4",
            duration_sec=15.0,
            metadata_json='{"title":"founder speaking to camera in office"}',
        ),
        _asset(
            "supporting-asset",
            "warehouse-dashboard-closeup.mp4",
            duration_sec=4.0,
            metadata_json='{"tags":["warehouse","dashboard","workflow","screen","close up"]}',
        ),
    ]

    ranked = _rank_candidates(
        assets=assets,
        transcript_asset_id="transcript-asset",
        concept_tokens=["warehouse", "dashboard", "workflow"],
        candidates_per_slot=2,
        slot_duration=2.0,
        shot_style="detail",
        visual_intent="process_step",
    )

    assert ranked[0][0].id == "supporting-asset"
    assert "metadata_match" in ranked[0][2]["tags"]


def test_sequence_diversify_candidates_penalizes_recent_repeat() -> None:
    state = _empty_sequence_state()
    repeated = (
        "project_asset",
        "asset-repeat",
        None,
        "office-team-meeting.mp4",
        0.86,
        {
            "query_mode": "environment",
            "search_concept": "office team meeting",
            "score_breakdown": {"semantic": 0.7},
            "confidence": 0.82,
        },
    )
    _remember_sequence_candidate(state, repeated)

    diversified = _sequence_diversify_candidates(
        [
            repeated,
            (
                "project_asset",
                "asset-fresh",
                None,
                "dashboard-workflow-closeup.mp4",
                0.8,
                {
                    "query_mode": "process",
                    "search_concept": "dashboard workflow",
                    "score_breakdown": {"semantic": 0.68},
                    "confidence": 0.8,
                },
            ),
        ],
        sequence_state=state,
    )

    assert diversified[0][1] == "asset-fresh"
    assert diversified[0][5]["score_breakdown"]["diversity"] > diversified[1][5]["score_breakdown"]["diversity"]
