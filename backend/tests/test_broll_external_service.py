from app.broll_external_service import _pick_pexels_file, _pick_pixabay_video


def test_pick_pexels_file_prefers_reasonable_hd_stream_over_4k() -> None:
    picked = _pick_pexels_file(
        [
            {"link": "https://cdn.example.com/4k.mp4", "width": 3840, "height": 2160, "file_type": "video/mp4"},
            {"link": "https://cdn.example.com/1080.mp4", "width": 1920, "height": 1080, "file_type": "video/mp4"},
            {"link": "https://cdn.example.com/720.mp4", "width": 1280, "height": 720, "file_type": "video/mp4"},
        ],
        target_orientation="landscape",
    )

    assert picked is not None
    assert picked["link"] == "https://cdn.example.com/1080.mp4"


def test_pick_pixabay_video_prefers_reasonable_hd_stream_over_4k() -> None:
    picked = _pick_pixabay_video(
        {
            "large": {"url": "https://cdn.example.com/4k.mp4", "width": 3840, "height": 2160},
            "medium": {"url": "https://cdn.example.com/1080.mp4", "width": 1920, "height": 1080},
            "small": {"url": "https://cdn.example.com/720.mp4", "width": 1280, "height": 720},
        },
        target_orientation="landscape",
    )

    assert picked == ("https://cdn.example.com/1080.mp4", 1920, 1080)
