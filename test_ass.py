import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.render_service import _build_ass_subtitle_file

text_overlays = [
    {
        "text": "Hello world",
        "start": 0.0,
        "end": 2.0,
        "color": "&H00FFFFFF",
        "highlight_color": "&H00FF0000",
        "outline_color": "&H00FF0000",
        "outline_width": 2,
        "margin_v": 50,
        "alignment": 2,
        "font_size": 24,
        "word_timings": [
            {"text": "Hello", "start_tl": 0.0, "end_tl": 1.0},
            {"text": "world", "start_tl": 1.0, "end_tl": 2.0}
        ]
    }
]

ass_path = _build_ass_subtitle_file(text_overlays, 1080, 1920)
with open(ass_path, "r") as f:
    print(f.read())
