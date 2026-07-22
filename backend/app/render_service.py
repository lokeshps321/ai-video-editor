from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from .config import get_settings
from .schemas import Clip, ExportSettings, Resolution, TimelineState

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _ffmpeg_supports_encoder(encoder_name: str) -> bool:
    try:
        process = subprocess.run(
            [settings.ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if process.returncode != 0:
        return False
    output = f"{process.stdout}\n{process.stderr}"
    return bool(re.search(rf"^\s*V\S*\s+{re.escape(encoder_name)}\b", output, flags=re.MULTILINE))


@lru_cache(maxsize=8)
def _ffmpeg_encoder_usable(encoder_name: str) -> bool:
    if not _ffmpeg_supports_encoder(encoder_name):
        return False
    try:
        process = subprocess.run(
            [
                settings.ffmpeg_bin,
                "-hide_banner",
                "-f",
                "lavfi",
                "-i",
                "color=size=16x16:rate=1:color=black",
                "-frames:v",
                "1",
                "-c:v",
                encoder_name,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return process.returncode == 0


def _resolve_h264_video_encoder() -> str:
    configured = (settings.render_video_encoder or "auto").strip().lower()
    if configured in {"libx264", "h264_nvenc"}:
        return configured
    if configured == "auto" and _ffmpeg_encoder_usable("h264_nvenc"):
        return "h264_nvenc"
    return "libx264"


def _quality_to_nvenc_preset(quality: str) -> str:
    return {
        "low": "p1",
        "medium": "p4",
        "high": "p5",
        "max": "p7",
    }.get(quality, "p5")


def _quality_to_nvenc_cq(quality: str) -> int:
    return {
        "low": 30,
        "medium": 25,
        "high": 21,
        "max": 18,
    }.get(quality, 21)

# Convert hex #RRGGBB or named colors to ASS &HAABBGGRR format
def _color_to_ass(color: str | None, fallback: str = "&H00FFFFFF") -> str:
    if not color:
        return fallback
    raw = str(color).strip()
    # Already ASS format
    if raw.upper().startswith("&H"):
        return raw.upper()
        
    alpha_hex = "00" # default opaque
    if "@" in raw:
        base_color, alpha_str = raw.split("@", 1)
        raw = base_color
        try:
            alpha_float = float(alpha_str)
            # ASS alpha: 00 is opaque, FF is transparent.
            # So 0.5 opacity = 50% transparent = 127 = 7F
            alpha_int = int((1.0 - max(0.0, min(1.0, alpha_float))) * 255)
            alpha_hex = f"{alpha_int:02X}"
        except ValueError:
            pass

    # Named white/black/yellow shortcuts
    name_map = {
        "white": "FFFFFF", "black": "000000",
        "yellow": "00FFFF", "red": "0000FF",
        "blue": "FF0000", "cyan": "FFFF00",
        "green": "00FF00", "magenta": "FF00FF",
    }
    if raw.lower() in name_map:
        return f"&H{alpha_hex}{name_map[raw.lower()]}"

    # Convert #RRGGBB or #RRGGBBAA
    hex_raw = raw.lstrip("#")
    if len(hex_raw) in (6, 8):
        try:
            r = int(hex_raw[0:2], 16)
            g = int(hex_raw[2:4], 16)
            b = int(hex_raw[4:6], 16)
            
            # If 8 digits and no @ was provided, parse the alpha from the hex
            if len(hex_raw) == 8 and alpha_hex == "00":
                # #RRGGBBAA where AA is opacity (FF=opaque, 00=transparent)
                # ASS needs transparency (00=opaque, FF=transparent)
                a = int(hex_raw[6:8], 16)
                ass_alpha = 255 - a
                alpha_hex = f"{ass_alpha:02X}"
                
            return f"&H{alpha_hex}{b:02X}{g:02X}{r:02X}"
        except ValueError:
            pass
    return fallback


def _ass_color_with_alpha(color: str, alpha_hex: str, fallback: str = "&HA0000000") -> str:
    raw = str(color or "").strip().upper()
    match = re.match(r"^&H([0-9A-F]{8})$", raw)
    if not match:
        return fallback
    packed = match.group(1)
    return f"&H{alpha_hex.upper()}{packed[2:]}"


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _scale_ass_caption_metrics(
    font_size: int,
    margin_v: int,
    outline_w: int,
    shadow: int,
    alignment: int,
    out_w: int,
    out_h: int,
) -> tuple[int, int, int, int]:
    portrait = out_h >= out_w
    ref_w, ref_h = (360.0, 640.0) if portrait else (640.0, 360.0)
    scale = max(1.0, min(out_w / ref_w, out_h / ref_h))

    scaled_font = max(18, int(round(font_size * scale)))
    if portrait:
        min_font = max(40, int(round(out_h * 0.032)))
        max_font = max(min_font, int(round(out_h * 0.056)))
    else:
        min_font = max(24, int(round(out_h * 0.040)))
        max_font = max(min_font, int(round(out_h * 0.085)))
    scaled_font = _clamp_int(scaled_font, min_font, max_font)

    scaled_outline = max(1, int(round(outline_w * scale * (0.65 if portrait else 0.75))))
    scaled_shadow = max(0, int(round(shadow * scale * (0.40 if portrait else 0.55))))

    if alignment in {1, 2, 3}:
        if portrait:
            # Portrait exports need a lower-third safe area, not a linear margin
            # scale that drifts captions toward mid-frame on taller outputs.
            scaled_margin = int(round(margin_v * (out_h / 1080.0)))
            min_margin = int(round(out_h * 0.085))
            max_margin = int(round(out_h * 0.130))
        else:
            scaled_margin = int(round(margin_v * scale))
            min_margin = int(round(out_h * 0.055))
            max_margin = int(round(out_h * 0.180))
        scaled_margin = _clamp_int(scaled_margin, min_margin, max_margin)
    else:
        scaled_margin = max(0, int(round(margin_v * scale)))

    return scaled_font, scaled_margin, scaled_outline, scaled_shadow


def _ass_ms(seconds: float) -> int:
    return max(0, int(round(max(seconds, 0.0) * 1000)))


def _ass_style_motion_tags(
    style: str,
    duration_sec: float,
    *,
    indic_safe: bool = False,
    karaoke_segment: bool = False,
) -> str:
    """ASS override tags that mirror caption preset motion using libass-safe fades."""
    if karaoke_segment:
        return ""

    normalized = _normalize_caption_style(style)
    dur_ms = max(50, _ass_ms(duration_sec))

    def _fad(in_ms: int, out_ms: int) -> str:
        fade_in = max(20, min(in_ms, max(20, dur_ms // 2)))
        fade_out = max(20, min(out_ms, max(20, dur_ms // 2)))
        return f"{{\\fad({fade_in},{fade_out})}}"

    if indic_safe:
        kinetic_styles = {
            "pop",
            "hormozi_bold",
            "hormozi_green",
            "shorts_viral",
            "orange_fire",
            "pop_color",
            "street_impact",
            "neon_gamer",
            "retro_vhs",
            "bounce",
            "karaoke",
        }
        if normalized in kinetic_styles or normalized in {
            "fade",
            "cinematic_serif",
            "elegant_gold",
            "creator",
            "minimalist",
            "typewriter",
            "basic_white",
        }:
            return _fad(120, 100)
        return ""

    if normalized in {
        "pop",
        "hormozi_bold",
        "pop_color",
        "street_impact",
        "orange_fire",
    }:
        snap_in = 70 if normalized == "street_impact" else 90
        return _fad(snap_in, 80)

    if normalized in {"fade", "cinematic_serif"}:
        return _fad(280, 260)

    if normalized == "elegant_gold":
        return _fad(360, 300)

    if normalized in {"typewriter", "retro_vhs"}:
        return _fad(60, 70)

    if normalized in {"karaoke", "neon_gamer", "bounce"}:
        return _fad(80, 90)

    if normalized == "creator":
        return _fad(90, 120)

    if normalized == "minimalist":
        return _fad(100, 90)

    return ""


def _resolve_ass_font_name(value: object) -> str:
    raw = str(value).strip() if value not in (None, "") else ""
    if not raw or raw.lower() == "none":
        return "DejaVu Sans"
    family = raw.split("-")[0].split(" ")[0]
    if family.lower() in (
        "arial",
        "helvetica",
        "inter",
        "roboto",
        "montserrat",
        "poppins",
        "impact",
        "orbitron",
        "courier",
        "playfair",
        "georgia",
    ):
        return "DejaVu Sans"
    return family


def _build_ass_subtitle_file(
    text_overlays: list[dict],
    out_w: int,
    out_h: int,
) -> str:
    """Write an ASS subtitle file to a temp path and return the path.
    Uses a single ASS file instead of 100+ drawtext filters for massive speed gains.
    """
    def _ts(sec: float) -> str:
        """Return ASS timestamp in H:MM:SS.CS format (centiseconds)."""
        total_cs = max(0, int(round(max(sec, 0.0) * 100)))
        h = total_cs // 360000
        rem = total_cs % 360000
        m = rem // 6000
        rem %= 6000
        s = rem // 100
        cs = rem % 100
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    # Build events from first overlay to estimate global style
    margin_v = 60
    alignment = 2
    font_size = 48
    primary = "&H00FFFFFF"
    outline_color = "&H00000000"
    back_color = "&HA0000000"
    outline_w = 2
    shadow = 1
    font_name = "DejaVu Sans" # Better fallback for Linux than Arial
    bold = 0
    indic_font_name: str | None = None

    if text_overlays:
        sample = text_overlays[0]
        sample_style = _normalize_caption_style(str(sample.get("style", "")))
        font_size = int(sample.get("font_size", 48))
        primary = _color_to_ass(str(sample.get("color", "white")))
        outline_color = _color_to_ass(str(sample.get("outline_color", "black")))
        outline_w = int(sample.get("outline_width", 2))
        shadow = int(sample.get("shadow", 1))
        margin_v = int(sample.get("margin_v", 60))
        alignment = int(sample.get("alignment", 2))
        
        font_name = _resolve_ass_font_name(sample.get("font_name"))
        raw_font = str(sample.get("font_name") or "")
        if "bold" in raw_font.lower():
            bold = -1

        # Detect Indic scripts and override font if needed
        all_text = " ".join([str(o.get("text", "")) for o in text_overlays])
        indic_font_name = _pick_ass_font_name(all_text)
        if indic_font_name:
            font_name = indic_font_name
            # Synthetic bold breaks shaping and can clip diacritics on complex-script fallback fonts.
            bold = 0
            # Indic shaping is stable only when we avoid inline karaoke color tags.
            # Preserve style identity only for explicitly color-led presets by promoting
            # the preset highlight color to the whole line. Neutral presets such as
            # basic_white must keep their primary color.
            sample_highlight = _color_to_ass(str(sample.get("highlight_color", "")), fallback=primary)
            if sample_highlight and sample_highlight != primary:
                back_color = _ass_color_with_alpha(sample_highlight, "96")
                shadow = max(shadow, 3)
                outline_w = max(outline_w, 3)
                if sample_style in _INDIC_HIGHLIGHT_TO_PRIMARY_STYLES:
                    primary = sample_highlight

        font_size, margin_v, outline_w, shadow = _scale_ass_caption_metrics(
            font_size,
            margin_v,
            outline_w,
            shadow,
            alignment,
            out_w,
            out_h,
        )
            
    logger.debug(
        "ASS subtitle: font=%r size=%d color=%s align=%d margin_v=%d overlays=%d",
        font_name, font_size, primary, alignment, margin_v, len(text_overlays),
    )

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {out_w}",
        f"PlayResY: {out_h}",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},{font_size},{primary},&H000000FF,{outline_color},{back_color},{bold},0,0,0,100,100,0,0,1,{outline_w},{shadow},{alignment},10,10,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for overlay in text_overlays:
        start_sec = float(overlay.get("start", 0.0))
        end_sec = float(overlay.get("end", start_sec + 1.0))
        overlay_style = str(overlay.get("style", "") or "")
        raw_text = str(overlay.get("text", "")).replace("\n", "\\N")
        # Escape special ASS characters
        raw_text = raw_text.replace("{", "").replace("}", "")

        words = overlay.get("word_timings", [])
        highlight_color = overlay.get("highlight_color")
        overlay_uses_indic_font = indic_font_name is not None or _pick_ass_font_name(raw_text.replace("\\N", " ")) is not None

        # Disable karaoke highlights for Indic languages due to a known libass bug
        # where inline ASS tags (like {\c}) permanently break HarfBuzz complex text 
        # shaping for the adjacent text segment, causing disjointed characters.
        if overlay_uses_indic_font:
            highlight_color = None

        # If we have word timings AND a highlight color, do a word-by-word karaoke style reveal!
        # IMPORTANT: Limit karaoke entries to prevent OOM crashes in FFmpeg's libass renderer
        max_karaoke_words = int(os.environ.get("RENDER_MAX_KARAOKE_WORDS", "50"))
        
        if words and highlight_color and len(words) <= max_karaoke_words:
            hl_ass_color = _color_to_ass(str(highlight_color), fallback=primary)
            
            # The raw_text is exactly the words joined by spaces (maybe some uppercasing).
            # We split it into parts matching the words array.
            final_words = raw_text.split(" ")
            
            # Failsafe: if parts don't match (e.g. custom text edits), fallback to standard block
            if len(final_words) == len(words):
                for i in range(len(words)):
                    # Word segment covers the time until the START of the next word
                    seg_start = start_sec if i == 0 else float(words[i]["start_tl"])
                    seg_end = end_sec if i == len(words) - 1 else float(words[i+1]["start_tl"])
                    seg_motion = _ass_style_motion_tags(
                        overlay_style,
                        max(seg_end - seg_start, 0.05),
                        indic_safe=overlay_uses_indic_font,
                        karaoke_segment=True,
                    )
                    
                    colored_line = []
                    for j, w in enumerate(final_words):
                        if i == j:
                            colored_line.append(f"{{\\c{hl_ass_color}&}}{w}{{\\c{primary}&}}")
                        else:
                            colored_line.append(w)
                    
                    full_line = f"{seg_motion}{' '.join(colored_line)}"
                    # Use Layer 1 to ensure it's above the video
                    lines.append(f"Dialogue: 1,{_ts(seg_start)},{_ts(seg_end)},Default,,0,0,0,,{full_line}")
                continue
                
        # Standard fallback (no highlights, just plain text block)
        block_motion = _ass_style_motion_tags(
            overlay_style,
            max(end_sec - start_sec, 0.05),
            indic_safe=overlay_uses_indic_font,
        )
        lines.append(
            f"Dialogue: 1,{_ts(start_sec)},{_ts(end_sec)},Default,,0,0,0,,{block_motion}{raw_text}"
        )

    content = "\n".join(lines)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".ass", delete=False
    )
    tmp.write(content)
    tmp.close()
    return tmp.name

_SCRIPT_FONT_CANDIDATES: list[tuple[tuple[tuple[int, int], ...], list[str]]] = [
    (
        ((0x0C80, 0x0CFF),),  # Kannada
        ["/usr/share/fonts/truetype/lohit-kannada/Lohit-Kannada.ttf"],
    ),
    (
        ((0x0900, 0x097F),),  # Devanagari
        ["/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"],
    ),
    (
        ((0x0B80, 0x0BFF),),  # Tamil
        [
            "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
            "/usr/share/fonts/truetype/samyak-fonts/Samyak-Tamil.ttf",
        ],
    ),
    (
        ((0x0C00, 0x0C7F),),  # Telugu
        ["/usr/share/fonts/truetype/lohit-telugu/Lohit-Telugu.ttf"],
    ),
    (
        ((0x0D00, 0x0D7F),),  # Malayalam
        ["/usr/share/fonts/truetype/lohit-malayalam/Lohit-Malayalam.ttf"],
    ),
    (
        ((0x0980, 0x09FF),),  # Bengali/Assamese
        [
            "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
            "/usr/share/fonts/truetype/lohit-assamese/Lohit-Assamese.ttf",
        ],
    ),
    (
        ((0x0A80, 0x0AFF),),  # Gujarati
        ["/usr/share/fonts/truetype/lohit-gujarati/Lohit-Gujarati.ttf"],
    ),
    (
        ((0x0A00, 0x0A7F),),  # Gurmukhi (Punjabi)
        ["/usr/share/fonts/truetype/lohit-punjabi/Lohit-Gurmukhi.ttf"],
    ),
    (
        ((0x0B00, 0x0B7F),),  # Odia
        ["/usr/share/fonts/truetype/lohit-oriya/Lohit-Odia.ttf"],
    ),
]

_GENERIC_FONT_CANDIDATES: list[str] = [
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_MAX_INLINE_FILTER_COMPLEX_CHARS = 20_000
_ASS_COLOR_RE = re.compile(r"^&H([0-9A-Fa-f]{8})$")
_CAPTION_STYLE_ALIASES = {
    "kinetic": "hormozi_bold",
    "minimal": "minimalist",
    "bold": "hormozi_bold",
    "gradient": "neon_gamer",
    "outline": "neon_gamer",
    "cinema": "cinematic_serif",
    "cinematic": "cinematic_serif",
    "serif": "cinematic_serif",
    "retro": "retro_vhs",
    "vhs": "retro_vhs",
    "pop": "pop_color",
    "colorful": "pop_color",
    "gold": "elegant_gold",
    "elegant": "elegant_gold",
    "luxury": "elegant_gold",
    "street": "street_impact",
    "impact": "street_impact",
    "urban": "street_impact",
    "fire": "orange_fire",
    "orange": "orange_fire",
}
_INDIC_HIGHLIGHT_TO_PRIMARY_STYLES = {
    "hormozi_bold",
    "hormozi_green",
    "shorts_viral",
    "orange_fire",
    "neon_gamer",
    "retro_vhs",
    "pop_color",
    "pop_cyan",
    "elegant_gold",
    "street_impact",
}


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _resolution_dims(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    dims_map = {
        "16:9": {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
        },
        "9:16": {
            "720p": (720, 1280),
            "1080p": (1080, 1920),
            "4k": (2160, 3840),
        },
    }
    width, height = dims_map[aspect_ratio][resolution]
    return _even(width), _even(height)


def _quality_to_crf(quality: str) -> int:
    mapping = {
        "low": 30,
        "medium": 25,
        "high": 20,
        "max": 16,
    }
    return mapping[quality]


def _quality_to_x264_preset(quality: str) -> str:
    mapping = {
        "low": "ultrafast",
        "medium": "veryfast",
        "high": "medium",
        "max": "slow",
    }
    return mapping[quality]


def _clip_duration(clip: Clip) -> float:
    return (clip.end_sec - clip.start_sec) / max(clip.speed, 0.01)


def _escape_drawtext(text: str) -> str:
    # Older ffmpeg builds are fragile around single quotes inside drawtext text values.
    # Normalize apostrophes out to avoid breaking filter parsing.
    normalized = text.replace("'", "")
    return (
        normalized.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("%", "\\%")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(";", "\\;")
    )


def _escape_drawtext_expr(expr: str) -> str:
    return (
        expr.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("'", "\\'")
    )


def _font_exists(path: str) -> bool:
    return Path(path).is_file()


def _contains_script(text: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    for char in text:
        code = ord(char)
        for start, end in ranges:
            if start <= code <= end:
                return True
    return False


def _pick_drawtext_fontfile(text: str) -> str | None:
    override = (os.getenv("RENDER_SUBTITLE_FONTFILE", "") or "").strip()
    if override and _font_exists(override):
        return override

    for ranges, candidates in _SCRIPT_FONT_CANDIDATES:
        if not _contains_script(text, ranges):
            continue
        for path in candidates:
            if _font_exists(path):
                return path

    for path in _GENERIC_FONT_CANDIDATES:
        if _font_exists(path):
            return path
    return None


def _pick_ass_font_name(text: str) -> str | None:
    for ranges, candidates in _SCRIPT_FONT_CANDIDATES:
        if not _contains_script(text, ranges):
            continue
        for path in candidates:
            if _font_exists(path):
                # E.g. /usr/share/fonts/truetype/lohit-kannada/Lohit-Kannada.ttf -> "Lohit Kannada"
                filename = Path(path).stem
                return filename.replace("-", " ")
    return None


def _normalize_caption_style(style: str) -> str:
    normalized = str(style or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _CAPTION_STYLE_ALIASES.get(normalized, normalized)


def _ass_color_to_drawtext(value: str) -> str | None:
    match = _ASS_COLOR_RE.match(str(value or "").strip())
    if not match:
        return None
    packed = match.group(1)
    alpha = int(packed[0:2], 16)
    blue = int(packed[2:4], 16)
    green = int(packed[4:6], 16)
    red = int(packed[6:8], 16)
    opacity = max(0.0, min(1.0, 1.0 - (alpha / 255.0)))
    rgb = f"#{red:02X}{green:02X}{blue:02X}"
    if opacity >= 0.995:
        return rgb
    if opacity <= 0.005:
        return f"{rgb}@0"
    return f"{rgb}@{opacity:.3f}".rstrip("0").rstrip(".")


def _resolve_drawtext_color(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    raw = str(value).strip()
    if not raw:
        return fallback
    converted = _ass_color_to_drawtext(raw)
    return converted if converted else raw


def _drawtext_stroke_shadow_options(
    *,
    outline_color: str,
    outline_width: int,
    shadow: int,
) -> list[str]:
    options: list[str] = []
    if outline_width > 0:
        options.append(f"borderw={outline_width}")
        options.append(f"bordercolor={outline_color}")
    if shadow > 0:
        options.append(f"shadowcolor={outline_color}")
        options.append("shadowx=0")
        options.append(f"shadowy={shadow}")
    return options


def _float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _atempo_chain(speed: float) -> str:
    # FFmpeg atempo accepts [0.5, 2.0], so decompose out-of-range speeds.
    if speed <= 0:
        return "atempo=1.0"
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.5f}".rstrip("0").rstrip(".") for factor in factors)


def _apply_preset_filters(chain: list[str], preset: str | None) -> None:
    if not preset:
        return
    key = preset.strip().lower()
    if key == "warm":
        chain.append("colorbalance=rs=0.08:bs=-0.05")
    elif key == "cool":
        chain.append("colorbalance=rs=-0.06:bs=0.08")
    elif key == "cinematic":
        chain.append("eq=contrast=1.15:saturation=0.9")
        chain.append("vignette=angle=PI/4")
    elif key == "vintage":
        chain.append("curves=preset=vintage")
    elif key in {"mono", "blackwhite", "b&w"}:
        chain.append("hue=s=0")
    elif key == "text_safe_mild":
        chain.append("eq=brightness=-0.055:contrast=0.96:saturation=0.95")
    elif key == "text_safe_soft":
        chain.append("boxblur=luma_radius=2:luma_power=1")
        chain.append("eq=brightness=-0.085:contrast=0.94:saturation=0.92")


def _crop_x_expression(crop_width: int, keyframes: list[tuple[float, int]]) -> str:
    if not keyframes:
        return "0"
    points = sorted(keyframes, key=lambda item: item[0])
    deduped: list[tuple[float, int]] = []
    for time_sec, x in points:
        clamped_x = max(0, int(x))
        if deduped and abs(deduped[-1][0] - time_sec) < 1e-6:
            deduped[-1] = (time_sec, clamped_x)
        else:
            deduped.append((time_sec, clamped_x))
    if len(deduped) == 1:
        return f"max(0,min(iw-{crop_width},{deduped[0][1]}))"
    expr = f"{deduped[-1][1]}"
    for idx in range(len(deduped) - 2, -1, -1):
        t0, x0 = deduped[idx]
        t1, x1 = deduped[idx + 1]
        if t1 <= t0:
            continue
        span = max(t1 - t0, 0.001)
        lerp = f"({x0}+({x1 - x0})*(t-{t0:.3f})/{span:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{lerp},{expr})"
    return f"max(0,min(iw-{crop_width},{expr}))"


def _is_auto_frame_crop(crop: object) -> bool:
    """Identify the 9:16 crop shape produced by the prior Smart Reframe tool."""

    try:
        width = float(getattr(crop, "width"))
        height = float(getattr(crop, "height"))
    except (AttributeError, TypeError, ValueError):
        return False
    return width > 0 and height > 0 and abs((width / height) - (9 / 16)) < 0.01


def _video_filters_for_clip(
    clip: Clip,
    out_w: int,
    out_h: int,
    fps: int,
    *,
    cover_output: bool = False,
    apply_auto_frame_crop: bool = True,
) -> str:
    chain: list[str] = []
    crop = clip.transform.crop
    # Smart Reframe stored its subject crop on the clip. Treat those 9:16
    # crops as auto-framing: keep them only while Auto Frame is active, which
    # also makes previously saved reframes disappear in 16:9 renders.
    if crop and (apply_auto_frame_crop or not _is_auto_frame_crop(crop)):
        if clip.transform.crop_keyframes:
            keyframes = [
                (max(0.0, float(item.time_sec)), int(item.x))
                for item in clip.transform.crop_keyframes
            ]
            x_expr = _crop_x_expression(max(2, int(crop.width)), keyframes)
            chain.append(f"crop={crop.width}:{crop.height}:'{x_expr}':{int(crop.y)}")
        else:
            chain.append(f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y}")
    if clip.transform.rotate == 90:
        chain.append("transpose=1")
    elif clip.transform.rotate == 180:
        chain.append("transpose=1,transpose=1")
    elif clip.transform.rotate == 270:
        chain.append("transpose=2")
    if clip.transform.flip == "horizontal":
        chain.append("hflip")
    elif clip.transform.flip == "vertical":
        chain.append("vflip")
    if clip.speed != 1:
        chain.append(f"setpts=PTS/{clip.speed}")

    adj = clip.adjustments
    brightness = _float(adj.brightness + (adj.exposure * 0.35), -1.0, 1.0)
    contrast = _float(adj.contrast, 0.0, 3.0)
    saturation = _float(adj.saturation, 0.0, 4.0)
    chain.append(
        "eq="
        f"brightness={brightness:.4f}:"
        f"contrast={contrast:.4f}:"
        f"saturation={saturation:.4f}"
    )

    _apply_preset_filters(chain, adj.preset)
    temperature = _float(adj.temperature, -1.0, 1.0)
    if abs(temperature) > 0.001:
        rs = _float(temperature * 0.35, -1.0, 1.0)
        bs = _float(-temperature * 0.35, -1.0, 1.0)
        chain.append(f"colorbalance=rs={rs:.4f}:bs={bs:.4f}")

    chain.append(f"fps={fps}")
    if cover_output:
        chain.append(f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase")
        chain.append(f"crop={out_w}:{out_h}:(iw-ow)/2:(ih-oh)/2")
    else:
        chain.append(f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease")
        chain.append(f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2")
    chain.append("format=yuv420p")
    return ",".join(chain)


def _transition_duration(previous: Clip, current: Clip) -> float:
    if not current.transition:
        return 0.0
    raw = max(0.0, float(current.transition.duration_sec))
    max_allowed = min(_clip_duration(previous), _clip_duration(current)) * 0.45
    return _float(raw, 0.0, max_allowed)


def _timeline_layout(clips: list[Clip]) -> tuple[list[float], list[float]]:
    starts: list[float] = []
    transitions: list[float] = []
    if not clips:
        return starts, transitions
    starts.append(0.0)
    transitions.append(0.0)
    for idx in range(1, len(clips)):
        prev = clips[idx - 1]
        current = clips[idx]
        d = _transition_duration(prev, current)
        start = starts[idx - 1] + _clip_duration(prev) - d
        starts.append(max(0.0, round(start, 3)))
        transitions.append(round(d, 3))
    return starts, transitions


def _collect_text_overlays(clips: Iterable[Clip], render_starts: dict[str, float]) -> list[dict[str, object]]:
    overlays: list[dict[str, object]] = []
    for clip in clips:
        base = render_starts.get(clip.id, clip.timeline_start_sec)
        speed = float(getattr(clip, "speed", 1.0))
        if speed <= 0:
            speed = 1.0
        for item in clip.text_overlays:
            start = base + (item.start_sec / speed)
            end = base + ((item.start_sec + item.duration_sec) / speed)
            overlay_dict = {
                "text": item.text,
                "start": start,
                "end": end,
                "x": item.x,
                "y": item.y,
                "font_size": item.font_size,
                "style": item.style,
                "color": item.color,
                "highlight_color": item.highlight_color,
                "outline_color": item.outline_color,
                "outline_width": item.outline_width,
                "shadow": item.shadow,
                "font_name": item.font_name,
                "alignment": getattr(item, "alignment", 2),
                "margin_v": getattr(item, "margin_v", 80),
            }
            
            # Process word timings into timeline space
            raw_words = getattr(item, "word_timings", [])
            mapped_words = []
            for w in raw_words:
                w_src_start = float(w.get("start_sec", 0.0))
                w_src_end = float(w.get("end_sec", 0.0))
                # Convert to timeline space
                w_tl_start = base + (max(w_src_start - clip.start_sec, 0.0) / speed)
                w_tl_end = base + (max(w_src_end - clip.start_sec, 0.0) / speed)
                mapped_words.append({
                    "text": w.get("text", ""),
                    "start_tl": w_tl_start,
                    "end_tl": w_tl_end
                })
            overlay_dict["word_timings"] = mapped_words
            overlays.append(overlay_dict)
    return overlays


def _style_drawtext_options(
    style: str,
    start: float,
    end: float,
    font_size: int,
    x: str,
    y: str,
    color: str,
    *,
    highlight_color: str | None = None,
    outline_color: str = "black@0.5",
    outline_width: int = 2,
    shadow: int = 0,
) -> str:
    normalized = _normalize_caption_style(style)
    primary = _resolve_drawtext_color(color, "white")
    highlight = _resolve_drawtext_color(highlight_color, primary)
    stroke_color = _resolve_drawtext_color(outline_color, "black@0.5")
    stroke_width = max(int(outline_width), 0)
    shadow_size = max(int(shadow), 0)

    def _with_base(parts: list[str], *, border_width: int | None = None, shadow_px: int | None = None) -> str:
        parts.extend(
            _drawtext_stroke_shadow_options(
                outline_color=stroke_color,
                outline_width=stroke_width if border_width is None else max(border_width, 0),
                shadow=shadow_size if shadow_px is None else max(shadow_px, 0),
            )
        )
        return ":".join(parts)

    if normalized == "fade":
        fade_in = min(0.25, max(end - start, 0.1) * 0.3)
        fade_out = min(0.25, max(end - start, 0.1) * 0.3)
        alpha = (
            f"if(lt(t,{start:.3f}),0,"
            f"if(lt(t,{start + fade_in:.3f}),(t-{start:.3f})/{fade_in:.3f},"
            f"if(lt(t,{max(end - fade_out, start):.3f}),1,({end:.3f}-t)/{fade_out:.3f})))"
        )
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize={font_size}",
                f"fontcolor={primary}",
                f"alpha='{_escape_drawtext_expr(alpha)}'",
            ]
        )
    if normalized in {"pop", "hormozi_bold", "orange_fire"}:
        pop_end = start + (
            0.12 if normalized in {"hormozi_bold", "orange_fire"} else 0.3
        )
        start_scale = (
            1.24
            if normalized == "orange_fire"
            else 1.22 if normalized == "hormozi_bold" else 1.35
        )
        duration = (
            0.12 if normalized in {"hormozi_bold", "orange_fire"} else 0.30
        )
        size_expr = (
            f"if(lt(t,{pop_end:.3f}),"
            f"{font_size}*({start_scale:.2f}-{(start_scale - 1.0):.2f}*((t-{start:.3f})/{duration:.2f})),{font_size})"
        )
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize='{_escape_drawtext_expr(size_expr)}'",
                f"fontcolor={highlight if normalized == 'orange_fire' else primary}",
            ],
            border_width=max(stroke_width, 2 if normalized == "orange_fire" else stroke_width),
            shadow_px=max(shadow_size, 2 if normalized == "orange_fire" else shadow_size),
        )
    if normalized == "bounce":
        y_expr = f"{y}+18*sin((t-{start:.3f})*12)"
        return _with_base(
            [
                f"x={x}",
                f"y='{_escape_drawtext_expr(y_expr)}'",
                f"fontsize={font_size}",
                f"fontcolor={primary}",
            ]
        )
    if normalized == "typewriter":
        alpha = f"if(lt(t,{start + 0.08:.3f}),0,1)"
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize={font_size}",
                f"fontcolor={primary}",
                f"alpha='{_escape_drawtext_expr(alpha)}'",
            ]
        )
    if normalized in {"karaoke", "neon_gamer"}:
        # Avoid fontcolor_expr here: some ffmpeg builds accept it but render nothing.
        # Pulse alpha instead, while keeping a standard fontcolor path.
        pulse_alpha = f"if(lt(mod(t-{start:.3f},0.45),0.22),1,0.78)"
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize={font_size}",
                f"fontcolor={highlight}",
                f"alpha='{_escape_drawtext_expr(pulse_alpha)}'",
            ],
            border_width=max(stroke_width, 1),
            shadow_px=max(shadow_size, 2 if normalized == "neon_gamer" else 0),
        )
    if normalized == "creator":
        # Lightweight creator look: pop-in + readable stroke/shadow.
        # Keep math simple so long-caption renders don't overload ffmpeg.
        punch_end = start + 0.18
        fade_in = 0.06
        fade_out = 0.10
        fade_out_start = max(start + fade_in, end - fade_out)
        size_expr = (
            f"if(lt(t,{punch_end:.3f}),"
            f"{font_size}*(1.16-0.16*((t-{start:.3f})/0.18)),{font_size})"
        )
        alpha_expr = (
            f"if(lt(t,{start:.3f}),0,"
            f"if(lt(t,{start + fade_in:.3f}),(t-{start:.3f})/{fade_in:.3f},"
            f"if(lt(t,{fade_out_start:.3f}),1,"
            f"if(lt(t,{end:.3f}),({end:.3f}-t)/{fade_out:.3f},0))))"
        )
        creator_border_color = _resolve_drawtext_color(outline_color, "black@0.85")
        if str(outline_color).strip().lower() == "black@0.5":
            creator_border_color = "black@0.85"
        creator_shadow_color = "black@0.72" if shadow_size == 0 else creator_border_color
        return (
            f"x={x}:y={y}:"
            f"fontsize='{_escape_drawtext_expr(size_expr)}':"
            f"fontcolor={primary}:"
            f"alpha='{_escape_drawtext_expr(alpha_expr)}':"
            f"borderw={max(stroke_width, 3)}:bordercolor={creator_border_color}:"
            f"shadowcolor={creator_shadow_color}:shadowx=0:shadowy={max(shadow_size, 3)}"
        )
    if normalized == "minimalist":
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize={font_size}",
                f"fontcolor={primary}",
            ]
        )
    if normalized == "cinematic_serif":
        # Gentle fade-in / fade-out like classic movie subtitles.
        fade_in = min(0.30, max(end - start, 0.15) * 0.25)
        fade_out = min(0.30, max(end - start, 0.15) * 0.25)
        fade_out_start = max(start + fade_in, end - fade_out)
        alpha = (
            f"if(lt(t,{start:.3f}),0,"
            f"if(lt(t,{start + fade_in:.3f}),(t-{start:.3f})/{fade_in:.3f},"
            f"if(lt(t,{fade_out_start:.3f}),1,({end:.3f}-t)/{fade_out:.3f})))"
        )
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize={font_size}",
                f"fontcolor={primary}",
                f"alpha='{_escape_drawtext_expr(alpha)}'",
            ],
            shadow_px=max(shadow_size, 2),
        )
    if normalized == "retro_vhs":
        # Typewriter snap-in with a subtle alpha flicker for VHS feel.
        snap_delay = 0.06
        flicker = f"if(lt(t,{start + snap_delay:.3f}),0,if(lt(mod(t-{start:.3f},0.55),0.04),0.7,1))"
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize={font_size}",
                f"fontcolor={highlight}",
                f"alpha='{_escape_drawtext_expr(flicker)}'",
            ],
            border_width=max(stroke_width, 2),
        )
    if normalized == "pop_color":
        # Scale pop-in (1.3x → 1x) with snappy timing for social media energy.
        pop_dur = 0.15
        pop_end = start + pop_dur
        size_expr = (
            f"if(lt(t,{pop_end:.3f}),"
            f"{font_size}*(1.30-0.30*((t-{start:.3f})/{pop_dur:.2f})),{font_size})"
        )
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize='{_escape_drawtext_expr(size_expr)}'",
                f"fontcolor={primary}",
            ]
        )
    if normalized == "elegant_gold":
        # Slow, graceful fade-in with extended sustain and deeper shadow.
        fade_in = min(0.40, max(end - start, 0.2) * 0.30)
        fade_out = min(0.35, max(end - start, 0.2) * 0.25)
        fade_out_start = max(start + fade_in, end - fade_out)
        alpha = (
            f"if(lt(t,{start:.3f}),0,"
            f"if(lt(t,{start + fade_in:.3f}),(t-{start:.3f})/{fade_in:.3f},"
            f"if(lt(t,{fade_out_start:.3f}),1,({end:.3f}-t)/{fade_out:.3f})))"
        )
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize={font_size}",
                f"fontcolor={primary}",
                f"alpha='{_escape_drawtext_expr(alpha)}'",
            ],
            shadow_px=max(shadow_size, 3),
        )
    if normalized == "street_impact":
        # Hard slam-in with fast scale punch for maximum visual impact.
        slam_dur = 0.08
        slam_end = start + slam_dur
        size_expr = (
            f"if(lt(t,{slam_end:.3f}),"
            f"{font_size}*(1.40-0.40*((t-{start:.3f})/{slam_dur:.2f})),{font_size})"
        )
        return _with_base(
            [
                f"x={x}",
                f"y={y}",
                f"fontsize='{_escape_drawtext_expr(size_expr)}'",
                f"fontcolor={primary}",
            ],
            border_width=max(stroke_width, 4),
        )
    return _with_base(
        [
            f"x={x}",
            f"y={y}",
            f"fontsize={font_size}",
            f"fontcolor={primary}",
        ]
    )


def _volume_expression(clip: Clip) -> str:
    if clip.audio.mute:
        return "0"
    if not clip.audio.keyframes:
        return f"{max(clip.audio.volume, 0.0):.4f}"
    points = [(0.0, max(clip.audio.volume, 0.0))]
    for keyframe in sorted(clip.audio.keyframes, key=lambda item: item.time_sec):
        points.append((max(0.0, keyframe.time_sec), max(0.0, keyframe.volume)))
    dedup: list[tuple[float, float]] = []
    for time_sec, volume in points:
        if dedup and abs(dedup[-1][0] - time_sec) < 1e-6:
            dedup[-1] = (time_sec, volume)
        else:
            dedup.append((time_sec, volume))
    points = dedup
    if len(points) == 1:
        return f"{points[0][1]:.4f}"
    expr = f"{points[-1][1]:.4f}"
    for idx in range(len(points) - 2, -1, -1):
        t0, v0 = points[idx]
        t1, v1 = points[idx + 1]
        if t1 <= t0:
            continue
        span = max(t1 - t0, 0.001)
        lerp = f"({v0:.4f}+({(v1 - v0):.4f})*(t-{t0:.3f})/{span:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{lerp},{expr})"
    return expr


def _xfade_transition(name: str | None) -> str:
    mapping = {
        "fade": "fade",
        "dissolve": "dissolve",
        "slide_left": "slideleft",
        "slide_right": "slideright",
        "slide_up": "slideup",
        "slide_down": "slidedown",
        "wipe": "wipeleft",
        "zoom": "zoomin",
    }
    if not name:
        return "fade"
    return mapping.get(name, "fade")


def build_ffmpeg_command(
    timeline: TimelineState,
    clip_inputs: list[tuple[Clip, str]],
    clip_has_audio_flags: list[bool],
    bg_audio_inputs: list[tuple[Clip, str]],
    bg_has_audio_flags: list[bool],
    output_path: str,
    export_settings: ExportSettings,
    overlay_inputs: list[tuple[Clip, str]] | None = None,
    overlay_has_video_flags: list[bool] | None = None,
) -> list[str]:
    out_w, out_h = _resolution_dims(export_settings.resolution, export_settings.aspect_ratio)
    fps = export_settings.fps
    overlay_inputs = list(overlay_inputs or [])
    if overlay_has_video_flags is None:
        overlay_has_video_flags = [True for _ in overlay_inputs]
    else:
        overlay_has_video_flags = list(overlay_has_video_flags)

    if not clip_inputs:
        raise ValueError("No video clips in timeline")
    if len(clip_inputs) != len(clip_has_audio_flags):
        raise ValueError("clip_has_audio_flags length mismatch")
    if len(overlay_inputs) != len(overlay_has_video_flags):
        raise ValueError("overlay_has_video_flags length mismatch")
    if len(bg_audio_inputs) != len(bg_has_audio_flags):
        raise ValueError("bg_has_audio_flags length mismatch")

    clip_pairs = sorted(zip(clip_inputs, clip_has_audio_flags, strict=True), key=lambda item: item[0][0].timeline_start_sec)
    clip_inputs = [pair[0] for pair in clip_pairs]
    clip_has_audio_flags = [pair[1] for pair in clip_pairs]
    overlay_pairs = sorted(
        zip(overlay_inputs, overlay_has_video_flags, strict=True),
        key=lambda item: item[0][0].timeline_start_sec,
    )
    overlay_inputs = [pair[0] for pair in overlay_pairs]
    overlay_has_video_flags = [pair[1] for pair in overlay_pairs]
    bg_pairs = sorted(zip(bg_audio_inputs, bg_has_audio_flags, strict=True), key=lambda item: item[0][0].timeline_start_sec)
    bg_audio_inputs = [pair[0] for pair in bg_pairs]
    bg_has_audio_flags = [pair[1] for pair in bg_pairs]
    ordered_clips = [clip for clip, _ in clip_inputs]
    clip_starts, transition_durations = _timeline_layout(ordered_clips)
    render_starts = {clip.id: clip_starts[idx] for idx, clip in enumerate(ordered_clips)}

    cmd = [settings.ffmpeg_bin, "-y"]
    for clip, src in clip_inputs:
        cmd.extend(
            [
                "-ss",
                f"{clip.start_sec}",
                "-to",
                f"{clip.end_sec}",
                "-i",
                src,
            ]
        )
    for clip, src in overlay_inputs:
        cmd.extend(
            [
                "-ss",
                f"{clip.start_sec}",
                "-to",
                f"{clip.end_sec}",
                "-i",
                src,
            ]
        )
    for clip, src in bg_audio_inputs:
        cmd.extend(
            [
                "-ss",
                f"{clip.start_sec}",
                "-to",
                f"{clip.end_sec}",
                "-i",
                src,
            ]
        )

    filter_parts: list[str] = []
    # Auto framing is deliberately opt-in. The cover path preserves aspect
    # ratio, then centre-crops the excess width (or height); without it, clips
    # retain their full image inside the requested canvas. It is never used for
    # a landscape render, even if a client sends auto_frame=true.
    main_cover_output = (
        export_settings.aspect_ratio == "9:16" and export_settings.auto_frame
    )
    for idx, (clip, _src) in enumerate(clip_inputs):
        vf = _video_filters_for_clip(
            clip,
            out_w,
            out_h,
            fps,
            cover_output=main_cover_output,
            apply_auto_frame_crop=main_cover_output,
        )
        filter_parts.append(f"[{idx}:v]{vf}[v{idx}]")
        duration = max(_clip_duration(clip), 0.1)
        if clip_has_audio_flags[idx]:
            af = f"[{idx}:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS"
            if clip.speed != 1:
                af += f",{_atempo_chain(clip.speed)}"
            af += ",aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
            af += f",volume='{_volume_expression(clip)}'"
            if clip.audio.fade_in_sec > 0:
                af += f",afade=t=in:st=0:d={clip.audio.fade_in_sec:.3f}"
            if clip.audio.fade_out_sec > 0:
                fade_start = max(duration - clip.audio.fade_out_sec, 0.0)
                af += f",afade=t=out:st={fade_start:.3f}:d={clip.audio.fade_out_sec:.3f}"
            af += f"[va{idx}]"
        else:
            af = f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.3f},aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[va{idx}]"
        filter_parts.append(af)

    if len(clip_inputs) == 1:
        filter_parts.append("[v0]null[vmain]")
        filter_parts.append("[va0]anull[amain]")
    else:
        previous_video = "v0"
        previous_audio = "va0"
        current_end = _clip_duration(ordered_clips[0])
        for idx in range(1, len(clip_inputs)):
            duration = transition_durations[idx]
            current_video = f"v{idx}"
            current_audio = f"va{idx}"
            next_video = f"vx{idx}"
            next_audio = f"ax{idx}"
            if duration > 0.03:
                transition = _xfade_transition(ordered_clips[idx].transition.type if ordered_clips[idx].transition else None)
                offset = max(current_end - duration, 0.0)
                filter_parts.append(
                    f"[{previous_video}][{current_video}]xfade=transition={transition}:duration={duration:.3f}:offset={offset:.3f}[{next_video}]"
                )
                filter_parts.append(
                    f"[{previous_audio}][{current_audio}]acrossfade=d={duration:.3f}:c1=tri:c2=tri[{next_audio}]"
                )
                current_end = current_end + _clip_duration(ordered_clips[idx]) - duration
            else:
                filter_parts.append(
                    f"[{previous_video}][{previous_audio}][{current_video}][{current_audio}]concat=n=2:v=1:a=1[{next_video}][{next_audio}]"
                )
                current_end = current_end + _clip_duration(ordered_clips[idx])
            previous_video = next_video
            previous_audio = next_audio
        filter_parts.append(f"[{previous_video}]null[vmain]")
        filter_parts.append(f"[{previous_audio}]anull[amain]")

    last_video_stream = "vmain"
    if overlay_inputs:
        overlay_base_index = len(clip_inputs)
        for idx, (clip, _src) in enumerate(overlay_inputs):
            if not overlay_has_video_flags[idx]:
                continue
            source_stream_index = overlay_base_index + idx
            overlay_stream = f"ov{idx}"
            vf = _video_filters_for_clip(
                clip,
                out_w,
                out_h,
                fps,
                cover_output=True,
            )
            filter_parts.append(f"[{source_stream_index}:v]{vf}[{overlay_stream}]")

            opacity = _float(clip.broll_opacity, 0.0, 1.0)
            if opacity < 0.999:
                mixed_stream = f"ovm{idx}"
                filter_parts.append(
                    f"[{overlay_stream}]format=rgba,colorchannelmixer=aa={opacity:.3f}[{mixed_stream}]"
                )
                overlay_stream = mixed_stream

            start = max(0.0, float(clip.timeline_start_sec))
            end = start + max(_clip_duration(clip), 0.1)
            next_stream = f"vov{idx}"
            overlay_enable = _escape_drawtext_expr(f"between(t,{start:.3f},{end:.3f})")
            filter_parts.append(
                f"[{last_video_stream}][{overlay_stream}]"
                f"overlay=(W-w)/2:(H-h)/2:enable='{overlay_enable}'"
                f"[{next_stream}]"
            )
            last_video_stream = next_stream

    text_overlays = _collect_text_overlays([clip for clip, _ in clip_inputs], render_starts)
    text_overlays = sorted(text_overlays, key=lambda item: float(item["start"]))
    # Use a SINGLE ASS subtitle file instead of N chained drawtext filters.
    # This is dramatically faster (one pass vs N passes per frame) and avoids OOM
    # kills on longer videos with many captions.
    _ass_subtitle_path: str | None = None
    if text_overlays:
        _ass_subtitle_path = _build_ass_subtitle_file(text_overlays, out_w, out_h)
        src = last_video_stream
        dst = "vtxt_ass"
        # Escape path for ffmpeg filter string (backslashes and colons)
        esc_path = _ass_subtitle_path.replace("\\", "\\\\").replace(":", "\\:")
        filter_parts.append(f"[{src}]ass='{esc_path}':fontsdir=/usr/share/fonts:shaping=complex[{dst}]")
        last_video_stream = dst

    has_audio = True
    if bg_audio_inputs:
        base_index = len(clip_inputs) + len(overlay_inputs)
        mix_parts = ["[amain]"]
        for offset, (clip, _src) in enumerate(bg_audio_inputs):
            stream_idx = base_index + offset
            duration = max(_clip_duration(clip), 0.1)
            label = f"bg{offset}"
            if bg_has_audio_flags[offset]:
                chain = f"[{stream_idx}:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS"
            else:
                chain = f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.3f}"
            if clip.speed != 1:
                chain += f",{_atempo_chain(clip.speed)}"
            chain += ",aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
            chain += f",volume='{_volume_expression(clip)}'"
            if clip.audio.fade_in_sec > 0:
                chain += f",afade=t=in:st=0:d={clip.audio.fade_in_sec:.3f}"
            if clip.audio.fade_out_sec > 0:
                fade_start = max(duration - clip.audio.fade_out_sec, 0.0)
                chain += f",afade=t=out:st={fade_start:.3f}:d={clip.audio.fade_out_sec:.3f}"
            if clip.timeline_start_sec > 0:
                delay_ms = int(round(clip.timeline_start_sec * 1000))
                chain += f",adelay={delay_ms}|{delay_ms}"
            chain += f"[{label}]"
            filter_parts.append(chain)
            mix_parts.append(f"[{label}]")
        # Normalize mixed tracks to avoid clipping/distortion when multiple
        # sources overlap (voice + music + overlays).
        filter_parts.append(f"{''.join(mix_parts)}amix=inputs={len(mix_parts)}:duration=longest:normalize=1[aout]")
    else:
        filter_parts.append("[amain]anull[aout]")

    filter_complex = ";".join(filter_parts)
    if len(filter_complex) > _MAX_INLINE_FILTER_COMPLEX_CHARS:
        tmp_dir = Path(settings.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"ffmpeg-filter-{Path(output_path).stem[:24]}-"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            prefix=prefix,
            dir=tmp_dir,
            delete=False,
        ) as filter_file:
            filter_file.write(filter_complex)
            filter_script_path = filter_file.name
        cmd.extend(["-filter_complex_script", filter_script_path])
    else:
        cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(["-map", f"[{last_video_stream}]"])
    if has_audio:
        cmd.extend(["-map", "[aout]"])
    cmd.extend(["-r", str(fps)])
    if export_settings.bitrate:
        cmd.extend(["-b:v", export_settings.bitrate])
    if export_settings.format == "webm":
        cmd.extend(["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p"])
        cmd.extend(["-crf", str(_quality_to_crf(export_settings.quality))])
        if has_audio:
            cmd.extend(["-c:a", "libopus", "-b:a", "160k"])
        else:
            cmd.extend(["-an"])
    else:
        video_encoder = _resolve_h264_video_encoder()
        if video_encoder == "h264_nvenc":
            cmd.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    _quality_to_nvenc_preset(export_settings.quality),
                    "-cq",
                    str(_quality_to_nvenc_cq(export_settings.quality)),
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        else:
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    _quality_to_x264_preset(export_settings.quality),
                    "-crf",
                    str(_quality_to_crf(export_settings.quality)),
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        if has_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-an"])
        # Make MP4 previews seekable immediately in browser players.
        cmd.extend(["-movflags", "+faststart"])
    # Only apply -shortest when there are no short B-roll overlay inputs,
    # as their short durations could prematurely truncate the output.
    if not overlay_inputs:
        cmd.append("-shortest")
    cmd.append(output_path)
    logger.debug("FFmpeg command: %s", " ".join(shlex.quote(p) for p in cmd))
    return cmd


def _parse_ffmpeg_out_time_seconds(progress_fields: dict[str, str]) -> float | None:
    out_time = progress_fields.get("out_time")
    if out_time:
        parts = out_time.strip().split(":")
        if len(parts) == 3:
            try:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return (hours * 3600.0) + (minutes * 60.0) + seconds
            except ValueError:
                pass

    for key in ("out_time_us", "out_time_ms"):
        raw_value = progress_fields.get(key)
        if not raw_value:
            continue
        try:
            return float(raw_value) / 1_000_000.0
        except ValueError:
            continue

    return None


def run_ffmpeg(
    command: list[str],
    *,
    duration_sec: float | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    # Memory check before FFmpeg to prevent OOM kills
    min_available_mb = int(os.environ.get("RENDER_MIN_AVAILABLE_MEMORY_MB", "500"))
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        if available_mb < min_available_mb:
            raise RuntimeError(
                f"Insufficient memory for render: {available_mb:.0f}MB available, "
                f"need at least {min_available_mb}MB. "
                "Close other applications or wait for other renders to finish."
            )
    except ImportError:
        logger.debug("psutil not installed; skipping render memory pre-check")
    except Exception as exc:
        logger.warning("Render memory pre-check failed; continuing anyway: %s", exc)
    
    # Garbage collect before FFmpeg to free any lingering memory
    import gc
    gc.collect()
    
    filter_script_paths: list[Path] = []
    ass_paths: list[Path] = []
    for idx, part in enumerate(command):
        if part == "-filter_complex_script" and idx + 1 < len(command):
            filter_script_paths.append(Path(command[idx + 1]))
        # Detect temp ASS files embedded in -filter_complex values
        if part == "-filter_complex" and idx + 1 < len(command):
            fc = command[idx + 1]
            import re as _re
            for match in _re.finditer(r"(?:subtitles|ass)='([^']+\.ass)'", fc):
                ass_paths.append(Path(match.group(1).replace("\\:", ":").replace("\\\\", "\\")))
    try:
        ffmpeg_command = command[:-1] + ["-progress", "pipe:1", "-nostats", command[-1]]
        process = subprocess.Popen(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        formatted = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(
            f"ffmpeg invocation failed: {exc}\n"
            f"command: {formatted}"
        ) from exc

    stderr_lines: list[str] = []

    def collect_stderr() -> None:
        if not process.stderr:
            return
        for line in process.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=collect_stderr, name="ffmpeg-stderr", daemon=True)
    stderr_thread.start()

    progress_fields: dict[str, str] = {}
    last_fraction = -1.0
    try:
        if process.stdout:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                progress_fields[key] = value
                if not progress_callback or not duration_sec or duration_sec <= 0:
                    continue
                rendered_sec = _parse_ffmpeg_out_time_seconds(progress_fields)
                if rendered_sec is None:
                    continue
                normalized = max(0.0, min(1.0, rendered_sec / duration_sec))
                if normalized > last_fraction:
                    last_fraction = normalized
                    progress_callback(normalized)
        returncode = process.wait()
    finally:
        if process.stdout:
            process.stdout.close()
        stderr_thread.join(timeout=1.0)
        if process.stderr:
            process.stderr.close()
        for path in filter_script_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # Best effort cleanup for temp filter scripts.
                pass
        for path in ass_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    if progress_callback and duration_sec and duration_sec > 0:
        progress_callback(1.0)

    if returncode != 0:
        formatted = " ".join(shlex.quote(part) for part in command)
        # Detect if ffmpeg was killed by a signal (e.g. SIGINT from server restart)
        stderr_output = "".join(stderr_lines)
        stderr_lower = stderr_output.lower()
        if returncode in (-2, 255) or "received signal 2" in stderr_lower:
            raise RuntimeError(
                f"ffmpeg was interrupted (signal 2 / SIGINT) — likely caused by a server restart. "
                f"Please retry the render.\n"
                f"command: {formatted}\n"
                f"stderr: {stderr_output.strip()}"
            )
        if returncode in (-9, 137):
            raise RuntimeError(
                "ffmpeg was killed by the OS (signal 9 / likely out-of-memory). "
                "Try a simpler caption render (fewer caption blocks) or rerun after pending renders finish.\n"
                f"command: {formatted}\n"
                f"stderr: {stderr_output.strip()}"
            )
        raise RuntimeError(
            f"ffmpeg failed ({returncode})\n"
            f"command: {formatted}\n"
            f"stderr: {stderr_output.strip()}"
        )


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
