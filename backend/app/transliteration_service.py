"""
Transliteration service for Indian languages.

Converts Indian script text (Kannada, Hindi, Tamil, etc.) to Roman/Latin letters
so users who can't read the script can follow along phonetically - like YouTube
lyric videos showing "Nee sigadere nee navladare" instead of "ನೀ ಸಿಗದೆರೆ ನೀ ನವ್ಲದರೆ".

Uses a multi-tier approach:
1. Gemini batch transliteration (best quality, context-aware) — off by
   default; the free-tier key is capped at 5 requests/minute, so it is
   opt-in via TRANSLITERATE_USE_LLM=true rather than the default path.
2. IndicXlit (AI4Bharat) — offline neural model, near-Gemini quality, no
   rate limit. Default engine.
3. indic-transliteration library + post-processing (rule-based, offline)
4. Character map fallback (basic, always available)
"""

from __future__ import annotations

import json
import os
import re
import time
import hashlib
import logging
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Use Gemini for transliteration. Off by default: the free-tier key is capped
# at 5 requests/minute, which made this an unreliable primary path in
# practice. IndicXlit (below) gives near-Gemini quality with no rate limit,
# so it is the default engine; set TRANSLITERATE_USE_LLM=true to try Gemini
# first again (e.g. with a paid key) and fall back to IndicXlit only when it
# fails.
USE_LLM_TRANSLITERATION = _env_bool("TRANSLITERATE_USE_LLM", False)

# Gemini model for transliteration
GEMINI_TRANSLITERATION_MODEL = os.getenv(
    "TRANSLITERATION_GEMINI_MODEL", "gemini-3.5-flash"
)

# Cache directory for LLM results
TRANSLITERATION_CACHE_DIR = Path(
    os.getenv("TRANSLITERATION_CACHE_DIR", "./tmp/transliteration_cache")
)

# Try to import the professional transliteration library
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as indic_transliterate

    INDIC_LIB_AVAILABLE = True
except ImportError:
    INDIC_LIB_AVAILABLE = False
    logger.warning("indic-transliteration library not available, using basic fallback")


@lru_cache(maxsize=1)
def _get_genai_client(api_key: str):
    """Build the Gemini client once per API key.

    Uses `google-genai`, not the deprecated `google-generativeai` package --
    the old SDK stopped receiving updates and bug fixes.
    """
    from google import genai

    return genai.Client(api_key=api_key)


# Free-tier Gemini allows only a handful of requests per minute, and a
# transcript is transliterated in several chunks back to back. Without this the
# first 429 silently demotes the whole transcript to rule-based romanization.
_GEMINI_MAX_RETRIES = int(os.getenv("TRANSLITERATION_GEMINI_MAX_RETRIES", "2"))
_GEMINI_MAX_RETRY_WAIT_SEC = float(
    os.getenv("TRANSLITERATION_GEMINI_MAX_RETRY_WAIT_SEC", "15")
)


def _build_generation_config(types_mod, *, temperature: float, output_tokens: int):
    """Config for a transliteration call.

    Gemini 2.5/3.x are *thinking* models: reasoning tokens are drawn from
    `max_output_tokens`. Measured on gemini-3.5-flash, a 256-token budget was
    spent 244-on-thinking / 8-on-answer, so the reply came back truncated
    mid-word and the caller silently demoted to rule-based romanization.
    Transliteration is mechanical, so thinking is switched off outright and the
    whole budget goes to the answer.
    """
    kwargs = {
        "temperature": temperature,
        "max_output_tokens": output_tokens,
    }
    thinking_config_cls = getattr(types_mod, "ThinkingConfig", None)
    if thinking_config_cls is not None:
        try:
            kwargs["thinking_config"] = thinking_config_cls(thinking_budget=0)
        except Exception:  # noqa: BLE001 - older SDKs / models without thinking
            pass
    return types_mod.GenerateContentConfig(**kwargs)


def _transliteration_output_tokens(text: str) -> int:
    """Token budget for a transliteration answer, with headroom."""
    return max(512, len(text) * 4)


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _rate_limit_retry_delay(exc: Exception) -> float:
    """Seconds to wait, preferring the server's own retryDelay hint."""
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(exc))
    if match:
        try:
            return min(float(match.group(1)) + 1.0, _GEMINI_MAX_RETRY_WAIT_SEC)
        except ValueError:
            pass
    return min(5.0, _GEMINI_MAX_RETRY_WAIT_SEC)


def _generate_with_retry(client, **kwargs):
    """Call Gemini, retrying only on rate-limit errors.

    Any other failure is raised immediately -- retrying a bad request or a bad
    API key just wastes time before the caller's fallback runs.
    """
    attempt = 0
    while True:
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised below unless 429
            if attempt >= _GEMINI_MAX_RETRIES or not _is_rate_limit_error(exc):
                raise
            delay = _rate_limit_retry_delay(exc)
            attempt += 1
            logger.info(
                "Gemini rate-limited, retrying in %.1fs (attempt %d/%d)",
                delay,
                attempt,
                _GEMINI_MAX_RETRIES,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# IndicXlit: offline neural transliteration (AI4Bharat)
# ---------------------------------------------------------------------------
#
# Sits between Gemini and the rule-based library: near-Gemini "natural" output
# quality (trained on 26M real romanization pairs), but fully local -- no API
# key, no rate limit, no network dependency. Word-level by design, so it
# preserves exact word count/order, unlike Gemini (thinking-token truncation,
# free-tier 429s) or Sarvam's transliterate endpoint (reinterprets whole
# sentences and does not preserve word boundaries once given >1 word).

USE_INDICXLIT = _env_bool("TRANSLITERATE_USE_INDICXLIT", True)

# Our script names -> IndicXlit's 2/3-letter language codes.
_INDICXLIT_LANG_CODES: dict[str, str] = {
    "kannada": "kn",
    "devanagari": "hi",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "bengali": "bn",
    "gujarati": "gu",
    "punjabi": "pa",
    "odia": "or",
}


@lru_cache(maxsize=1)
def _indicxlit_engine():
    """Load the IndicXlit indic->English model once per process.

    Returns None (never raises) when the package/model isn't available, so
    every caller can treat "no engine" as just another fallback trigger.
    """
    try:
        import argparse

        import torch

        # fairseq (IndicXlit's dependency, unmaintained since ~2022) calls
        # torch.load() without weights_only=False. PyTorch >=2.6 defaults
        # weights_only=True and rejects argparse.Namespace in the checkpoint.
        # Allowlisting only this one known-safe stdlib class avoids disabling
        # pickle safety entirely.
        torch.serialization.add_safe_globals([argparse.Namespace])

        from ai4bharat.transliteration import XlitEngine
    except Exception as exc:  # noqa: BLE001
        logger.warning("IndicXlit unavailable, skipping: %s", exc)
        return None
    try:
        return XlitEngine(beam_width=4, src_script_type="indic")
    except Exception as exc:  # noqa: BLE001
        logger.warning("IndicXlit failed to load: %s", exc)
        return None


def _transliterate_words_with_indicxlit(
    words: list[dict], script: str
) -> list[dict] | None:
    """Word-level transliteration via IndicXlit. Always preserves word count."""
    if not USE_INDICXLIT or not words:
        return None
    lang_code = _INDICXLIT_LANG_CODES.get(script)
    if lang_code is None:
        return None
    engine = _indicxlit_engine()
    if engine is None:
        return None

    results: list[dict] = []
    for word in words:
        original_text = str(word.get("text", ""))
        new_word = dict(word)
        new_word["original_text"] = original_text
        token = original_text.strip()
        if not token:
            new_word["text"] = original_text
            results.append(new_word)
            continue
        try:
            candidates = engine.translit_word(token, lang_code=lang_code, topk=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("IndicXlit word transliteration failed: %s", exc)
            return None
        if not candidates:
            new_word["text"] = original_text
        else:
            new_word["text"] = str(candidates[0]).lower()
        results.append(new_word)
    return results


def _transliterate_with_indicxlit(text: str, script: str) -> str | None:
    """Single-string transliteration via IndicXlit, for the non-batch path."""
    if not text.strip():
        return None
    words = [{"id": str(i), "text": token} for i, token in enumerate(text.split())]
    result = _transliterate_words_with_indicxlit(words, script)
    if result is None:
        return None
    return " ".join(str(w["text"]) for w in result)


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

INDIC_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "kannada": ((0x0C80, 0x0CFF),),
    "devanagari": ((0x0900, 0x097F),),  # Hindi, Marathi, Sanskrit
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "bengali": ((0x0980, 0x09FF),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "punjabi": ((0x0A00, 0x0A7F),),  # Gurmukhi
    "odia": ((0x0B00, 0x0B7F),),
}

# Map our script names to indic-transliteration sanscript constants
SCRIPT_TO_SANSCRIPT: dict[str, Any] = {}
if INDIC_LIB_AVAILABLE:
    SCRIPT_TO_SANSCRIPT = {
        "kannada": sanscript.KANNADA,
        "devanagari": sanscript.DEVANAGARI,
        "tamil": sanscript.TAMIL,
        "telugu": sanscript.TELUGU,
        "malayalam": sanscript.MALAYALAM,
        "bengali": sanscript.BENGALI,
        "gujarati": sanscript.GUJARATI,
        "punjabi": sanscript.GURMUKHI,
        "odia": sanscript.ORIYA,
    }


def detect_indic_script(text: str) -> str | None:
    """Detect which Indic script is dominant in the text."""
    if not text:
        return None

    script_counts: dict[str, int] = {script: 0 for script in INDIC_SCRIPT_RANGES}

    for char in text:
        code = ord(char)
        for script, ranges in INDIC_SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code <= end:
                    script_counts[script] += 1
                    break

    max_script = max(script_counts, key=lambda k: script_counts[k])
    if script_counts[max_script] > 0:
        return max_script
    return None


def contains_indic_script(text: str) -> bool:
    """Check if text contains any Indic script characters."""
    for char in text:
        code = ord(char)
        for ranges in INDIC_SCRIPT_RANGES.values():
            for start, end in ranges:
                if start <= code <= end:
                    return True
    return False


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _get_cache_path(text: str, script: str) -> Path:
    """Get cache file path for a given text."""
    text_hash = hashlib.md5(f"{script}:{text}".encode(), usedforsecurity=False).hexdigest()
    TRANSLITERATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSLITERATION_CACHE_DIR / f"{text_hash}.json"


def _load_from_cache(text: str, script: str) -> str | None:
    """Load transliteration from cache if available."""
    cache_path = _get_cache_path(text, script)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return data.get("transliteration")
        except Exception as exc:
            logger.debug("Transliteration cache read failed: %s", exc)
    return None


def _save_to_cache(text: str, script: str, transliteration: str) -> None:
    """Save transliteration to cache."""
    try:
        cache_path = _get_cache_path(text, script)
        cache_path.write_text(
            json.dumps(
                {"original": text, "script": script, "transliteration": transliteration}
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Failed to cache transliteration: %s", e)


def _get_script_language_name(script: str) -> str:
    """Get human-readable language name for a script."""
    script_names = {
        "kannada": "Kannada",
        "devanagari": "Hindi",
        "tamil": "Tamil",
        "telugu": "Telugu",
        "malayalam": "Malayalam",
        "bengali": "Bengali",
        "gujarati": "Gujarati",
        "punjabi": "Punjabi",
        "odia": "Odia",
    }
    return script_names.get(script, script.title())


# ---------------------------------------------------------------------------
# LLM-based transliteration (best quality - YouTube-style)
# ---------------------------------------------------------------------------

_LLM_TRANSLITERATION_PROMPT = """You are a professional transliteration engine for {language} song lyrics and speech.

TASK: Convert the {language} text below into English letters (romanization).

CRITICAL RULES:
1. This is TRANSLITERATION, NOT translation. Keep the original sounds, just write them in English letters.
2. Write it EXACTLY how a native speaker would type it in WhatsApp/Instagram — casual, natural, easy to read and sing along.
3. Use common, intuitive spellings that anyone can pronounce:
   - Double vowels for long sounds: "aa", "ee", "oo" (not "ā", "ī", "ū")
   - "sh" for श/ಶ, "ch" for च/ಚ, "th" for थ/ಥ
   - No diacritics, no special characters — ONLY a-z letters
4. Keep the EXACT same number of words as the input. Each input word = exactly one output word.
5. Preserve word boundaries precisely — don't merge or split words.
6. Lowercase everything unless it's a proper name.
7. NEVER collapse or deduplicate repeated words. Song lyrics repeat lines on
   purpose. If a word or phrase appears 3 times in the input, it MUST appear 3
   times in the output, in the same positions.
   Example: "தேன் சுடரே தேன் சுடரே" (4 words) → "thaen sudarae thaen sudarae" (4 words),
   NOT "thaen sudarae" (2 words).

EXAMPLES of good romanized output:
- Hindi: "दिल" → "dil", "प्यार" → "pyaar", "मोहब्बत" → "mohabbat", "ख्वाहिश" → "khwahish"
- Kannada: "ನಾನು" → "naanu", "ಹೃದಯ" → "hrudaya", "ಬೆಳಕು" → "belaku"  
- Tamil: "காதல்" → "kaadhal", "வணக்கம்" → "vanakkam", "நன்றி" → "nandri"
- Telugu: "ప్రేమ" → "prema", "అందం" → "andam", "హృదయం" → "hrudayam"

INPUT ({num_words} words):
{text}

OUTPUT (exactly {num_words} romanized words, space-separated):"""


def _transliterate_with_llm(text: str, script: str) -> str | None:
    """
    Use Gemini API for high-quality, natural transliteration.
    This produces YouTube-style romanization that sounds natural when read aloud.
    """
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        from google.genai import types
    except ImportError:
        logger.warning("google-genai package missing, falling back to library transliteration")
        return None

    # Check cache first
    cached = _load_from_cache(text, script)
    if cached:
        return cached

    language_name = _get_script_language_name(script)

    prompt = f"""Transliterate the following {language_name} text to English letters (romanization).

IMPORTANT RULES:
1. This is TRANSLITERATION, not translation - keep the same sounds, just write in English letters
2. Make it natural and easy to read/sing along - like YouTube lyric videos
3. Use simple, common spelling (not scholarly notation)
4. Keep word boundaries exactly as in original
5. No diacritics or special characters - only basic English letters
6. Lowercase unless it's a name or start of line
7. Use double vowels for long sounds: aa, ee, oo, ii, uu

Examples of good transliteration style:
- "ನಾನು" → "naanu" (not "nānu" or "NAANU")
- "प्यार" → "pyaar" (not "pyāra" or "pyar")
- "காதல்" → "kaadhal" (not "kātaḷ")
- "దిల్" → "dil" (not "dil" - same but no diacritics)

Text to transliterate:
{text}

Return ONLY the transliterated text, nothing else."""

    try:
        client = _get_genai_client(api_key)
        response = _generate_with_retry(
            client,
            model=GEMINI_TRANSLITERATION_MODEL,
            contents=prompt,
            config=_build_generation_config(
                types,
                temperature=0.1,
                output_tokens=_transliteration_output_tokens(text),
            ),
        )

        # Clean up markdown/quotes/preamble BEFORE validating. The model
        # sometimes answers with a bulleted list that echoes the original
        # script; validating first rejected those as "not ASCII enough" even
        # though the cleaned line was a perfectly good romanization.
        result = _clean_llm_response((response.text or "").strip())

        # Basic validation - result should be mostly ASCII
        if result and sum(1 for c in result if ord(c) < 128) / len(result) > 0.9:
            # Cache the result
            _save_to_cache(text, script, result)
            return result

        logger.warning(
            "LLM transliteration returned unexpected result: %s", result[:100]
        )
        return None

    except Exception as e:
        logger.warning("LLM transliteration failed: %s", e)
        return None


def _clean_llm_response(text: str) -> str:
    """Clean up LLM response artifacts."""
    # Remove markdown quotes, backticks, etc.
    text = text.strip("`'\"")
    # Remove leading/trailing whitespace and newlines
    text = text.strip()
    # If multi-line, take first non-empty line
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        # Take the longest line (likely the actual transliteration)
        text = max(lines, key=len)
    return text


def _transliterate_words_with_llm(words: list[dict], script: str) -> list[dict] | None:
    """
    Transliterate multiple words using Gemini in a single call for context-aware results.
    
    Processes words in manageable chunks (40-60 words) for context while keeping
    token counts reasonable. This is the highest quality approach.
    """
    if not words:
        return None

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        from google.genai import types
    except ImportError:
        return None

    language_name = _get_script_language_name(script)

    # Process in chunks of 50 words for context
    chunk_size = 50
    all_results: list[dict] = []

    # Build the client once, outside the chunk loop
    client = _get_genai_client(api_key)

    for chunk_start in range(0, len(words), chunk_size):
        chunk = words[chunk_start : chunk_start + chunk_size]
        original_texts = [str(w.get("text", "")) for w in chunk]
        combined_text = " ".join(original_texts)

        # Check cache for this chunk
        cached = _load_from_cache(combined_text, script)
        if cached:
            cached_words = cached.split()
            if len(cached_words) == len(original_texts):
                for word, transliterated in zip(chunk, cached_words):
                    new_word = dict(word)
                    new_word["text"] = transliterated
                    new_word["original_text"] = str(word.get("text", ""))
                    all_results.append(new_word)
                continue

        prompt = _LLM_TRANSLITERATION_PROMPT.format(
            language=language_name,
            num_words=len(original_texts),
            text=combined_text,
        )

        try:
            response = _generate_with_retry(
                client,
                model=GEMINI_TRANSLITERATION_MODEL,
                contents=prompt,
                config=_build_generation_config(
                    types,
                    temperature=0.05,
                    output_tokens=_transliteration_output_tokens(combined_text),
                ),
            )

            result_text = _clean_llm_response((response.text or "").strip())
            result_words = result_text.split()

            # One corrective retry before demoting the chunk. The usual cause is
            # the model deduplicating a repeated lyric line, and naming the
            # actual counts back to it reliably fixes that.
            if len(result_words) != len(original_texts):
                logger.info(
                    "LLM returned %d words, expected %d; retrying with an "
                    "explicit count correction.",
                    len(result_words), len(original_texts),
                )
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous answer had {len(result_words)} words but the "
                    f"input has exactly {len(original_texts)} words. Do NOT merge, "
                    f"drop, or deduplicate repeated words. Output exactly "
                    f"{len(original_texts)} space-separated romanized words."
                )
                try:
                    retry_response = _generate_with_retry(
                        client,
                        model=GEMINI_TRANSLITERATION_MODEL,
                        contents=retry_prompt,
                        config=_build_generation_config(
                            types,
                            temperature=0.0,
                            output_tokens=_transliteration_output_tokens(
                                combined_text
                            ),
                        ),
                    )
                    retry_text = _clean_llm_response(
                        (retry_response.text or "").strip()
                    )
                    retry_words = retry_text.split()
                    if len(retry_words) == len(original_texts):
                        result_text, result_words = retry_text, retry_words
                except Exception as exc:  # noqa: BLE001 - fall through below
                    logger.warning("Transliteration count-retry failed: %s", exc)

            # Validate word count matches
            if len(result_words) != len(original_texts):
                logger.warning(
                    "LLM returned %d words, expected %d. "
                    "Falling back to per-word transliteration for this chunk.",
                    len(result_words), len(original_texts),
                )
                # Fall back to library for this chunk
                for word in chunk:
                    new_word = dict(word)
                    original_text = str(word.get("text", ""))
                    transliterated = transliterate_text(original_text, script)
                    new_word["text"] = transliterated
                    new_word["original_text"] = original_text
                    all_results.append(new_word)
                continue

            # Cache the combined result
            _save_to_cache(combined_text, script, result_text)

            # Build results
            for word, transliterated in zip(chunk, result_words):
                new_word = dict(word)
                # Normalize: lowercase, remove stray punctuation that LLM might add
                cleaned = transliterated.lower().strip(".,;:!?'\"")
                new_word["text"] = cleaned
                new_word["original_text"] = str(word.get("text", ""))
                all_results.append(new_word)

        except Exception as e:
            logger.warning("LLM batch transliteration failed for chunk: %s", e)
            # Fall back to library for this chunk
            for word in chunk:
                new_word = dict(word)
                original_text = str(word.get("text", ""))
                transliterated = transliterate_text(original_text, script)
                new_word["text"] = transliterated
                new_word["original_text"] = original_text
                all_results.append(new_word)

    return all_results if len(all_results) == len(words) else None


# ---------------------------------------------------------------------------
# Post-processing for readability
# ---------------------------------------------------------------------------


def _clean_transliteration_for_display(text: str) -> str:
    """
    Clean up ITRANS output to be more readable for general audiences.
    
    ITRANS uses special markers that scholars understand but look weird
    to regular users. This makes it more natural - like how people actually
    type in Roman script on WhatsApp/Instagram.
    """
    if not text:
        return text

    # ---- Phase 1: Replace scholarly/ITRANS markers with natural forms ----
    
    # Diacritical characters → simple equivalents
    diacritics = [
        ("ā", "aa"), ("ī", "ee"), ("ū", "oo"), ("ē", "e"), ("ō", "o"),
        ("ñ", "n"), ("ṃ", "m"), ("ṁ", "m"), ("ṅ", "n"), ("ṇ", "n"),
        ("ḥ", "h"), ("ś", "sh"), ("ṣ", "sh"), ("ṛ", "ri"), ("ḍ", "d"),
        ("ṭ", "t"), ("ḷ", "l"), ("è", "e"), ("ò", "o"),
        ("É", "E"), ("Ó", "O"),
    ]
    
    result = text
    for old, new in diacritics:
        result = result.replace(old, new)

    # ITRANS special sequences
    itrans_fixes = [
        ("RRI", "ri"), ("RR", "r"),
        ("~N", "n"), ("~n", "n"),
        (".h", ""), (".n", "n"), ("N^", "n"),
        ("chh", "ch"), ("thh", "th"),
    ]
    for old, new in itrans_fixes:
        result = result.replace(old, new)

    # Tamil residual characters
    tamil_chars = [
        ("ன", "n"), ("ற", "r"), ("ழ", "zh"), ("ள", "l"),
    ]
    for old, new in tamil_chars:
        result = result.replace(old, new)

    # ---- Phase 2: Handle ITRANS uppercase conventions ----
    
    # Retroflex consonants (ITRANS uppercase)
    result = result.replace("D", "d")
    result = result.replace("T", "t")
    result = result.replace("N", "n")
    result = result.replace("S", "sh")
    result = result.replace("Ch", "ch")
    result = result.replace("R", "r")

    # Anusvara M → m
    result = re.sub(r"M(?=[^aeiouAEIOU]|$)", "m", result)
    result = result.replace("M", "m")

    # Long vowels (ITRANS uppercase)
    result = result.replace("A", "aa")
    result = result.replace("I", "ee")
    result = result.replace("U", "oo")
    result = result.replace("E", "e")
    result = result.replace("O", "o")

    # ---- Phase 3: Clean up awkward patterns ----
    
    # Collapse excessive vowel repetitions
    result = re.sub(r"a{3,}", "aa", result)
    result = re.sub(r"e{3,}", "ee", result)
    result = re.sub(r"i{3,}", "ee", result)
    result = re.sub(r"o{3,}", "oo", result)
    result = re.sub(r"u{3,}", "oo", result)

    # Fix common awkward consonant clusters
    result = result.replace("ghgh", "kk")
    result = result.replace("thth", "th")

    # Remove orphaned virama/schwa markers that leaked through
    result = re.sub(r"[््]", "", result)

    # Clean double spaces
    result = re.sub(r"  +", " ", result)

    return result.strip()


def _capitalize_sentences(text: str) -> str:
    """Capitalize first letter of sentences for better readability."""
    if not text:
        return text

    result = re.sub(
        r"([.!?]\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
    )

    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    return result


# ---------------------------------------------------------------------------
# Core transliteration using indic-transliteration library
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2048)
def _transliterate_with_library(text: str, script: str) -> str:
    """
    High-quality transliteration using indic-transliteration library.
    Uses ITRANS scheme which is most readable for general audiences.
    """
    if not INDIC_LIB_AVAILABLE:
        return text

    source_script = SCRIPT_TO_SANSCRIPT.get(script)
    if source_script is None:
        return text

    try:
        result = indic_transliterate(text, source_script, sanscript.ITRANS)
        return result
    except Exception as e:
        logger.warning("Library transliteration failed for %s: %s", script, e)
        return text


@lru_cache(maxsize=8192)
def romanize_token_for_matching(token: str) -> str:
    """Deterministic, LLM-free romanization for fuzzy lyric matching.

    Returns lowercase ASCII text; returns the input unchanged when it
    contains no Indic script. Never calls the LLM tiers, so it is safe
    in matching hot paths.
    """
    text = str(token or "")
    if not text or not contains_indic_script(text):
        return text
    script = detect_indic_script(text)
    if script is not None:
        text = _transliterate_with_library(text, script)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def transliterate_text(text: str, script: str | None = None) -> str:
    """
    Transliterate Indian script text to Roman/Latin letters.

    Args:
        text: Text in Indian script
        script: Script name (kannada, devanagari, tamil, etc.)

    Returns:
        Romanized text that phonetically matches the original.
    """
    if not text:
        return text

    if script is None:
        script = detect_indic_script(text)

    if script is None:
        return text

    # IndicXlit: offline, near-Gemini quality, no rate limits.
    xlit_result = _transliterate_with_indicxlit(text, script)
    if xlit_result is not None:
        return xlit_result

    # Use library if available, with post-processing
    if INDIC_LIB_AVAILABLE and script in SCRIPT_TO_SANSCRIPT:
        raw_result = _transliterate_with_library(text, script)
        return _clean_transliteration_for_display(raw_result)

    # Fallback to basic mapping
    return _transliterate_fallback(text, script)


def transliterate_words(
    words: list[dict],
    script: str | None = None,
) -> list[dict]:
    """
    Transliterate a list of transcript words.
    
    Uses batch LLM transliteration for context-aware, natural romanization.
    Falls back to library-based transliteration if LLM is unavailable.

    Args:
        words: List of word dicts with 'text' field
        script: Script name or None for auto-detect

    Returns:
        New list with transliterated 'text' fields (other fields preserved).
        Original text is saved in 'original_text' field.
    """
    if not words:
        return words

    # Auto-detect script from first few words
    if script is None:
        sample_text = " ".join(str(w.get("text", "")) for w in words[:40])
        script = detect_indic_script(sample_text)

    if script is None:
        return words

    # Strategy 1: Try batch LLM transliteration (best quality, context-aware)
    if USE_LLM_TRANSLITERATION:
        llm_result = _transliterate_words_with_llm(words, script)
        if llm_result is not None:
            return llm_result

    # Strategy 2: IndicXlit -- offline, near-Gemini quality, no rate limits,
    # and word-level by design so it always preserves count/order.
    xlit_result = _transliterate_words_with_indicxlit(words, script)
    if xlit_result is not None:
        return xlit_result

    # Strategy 3: Per-word library transliteration with post-processing
    result = []
    for word in words:
        new_word = dict(word)
        original_text = str(word.get("text", ""))
        transliterated = transliterate_text(original_text, script)
        new_word["text"] = transliterated
        new_word["original_text"] = original_text
        result.append(new_word)

    return result


# ---------------------------------------------------------------------------
# Fallback basic transliteration (when library not available)
# ---------------------------------------------------------------------------

KANNADA_MAP_FALLBACK: dict[str, str] = {
    "ಅ": "a", "ಆ": "aa", "ಇ": "i", "ಈ": "ee", "ಉ": "u", "ಊ": "oo",
    "ಋ": "ru", "ಎ": "e", "ಏ": "ae", "ಐ": "ai", "ಒ": "o", "ಓ": "o", "ಔ": "au",
    "ಾ": "aa", "ಿ": "i", "ೀ": "ee", "ು": "u", "ೂ": "oo", "ೃ": "ru",
    "ೆ": "e", "ೇ": "ae", "ೈ": "ai", "ೊ": "o", "ೋ": "o", "ೌ": "au",
    "ಂ": "m", "ಃ": "h", "್": "",
    "ಕ": "ka", "ಖ": "kha", "ಗ": "ga", "ಘ": "gha", "ಙ": "nga",
    "ಚ": "cha", "ಛ": "chha", "ಜ": "ja", "ಝ": "jha", "ಞ": "nya",
    "ಟ": "ta", "ಠ": "tha", "ಡ": "da", "ಢ": "dha", "ಣ": "na",
    "ತ": "tha", "ಥ": "thha", "ದ": "da", "ಧ": "dha", "ನ": "na",
    "ಪ": "pa", "ಫ": "pha", "ಬ": "ba", "ಭ": "bha", "ಮ": "ma",
    "ಯ": "ya", "ರ": "ra", "ಲ": "la", "ವ": "va",
    "ಶ": "sha", "ಷ": "sha", "ಸ": "sa", "ಹ": "ha", "ಳ": "la",
}

DEVANAGARI_MAP_FALLBACK: dict[str, str] = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ं": "n", "ः": "h", "्": "",
    "क": "ka", "ख": "kha", "ग": "ga", "घ": "gha", "ङ": "nga",
    "च": "cha", "छ": "chha", "ज": "ja", "झ": "jha", "ञ": "nya",
    "ट": "ta", "ठ": "tha", "ड": "da", "ढ": "dha", "ण": "na",
    "त": "ta", "थ": "tha", "द": "da", "ध": "dha", "न": "na",
    "प": "pa", "फ": "pha", "ब": "ba", "भ": "bha", "म": "ma",
    "य": "ya", "र": "ra", "ल": "la", "व": "va",
    "श": "sha", "ष": "sha", "स": "sa", "ह": "ha",
}

TAMIL_MAP_FALLBACK: dict[str, str] = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo",
    "எ": "e", "ஏ": "ae", "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au",
    "ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo",
    "ெ": "e", "ே": "ae", "ை": "ai", "ொ": "o", "ோ": "o", "ௌ": "au",
    "ஂ": "m", "ஃ": "h", "்": "",
    "க": "ka", "ங": "nga", "ச": "sa", "ஞ": "nya",
    "ட": "da", "ண": "na", "த": "tha", "ந": "na", "ப": "pa", "ம": "ma",
    "ய": "ya", "ர": "ra", "ல": "la", "வ": "va",
    "ழ": "zha", "ள": "la", "ற": "ra", "ன": "na",
    "ஜ": "ja", "ஷ": "sha", "ஸ": "sa", "ஹ": "ha",
}

TELUGU_MAP_FALLBACK: dict[str, str] = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ee", "ఉ": "u", "ఊ": "oo",
    "ఋ": "ri", "ఎ": "e", "ఏ": "ae", "ఐ": "ai", "ఒ": "o", "ఓ": "o", "ఔ": "au",
    "ా": "aa", "ి": "i", "ీ": "ee", "ు": "u", "ూ": "oo", "ృ": "ri",
    "ె": "e", "ే": "ae", "ై": "ai", "ొ": "o", "ో": "o", "ౌ": "au",
    "ం": "m", "ః": "h", "్": "",
    "క": "ka", "ఖ": "kha", "గ": "ga", "ఘ": "gha", "ఙ": "nga",
    "చ": "cha", "ఛ": "chha", "జ": "ja", "ఝ": "jha", "ఞ": "nya",
    "ట": "ta", "ఠ": "tha", "డ": "da", "ఢ": "dha", "ణ": "na",
    "త": "ta", "థ": "tha", "ద": "da", "ధ": "dha", "న": "na",
    "ప": "pa", "ఫ": "pha", "బ": "ba", "భ": "bha", "మ": "ma",
    "య": "ya", "ర": "ra", "ల": "la", "వ": "va",
    "శ": "sha", "ష": "sha", "స": "sa", "హ": "ha", "ళ": "la",
}

FALLBACK_MAPS: dict[str, dict[str, str]] = {
    "kannada": KANNADA_MAP_FALLBACK,
    "devanagari": DEVANAGARI_MAP_FALLBACK,
    "tamil": TAMIL_MAP_FALLBACK,
    "telugu": TELUGU_MAP_FALLBACK,
}


def _transliterate_fallback(text: str, script: str) -> str:
    """Basic character-by-character transliteration as fallback."""
    char_map = FALLBACK_MAPS.get(script, {})
    if not char_map:
        return text

    result = []
    for char in text:
        if char in char_map:
            result.append(char_map[char])
        else:
            result.append(char)

    return "".join(result)
