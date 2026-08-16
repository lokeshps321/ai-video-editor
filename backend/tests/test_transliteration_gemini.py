"""Gemini-backed transliteration.

These pin the failure modes that made Gemini silently dead in production:
  * the SDK package was never installed (bare `except ImportError` -> fallback),
  * markdown-wrapped answers were rejected before they were cleaned,
  * a single free-tier 429 demoted the whole transcript to rule-based output.

All tests stub the network - nothing here calls the real API.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/prompt_video_editor_test.db")
os.environ.setdefault("UPLOAD_DIR", "/tmp/prompt_video_editor_uploads")
os.environ.setdefault("RENDER_DIR", "/tmp/prompt_video_editor_renders")
os.environ.setdefault("TMP_DIR", "/tmp/prompt_video_editor_tmp")

from app import transliteration_service as ts


class _FakeResponse:
    def __init__(self, text: str | None):
        self.text = text


class _FakeModels:
    def __init__(self, script):
        self._script = script
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._script(len(self.calls))


class _FakeClient:
    def __init__(self, script):
        self.models = _FakeModels(script)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Give each test its own cache dir and a key, and drop the client cache."""
    monkeypatch.setattr(ts, "TRANSLITERATION_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _clear_client_cache()
    yield
    # Tests replace _get_genai_client with a plain stub, so the attribute may
    # no longer be the lru_cache-wrapped original by teardown time.
    _clear_client_cache()


def _clear_client_cache() -> None:
    clear = getattr(ts._get_genai_client, "cache_clear", None)
    if clear is not None:
        clear()


def _install_client(monkeypatch, script) -> _FakeClient:
    client = _FakeClient(script)
    monkeypatch.setattr(ts, "_get_genai_client", lambda _key: client)
    return client


# ---------------------------------------------------------------------------
# The SDK must be the maintained one
# ---------------------------------------------------------------------------


def test_uses_the_maintained_google_genai_sdk():
    """`google-generativeai` is retired; importing it is the old dead path."""
    from google import genai  # noqa: F401

    source = (
        ts.__file__.replace(".pyc", ".py")
    )
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "import google.generativeai" not in text


def test_client_is_cached_not_rebuilt_per_call():
    """The client is built once per key; rebuilding it per word would be slow."""
    assert hasattr(ts._get_genai_client, "cache_clear"), "should be lru_cache'd"


# ---------------------------------------------------------------------------
# Markdown-wrapped answers must be cleaned BEFORE the ASCII check
# ---------------------------------------------------------------------------


def test_markdown_wrapped_answer_is_cleaned_not_rejected(monkeypatch):
    """Real observed failure: the model answered with a bulleted list that
    echoed the Tamil source, so the ASCII check rejected a good answer."""
    messy = (
        "Here is the transliteration:\n"
        "**Text:**\n"
        '*   "என்"\n'
        "en uyirai unnum thaen sudarae\n"
    )
    _install_client(monkeypatch, lambda _n: _FakeResponse(messy))

    result = ts._transliterate_with_llm("என் உயிரை உண்ணும் தேன் சுடரே", "tamil")

    assert result == "en uyirai unnum thaen sudarae"


def test_clean_answer_passes_through(monkeypatch):
    _install_client(monkeypatch, lambda _n: _FakeResponse("nee sigadere"))

    assert ts._transliterate_with_llm("ನೀ ಸಿಗದೆರೆ", "kannada") == "nee sigadere"


def test_genuinely_non_ascii_answer_is_still_rejected(monkeypatch):
    """The ASCII guard must still catch a model that returned raw script."""
    _install_client(monkeypatch, lambda _n: _FakeResponse("என் உயிரை உண்ணும்"))

    assert ts._transliterate_with_llm("என் உயிரை உண்ணும்", "tamil") is None


def test_empty_response_is_rejected(monkeypatch):
    _install_client(monkeypatch, lambda _n: _FakeResponse(None))

    assert ts._transliterate_with_llm("என்", "tamil") is None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(ts.time, "sleep", lambda _s: None)

    def script(call_number: int):
        if call_number == 1:
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED {'retryDelay': '11s'}"
            )
        return _FakeResponse("en uyirai")

    client = _install_client(monkeypatch, script)

    assert ts._transliterate_with_llm("என் உயிரை", "tamil") == "en uyirai"
    assert len(client.models.calls) == 2, "should have retried once"


def test_rate_limit_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(ts.time, "sleep", lambda _s: None)

    def script(_n: int):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    client = _install_client(monkeypatch, script)

    assert ts._transliterate_with_llm("என்", "tamil") is None
    assert len(client.models.calls) == ts._GEMINI_MAX_RETRIES + 1


def test_non_rate_limit_error_is_not_retried(monkeypatch):
    """A bad key or bad request fails the same way every time -- retrying it
    only delays the caller's fallback."""
    monkeypatch.setattr(ts.time, "sleep", lambda _s: None)

    def script(_n: int):
        raise RuntimeError("400 INVALID_ARGUMENT: bad model name")

    client = _install_client(monkeypatch, script)

    assert ts._transliterate_with_llm("என்", "tamil") is None
    assert len(client.models.calls) == 1, "must not retry a non-429 failure"


def test_retry_delay_prefers_server_hint():
    exc = RuntimeError("429 RESOURCE_EXHAUSTED {'retryDelay': '11s'}")

    assert ts._rate_limit_retry_delay(exc) == pytest.approx(12.0)


def test_retry_delay_is_capped(monkeypatch):
    monkeypatch.setattr(ts, "_GEMINI_MAX_RETRY_WAIT_SEC", 3.0)
    exc = RuntimeError("429 RESOURCE_EXHAUSTED {'retryDelay': '600s'}")

    assert ts._rate_limit_retry_delay(exc) == pytest.approx(3.0)


def test_rate_limit_detection():
    assert ts._is_rate_limit_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert ts._is_rate_limit_error(RuntimeError("RESOURCE_EXHAUSTED"))
    assert not ts._is_rate_limit_error(RuntimeError("500 internal"))


# ---------------------------------------------------------------------------
# Batch/word path - what the transcript UI actually calls
# ---------------------------------------------------------------------------


def test_batch_path_maps_words_and_keeps_originals(monkeypatch):
    _install_client(monkeypatch, lambda _n: _FakeResponse("en uyirai unnum"))
    words = [
        {"id": "w0", "text": "என்"},
        {"id": "w1", "text": "உயிரை"},
        {"id": "w2", "text": "உண்ணும்"},
    ]

    out = ts._transliterate_words_with_llm(words, "tamil")

    assert out is not None
    assert [w["text"] for w in out] == ["en", "uyirai", "unnum"]
    assert [w["original_text"] for w in out] == ["என்", "உயிரை", "உண்ணும்"]
    assert [w["id"] for w in out] == ["w0", "w1", "w2"]


def test_batch_path_without_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert ts._transliterate_words_with_llm([{"id": "w0", "text": "என்"}], "tamil") is None


def test_single_path_without_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert ts._transliterate_with_llm("என்", "tamil") is None


def test_batch_retries_once_when_model_deduplicates_repeats(monkeypatch):
    """Songs repeat lines; the model tends to collapse them. One corrective
    retry should recover the chunk instead of demoting it to the library."""
    def script(call_number: int):
        if call_number == 1:
            # Deduplicated: 2 words for a 4-word input.
            return _FakeResponse("thaen sudarae")
        return _FakeResponse("thaen sudarae thaen sudarae")

    client = _install_client(monkeypatch, script)
    words = [
        {"id": "w0", "text": "தேன்"},
        {"id": "w1", "text": "சுடரே"},
        {"id": "w2", "text": "தேன்"},
        {"id": "w3", "text": "சுடரே"},
    ]

    out = ts._transliterate_words_with_llm(words, "tamil")

    assert out is not None
    assert [w["text"] for w in out] == ["thaen", "sudarae", "thaen", "sudarae"]
    assert len(client.models.calls) == 2, "should have issued one corrective retry"


def test_batch_falls_back_to_library_when_retry_also_mismatches(monkeypatch):
    def script(_n: int):
        return _FakeResponse("only two")

    client = _install_client(monkeypatch, script)
    words = [
        {"id": "w0", "text": "தேன்"},
        {"id": "w1", "text": "சுடரே"},
        {"id": "w2", "text": "தேன்"},
    ]

    out = ts._transliterate_words_with_llm(words, "tamil")

    assert out is not None, "must still return words, via the library fallback"
    assert len(out) == 3
    assert len(client.models.calls) == 2, "one attempt + one corrective retry"
    # Library output, not the bad LLM answer
    assert [w["text"] for w in out] != ["only", "two"]


# ---------------------------------------------------------------------------
# Thinking models must not eat the output budget
# ---------------------------------------------------------------------------


def test_generation_config_disables_thinking():
    """gemini-3.5-flash spent 244 of a 256-token budget on reasoning and
    returned a truncated 8-token answer. Transliteration needs no reasoning."""
    from google.genai import types

    config = ts._build_generation_config(types, temperature=0.1, output_tokens=512)

    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 0


def test_generation_config_survives_sdk_without_thinking_support():
    """Older SDKs / non-thinking models have no ThinkingConfig; don't crash."""

    class _TypesWithoutThinking:
        @staticmethod
        def GenerateContentConfig(**kwargs):
            assert "thinking_config" not in kwargs
            return kwargs

    config = ts._build_generation_config(
        _TypesWithoutThinking, temperature=0.1, output_tokens=512
    )

    assert config["max_output_tokens"] == 512


def test_output_token_budget_has_headroom():
    """The old max(256, len*3) floor was too small once thinking was counted."""
    assert ts._transliteration_output_tokens("x") >= 512
    long_text = "என் உயிரை உண்ணும் தேன் சுடரே " * 20
    assert ts._transliteration_output_tokens(long_text) > len(long_text)


def test_batch_call_passes_thinking_disabled_config(monkeypatch):
    _install_client(monkeypatch, lambda _n: _FakeResponse("en uyirai"))
    client = ts._get_genai_client("k")

    ts._transliterate_words_with_llm(
        [{"id": "w0", "text": "என்"}, {"id": "w1", "text": "உயிரை"}], "tamil"
    )

    sent = client.models.calls[0]["config"]
    assert sent.thinking_config.thinking_budget == 0


# ---------------------------------------------------------------------------
# IndicXlit (offline neural transliteration) - the tier between Gemini and
# the rule-based library. Stubs the engine; never loads the real model in
# unit tests (conftest.py also disables it globally as a second safety net).
# ---------------------------------------------------------------------------


class _FakeXlitEngine:
    def __init__(self, script=None):
        self._script = script or (lambda word, lang_code: [word.lower()])
        self.calls: list[tuple[str, str]] = []

    def translit_word(self, word, lang_code, topk=1):
        self.calls.append((word, lang_code))
        return self._script(word, lang_code)


def test_indicxlit_disabled_by_default_in_tests():
    """conftest.py forces this off so tests never touch the real model."""
    assert ts.USE_INDICXLIT is False


def test_indicxlit_batch_preserves_word_count(monkeypatch):
    monkeypatch.setattr(ts, "USE_INDICXLIT", True)
    engine = _FakeXlitEngine(lambda w, lc: [f"{w}-{lc}"])
    monkeypatch.setattr(ts, "_indicxlit_engine", lambda: engine)
    words = [
        {"id": "w0", "text": "தேன்"},
        {"id": "w1", "text": "சுடரே"},
    ]

    out = ts._transliterate_words_with_indicxlit(words, "tamil")

    assert out is not None
    assert [w["text"] for w in out] == ["தேன்-ta", "சுடரே-ta"]
    assert [w["original_text"] for w in out] == ["தேன்", "சுடரே"]
    assert engine.calls == [("தேன்", "ta"), ("சுடரே", "ta")]


def test_indicxlit_maps_script_names_to_language_codes(monkeypatch):
    monkeypatch.setattr(ts, "USE_INDICXLIT", True)
    engine = _FakeXlitEngine(lambda w, lc: [lc])
    monkeypatch.setattr(ts, "_indicxlit_engine", lambda: engine)

    cases = {
        "kannada": "kn", "devanagari": "hi", "tamil": "ta", "telugu": "te",
        "malayalam": "ml", "bengali": "bn", "gujarati": "gu",
        "punjabi": "pa", "odia": "or",
    }
    for script, expected_code in cases.items():
        out = ts._transliterate_words_with_indicxlit(
            [{"id": "w0", "text": "x"}], script
        )
        assert out[0]["text"] == expected_code, script


def test_indicxlit_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(ts, "USE_INDICXLIT", False)

    assert ts._transliterate_words_with_indicxlit(
        [{"id": "w0", "text": "x"}], "tamil"
    ) is None


def test_indicxlit_returns_none_for_unsupported_script(monkeypatch):
    monkeypatch.setattr(ts, "USE_INDICXLIT", True)

    assert ts._transliterate_words_with_indicxlit(
        [{"id": "w0", "text": "x"}], "arabic"
    ) is None


def test_indicxlit_returns_none_when_engine_unavailable(monkeypatch):
    monkeypatch.setattr(ts, "USE_INDICXLIT", True)
    monkeypatch.setattr(ts, "_indicxlit_engine", lambda: None)

    assert ts._transliterate_words_with_indicxlit(
        [{"id": "w0", "text": "x"}], "tamil"
    ) is None


def test_indicxlit_falls_through_on_word_error(monkeypatch):
    monkeypatch.setattr(ts, "USE_INDICXLIT", True)

    def blow_up(word, lang_code, topk=1):
        raise RuntimeError("model exploded")

    engine = _FakeXlitEngine()
    engine.translit_word = blow_up
    monkeypatch.setattr(ts, "_indicxlit_engine", lambda: engine)

    assert ts._transliterate_words_with_indicxlit(
        [{"id": "w0", "text": "x"}], "tamil"
    ) is None


def test_indicxlit_keeps_original_text_when_no_candidates(monkeypatch):
    monkeypatch.setattr(ts, "USE_INDICXLIT", True)
    engine = _FakeXlitEngine(lambda w, lc: [])
    monkeypatch.setattr(ts, "_indicxlit_engine", lambda: engine)

    out = ts._transliterate_words_with_indicxlit(
        [{"id": "w0", "text": "தேன்"}], "tamil"
    )

    assert out[0]["text"] == "தேன்"


def test_indicxlit_single_text_path_preserves_word_order(monkeypatch):
    monkeypatch.setattr(ts, "USE_INDICXLIT", True)
    engine = _FakeXlitEngine(lambda w, lc: [w.lower() + "!"])
    monkeypatch.setattr(ts, "_indicxlit_engine", lambda: engine)

    result = ts._transliterate_with_indicxlit("தேன் சுடரே", "tamil")

    assert result == "தேன்! சுடரே!"


def test_transliterate_words_tries_indicxlit_between_gemini_and_library(
    monkeypatch,
):
    """Full fallback order: Gemini disabled -> IndicXlit -> (library not
    reached)."""
    monkeypatch.setattr(ts, "USE_LLM_TRANSLITERATION", False)
    monkeypatch.setattr(ts, "USE_INDICXLIT", True)
    engine = _FakeXlitEngine(lambda w, lc: [f"xlit-{w}"])
    monkeypatch.setattr(ts, "_indicxlit_engine", lambda: engine)
    monkeypatch.setattr(
        ts,
        "_transliterate_with_library",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("library should not be reached")
        ),
    )

    out = ts.transliterate_words([{"id": "w0", "text": "தேன்"}], "tamil")

    assert out[0]["text"] == "xlit-தேன்"


def test_transliterate_words_falls_back_to_library_when_indicxlit_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(ts, "USE_LLM_TRANSLITERATION", False)
    monkeypatch.setattr(ts, "USE_INDICXLIT", True)
    monkeypatch.setattr(ts, "_indicxlit_engine", lambda: None)

    out = ts.transliterate_words([{"id": "w0", "text": "தேன்"}], "tamil")

    assert out[0]["text"], "must still produce output via the library fallback"
    assert out[0]["text"] != "தேன்"


def test_engine_uses_multilingual_model_not_reloaded_per_language():
    """A single cached engine handles every language -- verified against the
    real package's constructor signature so a future upgrade can't silently
    change this without failing a test."""
    import inspect

    from ai4bharat.transliteration import XlitEngine

    sig = inspect.signature(XlitEngine)
    assert "src_script_type" in sig.parameters
