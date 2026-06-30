from __future__ import annotations

DEFAULT_MUSIC_RETRY_PROMPT = (
    "Transcribe only clearly audible speech and sung lyrics verbatim in the original "
    "language. Preserve repeated chorus lines and ad-libs. If a word is unclear, omit "
    "it instead of guessing. Do not paraphrase. Do not translate. Do not add "
    "commentary, summaries, or intro/outro phrases that are not present in the audio."
)
_PROMPT_LEAKAGE_PHRASES: tuple[tuple[str, ...], ...] = (
    ("transcribe", "speech", "and", "sung", "lyrics"),
    ("transcribe", "speech", "and", "analysis"),
    ("preserve", "repeated", "chorus", "lines"),
    ("do", "not", "paraphrase"),
    ("do", "not", "translate"),
    ("sung", "lyrics"),
    ("ad", "libs"),
)
_PROMPT_LEAKAGE_SINGLETONS: set[str] = {"transcribe", "paraphrase", "translate"}

# ---------------------------------------------------------------------------
# Filler words (used by vibe auto-cut and hallucination heuristic)
# ---------------------------------------------------------------------------
FILLER_WORDS: set[str] = {
    "um",
    "uh",
    "uhm",
    "umm",
    "hmm",
    "hm",
    "ah",
    "er",
    "eh",
    "like",
    "basically",
    "literally",
    "actually",
    "right",
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "so yeah",
}

_HALLUCINATION_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "i'm",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "we",
    "you",
    # Common hallucination words — frequently appear in silence/music gaps
    "thank",
    "thanks",
    "watching",
    "listen",
    "listening",
    "bye",
    "goodbye",
    "subscribe",
    "subscribed",
    "end",
}

_TAIL_HALLUCINATION_PHRASES: set[tuple[str, ...]] = {
    ("thank", "you"),
    ("thank", "you", "so", "much"),
    ("thank", "you", "for", "watching"),
    ("thank", "you", "for", "listening"),
    ("thank", "you", "very", "much"),
    ("thanks", "for", "watching"),
    ("thanks", "for", "listening"),
    ("the", "end"),
    ("the", "end", "thank", "you"),
    ("bye", "bye"),
    ("good", "bye"),
    ("see", "you", "next", "time"),
    ("see", "you", "later"),
    ("see", "you", "in", "the", "next", "video"),
    ("that's", "all", "for", "today"),
    ("that's", "it", "for", "today"),
    ("please", "subscribe"),
    ("don't", "forget", "to", "subscribe"),
    ("like", "and", "subscribe"),
    ("like", "comment", "and", "subscribe"),
    ("if", "you", "enjoyed", "this", "video"),
}

# Head hallucination phrases - common YouTube intro/outro phrases that appear at START
_HEAD_HALLUCINATION_PHRASES: set[tuple[str, ...]] = {
    ("thank", "you"),
    ("thank", "you", "for", "watching"),
    ("thank", "you", "for", "listening"),
    ("thanks", "for", "watching"),
    ("thank", "you", "so", "much"),
    ("hey", "guys"),
    ("hey", "everyone"),
    ("hello", "everyone"),
    ("hi", "everyone"),
    ("hi", "guys"),
    ("what's", "up"),
    ("what's", "up", "guys"),
    ("welcome", "back"),
    ("welcome", "back", "everyone"),
    ("welcome", "to", "the", "channel"),
    ("subscribe",),
    ("please", "subscribe"),
    ("don't", "forget", "to", "subscribe"),
    ("like", "and", "subscribe"),
    ("like", "comment", "and", "subscribe"),
    ("good", "morning", "everyone"),
    ("good", "evening", "everyone"),
}

# Phrases that are almost certainly hallucinations *anywhere* in the transcript
# when they appear surrounded by sufficient silence gaps.
_ANYWHERE_HALLUCINATION_PHRASES: set[tuple[str, ...]] = {
    ("thank", "you", "for", "watching"),
    ("thank", "you", "for", "listening"),
    ("thank", "you", "very", "much", "for", "watching"),
    ("thanks", "for", "watching"),
    ("thanks", "for", "listening"),
    ("don't", "forget", "to", "subscribe"),
    ("please", "subscribe"),
    ("like", "and", "subscribe"),
    ("like", "comment", "and", "subscribe"),
    ("hit", "the", "subscribe", "button"),
    ("see", "you", "in", "the", "next", "video"),
    ("see", "you", "next", "time"),
    ("see", "you", "guys", "next", "time"),
    ("that's", "all", "for", "today"),
    ("that's", "it", "for", "today"),
    ("if", "you", "enjoyed", "this", "video"),
    ("thank", "you", "so", "much"),
}

# Single tokens that are almost always hallucinated when isolated inside silence gaps.
_HALLUCINATION_SINGLETONS_IN_GAPS: set[str] = {
    "subscribe",
    "subscribed",
}
