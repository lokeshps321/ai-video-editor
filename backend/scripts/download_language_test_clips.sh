#!/usr/bin/env bash
# Download ~30s audio clips for live transcript language audit.
# Clips are stored under backend/tests/fixtures/language_clips/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/tests/fixtures/language_clips"
mkdir -p "$OUT"

download_clip() {
  local code="$1"
  local url="$2"
  local out="$OUT/${code}.wav"
  if [[ -f "$out" ]]; then
    echo "skip $code (exists)"
    return 0
  fi
  echo "download $code ..."
  yt-dlp -f "bestaudio/best" --extract-audio --audio-format wav \
    --postprocessor-args "ffmpeg:-ac 1 -ar 16000 -t 30" \
    -o "$OUT/${code}.%(ext)s" "$url"
  if [[ -f "$OUT/${code}.wav" ]]; then
    echo "ok $code"
  else
    echo "fail $code" >&2
    return 1
  fi
}

# Curated official / widely-available music samples (first 30s used).
download_clip "hi" "https://www.youtube.com/watch?v=Gz38C23UHPQ" || true  # Kesariya
download_clip "te" "https://www.youtube.com/watch?v=Zd7ieVUErO0" || true  # Butta Bomma
download_clip "ml" "https://www.youtube.com/watch?v=1--qqQqGJ_M" || true  # Malare
download_clip "mr" "https://www.youtube.com/watch?v=0NF7EHAFSIM" || true  # Zingaat (Marathi)
download_clip "bn" "https://www.youtube.com/watch?v=kQ2azj_nCHE" || true  # Tumpa Sona
download_clip "gu" "https://www.youtube.com/watch?v=9vJRopWd0-A" || true  # Chogada
download_clip "pa" "https://www.youtube.com/watch?v=4tywp83Xky8" || true  # Laembadgini
download_clip "ur" "https://www.youtube.com/watch?v=9vJRopWd0-A" || true  # placeholder reuse if needed

echo "Done. Clips in $OUT"
