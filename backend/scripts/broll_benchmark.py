from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.broll_ai_service import CandidateRow, rerank_broll_candidates


def _candidate_row(payload: dict[str, Any]) -> CandidateRow:
    reason = payload.get("reason", {})
    if not isinstance(reason, dict):
        reason = {}
    return (
        str(payload.get("source_type") or "project_asset"),
        str(payload["asset_id"]) if payload.get("asset_id") else None,
        str(payload["source_url"]) if payload.get("source_url") else None,
        str(payload["source_label"]) if payload.get("source_label") else None,
        float(payload.get("score", 0.0)),
        reason,
    )


def _contains_positive_terms(row: CandidateRow, positive_terms: list[str]) -> bool:
    source_type, _asset_id, source_url, source_label, _score, reason = row
    haystack = " ".join(
        [
            source_type or "",
            source_label or "",
            source_url or "",
            " ".join(str(item) for item in reason.get("keyword_hits", []) if str(item).strip())
            if isinstance(reason.get("keyword_hits"), list)
            else "",
            str(reason.get("query", "")),
            str(reason.get("page_url", "")),
        ]
    ).lower()
    for term in positive_terms:
        normalized = term.strip().lower()
        if normalized and normalized in haystack:
            return True
    return False


def _evaluate(rows: list[CandidateRow], positive_terms: list[str]) -> tuple[float, float]:
    if not rows:
        return (0.0, 0.0)
    top1 = 1.0 if _contains_positive_terms(rows[0], positive_terms) else 0.0
    top3 = 1.0 if any(_contains_positive_terms(row, positive_terms) for row in rows[:3]) else 0.0
    return (top1, top3)


def run(dataset_path: Path) -> int:
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to read dataset: {exc}")
        return 2
    if not isinstance(payload, list):
        print("Dataset must be a JSON array.")
        return 2

    baseline_top1 = 0.0
    baseline_top3 = 0.0
    ai_top1 = 0.0
    ai_top3 = 0.0
    total = 0

    for item in payload:
        if not isinstance(item, dict):
            continue
        rows = [_candidate_row(entry) for entry in item.get("candidates", []) if isinstance(entry, dict)]
        if not rows:
            continue

        chunk_text = str(item.get("slot_text") or "")
        concept_text = str(item.get("concept_text") or chunk_text)
        concept_tokens = [str(token).strip().lower() for token in item.get("concept_tokens", []) if str(token).strip()]
        if not concept_tokens:
            concept_tokens = [token for token in concept_text.lower().split() if token]
        slot_duration_sec = float(item.get("slot_duration_sec", 3.0))
        positive_terms = [str(term).strip().lower() for term in item.get("positive_terms", []) if str(term).strip()]

        baseline_rows = sorted(rows, key=lambda entry: entry[4], reverse=True)
        ai_rows = rerank_broll_candidates(
            chunk_text=chunk_text,
            concept_text=concept_text,
            concept_tokens=concept_tokens,
            slot_duration_sec=slot_duration_sec,
            candidates=rows,
            assets_by_id={},
        )

        b1, b3 = _evaluate(baseline_rows, positive_terms)
        a1, a3 = _evaluate(ai_rows, positive_terms)
        baseline_top1 += b1
        baseline_top3 += b3
        ai_top1 += a1
        ai_top3 += a3
        total += 1

    if total == 0:
        print("No valid samples in dataset.")
        return 2

    print(f"Samples: {total}")
    print(f"Baseline Top-1: {(baseline_top1 / total) * 100:.1f}%")
    print(f"Baseline Top-3: {(baseline_top3 / total) * 100:.1f}%")
    print(f"AI Top-1: {(ai_top1 / total) * 100:.1f}%")
    print(f"AI Top-3: {(ai_top3 / total) * 100:.1f}%")
    print(f"Top-1 Delta: {((ai_top1 - baseline_top1) / total) * 100:.1f} points")
    print(f"Top-3 Delta: {((ai_top3 - baseline_top3) / total) * 100:.1f} points")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight B-roll ranking benchmark")
    parser.add_argument(
        "--dataset",
        default="tests/data/broll_benchmark_sample.json",
        help="Path to benchmark JSON dataset",
    )
    args = parser.parse_args()
    return run(Path(args.dataset))


if __name__ == "__main__":
    raise SystemExit(main())
