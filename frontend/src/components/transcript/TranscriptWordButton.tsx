import React from "react";
import type { TranscriptWord } from "../../types";

const SCRIPT_TAG_LABELS: Record<
  NonNullable<TranscriptWord["script_tag"]>,
  string
> = {
  latin: "Latin",
  indic: "Indic",
  arabic: "Arabic",
  mixed: "Mixed",
  other: "Other",
};

export interface TranscriptWordButtonProps {
  word: TranscriptWord;
  displayText: string;
  showRomanized: boolean;
  isDeleted: boolean;
  isSelected: boolean;
  isActive: boolean;
  isFiller: boolean;
  isSearchMatch: boolean;
  isCurrentMatch: boolean;
  hasLowConfidence: boolean;
  isWeakRegionWord: boolean;
  speakerSlot: number | null;
  activeWordRef: React.RefObject<HTMLButtonElement | null>;
  isDraggingRef: React.MutableRefObject<boolean>;
  dragStartWordIdRef: React.MutableRefObject<string | null>;
  selectWord: (id: string, shiftHeld: boolean) => void;
  seekToWord: (word: TranscriptWord) => void;
  selectWordRange: (fromId: string, toId: string) => void;
  startEditing: (word: TranscriptWord) => void;
  formatPreciseSeconds: (value: number) => string;
}

export const TranscriptWordButton = React.memo(function TranscriptWordButton({
  word,
  displayText,
  showRomanized,
  isDeleted,
  isSelected,
  isActive,
  isFiller,
  isSearchMatch,
  isCurrentMatch,
  hasLowConfidence,
  isWeakRegionWord,
  speakerSlot,
  activeWordRef,
  isDraggingRef,
  dragStartWordIdRef,
  selectWord,
  seekToWord,
  selectWordRange,
  startEditing,
  formatPreciseSeconds,
}: TranscriptWordButtonProps) {
  const className = [
    "word",
    isDeleted ? "deleted" : "",
    isSelected ? "selected" : "",
    isActive ? "active" : "",
    isFiller ? "filler" : "",
    isSearchMatch ? "searchMatch" : "",
    isCurrentMatch ? "currentMatch" : "",
    hasLowConfidence ? "lowConfidence" : "",
    isWeakRegionWord ? "weakRegion" : "",
    speakerSlot === 0 ? "speakerA" : "",
    speakerSlot === 1 ? "speakerB" : "",
    speakerSlot !== null && speakerSlot >= 2 ? "speakerExtra" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const confidenceHint =
    typeof word.confidence === "number"
      ? ` · ${(word.confidence * 100).toFixed(0)}%`
      : "";
  const qualityHint =
    typeof word.quality_score === "number"
      ? ` · quality ${(word.quality_score * 100).toFixed(0)}%`
      : "";
  const passHint = word.source_pass ? ` · ${word.source_pass}` : "";
  const labelHint = word.quality_label ? ` · ${word.quality_label}` : "";
  const speakerHint = word.speaker_label ? ` · ${word.speaker_label}` : "";
  const scriptHint = word.script_tag
    ? ` · script ${SCRIPT_TAG_LABELS[word.script_tag] ?? word.script_tag}`
    : "";
  const languageHint = word.language_hint ? ` · lang ${word.language_hint}` : "";
  const originalHint =
    showRomanized && displayText !== word.text ? ` · original ${word.text}` : "";

  return (
    <button
      id={`word-${word.id}`}
      type="button"
      className={className}
      ref={isActive ? activeWordRef : undefined}
      onMouseDown={(event) => {
        if (event.detail >= 2) return;
        isDraggingRef.current = true;
        dragStartWordIdRef.current = word.id;
        selectWord(word.id, event.shiftKey);
        seekToWord(word);
      }}
      onMouseEnter={() => {
        if (isDraggingRef.current && dragStartWordIdRef.current) {
          selectWordRange(dragStartWordIdRef.current, word.id);
        }
      }}
      onDoubleClick={() => startEditing(word)}
      title={`${formatPreciseSeconds(word.start_sec)} – ${formatPreciseSeconds(word.end_sec)}${speakerHint}${scriptHint}${languageHint}${originalHint}${confidenceHint}${qualityHint}${labelHint}${passHint}`}
    >
      {displayText}
    </button>
  );
});
