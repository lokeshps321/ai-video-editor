import { AlertTriangle, Captions, CheckCircle2, Globe2 } from "lucide-react";
import type { Transcript, TranscriptRegion } from "../../types";
import "./TranscriptQualityPanel.css";

type TranscriptQualityPanelProps = {
  transcript: Transcript | null;
  selectedLanguage: string;
  scriptSummary: string[];
  captionBlockCount: number;
  reviewWordCount: number;
  weakQualityCount: number;
  lowConfidenceOnlyCount: number;
  lowConfidenceCount: number;
  lowConfidenceRatio: number;
  shouldWarnLowConfidence: boolean;
  issueRegions: TranscriptRegion[];
  surface?: "transcript" | "captions" | "export";
  compact?: boolean;
  onReviewWeakWords?: () => void;
  onReviewNextWeakWord?: () => void;
  onFocusRegion?: (region: TranscriptRegion) => void;
  formatSeconds: (value: number) => string;
  regionLabel: (region: TranscriptRegion) => string;
};

function clamp(value: number, minValue: number, maxValue: number): number {
  return Math.max(minValue, Math.min(maxValue, value));
}

function normalizeLanguage(value: string | null | undefined): string {
  return String(value ?? "").trim().toLowerCase();
}

function formatPercent(value: number): string {
  return `${Math.round(clamp(value, 0, 1) * 100)}%`;
}

function surfaceTitle(surface: TranscriptQualityPanelProps["surface"]): string {
  if (surface === "captions") return "Caption Readiness";
  if (surface === "export") return "Export Quality Check";
  return "Transcript Quality";
}

export function TranscriptQualityPanel({
  transcript,
  selectedLanguage,
  scriptSummary,
  captionBlockCount,
  reviewWordCount,
  weakQualityCount,
  lowConfidenceOnlyCount,
  lowConfidenceCount,
  lowConfidenceRatio,
  shouldWarnLowConfidence,
  issueRegions,
  surface = "transcript",
  compact = false,
  onReviewWeakWords,
  onReviewNextWeakWord,
  onFocusRegion,
  formatSeconds,
  regionLabel,
}: TranscriptQualityPanelProps) {
  const wordCount = transcript?.words.length ?? 0;
  const detectedLanguage = normalizeLanguage(transcript?.language);
  const requestedLanguage = normalizeLanguage(selectedLanguage);
  const languageMismatch =
    !!transcript &&
    !!detectedLanguage &&
    !!requestedLanguage &&
    requestedLanguage !== "auto" &&
    requestedLanguage !== detectedLanguage;
  const hasMixedScript = !!transcript?.mixed_script && scriptSummary.length > 1;
  const weakRatio = wordCount > 0 ? reviewWordCount / wordCount : 0;
  const backendScore =
    typeof transcript?.quality_score === "number"
      ? clamp(transcript.quality_score, 0, 1) * 100
      : 100;
  const readinessScore = !transcript
    ? 0
    : Math.round(
        clamp(
          backendScore -
            (transcript.is_mock ? 45 : 0) -
            Math.min(issueRegions.length * 8, 30) -
            Math.min(lowConfidenceRatio * 90, 25) -
            Math.min(weakRatio * 60, 20) -
            (languageMismatch ? 18 : 0) -
            (hasMixedScript ? 6 : 0),
          0,
          100,
        ),
      );

  const tier = !transcript
    ? "empty"
    : transcript.is_mock || readinessScore < 55
      ? "risk"
      : shouldWarnLowConfidence ||
          reviewWordCount > 0 ||
          issueRegions.length > 0 ||
          languageMismatch ||
          hasMixedScript ||
          readinessScore < 82
        ? "review"
        : "ready";

  const summary =
    tier === "empty"
      ? "Generate a transcript before applying captions or judging B-roll meaning."
      : tier === "ready"
        ? "Timing, language, and confidence look ready for creator captions."
        : tier === "risk"
          ? "Fix transcript quality before relying on captions or visual matches."
          : "Review highlighted words and regions before final captions or export.";

  const detailItems: string[] = [];
  if (!transcript) {
    detailItems.push("No transcript loaded");
  } else {
    if (transcript.is_mock) detailItems.push("Fallback transcript");
    if (languageMismatch) {
      detailItems.push(
        `Requested ${selectedLanguage}, detected ${transcript.language}`,
      );
    }
    if (hasMixedScript) {
      detailItems.push(`Mixed script: ${scriptSummary.join(" + ")}`);
    }
    if (reviewWordCount > 0) {
      detailItems.push(`${reviewWordCount} words need review`);
    }
    if (issueRegions.length > 0) {
      detailItems.push(`${issueRegions.length} watchlist regions`);
    }
    if (captionBlockCount > 0) {
      detailItems.push(`${captionBlockCount} caption blocks on timeline`);
    } else if (surface !== "transcript") {
      detailItems.push("No caption blocks on timeline");
    }
    if (!detailItems.length) detailItems.push("No blocking quality issues");
  }

  const Icon = tier === "ready" ? CheckCircle2 : tier === "empty" ? Captions : AlertTriangle;

  return (
    <section className={`qualityPanel ${tier} ${compact ? "compact" : ""}`}>
      <div className="qualityPanelHead">
        <div className="qualityTitleGroup">
          <span className="qualityIcon" aria-hidden="true">
            <Icon size={16} strokeWidth={2.1} />
          </span>
          <div>
            <p className="qualityEyebrow">{surfaceTitle(surface)}</p>
            <h4>
              {tier === "empty"
                ? "Transcript needed"
                : tier === "ready"
                  ? "Ready"
                  : tier === "risk"
                    ? "Needs fixing"
                    : "Needs review"}
            </h4>
          </div>
        </div>
        <div className="qualityScore" aria-label="Readiness score">
          <strong>{readinessScore}</strong>
          <span>/100</span>
        </div>
      </div>

      <p className="qualitySummary">{summary}</p>

      <div className="qualityStats">
        <span>
          <strong>{wordCount}</strong>
          words
        </span>
        <span>
          <strong>{reviewWordCount}</strong>
          review
        </span>
        <span>
          <strong>{formatPercent(lowConfidenceRatio)}</strong>
          low conf
        </span>
        <span>
          <strong>{captionBlockCount}</strong>
          captions
        </span>
      </div>

      <div className="qualityDetailList">
        {detailItems.slice(0, compact ? 3 : 6).map((item) => (
          <span key={item}>{item}</span>
        ))}
      </div>

      {!!transcript?.language && !compact && (
        <div className="qualityLanguageLine">
          <Globe2 size={13} aria-hidden="true" />
          <span>Detected {transcript.language}</span>
          {scriptSummary.length > 0 && <span>{scriptSummary.join(" + ")}</span>}
        </div>
      )}

      {!compact && reviewWordCount > 0 && onReviewWeakWords && (
        <div className="qualityActions">
          <button type="button" onClick={onReviewWeakWords}>
            Review weak words
          </button>
          {onReviewNextWeakWord && (
            <button type="button" onClick={onReviewNextWeakWord}>
              Next weak word
            </button>
          )}
        </div>
      )}

      {!compact && issueRegions.length > 0 && (
        <div className="qualityRegions">
          <div className="qualityRegionsHead">
            <strong>Watchlist</strong>
            <span>
              {issueRegions.length} region
              {issueRegions.length === 1 ? "" : "s"}
            </span>
          </div>
          <div className="transcriptRegionBar">
            {issueRegions.slice(0, 8).map((region, index) => (
              <button
                key={`${region.status}-${region.start_sec}-${region.end_sec}-${index}`}
                type="button"
                className={`transcriptRegionChip ${region.status}`}
                onClick={() => onFocusRegion?.(region)}
                title={`${regionLabel(region)} · ${formatSeconds(region.start_sec)} - ${formatSeconds(region.end_sec)}${region.reason ? ` · ${region.reason}` : ""}`}
              >
                <span>{regionLabel(region)}</span>
                <span>
                  {formatSeconds(region.start_sec)}-{formatSeconds(region.end_sec)}
                </span>
              </button>
            ))}
            {issueRegions.length > 8 && (
              <span className="muted transcriptRegionOverflow">
                +{issueRegions.length - 8} more
              </span>
            )}
          </div>
        </div>
      )}

      {!compact && (weakQualityCount > 0 || lowConfidenceOnlyCount > 0) && (
        <p className="qualityFootnote">
          {weakQualityCount} weak-label · {lowConfidenceOnlyCount} low-confidence
          {lowConfidenceCount > 0
            ? ` · ${formatPercent(lowConfidenceRatio)} below target`
            : ""}
        </p>
      )}
    </section>
  );
}
