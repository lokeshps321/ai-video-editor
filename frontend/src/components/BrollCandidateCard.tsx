import { memo, useEffect, useState } from "react";
import "./BrollCandidateCard.css";

type BrollCandidateCardProps = {
  label: string;
  sourceTag: string;
  metaLine: string | null;
  confidencePercent: string | null;
  confidenceTier: string;
  scoreLabel: string;
  breakdownChips: string[];
  previewUrl: string | null;
  previewType: "image" | "video";
  chosen: boolean;
  busy: boolean;
  locked: boolean;
  onClick: () => void;
};

export const BrollCandidateCard = memo(function BrollCandidateCard(props: BrollCandidateCardProps) {
  const {
    label,
    sourceTag,
    metaLine,
    confidencePercent,
    confidenceTier,
    scoreLabel,
    breakdownChips,
    previewUrl,
    previewType,
    chosen,
    busy,
    locked,
    onClick,
  } = props;

  const disabled = busy || locked;
  const [previewFailed, setPreviewFailed] = useState(false);
  const normalizedPreviewUrl = previewUrl?.trim() ? previewUrl.trim() : null;

  useEffect(() => {
    setPreviewFailed(false);
  }, [normalizedPreviewUrl]);

  return (
    <button
      type="button"
      className={`brollCandidateBtn brollCandidateCard ${chosen ? "chosen" : ""}`}
      onClick={onClick}
      disabled={disabled}
      title={scoreLabel ? `Match ${scoreLabel}` : undefined}
    >
      <div className="brollCandidateThumb" aria-hidden="true">
        {normalizedPreviewUrl && !previewFailed ? (
          previewType === "image" ? (
            <img
              className="brollCandidatePreview"
              src={normalizedPreviewUrl}
              alt={label}
              onError={() => setPreviewFailed(true)}
            />
          ) : (
            <video
              className="brollCandidatePreview"
              src={normalizedPreviewUrl}
              muted
              loop
              autoPlay
              playsInline
              preload="metadata"
              onError={() => setPreviewFailed(true)}
            />
          )
        ) : (
          <div className="brollCandidateThumbInner">
            <span className="brollCandidateThumbBadge">{sourceTag}</span>
          </div>
        )}
      </div>
      <div className="brollCandidateBody">
        <div className="brollCandidateMain">
          <span className="brollCandidateLabel">{label}</span>
          {metaLine && <span className="brollCandidateMeta">{metaLine}</span>}
        </div>
        <div className="brollCandidateSide">
          {confidencePercent && confidenceTier !== "unknown" && (
            <span className={`brollConfidence ${confidenceTier}`}>
              {confidenceTier} {confidencePercent}
            </span>
          )}
          <span>{busy ? "…" : scoreLabel}</span>
        </div>
        {!!breakdownChips.length && (
          <span className="brollReasonChips">
            {breakdownChips.join(" · ")}
          </span>
        )}
      </div>
    </button>
  );
});

BrollCandidateCard.displayName = "BrollCandidateCard";
