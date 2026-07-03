import { AlertTriangle, CheckCircle2, Layers3 } from "lucide-react";
import type { BrollCandidate, BrollSlot } from "../../types";

type BrollPlanSummary = {
  modeLabel: string;
  runtimeHint: string;
  maxSlots: number;
  includeExternalSources: boolean;
  aiRerank: boolean;
  minConfidence: number;
};

type BrollTrustSummaryProps = {
  slots: BrollSlot[];
  overlayClipCount: number;
  selectedPlan: BrollPlanSummary | null;
  formatSeconds: (value: number) => string;
  reasonCodeLabel: (code: string) => string;
  onFocusSlot: (slot: BrollSlot, noticeMessage?: string) => void;
};

function chosenOrPrimaryCandidate(slot: BrollSlot): BrollCandidate | null {
  return (
    slot.candidates.find((candidate) => candidate.id === slot.chosen_candidate_id) ??
    slot.candidates[0] ??
    null
  );
}

function reasonText(
  reason: Record<string, unknown> | null | undefined,
  key: string,
): string | null {
  const value = reason?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function humanize(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function isReady(slot: BrollSlot): boolean {
  return ["ready", "approved"].includes(slot.review_status ?? "");
}

function confidenceForSlot(slot: BrollSlot): number | null {
  const candidate = chosenOrPrimaryCandidate(slot);
  return typeof candidate?.confidence === "number" ? candidate.confidence : null;
}

function sectionForSlot(slot: BrollSlot): string {
  const candidate = chosenOrPrimaryCandidate(slot);
  return (
    reasonText(candidate?.reason, "section_label") ??
    reasonText(candidate?.reason, "intent_label") ??
    "unreviewed"
  );
}

export function BrollTrustSummary({
  slots,
  overlayClipCount,
  selectedPlan,
  formatSeconds,
  reasonCodeLabel,
  onFocusSlot,
}: BrollTrustSummaryProps) {
  const chosenCount = slots.filter((slot) => !!slot.chosen_candidate_id).length;
  const readyCount = slots.filter(
    (slot) => isReady(slot) && !!slot.chosen_candidate_id,
  ).length;
  const needsReviewCount = slots.filter(
    (slot) => (slot.review_status ?? "needs_review") === "needs_review",
  ).length;
  const weakSlots = slots.filter(
    (slot) => (slot.weak_reason_codes?.length ?? 0) > 0,
  );
  const confidenceValues = slots
    .map(confidenceForSlot)
    .filter((value): value is number => typeof value === "number");
  const averageConfidence =
    confidenceValues.length > 0
      ? confidenceValues.reduce((sum, value) => sum + value, 0) /
        confidenceValues.length
      : null;
  const safeConfidenceCount = slots.filter((slot) => {
    const confidence = confidenceForSlot(slot);
    return (
      confidence !== null &&
      confidence >= (selectedPlan?.minConfidence ?? 0.72)
    );
  }).length;

  const grouped = slots.reduce<Record<string, BrollSlot[]>>((acc, slot) => {
    const key = sectionForSlot(slot);
    acc[key] = acc[key] ?? [];
    acc[key].push(slot);
    return acc;
  }, {});
  const groupEntries = Object.entries(grouped).sort((left, right) => {
    const order = ["hook", "setup", "body", "payoff", "outro", "unreviewed"];
    const leftIndex = order.indexOf(left[0]);
    const rightIndex = order.indexOf(right[0]);
    return (
      (leftIndex === -1 ? order.length : leftIndex) -
      (rightIndex === -1 ? order.length : rightIndex)
    );
  });

  const readinessRatio = slots.length > 0 ? readyCount / slots.length : 0;
  const trustTier =
    slots.length === 0
      ? "empty"
      : readinessRatio >= 0.7 && weakSlots.length === 0
        ? "ready"
        : readyCount > 0 || chosenCount > 0
          ? "review"
          : "risk";
  const TrustIcon =
    trustTier === "ready"
      ? CheckCircle2
      : trustTier === "empty"
        ? Layers3
        : AlertTriangle;
  const trustCopy =
    trustTier === "empty"
      ? "Generate B-roll slots to review visual meaning and coverage."
      : trustTier === "ready"
        ? "Most chosen slots are safe to sync. Review timing before export."
        : trustTier === "risk"
          ? "Candidates exist, but none are ready enough for a trusted sync."
          : "Some slots are chosen, but weak or unreviewed slots still need attention.";

  return (
    <section className={`brollTrustCard ${trustTier}`}>
      <div className="brollTrustHead">
        <div className="brollTrustTitle">
          <span className="brollTrustIcon" aria-hidden="true">
            <TrustIcon size={16} strokeWidth={2.1} />
          </span>
          <div>
            <p className="brollTrustEyebrow">B-roll Trust</p>
            <h4>
              {trustTier === "empty"
                ? "No slots yet"
                : trustTier === "ready"
                  ? "Ready to sync"
                  : trustTier === "risk"
                    ? "Not ready"
                    : "Review needed"}
            </h4>
          </div>
        </div>
        {averageConfidence !== null && (
          <span className="brollTrustConfidence">
            {(averageConfidence * 100).toFixed(0)}% avg
          </span>
        )}
      </div>

      <p className="brollTrustCopy">{trustCopy}</p>

      {selectedPlan && (
        <p className="brollTrustPlan">
          {selectedPlan.modeLabel} · {selectedPlan.runtimeHint} · up to{" "}
          {selectedPlan.maxSlots} slots ·{" "}
          {selectedPlan.includeExternalSources ? "stock on" : "local only"} ·{" "}
          {selectedPlan.aiRerank ? "AI rerank" : "fast rank"}
        </p>
      )}

      <div className="brollTrustStats">
        <span>
          <strong>{slots.length}</strong>
          slots
        </span>
        <span>
          <strong>{readyCount}</strong>
          ready
        </span>
        <span>
          <strong>{needsReviewCount}</strong>
          review
        </span>
        <span>
          <strong>{overlayClipCount}</strong>
          overlays
        </span>
      </div>

      {!!slots.length && (
        <div className="brollTrustGroups" aria-label="B-roll coverage by beat">
          {groupEntries.slice(0, 6).map(([section, sectionSlots]) => {
            const readyInSection = sectionSlots.filter(isReady).length;
            return (
              <button
                key={section}
                type="button"
                className="brollTrustGroup"
                onClick={() =>
                  onFocusSlot(
                    sectionSlots[0],
                    `${humanize(section)} B-roll group selected.`,
                  )
                }
                title={`Jump to ${humanize(section)} B-roll group`}
              >
                <span>{humanize(section)}</span>
                <strong>
                  {readyInSection}/{sectionSlots.length}
                </strong>
              </button>
            );
          })}
        </div>
      )}

      {!!weakSlots.length && (
        <div className="brollTrustWeakList">
          <strong>Needs meaning review</strong>
          {weakSlots.slice(0, 3).map((slot) => (
            <button
              key={slot.id}
              type="button"
              onClick={() =>
                onFocusSlot(slot, "Weak B-roll slot selected for review.")
              }
            >
              <span>
                {formatSeconds(slot.start_sec)}-{formatSeconds(slot.end_sec)}
              </span>
              <span>
                {(slot.weak_reason_codes ?? [])
                  .slice(0, 2)
                  .map(reasonCodeLabel)
                  .join(" · ")}
              </span>
            </button>
          ))}
        </div>
      )}

      {!!slots.length && (
        <p className="brollTrustFootnote">
          {safeConfidenceCount} slot
          {safeConfidenceCount === 1 ? "" : "s"} meet the current confidence
          target before manual review.
        </p>
      )}
    </section>
  );
}
