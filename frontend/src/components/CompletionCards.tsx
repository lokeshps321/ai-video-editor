import { Check, Download } from "lucide-react";
import "./CompletionCards.css";

type QuickEditSummary = {
  cutDetails: string | null;
  captionDetails: string | null;
  removedDurationSec: number | null;
  removedWordCount: number | null;
  captionBlockCount: number | null;
  captionsAdded: boolean;
  finalDurationSec: number;
  nextStep: string;
};

type ExportCompletionSummary = {
  format: "mp4" | "mov" | "webm";
  aspectRatio: string;
  resolution: "720p" | "1080p" | "4k";
  fps: 24 | 30 | 60;
  quality: "low" | "medium" | "high" | "max";
  jobId: string;
  filename: string;
  outputPath: string | null;
  downloadError: string | null;
};

type QuickEditSummaryCardProps = {
  quickEditSummary: QuickEditSummary;
  formatSeconds: (value: number) => string;
  formatFixedSec: (value: number) => string;
};

type ExportCompletionCardProps = {
  exportCompletion: ExportCompletionSummary;
  downloadingExport: boolean;
  onDownload: () => void;
};

export function QuickEditSummaryCard({
  quickEditSummary,
  formatSeconds,
  formatFixedSec,
}: QuickEditSummaryCardProps) {
  return (
    <section className="completionCard quickEditSummaryCard" aria-live="polite">
      <div className="completionCardHeader">
        <span className="completionCardIcon">
          <Check size={16} strokeWidth={2.4} aria-hidden="true" />
        </span>
        <div>
          <h2>Quick Edit complete</h2>
          <p>
            Clean cut is ready for review
            {quickEditSummary.captionsAdded
              ? " with captions applied."
              : "."}
          </p>
        </div>
      </div>
      <div className="completionStats">
        {quickEditSummary.removedDurationSec !== null && (
          <span>
            <strong>
              {formatFixedSec(quickEditSummary.removedDurationSec)}s
            </strong>
            <small>removed</small>
          </span>
        )}
        {quickEditSummary.removedWordCount !== null && (
          <span>
            <strong>{quickEditSummary.removedWordCount}</strong>
            <small>
              filler word
              {quickEditSummary.removedWordCount === 1 ? "" : "s"}
            </small>
          </span>
        )}
        <span>
          <strong>
            {quickEditSummary.captionBlockCount !== null
              ? quickEditSummary.captionBlockCount
              : quickEditSummary.captionsAdded
                ? "Added"
                : "Skipped"}
          </strong>
          <small>captions</small>
        </span>
        <span>
          <strong>{formatSeconds(quickEditSummary.finalDurationSec)}</strong>
          <small>final length</small>
        </span>
      </div>
      <div className="completionDetails">
        {quickEditSummary.cutDetails && (
          <span>{quickEditSummary.cutDetails}</span>
        )}
        {quickEditSummary.captionDetails && (
          <span>{quickEditSummary.captionDetails}</span>
        )}
      </div>
      <p className="completionNextStep">
        <strong>Next:</strong> {quickEditSummary.nextStep}
      </p>
    </section>
  );
}

export function ExportCompletionCard({
  exportCompletion,
  downloadingExport,
  onDownload,
}: ExportCompletionCardProps) {
  return (
    <section className="completionCard exportCompletionCard" aria-live="polite">
      <div className="completionCardHeader">
        <span className="completionCardIcon">
          <Check size={16} strokeWidth={2.4} aria-hidden="true" />
        </span>
        <div>
          <h2>Export complete</h2>
          <p>Your video rendered successfully and is ready to share.</p>
        </div>
      </div>
      <div className="completionStats">
        <span>
          <strong>{exportCompletion.format.toUpperCase()}</strong>
          <small>format</small>
        </span>
        <span>
          <strong>{exportCompletion.aspectRatio}</strong>
          <small>aspect</small>
        </span>
        <span>
          <strong>{exportCompletion.resolution}</strong>
          <small>resolution</small>
        </span>
        <span>
          <strong>{exportCompletion.fps}</strong>
          <small>fps</small>
        </span>
      </div>
      <div className="exportCompletionFooter">
        <span className="exportCompletionFilename">
          {exportCompletion.filename}
        </span>
        {exportCompletion.outputPath && (
          <button
            className="primaryBtn exportDownloadBtn"
            type="button"
            onClick={onDownload}
            disabled={downloadingExport}
          >
            <Download size={14} aria-hidden="true" />
            {downloadingExport ? "Downloading..." : "Download again"}
          </button>
        )}
      </div>
      {exportCompletion.downloadError && (
        <p className="completionWarning">
          Download did not start automatically: {exportCompletion.downloadError}
        </p>
      )}
    </section>
  );
}
