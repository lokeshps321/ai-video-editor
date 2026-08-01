import { Fragment, useState, type CSSProperties, type RefObject } from "react";
import { FileVideo, Link2, UploadCloud, Zap } from "lucide-react";
import type { ExportAspectRatio, Job } from "../types";
import "./PreviewDock.css";

type LivePreviewCaption = {
  text: string;
  words: Array<{
    text: string;
    key: string;
    isActive: boolean;
    isPast: boolean;
  }>;
  fontName: string;
  color: string;
  outlineColor: string;
  outlineWidth: number;
  shadow: number;
  alignment: number;
  marginV: number;
  fontSize: number;
};

type PreviewDockProps = {
  previewSource: string | null;
  uploading: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
  previewFrameAspectRatio: ExportAspectRatio;
  exportAspectRatio: ExportAspectRatio;
  livePreviewCaption: LivePreviewCaption | null;
  shouldShowLiveCaptionOverlay: boolean;
  showExportFrameGuide: boolean;
  previewRenderBusy: boolean;
  previewBusyDetail: string;
  previewProgress: number;
  currentTimeSec: number;
  previewStatusText: string;
  previewJob: Job | null;
  previewUpdateQueued: boolean;
  queueingPreview: boolean;
  canRenderPreview: boolean;
  ingestingUrl: boolean;
  ingestProgress?: number;
  ingestStatusMessage?: string;
  onUploadVideo: (file: File) => void;
  onIngestUrl: (url: string) => void;
  onLoadedMetadata: () => void;
  onPlay: () => void;
  onPause: () => void;
  onSeeked: () => void;
  onEnded: () => void;
  onTimeUpdate: () => void;
  onFrameAspectRatioChange: (ratio: ExportAspectRatio) => void;
  onQueuePreview: () => void;
  formatPreciseSeconds: (value: number) => string;
};

function assColorToCss(color: string | undefined, fallback: string): string {
  if (!color) return fallback;
  if (!color.startsWith("&H")) return color;
  const raw = color.replace("&H", "").padStart(8, "0");
  const bb = raw.slice(2, 4);
  const gg = raw.slice(4, 6);
  const rr = raw.slice(6, 8);
  return `#${rr}${gg}${bb}`;
}

export function PreviewDock({
  previewSource,
  uploading,
  videoRef,
  previewFrameAspectRatio,
  exportAspectRatio,
  livePreviewCaption,
  shouldShowLiveCaptionOverlay,
  showExportFrameGuide,
  previewRenderBusy,
  previewBusyDetail,
  previewProgress,
  currentTimeSec,
  previewStatusText,
  previewJob,
  previewUpdateQueued,
  queueingPreview,
  canRenderPreview,
  ingestingUrl,
  ingestProgress = 0,
  ingestStatusMessage = "",
  onUploadVideo,
  onIngestUrl,
  onLoadedMetadata,
  onPlay,
  onPause,
  onSeeked,
  onEnded,
  onTimeUpdate,
  onFrameAspectRatioChange,
  onQueuePreview,
  formatPreciseSeconds,
}: PreviewDockProps) {
  const [videoDragOver, setVideoDragOver] = useState(false);
  const [ingestUrlValue, setIngestUrlValue] = useState("");

  const submitIngestUrl = () => {
    const url = ingestUrlValue.trim();
    if (!url || ingestingUrl || uploading) return;
    onIngestUrl(url);
    setIngestUrlValue("");
  };

  return (
    <section className="panel card editorPreviewDock">
      <div className="workspacePreviewBlock">
        <h2>Video Preview</h2>
        {!previewSource && (
          <div
            className={`onboardingDropZone ${videoDragOver ? "dragover" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setVideoDragOver(true);
            }}
            onDragLeave={() => setVideoDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setVideoDragOver(false);
              const file = e.dataTransfer.files[0];
              if (file?.type.startsWith("video/")) {
                onUploadVideo(file);
              }
            }}
          >
            <FileVideo size={40} className="onboardingIcon" />
            <h3 className="onboardingTitle">Get started in 3 steps</h3>
            <div className="onboardingSteps">
              <div className="onboardingStep">
                <span className="stepNum">1</span>
                <span>Upload or drag a video here</span>
              </div>
              <div className="onboardingStep">
                <span className="stepNum">2</span>
                <span>Generate transcript - edit by text</span>
              </div>
              <div className="onboardingStep">
                <span className="stepNum">3</span>
                <span>Quick Edit: Cut + Captions</span>
              </div>
            </div>
            <label className="uploadBtn primaryBtn onboardingUploadBtn">
              <input
                type="file"
                accept="video/*"
                disabled={uploading}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) onUploadVideo(file);
                  event.currentTarget.value = "";
                }}
              />
              <UploadCloud size={16} />
              {uploading ? "Uploading..." : "Choose Video"}
            </label>
            <div className="onboardingUrlRow">
              <input
                type="url"
                className="onboardingUrlInput"
                placeholder="or paste a video link (YouTube, .mp4 ...)"
                value={ingestUrlValue}
                disabled={ingestingUrl || uploading}
                onChange={(event) => setIngestUrlValue(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") submitIngestUrl();
                }}
              />
              <button
                type="button"
                className="onboardingUrlBtn"
                disabled={!ingestUrlValue.trim() || ingestingUrl || uploading}
                onClick={submitIngestUrl}
              >
                <Link2 size={14} />
                {ingestingUrl
                  ? `Fetching ${Math.max(0, Math.min(100, Math.round(ingestProgress)))}%`
                  : "Add from URL"}
              </button>
            </div>
            {ingestingUrl && (
              <div
                className="ingestProgressCard"
                role="status"
                aria-live="polite"
                aria-busy="true"
              >
                <div className="ingestProgressTop">
                  <span>
                    {ingestStatusMessage || "Fetching video from URL..."}
                  </span>
                  <strong>
                    {Math.max(0, Math.min(100, Math.round(ingestProgress)))}%
                  </strong>
                </div>
                <div
                  className="ingestProgressTrack"
                  aria-label="URL fetch progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.max(
                    0,
                    Math.min(100, Math.round(ingestProgress)),
                  )}
                  role="progressbar"
                >
                  <span
                    className="ingestProgressFill"
                    style={{
                      width: `${Math.max(0, Math.min(100, Math.round(ingestProgress)))}%`,
                    }}
                  />
                </div>
              </div>
            )}
            <p className="muted onboardingHint">
              Or press{" "}
              <span className="inlineIconLabel">
                <Zap size={12} aria-hidden="true" />
                Quick Edit
              </span>{" "}
              for transcript cut + captions (B-roll is optional in B-roll
              Studio)
            </p>
          </div>
        )}
        {previewSource && (
          <div
            className={`previewStage previewStage${previewFrameAspectRatio === "9:16" ? "Portrait" : "Landscape"}`}
          >
            <video
              ref={videoRef}
              key={previewSource}
              src={previewSource}
              controls
              crossOrigin="anonymous"
              className="previewVideo"
              onLoadedMetadata={onLoadedMetadata}
              onPlay={onPlay}
              onPause={onPause}
              onSeeked={onSeeked}
              onEnded={onEnded}
              onTimeUpdate={onTimeUpdate}
              onError={(e) => console.error("Video preview error:", e)}
            />
            {livePreviewCaption && (
              <div
                className="livePreviewCaption"
                aria-hidden="true"
                style={
                  {
                    "--caption-safe-bottom":
                      previewFrameAspectRatio === "9:16"
                        ? `clamp(72px, ${Math.max(8.8, Math.min(12.8, livePreviewCaption.marginV / 12))}%, 98px)`
                        : `${Math.max(
                            shouldShowLiveCaptionOverlay ? 84 : 54,
                            Math.min(
                              livePreviewCaption.marginV *
                                (shouldShowLiveCaptionOverlay ? 0.8 : 0.58),
                              shouldShowLiveCaptionOverlay ? 156 : 116,
                            ),
                          )}px`,
                    "--caption-max-width": shouldShowLiveCaptionOverlay
                      ? previewFrameAspectRatio === "9:16"
                        ? "84%"
                        : "54%"
                      : previewFrameAspectRatio === "16:9"
                        ? "72%"
                        : "88%",
                  } as CSSProperties
                }
              >
                <span
                  className="livePreviewCaptionText"
                  style={{
                    color: assColorToCss(livePreviewCaption.color, "#ffffff"),
                    fontFamily: `${livePreviewCaption.fontName.replace("-", " ")}, sans-serif`,
                    WebkitTextStroke: `${Math.min(Math.max(livePreviewCaption.outlineWidth, 1), 3)}px ${assColorToCss(livePreviewCaption.outlineColor, "#000000")}`,
                    textShadow:
                      livePreviewCaption.shadow > 0
                        ? `0 2px ${Math.min(livePreviewCaption.shadow * 2, 8)}px rgba(0,0,0,0.7)`
                        : "0 1px 2px rgba(0,0,0,0.55)",
                    fontSize: `clamp(0.92rem, ${Math.max(
                      1.2,
                      Math.min(
                        livePreviewCaption.fontSize / 18,
                        previewFrameAspectRatio === "9:16"
                          ? 2.3
                          : shouldShowLiveCaptionOverlay
                            ? 2.1
                            : 1.85,
                      ),
                    )}vw, ${previewFrameAspectRatio === "9:16" ? "1.36rem" : shouldShowLiveCaptionOverlay ? "1.35rem" : "1.55rem"})`,
                    background: "transparent",
                  }}
                >
                  {livePreviewCaption.words.length > 0
                    ? livePreviewCaption.words.map((word, index) => (
                        <Fragment key={word.key}>
                          {index > 0 ? " " : null}
                          <span
                            className={[
                              "livePreviewCaptionWord",
                              word.isActive ? "active" : "",
                              word.isPast ? "past" : "",
                            ]
                              .filter(Boolean)
                              .join(" ")}
                          >
                            {word.text}
                          </span>
                        </Fragment>
                      ))
                    : livePreviewCaption.text}
                </span>
              </div>
            )}
            {showExportFrameGuide &&
              previewFrameAspectRatio === "16:9" &&
              exportAspectRatio === "9:16" && (
                <div className="previewFrameGuide" aria-hidden="true">
                  <div className="previewFrameGuideWindow" />
                </div>
              )}
            {previewRenderBusy && (
              <div className="previewBusyBadge" aria-live="polite">
                <div className="previewBusyRow">
                  <span className="previewSpinner" aria-hidden="true" />
                  <span>{previewBusyDetail}</span>
                </div>
                <div
                  className="jobProgressBar previewJobProgressBar"
                  aria-hidden="true"
                >
                  <span
                    className="jobProgressFill"
                    style={{ width: `${previewProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}
        <div className="previewMeta">
          <span>Playhead: {formatPreciseSeconds(currentTimeSec)}</span>
          <span>Preview: {previewStatusText}</span>
          <span>Editor frame: {previewFrameAspectRatio}</span>
          {showExportFrameGuide &&
            previewFrameAspectRatio === "16:9" &&
            exportAspectRatio === "9:16" && <span>Portrait export guide on</span>}
          {previewRenderBusy && previewSource && (
            <span>Showing last rendered preview while update runs.</span>
          )}
          <span>
            Job:{" "}
            {previewJob
              ? `${previewJob.status} (${previewProgress}%)`
              : "not queued"}
            {previewUpdateQueued ? " · update queued" : ""}
          </span>
        </div>
        {previewJob?.status === "failed" && (
          <p className="warning">
            Preview failed: {previewJob.error ?? "Unknown render error"}
          </p>
        )}
        <div className="wordActions">
          <div
            className="previewAspectToggle"
            role="group"
            aria-label="Preview frame aspect"
          >
            {(["16:9", "9:16"] as const).map((ratio) => (
              <button
                key={ratio}
                className={`previewAspectBtn ${previewFrameAspectRatio === ratio ? "active" : ""}`}
                onClick={() => onFrameAspectRatioChange(ratio)}
                type="button"
              >
                {ratio}
              </button>
            ))}
          </div>
          <button onClick={onQueuePreview} disabled={!canRenderPreview || queueingPreview}>
            {queueingPreview ? "Queueing..." : "Render Preview"}
          </button>
        </div>
      </div>
    </section>
  );
}
