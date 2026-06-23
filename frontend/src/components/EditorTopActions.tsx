import {
  Captions,
  Clapperboard,
  Download,
  FolderOpen,
  Keyboard,
  Sparkles,
  UploadCloud,
  Zap,
} from "lucide-react";
import "./EditorTopActions.css";

type TopActionFeatureTab = "captions" | "broll_studio" | "ai_actions";

type EditorTopActionsProps = {
  uploading: boolean;
  selectedVideoFilename: string | null;
  timelineDurationSec: number;
  quickEditing: boolean;
  quickEditStage: string | null;
  quickEditRuntimeHint: string;
  generatingTranscript: boolean;
  runningAction: string | null;
  exportingVideo: boolean;
  onUploadVideo: (file: File) => void;
  onToggleProjects: () => void;
  onQuickEdit: () => void;
  onExport: () => void;
  onOpenFeatureDrawer: (tab: TopActionFeatureTab) => void;
  onShowShortcuts: () => void;
  formatSeconds: (value: number) => string;
};

export function EditorTopActions({
  uploading,
  selectedVideoFilename,
  timelineDurationSec,
  quickEditing,
  quickEditStage,
  quickEditRuntimeHint,
  generatingTranscript,
  runningAction,
  exportingVideo,
  onUploadVideo,
  onToggleProjects,
  onQuickEdit,
  onExport,
  onOpenFeatureDrawer,
  onShowShortcuts,
  formatSeconds,
}: EditorTopActionsProps) {
  return (
    <section className="controls card creatorTopActions">
      <label className="uploadBtn primaryBtn">
        <input
          type="file"
          accept="video/*"
          disabled={uploading}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onUploadVideo(file);
            }
            event.currentTarget.value = "";
          }}
        />
        <UploadCloud size={16} />
        {uploading ? "Uploading..." : "Upload Video"}
      </label>

      <button
        type="button"
        onClick={onToggleProjects}
        title="Open an existing local project"
      >
        <FolderOpen size={14} />
        Projects
      </button>

      <button
        className="primaryBtn quickEditBtn"
        onClick={onQuickEdit}
        disabled={
          !selectedVideoFilename ||
          quickEditing ||
          generatingTranscript ||
          !!runningAction
        }
        title={`One-click: Transcribe, auto-cut pauses/fillers, add captions. Estimated time: ${quickEditRuntimeHint}. B-roll is separate in B-roll Studio.`}
      >
        <Zap size={16} />
        {quickEditing ? quickEditStage || "Quick Editing..." : "Quick Edit"}
      </button>

      <button className="primaryBtn" onClick={onExport} disabled={exportingVideo}>
        <Download size={14} />
        {exportingVideo ? "Exporting..." : "Export"}
      </button>
      <button
        onClick={() => onOpenFeatureDrawer("captions")}
        title="Caption styles & settings"
      >
        <Captions size={14} />
        Captions
      </button>
      <button
        onClick={() => onOpenFeatureDrawer("broll_studio")}
        title="B-roll studio"
      >
        <Clapperboard size={14} />
        B-roll
      </button>
      <button
        onClick={() => onOpenFeatureDrawer("ai_actions")}
        title="AI editing tools"
      >
        <Sparkles size={14} />
        AI Tools
      </button>
      <button
        className="shortcutsHelpBtn"
        onClick={onShowShortcuts}
        title="Keyboard shortcuts (?)"
      >
        <Keyboard size={16} />
      </button>
      <p className="muted creatorTopMeta">
        <span>{selectedVideoFilename ?? "No video selected"}</span>
        <span>{formatSeconds(timelineDurationSec)}</span>
        {selectedVideoFilename && (
          <span>Quick Edit estimate: {quickEditRuntimeHint}</span>
        )}
      </p>
    </section>
  );
}
