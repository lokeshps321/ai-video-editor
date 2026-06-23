import { useRef, useState } from "react";
import { Check, Pencil, Plus, RefreshCw, Trash2, X } from "lucide-react";
import type { Project } from "../types";
import "./ProjectReopenPanel.css";

type ProjectReopenPanelProps = {
  project: Project;
  recentProjects: Project[];
  loadingProjects: boolean;
  openingProjectId: string | null;
  submittingRenameId: string | null;
  deletingProjectId: string | null;
  creatingProject: boolean;
  defaultProjectName: string;
  onRefresh: () => void;
  onCreate: (name: string) => void | Promise<void>;
  onOpen: (projectId: string) => void | Promise<void>;
  onRename: (projectId: string, name: string) => void | Promise<void>;
  onDelete: (projectId: string) => void | Promise<void>;
  formatSeconds: (value: number) => string;
};

export function ProjectReopenPanel({
  project,
  recentProjects,
  loadingProjects,
  openingProjectId,
  submittingRenameId,
  deletingProjectId,
  creatingProject,
  defaultProjectName,
  onRefresh,
  onCreate,
  onOpen,
  onRename,
  onDelete,
  formatSeconds,
}: ProjectReopenPanelProps) {
  const [renamingProjectId, setRenamingProjectId] = useState<string | null>(
    null,
  );
  const [renameText, setRenameText] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [newProjectName, setNewProjectName] = useState("");
  const [showNewProjectInput, setShowNewProjectInput] = useState(false);
  const renameInputRef = useRef<HTMLInputElement | null>(null);
  const newProjectInputRef = useRef<HTMLInputElement | null>(null);

  function startNewProject() {
    setShowNewProjectInput(true);
    setNewProjectName("");
    setTimeout(() => newProjectInputRef.current?.focus(), 50);
  }

  function cancelNewProject() {
    setShowNewProjectInput(false);
    setNewProjectName("");
  }

  function submitNewProject() {
    const name = newProjectName.trim() || defaultProjectName;
    setShowNewProjectInput(false);
    void onCreate(name);
  }

  function startRename(item: Project) {
    setRenamingProjectId(item.id);
    setRenameText(item.name || "");
    setTimeout(() => renameInputRef.current?.focus(), 50);
  }

  async function submitRename(projectId: string) {
    if (!renameText.trim()) return;
    try {
      await onRename(projectId, renameText);
    } finally {
      setRenamingProjectId(null);
      setRenameText("");
    }
  }

  function cancelRename() {
    setRenamingProjectId(null);
    setRenameText("");
  }

  async function confirmDelete(projectId: string) {
    try {
      await onDelete(projectId);
    } finally {
      setConfirmDeleteId(null);
    }
  }

  return (
    <section className="projectReopenPanel card" aria-label="Recent projects">
      <div className="projectReopenHeader">
        <div>
          <p className="inspectorEyebrow">Local workspace</p>
          <h2>Recent projects</h2>
        </div>
        <div className="projectHeaderActions">
          <button
            type="button"
            onClick={onRefresh}
            disabled={loadingProjects}
            title="Refresh project list"
          >
            <RefreshCw size={13} className={loadingProjects ? "spin" : ""} />
            {loadingProjects ? "Refreshing..." : "Refresh"}
          </button>
          <button
            type="button"
            className="primaryBtn newProjectBtn"
            onClick={startNewProject}
            disabled={creatingProject}
            title="Create a new project"
          >
            <Plus size={14} />
            New
          </button>
        </div>
      </div>

      {showNewProjectInput && (
        <div className="newProjectInputRow">
          <input
            ref={newProjectInputRef}
            type="text"
            className="controlInput newProjectInput"
            placeholder="Project name..."
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submitNewProject();
              if (e.key === "Escape") cancelNewProject();
            }}
            disabled={creatingProject}
          />
          <button
            type="button"
            className="primaryBtn"
            disabled={creatingProject}
            onClick={submitNewProject}
          >
            {creatingProject ? "Creating..." : "Create"}
          </button>
          <button
            type="button"
            className="projectActionBtnClose"
            onClick={cancelNewProject}
            title="Cancel"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {recentProjects.length === 0 ? (
        <p className="muted projectReopenEmpty">No saved projects found yet.</p>
      ) : (
        <div className="projectReopenList">
          {recentProjects.map((item) => {
            const isCurrent = item.id === project.id;
            const isRenaming = renamingProjectId === item.id;
            const isSubmittingRename = submittingRenameId === item.id;
            const videoClipCount = item.timeline.tracks
              .filter((track) => track.kind === "video")
              .reduce((count, track) => count + (track.clips?.length ?? 0), 0);
            const showConfirmDelete = confirmDeleteId === item.id;
            const isDeleting = deletingProjectId === item.id;

            return (
              <div
                key={item.id}
                className={`projectReopenItem ${isCurrent ? "active" : ""} ${showConfirmDelete ? "danger" : ""}`}
              >
                {isRenaming ? (
                  <div className="projectRenameRow">
                    <input
                      ref={renameInputRef}
                      type="text"
                      className="controlInput projectRenameInput"
                      value={renameText}
                      onChange={(e) => setRenameText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") void submitRename(item.id);
                        if (e.key === "Escape") cancelRename();
                      }}
                      disabled={isSubmittingRename}
                    />
                    <button
                      type="button"
                      className="projectActionBtnConfirm"
                      onClick={() => void submitRename(item.id)}
                      disabled={isSubmittingRename || !renameText.trim()}
                      title="Save name"
                    >
                      <Check size={13} />
                    </button>
                    <button
                      type="button"
                      className="projectActionBtnClose"
                      onClick={cancelRename}
                      title="Cancel"
                    >
                      <X size={13} />
                    </button>
                  </div>
                ) : showConfirmDelete ? (
                  <div className="projectDeleteConfirm">
                    <p className="projectDeleteWarning">
                      Delete "<strong>{item.name || "Untitled"}</strong>"? This
                      cannot be undone.
                    </p>
                    <div className="projectDeleteActions">
                      <button
                        type="button"
                        className="projectDeleteConfirmBtn"
                        onClick={() => void confirmDelete(item.id)}
                        disabled={isDeleting}
                      >
                        <Trash2 size={13} />
                        {isDeleting ? "Deleting..." : "Yes, Delete"}
                      </button>
                      <button
                        type="button"
                        className="projectActionBtnClose"
                        onClick={() => setConfirmDeleteId(null)}
                        disabled={isDeleting}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div
                      className="projectReopenClickArea"
                      role="button"
                      tabIndex={0}
                      onClick={() => {
                        if (!isCurrent && !openingProjectId) {
                          void onOpen(item.id);
                        }
                      }}
                      onKeyDown={(e) => {
                        if (
                          e.key === "Enter" &&
                          !isCurrent &&
                          !openingProjectId
                        ) {
                          void onOpen(item.id);
                        }
                      }}
                      title={
                        isCurrent
                          ? "This project is already open"
                          : "Open this project"
                      }
                    >
                      <span className="projectReopenName">
                        {item.name || "Untitled Project"}
                      </span>
                      <span className="projectReopenMeta">
                        {formatSeconds(item.timeline.duration_sec)} ·{" "}
                        {videoClipCount} clip
                        {videoClipCount === 1 ? "" : "s"}
                        {isCurrent ? " · current" : ""}
                      </span>
                    </div>
                    <div className="projectItemActions">
                      {openingProjectId === item.id ? (
                        <span className="projectReopenAction">Opening...</span>
                      ) : isCurrent ? (
                        <span className="projectReopenAction current">
                          Current
                        </span>
                      ) : (
                        <span className="projectReopenAction">Open</span>
                      )}
                      <button
                        type="button"
                        className="projectActionBtn"
                        onClick={(e) => {
                          e.stopPropagation();
                          startRename(item);
                        }}
                        title="Rename project"
                        disabled={!!openingProjectId || !!deletingProjectId}
                      >
                        <Pencil size={13} />
                      </button>
                      <button
                        type="button"
                        className="projectActionBtn danger"
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmDeleteId(item.id);
                        }}
                        title="Delete project"
                        disabled={!!openingProjectId || !!deletingProjectId}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
