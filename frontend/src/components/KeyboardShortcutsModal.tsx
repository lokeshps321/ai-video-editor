import { Keyboard } from "lucide-react";
import "./KeyboardShortcutsModal.css";

type KeyboardShortcutsModalProps = {
  onClose: () => void;
};

export function KeyboardShortcutsModal({ onClose }: KeyboardShortcutsModalProps) {
  return (
    <div className="shortcutsOverlay" onClick={onClose}>
      <div className="shortcutsModal" onClick={(e) => e.stopPropagation()}>
        <div className="shortcutsHeader">
          <h3>
            <Keyboard size={20} /> Keyboard Shortcuts
          </h3>
          <button onClick={onClose} className="shortcutsClose">
            &times;
          </button>
        </div>
        <div className="shortcutsGrid">
          <div className="shortcutGroup">
            <h4>Playback</h4>
            <div className="shortcutRow">
              <kbd>Space</kbd>
              <span>Play / Pause</span>
            </div>
            <div className="shortcutRow">
              <kbd>←</kbd> <kbd>→</kbd>
              <span>Seek ±5 seconds</span>
            </div>
            <div className="shortcutRow">
              <kbd>Shift+←</kbd> <kbd>Shift+→</kbd>
              <span>Seek ±1 second</span>
            </div>
          </div>
          <div className="shortcutGroup">
            <h4>Transcript Editing</h4>
            <div className="shortcutRow">
              <kbd>Click</kbd>
              <span>Select word &amp; seek</span>
            </div>
            <div className="shortcutRow">
              <kbd>Shift+Click</kbd>
              <span>Select range</span>
            </div>
            <div className="shortcutRow">
              <kbd>Shift/Alt+Drag</kbd>
              <span>Range-select by time</span>
            </div>
            <div className="shortcutRow">
              <kbd>Double-click</kbd>
              <span>Edit word text (transcript or TXT track)</span>
            </div>
            <div className="shortcutRow">
              <kbd>Del</kbd> / <kbd>Backspace</kbd>
              <span>Delete selection / clip / caption</span>
            </div>
            <div className="shortcutRow">
              <kbd>Ctrl+Z</kbd>
              <span>Undo</span>
            </div>
            <div className="shortcutRow">
              <kbd>Ctrl+Y</kbd> / <kbd>Ctrl+Shift+Z</kbd>
              <span>Redo</span>
            </div>
            <div className="shortcutRow">
              <kbd>Esc</kbd>
              <span>Deselect all</span>
            </div>
          </div>
          <div className="shortcutGroup">
            <h4>Timeline Clips</h4>
            <div className="shortcutRow">
              <kbd>Click clip</kbd>
              <span>Select clip</span>
            </div>
            <div className="shortcutRow">
              <kbd>Drag clip</kbd>
              <span>Move clip</span>
            </div>
            <div className="shortcutRow">
              <kbd>Drag edge</kbd>
              <span>Trim clip in / out</span>
            </div>
            <div className="shortcutRow">
              <kbd>S</kbd>
              <span>Split selected clip at playhead</span>
            </div>
            <div className="shortcutRow">
              <kbd>Right-click</kbd>
              <span>Context menu (split / delete / jump)</span>
            </div>
            <div className="shortcutRow">
              <kbd>Ctrl+Scroll</kbd>
              <span>Zoom in / out</span>
            </div>
          </div>
          <div className="shortcutGroup">
            <h4>Search</h4>
            <div className="shortcutRow">
              <kbd>Ctrl+F</kbd>
              <span>Search transcript</span>
            </div>
            <div className="shortcutRow">
              <kbd>Enter</kbd>
              <span>Next match</span>
            </div>
            <div className="shortcutRow">
              <kbd>Shift+Enter</kbd>
              <span>Previous match</span>
            </div>
            <div className="shortcutRow">
              <kbd>Esc</kbd>
              <span>Clear search</span>
            </div>
          </div>
          <div className="shortcutGroup">
            <h4>General</h4>
            <div className="shortcutRow">
              <kbd>?</kbd>
              <span>Toggle this help</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
