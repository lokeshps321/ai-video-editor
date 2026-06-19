# Manual Test Checklist

Use this checklist for focused manual validation after UI, transcription, or export changes. Do not commit `.env`, `.env.local`, or any file containing real API secrets.

## Phase 1: Upload, Quick Edit, Export

- Upload a supported video through the file picker and confirm it appears in the editor.
- Drag and drop a supported video onto the upload area and confirm the same ingest flow starts.
- Confirm the Quick Edit summary appears after analysis/transcription and describes the planned edit.
- Start an export and confirm progress reaches completion.
- Confirm the export completion card appears with a working download action.

## Phase 2: Empty Editor and Project Reopen

- Open the editor with no active project and confirm the basic "Get started in 3 steps" guidance appears.
- Confirm removed preset cards do not appear in the empty editor state.
- Create or open a project, leave the editor, then reopen it from Projects.
- Reopen a project from Recent Projects and confirm the video, transcript, and edit state are restored.

## Phase 3: Transcript, B-roll, Retry, Layout

- Generate a transcript and confirm weak-word review highlights uncertain words or regions.
- Edit transcript words and confirm timing/edit state stays usable after saving or continuing.
- Open B-roll review and confirm focus state is clear for the currently selected beat/region.
- Review B-roll candidates and confirm candidate selection updates the preview or plan.
- Use meaning reroll for a B-roll item and confirm the refreshed candidates match the new intent.
- Confirm time/cost/effort estimates are visible where retry or generation actions are offered.
- Trigger available retry actions and confirm the UI reports progress and final status.
- Check a laptop-sized layout and confirm controls remain reachable without horizontal overflow.

## Cloud Transcription Troubleshooting

- Groq transcription requires `GROQ_API_KEY` to be present and non-empty in the backend environment.
- Sarvam transcription accepts either `SARVAM_API_KEY` or `TRANSCRIBE_SARVAM_API_KEY`; the backend checks `SARVAM_API_KEY` first, then `TRANSCRIBE_SARVAM_API_KEY`.
- For local development, place real secrets in `backend/.env.local`; keep committed example/default files secret-free.
- Restart the backend after adding or changing env vars so the running process reloads them.
- If keys are present but transcription still falls back to a mock transcript, check network access, provider API quota/billing, provider service status, and request limits.
- After fixing keys or network/quota issues, regenerate the transcript from the UI.
- Never commit `.env` secrets; use `.env.example` or `.env.local.example` only for blank placeholders and setup guidance.
