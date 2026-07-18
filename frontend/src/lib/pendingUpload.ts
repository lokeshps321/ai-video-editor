/**
 * Module-level store for passing a File object from the landing page
 * to the editor without serialization (File can't go through sessionStorage).
 *
 * Landing page sets the file → navigates to /editor → App reads it on mount.
 */

let pendingFile: File | null = null;

export function setPendingUploadFile(file: File) {
  pendingFile = file;
}

export function hasPendingUploadFile(): boolean {
  return pendingFile !== null;
}

export function consumePendingUploadFile(): File | null {
  const file = pendingFile;
  pendingFile = null;
  return file;
}

/** Return the pending upload's filename without consuming the file. */
export function peekPendingUploadName(): string | null {
  return pendingFile?.name ?? null;
}

/**
 * Create a readable project name from an uploaded filename.
 * "interview_take-02.mp4" becomes "interview take 02".
 */
export function filenameToProjectName(
  fileName: string | null | undefined,
  fallback: string,
): string {
  if (!fileName) return fallback;
  const withoutExtension = fileName.replace(/\.[^./\\]+$/, "");
  const name = withoutExtension.replace(/[_-]+/g, " ").trim();
  return name || fallback;
}
