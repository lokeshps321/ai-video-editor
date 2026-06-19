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

export function consumePendingUploadFile(): File | null {
  const file = pendingFile;
  pendingFile = null;
  return file;
}
