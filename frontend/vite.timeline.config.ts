import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const outputDirectory =
  process.env.TIMELINE_HARNESS_OUT_DIR ?? "dist-timeline-harness";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: outputDirectory,
    emptyOutDir: true,
    rollupOptions: {
      input: {
        "tests/timeline-baseline/index": resolve(
          process.cwd(),
          "tests/timeline-baseline/index.html",
        ),
      },
    },
  },
});
