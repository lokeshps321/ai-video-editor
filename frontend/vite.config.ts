import react from "@vitejs/plugin-react";
import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    exclude: [...configDefaults.exclude, "tests/playwright/**"],
  },
  server: {
    port: 5173,
    // Avoid ENOSPC when the OS inotify watcher/instance limit is exhausted
    // (common with IDEs + many node tools). Polling uses no inotify watches.
    watch: {
      usePolling: true,
      interval: 300,
    },
  },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("@react-three") || id.includes("/three/")) return "vendor-3d";
          if (id.includes("framer-motion")) return "vendor-motion";
          if (id.includes("lucide-react")) return "vendor-icons";
          if (id.includes("/react/") || id.includes("/react-dom/") || id.includes("react-router-dom")) {
            return "vendor-react";
          }
          return undefined;
        }
      }
    }
  }
});
