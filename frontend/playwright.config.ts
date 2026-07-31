import { existsSync } from "node:fs";

import { chromium, defineConfig, devices } from "@playwright/test";

const installedChrome = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
const packageChromium = chromium.executablePath();
const chromeCandidates = [
  packageChromium,
  installedChrome,
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
].filter((candidate): candidate is string => Boolean(candidate));
const executablePath = chromeCandidates.find((candidate) => existsSync(candidate));

if (!executablePath) {
  throw new Error(
    [
      "Timeline browser gate requires Chrome or Playwright Chromium.",
      "Install it with `npx playwright install chromium`, install Google Chrome,",
      "or set PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH.",
    ].join(" "),
  );
}

process.env.TIMELINE_BROWSER_EXECUTABLE = executablePath;
process.env.TIMELINE_BROWSER_SOURCE =
  executablePath === packageChromium ? "playwright-chromium" : "explicit-fallback";

export default defineConfig({
  testDir: "./tests/playwright",
  outputDir: "./test-results/playwright",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["line"]],
  use: {
    ...devices["Desktop Chrome"],
    browserName: "chromium",
    deviceScaleFactor: 1,
    headless: true,
    launchOptions: { executablePath },
    viewport: { width: 1440, height: 900 },
  },
  webServer: [
    {
      command:
        "npm run build:timeline-harness:legacy && VITE_TIMELINE_CORE_V2=false TIMELINE_HARNESS_OUT_DIR=dist-timeline-legacy vite preview --config vite.timeline.config.ts --host 127.0.0.1 --port 41840 --strictPort",
      port: 41840,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command:
        "npm run build:timeline-harness:v2 && VITE_TIMELINE_CORE_V2=true TIMELINE_HARNESS_OUT_DIR=dist-timeline-v2 vite preview --config vite.timeline.config.ts --host 127.0.0.1 --port 41841 --strictPort",
      port: 41841,
      reuseExistingServer: false,
      timeout: 30_000,
    },
  ],
});
