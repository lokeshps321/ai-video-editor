import { spawn } from "node:child_process";
import { chromium } from "playwright";

const host = "127.0.0.1";
const port = 41741;
const baseUrl = `http://${host}:${port}`;
const browserExecutable =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || "/usr/bin/google-chrome";
const vite = spawn(
  process.platform === "win32"
    ? "node_modules/.bin/vite.cmd"
    : "node_modules/.bin/vite",
  ["--host", host, "--port", String(port), "--strictPort"],
  {
    cwd: process.cwd(),
    env: { ...process.env, VITE_TIMELINE_CORE_V2: "true" },
    stdio: ["ignore", "pipe", "pipe"],
  },
);

let viteOutput = "";
vite.stdout.on("data", (chunk) => {
  viteOutput += chunk;
});
vite.stderr.on("data", (chunk) => {
  viteOutput += chunk;
});

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function waitForServer() {
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    if (vite.exitCode !== null) {
      throw new Error(`Vite exited early (${vite.exitCode}):\n${viteOutput}`);
    }
    try {
      if ((await fetch(baseUrl)).ok) return;
    } catch {
      // Server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for Vite:\n${viteOutput}`);
}

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({
    executablePath: browserExecutable,
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto(`${baseUrl}/tests/timeline-baseline/index.html`);
  await page.waitForSelector(".timelineLane.video .timelineLaneClip");
  await page.waitForTimeout(200);

  const networkRequests = [];
  page.on("request", (request) => networkRequests.push(request.url()));
  const resetCounts = () =>
    page.evaluate(() => window.timelineBaseline.resetCallbackCounts());
  const counts = () =>
    page.evaluate(() => window.timelineBaseline.getCallbackCounts());

  const clip = page.locator(".timelineLane.video .timelineLaneClip").nth(5);
  await clip.scrollIntoViewIfNeeded();
  let box = await clip.boundingBox();
  assert(box, "Expected a visible unlocked main clip");

  await resetCounts();
  await page.mouse.move(box.x + 30, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + 50, box.y + box.height / 2);
  await page.waitForTimeout(20);
  await page.mouse.move(box.x + 30, box.y + box.height / 2);
  await page.mouse.up();
  await page.waitForTimeout(30);
  assert((await counts()).moveLane === 0, "Frame-identical drag committed");

  await resetCounts();
  box = await clip.boundingBox();
  const audioBox = await page.locator(".timelineLane.audio").first().boundingBox();
  assert(audioBox, "Expected a visible audio lane");
  await page.mouse.move(box.x + 30, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + 30, audioBox.y + audioBox.height / 2, {
    steps: 8,
  });
  await page.waitForTimeout(20);
  await page.mouse.up();
  await page.waitForTimeout(30);
  assert((await counts()).moveLane === 1, "Cross-lane drag did not commit once");

  const scroll = page.locator(".timelineScroll");
  const scrollBox = await scroll.boundingBox();
  assert(scrollBox, "Expected the timeline viewport");
  const playheadLeft = () =>
    page.locator(".timeline-playhead").evaluate((element) =>
      Number.parseFloat(getComputedStyle(element).left),
    );
  const clockBeforeCancel = await playheadLeft();
  await resetCounts();
  await page.mouse.move(scrollBox.x + 300, scrollBox.y + 12);
  await page.mouse.down();
  await page.mouse.move(scrollBox.x + 600, scrollBox.y + 12, { steps: 8 });
  await page.waitForTimeout(20);
  await page.keyboard.press("Escape");
  await page.mouse.up();
  await page.waitForTimeout(30);
  assert(
    Math.abs((await playheadLeft()) - clockBeforeCancel) < 0.01,
    "Cancelled scrub did not restore its pointer-down clock snapshot",
  );
  assert((await counts()).seek === 0, "Cancelled scrub called onSeek");

  await resetCounts();
  await page.keyboard.press("Control+Shift+ArrowRight");
  await page.keyboard.press("Meta+Shift+Alt+ArrowLeft");
  const modifiedArrowCounts = await counts();
  assert(
    modifiedArrowCounts.seek === 0 && modifiedArrowCounts.moveLane === 0,
    "Ctrl/Meta arrow triggered a timeline command",
  );

  const zoomBefore = (await page.evaluate(() => window.timelineBaseline.getState()))
    .pxPerSec;
  await page.mouse.move(scrollBox.x + 700, scrollBox.y + 80);
  await page.keyboard.down("Control");
  await page.mouse.wheel(0, -100);
  await page.keyboard.up("Control");
  await page.waitForTimeout(50);
  const zoomAfter = (await page.evaluate(() => window.timelineBaseline.getState()))
    .pxPerSec;
  assert(zoomAfter > zoomBefore, "Cursor-anchored V2 zoom did not change scale");
  assert(networkRequests.length === 0, "Timeline interactions issued network requests");
  assert(pageErrors.length === 0, `Browser errors: ${pageErrors.join("; ")}`);

  process.stdout.write(
    `${JSON.stringify({
      noOpSuppressed: true,
      crossLaneCommitCount: 1,
      scrubCancelRestored: true,
      modifiedArrowsSuppressed: true,
      zoomBefore,
      zoomAfter,
      networkRequests: networkRequests.length,
      pageErrors,
    })}\n`,
  );
} finally {
  await browser?.close();
  vite.kill("SIGTERM");
}
