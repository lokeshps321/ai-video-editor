import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import process from "node:process";
import { chromium } from "playwright";

const HOST = "127.0.0.1";
const PORT = 41739;
const BASE_URL = `http://${HOST}:${PORT}`;
const RESULT_PATH = new URL("./timeline-baseline.json", import.meta.url);
const BROWSER_EXECUTABLE =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || "/usr/bin/google-chrome";
const vite = spawn(
  process.platform === "win32"
    ? "node_modules/.bin/vite.cmd"
    : "node_modules/.bin/vite",
  ["--host", HOST, "--port", String(PORT), "--strictPort"],
  { cwd: process.cwd(), stdio: ["ignore", "pipe", "pipe"] },
);

let viteOutput = "";
vite.stdout.on("data", (chunk) => {
  viteOutput += chunk;
});
vite.stderr.on("data", (chunk) => {
  viteOutput += chunk;
});

async function waitForServer() {
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    if (vite.exitCode !== null) {
      throw new Error(`Vite exited early (${vite.exitCode}):\n${viteOutput}`);
    }
    try {
      const response = await fetch(BASE_URL);
      if (response.ok) return;
    } catch {
      // Server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for Vite:\n${viteOutput}`);
}

async function settledFrames(page, count = 3) {
  await page.evaluate(
    (frameCount) =>
      new Promise((resolve) => {
        let remaining = frameCount;
        const next = () => {
          remaining -= 1;
          if (remaining <= 0) resolve(undefined);
          else requestAnimationFrame(next);
        };
        requestAnimationFrame(next);
      }),
    count,
  );
}

function summarizeFrameDrift(samples) {
  assertFiniteSamples(
    "frameSamples",
    samples,
    ["frame", "timeMs", "expectedSec", "actualSec", "deltaSec"],
  );
  return Object.fromEntries(
    [24, 30, 60].map((fps) => [
      String(fps),
      (() => {
        const absoluteFrames = samples.map(
          (sample) => Math.abs(sample.deltaSec) * fps,
        );
        const roundedFrameDeltas = samples.map((sample) =>
          Math.abs(
            Math.round(sample.actualSec * fps) -
              Math.round(sample.expectedSec * fps),
          ),
        );
        return {
          sampleCount: samples.length,
          maxAbsoluteFrames: Number(
            Math.max(...absoluteFrames).toFixed(6),
          ),
          meanAbsoluteFrames: Number(
            (
              absoluteFrames.reduce((total, value) => total + value, 0) /
              absoluteFrames.length
            ).toFixed(6),
          ),
          maxRoundedFrameDelta: Math.max(...roundedFrameDeltas),
          samplesWithRoundedFrameDrift: roundedFrameDeltas.filter(
            (value) => value > 0,
          ).length,
        };
      })(),
    ]),
  );
}

function assertFiniteSamples(label, samples, fields) {
  if (!Array.isArray(samples) || samples.length === 0) {
    throw new Error(`${label} must contain at least one sample`);
  }
  for (const [index, sample] of samples.entries()) {
    for (const field of fields) {
      if (!Number.isFinite(sample[field])) {
        throw new Error(
          `${label}[${index}].${field} must be finite; received ${sample[field]}`,
        );
      }
    }
  }
}

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({
    executablePath: BROWSER_EXECUTABLE,
    headless: true,
  });
  const context = await browser.newContext({
    deviceScaleFactor: 1,
    viewport: { width: 1440, height: 900 },
  });
  const page = await context.newPage();
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  let activeRequests = null;
  page.on("request", (request) => {
    activeRequests?.push({
      method: request.method(),
      resourceType: request.resourceType(),
      url: request.url(),
    });
  });

  await page.goto(
    `${BASE_URL}/tests/timeline-baseline/index.html`,
    { waitUntil: "domcontentloaded" },
  );
  await page.waitForSelector('[data-testid="timeline-baseline-harness"]', {
    timeout: 30_000,
  });
  await settledFrames(page, 5);

  async function startMeasurement(label) {
    activeRequests = [];
    await page.evaluate(
      (measurementLabel) =>
        window.timelineBaseline.startMeasurement(measurementLabel),
      label,
    );
  }

  async function stopMeasurement() {
    await settledFrames(page);
    await page.waitForTimeout(75);
    const measurement = await page.evaluate(() =>
      window.timelineBaseline.stopMeasurement(),
    );
    const networkRequests = activeRequests ?? [];
    activeRequests = null;
    assertFiniteSamples(
      `${measurement.label}.commits`,
      measurement.commits,
      ["frame", "commitTimeMs", "actualDurationMs"],
    );
    return { ...measurement, networkRequests };
  }

  const scroll = page.locator(".timelineScroll");
  const scrollBox = await scroll.boundingBox();
  if (!scrollBox) throw new Error("Timeline scroll container has no box");

  await scroll.evaluate((element) => {
    element.scrollLeft = 12_000;
  });
  await settledFrames(page);

  const scrubStartX = scrollBox.x + 260;
  const scrubEndX = scrollBox.x + 1_100;
  const rulerY = scrollBox.y + 12;
  await page.mouse.move(scrubStartX, rulerY);
  await startMeasurement("sustained-scrub");
  await page.evaluate(
    (startClientX) =>
      window.timelineBaseline.startFrameSampling({
        mode: "scrub",
        startClientX,
      }),
    scrubStartX,
  );
  await page.mouse.down();
  for (let step = 1; step <= 120; step += 1) {
    const x = scrubStartX + ((scrubEndX - scrubStartX) * step) / 120;
    await page.mouse.move(x, rulerY);
    await page.waitForTimeout(4);
  }
  await page.mouse.up();
  const scrub = await stopMeasurement();
  assertFiniteSamples(
    "sustained-scrub.frameSamples",
    scrub.frameSamples,
    ["frame", "timeMs", "expectedSec", "actualSec", "deltaSec"],
  );
  scrub.frameDrift = summarizeFrameDrift(scrub.frameSamples);

  await scroll.evaluate((element) => {
    element.scrollLeft = 12_000;
  });
  await settledFrames(page);
  const dragPoint = await page.evaluate(() => {
    const viewport = document
      .querySelector(".timelineScroll")
      .getBoundingClientRect();
    const primaryLane = document.querySelector(".timelineLane.video");
    const clips = [...primaryLane.querySelectorAll(".timelineLaneClip")];
    const clip = clips.find((candidate) => {
      const rect = candidate.getBoundingClientRect();
      return rect.right > viewport.left + 140 && rect.left < viewport.right - 320;
    });
    if (!clip) throw new Error("No visible primary clip found");
    const rect = clip.getBoundingClientRect();
    return {
      x: Math.max(rect.left + 8, Math.min(rect.right - 8, rect.left + 40)),
      y: rect.top + rect.height / 2,
      initialTimelineSec:
        Number.parseFloat(clip.style.left) /
        (Number.parseFloat(
          document.querySelector(".timelineCanvas").style.width,
        ) /
          1_800),
    };
  });
  await page.mouse.move(dragPoint.x, dragPoint.y);
  await startMeasurement("sustained-clip-drag");
  await page.evaluate(
    ({ startClientX, initialTimelineSec }) =>
      window.timelineBaseline.startFrameSampling({
        mode: "drag",
        startClientX,
        initialTimelineSec,
      }),
    {
      startClientX: dragPoint.x,
      initialTimelineSec: dragPoint.initialTimelineSec,
    },
  );
  await page.mouse.down();
  for (let step = 1; step <= 120; step += 1) {
    await page.mouse.move(dragPoint.x + (240 * step) / 120, dragPoint.y);
    await page.waitForTimeout(4);
  }
  await page.mouse.up();
  const drag = await stopMeasurement();
  assertFiniteSamples(
    "sustained-clip-drag.frameSamples",
    drag.frameSamples,
    ["frame", "timeMs", "expectedSec", "actualSec", "deltaSec"],
  );
  drag.frameDrift = summarizeFrameDrift(drag.frameSamples);

  await scroll.evaluate((element) => {
    element.scrollLeft = 12_000;
  });
  await settledFrames(page);
  const zoomCursor = {
    x: scrollBox.x + 820,
    y: scrollBox.y + 90,
  };
  const zoomErrors = [];
  await startMeasurement("repeated-cursor-anchored-zoom");
  for (const deltaY of [-100, -100, -100, -100, -100, -100, 100, 100, 100, 100, 100, 100]) {
    const sample = await page.evaluate(
      async ({ clientX, clientY, wheelDeltaY, railWidth }) => {
        const element = document.querySelector(".timelineScroll");
        const canvas = document.querySelector(".timelineCanvas");
        const rect = element.getBoundingClientRect();
        const viewportX = clientX - rect.left;
        const beforePxPerSec =
          Number.parseFloat(canvas.style.width) / 1_800;
        const anchorSec =
          (viewportX + element.scrollLeft - railWidth) / beforePxPerSec;
        element.dispatchEvent(
          new WheelEvent("wheel", {
            bubbles: true,
            cancelable: true,
            clientX,
            clientY,
            ctrlKey: true,
            deltaY: wheelDeltaY,
          }),
        );
        await new Promise((resolve) =>
          requestAnimationFrame(() =>
            requestAnimationFrame(() => requestAnimationFrame(resolve)),
          ),
        );
        const afterPxPerSec = Number.parseFloat(canvas.style.width) / 1_800;
        const anchorViewportX =
          anchorSec * afterPxPerSec + railWidth - element.scrollLeft;
        return {
          beforePxPerSec,
          afterPxPerSec,
          errorPx: Math.abs(anchorViewportX - viewportX),
        };
      },
      {
        clientX: zoomCursor.x,
        clientY: zoomCursor.y,
        wheelDeltaY: deltaY,
        railWidth: 96,
      },
    );
    zoomErrors.push({
      beforePxPerSec: Number(sample.beforePxPerSec.toFixed(6)),
      afterPxPerSec: Number(sample.afterPxPerSec.toFixed(6)),
      errorPx: Number(sample.errorPx.toFixed(6)),
    });
  }
  const zoom = await stopMeasurement();
  zoom.anchorSamples = zoomErrors;
  zoom.maxAnchorErrorPx = Math.max(
    ...zoomErrors.map((sample) => sample.errorPx),
  );
  zoom.meanAnchorErrorPx = Number(
    (
      zoomErrors.reduce((total, sample) => total + sample.errorPx, 0) /
      zoomErrors.length
    ).toFixed(6),
  );
  assertFiniteSamples(
    "repeated-cursor-anchored-zoom.anchorSamples",
    zoom.anchorSamples,
    ["beforePxPerSec", "afterPxPerSec", "errorPx"],
  );

  const compactScrub = compactMeasurement(scrub);
  const compactDrag = compactMeasurement(drag);
  const compactZoom = compactMeasurement(zoom);
  const allLongTaskDurations = [scrub, drag, zoom].flatMap((entry) =>
    entry.longTasks.map((task) => task.durationMs),
  );
  const result = {
    capturedAt: new Date().toISOString(),
    environment: {
      browser: browser.version(),
      browserExecutable: BROWSER_EXECUTABLE,
      headless: true,
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1,
      fixture: fixtureMetadata(await page.evaluate(() => ({
        durationSec: 1_800,
        laneClips: document.querySelectorAll(".timelineLaneClip").length,
        transcriptWords: document.querySelectorAll(".tlWord").length,
        captions: document.querySelectorAll(".captionBlock").length,
        broll: document.querySelectorAll(".brollBlock").length,
      }))),
    },
    measurements: {
      scrub: compactScrub,
      drag: compactDrag,
      zoom: compactZoom,
    },
    summary: {
      maxCommitsPerFrame: Math.max(
        scrub.maxCommitsPerFrame,
        drag.maxCommitsPerFrame,
        zoom.maxCommitsPerFrame,
      ),
      networkCallsDuringGestures:
        scrub.networkRequests.length +
        drag.networkRequests.length +
        zoom.networkRequests.length,
      frameDrift: {
        scrub: scrub.frameDrift,
        drag: drag.frameDrift,
      },
      longTasksOver50Ms: {
        count: allLongTaskDurations.filter((duration) => duration > 50).length,
        maxDurationMs: Math.max(0, ...allLongTaskDurations),
        totalDurationMs: Number(
          allLongTaskDurations
            .reduce((total, duration) => total + duration, 0)
            .toFixed(3),
        ),
        byGesture: {
          scrub: compactScrub.longTasks,
          drag: compactDrag.longTasks,
          zoom: compactZoom.longTasks,
        },
      },
      repeatedZoomMaxErrorPx: zoom.maxAnchorErrorPx,
      repeatedZoomMeanErrorPx: zoom.meanAnchorErrorPx,
    },
    pageErrors,
  };

  await writeFile(RESULT_PATH, `${JSON.stringify(result, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} finally {
  await browser?.close();
  vite.kill("SIGTERM");
}

function fixtureMetadata(domCounts) {
  return {
    scenario: "30-minute interview with cutaways",
    ...domCounts,
    brollClips: 120,
    waveformPeaks: 9_000,
  };
}

function compactMeasurement(measurement) {
  const durations = measurement.longTasks.map((task) => task.durationMs);
  return {
    ...measurement,
    longTasks: {
      countOver50Ms: durations.filter((duration) => duration > 50).length,
      maxDurationMs: Math.max(0, ...durations),
      totalDurationMs: Number(
        durations
          .reduce((total, duration) => total + duration, 0)
          .toFixed(3),
      ),
    },
  };
}
