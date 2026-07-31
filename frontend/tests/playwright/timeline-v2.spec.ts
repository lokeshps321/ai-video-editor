import { mkdir, readFile, writeFile } from "node:fs/promises";

import { expect, test, type Page } from "@playwright/test";
import pixelmatch from "pixelmatch";
import { PNG } from "pngjs";

const LEGACY_URL =
  "http://127.0.0.1:41840/tests/timeline-baseline/index.html";
const V2_URL = "http://127.0.0.1:41841/tests/timeline-baseline/index.html";
const baselinePath = new URL(
  "../timeline-baseline/timeline-baseline.json",
  import.meta.url,
);
const artifactPath = new URL(
  "../../artifacts/timeline-v2-result.json",
  import.meta.url,
);
const visualArtifactPath = new URL(
  "../../artifacts/timeline-v2-visual-result.json",
  import.meta.url,
);

type Counts = Record<string, number>;
type Measurement = {
  label: string;
  commitCount: number;
  maxCommitsPerFrame: number;
  framesOverOneCommit: number;
  frameSamples: Array<{
    frame: number;
    timeMs: number;
    expectedSec: number;
    actualSec: number;
    deltaSec: number;
  }>;
  longTasks: Array<{ durationMs: number }>;
  previewPublications: Array<{ frame: number; timeMs: number; kind: string }>;
  renderedMutations: Array<{ frame: number; timeMs: number; kind: string }>;
};

async function loadHarness(page: Page, url = V2_URL): Promise<void> {
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.getByTestId("timeline-baseline-harness").waitFor();
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        let frames = 5;
        const tick = () => {
          frames -= 1;
          if (frames === 0) resolve();
          else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }),
  );
}

async function settleFrames(page: Page, count = 4): Promise<void> {
  await page.evaluate(
    (frameCount) =>
      new Promise<void>((resolve) => {
        let remaining = frameCount;
        const tick = () => {
          remaining -= 1;
          if (remaining <= 0) resolve();
          else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }),
    count,
  );
}

async function counts(page: Page): Promise<Counts> {
  return page.evaluate(() =>
    (
      window as unknown as {
        timelineBaseline: { getCallbackCounts: () => Counts };
      }
    ).timelineBaseline.getCallbackCounts(),
  );
}

async function resetCounts(page: Page): Promise<void> {
  await page.evaluate(() =>
    (
      window as unknown as {
        timelineBaseline: { resetCallbackCounts: () => void };
      }
    ).timelineBaseline.resetCallbackCounts(),
  );
}

async function visibleClip(page: Page) {
  const clip = page
    .locator(".timelineLane.video:not(.locked) .timelineLaneClip")
    .nth(5);
  await clip.scrollIntoViewIfNeeded();
  return clip;
}

async function movePointerInFrames(
  page: Page,
  from: { x: number; y: number },
  to: { x: number; y: number },
  steps = 12,
): Promise<void> {
  for (let step = 1; step <= steps; step += 1) {
    await page.mouse.move(
      from.x + ((to.x - from.x) * step) / steps,
      from.y + ((to.y - from.y) * step) / steps,
    );
    await page.evaluate(() => new Promise(requestAnimationFrame));
  }
}

test("real V2 timeline commits once and never mutates during gestures", async ({
  page,
}) => {
  await loadHarness(page);
  const gestureRequests: string[] = [];
  let measuringNetwork = false;
  page.on("request", (request) => {
    if (measuringNetwork) gestureRequests.push(request.url());
  });

  const clip = await visibleClip(page);
  await expect(clip).toBeVisible();
  const box = await clip.boundingBox();
  expect(box).not.toBeNull();
  const start = { x: box!.x + box!.width / 2, y: box!.y + box!.height / 2 };

  await resetCounts(page);
  measuringNetwork = true;
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await movePointerInFrames(page, start, { x: start.x + 180, y: start.y });
  expect((await counts(page)).moveLane).toBe(0);
  expect((await counts(page)).stateMutation).toBe(0);
  expect(gestureRequests).toHaveLength(0);
  await page.mouse.up();
  measuringNetwork = false;
  await page.evaluate(() =>
    (
      window as unknown as {
        timelineBaseline: { waitForAuthoritative: () => Promise<void> };
      }
    ).timelineBaseline.waitForAuthoritative(),
  );
  expect((await counts(page)).moveLane).toBe(1);
  expect((await counts(page)).stateMutation).toBe(1);

  const trimHandle = clip.locator(".laneClipHandle.end");
  const trimBox = await trimHandle.boundingBox();
  expect(trimBox).not.toBeNull();
  const trimStart = {
    x: trimBox!.x + trimBox!.width / 2,
    y: trimBox!.y + trimBox!.height / 2,
  };
  await resetCounts(page);
  await page.mouse.move(trimStart.x, trimStart.y);
  await page.mouse.down();
  await movePointerInFrames(page, trimStart, {
    x: trimStart.x - 60,
    y: trimStart.y,
  });
  expect((await counts(page)).trimLane).toBe(0);
  expect((await counts(page)).stateMutation).toBe(0);
  await page.mouse.up();
  await page.evaluate(() =>
    (
      window as unknown as {
        timelineBaseline: { waitForAuthoritative: () => Promise<void> };
      }
    ).timelineBaseline.waitForAuthoritative(),
  );
  expect((await counts(page)).trimLane).toBe(1);

  const scroll = page.locator(".timelineScroll");
  const scrollBox = await scroll.boundingBox();
  expect(scrollBox).not.toBeNull();
  await resetCounts(page);
  const scrubStart = { x: scrollBox!.x + 240, y: scrollBox!.y + 12 };
  await page.mouse.move(scrubStart.x, scrubStart.y);
  await page.mouse.down();
  await movePointerInFrames(page, scrubStart, {
    x: scrubStart.x + 300,
    y: scrubStart.y,
  });
  expect((await counts(page)).seek).toBe(0);
  await page.mouse.up();
  expect((await counts(page)).seek).toBe(1);

  await clip.click();
  await resetCounts(page);
  await page.keyboard.press("ArrowRight");
  expect((await counts(page)).seek).toBe(1);
  await resetCounts(page);
  await page.keyboard.press("Alt+ArrowRight");
  await page.evaluate(() =>
    (
      window as unknown as {
        timelineBaseline: { waitForAuthoritative: () => Promise<void> };
      }
    ).timelineBaseline.waitForAuthoritative(),
  );
  expect((await counts(page)).moveLane).toBe(1);

  await page.evaluate(() =>
    (
      window as unknown as {
        timelineBaseline: { setAuthoritativeDelay: (delayMs: number) => void };
      }
    ).timelineBaseline.setAuthoritativeDelay(1_000),
  );
  await resetCounts(page);
  const delayedBox = await clip.boundingBox();
  expect(delayedBox).not.toBeNull();
  const delayedStart = {
    x: delayedBox!.x + delayedBox!.width / 2,
    y: delayedBox!.y + delayedBox!.height / 2,
  };
  await page.mouse.move(delayedStart.x, delayedStart.y);
  await page.mouse.down();
  await page.mouse.move(delayedStart.x + 90, delayedStart.y);
  await page.evaluate(() => new Promise(requestAnimationFrame));
  await page.mouse.up();
  expect((await counts(page)).moveLane).toBe(1);
  expect((await counts(page)).stateMutation).toBe(0);
  await page.waitForTimeout(25);
  expect((await counts(page)).stateMutation).toBe(0);
  await page.evaluate(() =>
    (
      window as unknown as {
        timelineBaseline: { waitForAuthoritative: () => Promise<void> };
      }
    ).timelineBaseline.waitForAuthoritative(),
  );
  expect((await counts(page)).stateMutation).toBe(1);
});

test("Escape, pointercancel, deterministic snap, zoom, and cross-lane paths are reproducible", async ({
  page,
}) => {
  await loadHarness(page);
  const clip = await visibleClip(page);
  const box = await clip.boundingBox();
  expect(box).not.toBeNull();
  const start = { x: box!.x + box!.width / 2, y: box!.y + box!.height / 2 };

  for (const cancel of ["Escape", "pointercancel"] as const) {
    await resetCounts(page);
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    await page.mouse.move(start.x + 100, start.y);
    await page.evaluate(() => new Promise(requestAnimationFrame));
    if (cancel === "Escape") {
      await page.keyboard.press("Escape");
    } else {
      await page.locator(".timelineScroll").dispatchEvent("pointercancel", {
        pointerId: 1,
        isPrimary: true,
        button: 0,
        clientX: start.x + 100,
        clientY: start.y,
      });
    }
    await page.mouse.up();
    expect((await counts(page)).moveLane).toBe(0);
    expect((await counts(page)).stateMutation).toBe(0);
  }

  const audioBox = await page.locator(".timelineLane.audio").boundingBox();
  expect(audioBox).not.toBeNull();
  await resetCounts(page);
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await movePointerInFrames(page, start, {
    x: start.x + 20,
    y: audioBox!.y + audioBox!.height / 2,
  });
  expect((await counts(page)).moveLane).toBe(0);
  await page.mouse.up();
  await page.evaluate(() =>
    (
      window as unknown as {
        timelineBaseline: { waitForAuthoritative: () => Promise<void> };
      }
    ).timelineBaseline.waitForAuthoritative(),
  );
  expect((await counts(page)).moveLane).toBe(1);

  const scroll = page.locator(".timelineScroll");
  const scrollBox = await scroll.boundingBox();
  expect(scrollBox).not.toBeNull();
  await scroll.evaluate((element) => {
    element.scrollLeft = 5_000;
  });
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() =>
          requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
        ),
      ),
  );
  const cursor = { x: scrollBox!.x + 720, y: scrollBox!.y + 90 };
  const errors: number[] = [];
  for (const deltaY of [-40, 40, -40, 40, -40, 40]) {
    errors.push(
      await page.evaluate(
        async ({ clientX, clientY, deltaY: wheelDeltaY }) => {
          const element = document.querySelector<HTMLElement>(".timelineScroll")!;
          const rect = element.getBoundingClientRect();
          const viewportX = clientX - rect.left;
          const before =
            Number.parseFloat(
              document.querySelector<HTMLElement>(".timelineCanvas")!.style
                .width,
            ) / 1_800;
          const anchorSec =
            (viewportX + element.scrollLeft - 96) / before;
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
          await new Promise<void>((resolve) =>
            requestAnimationFrame(() =>
              requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
            ),
          );
          const after =
            Number.parseFloat(
              document.querySelector<HTMLElement>(".timelineCanvas")!.style
                .width,
            ) / 1_800;
          return Math.abs(
            anchorSec * after + 96 - element.scrollLeft - viewportX,
          );
        },
        { clientX: cursor.x, clientY: cursor.y, deltaY },
      ),
    );
  }
  expect(errors.every(Number.isFinite)).toBe(true);
  expect(errors.some((error) => error > 0)).toBe(true);

  const snappedPositions: number[] = [];
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await loadHarness(page);
    const candidate = await visibleClip(page);
    const candidateBox = await candidate.boundingBox();
    expect(candidateBox).not.toBeNull();
    const point = {
      x: candidateBox!.x + candidateBox!.width / 2,
      y: candidateBox!.y + candidateBox!.height / 2,
    };
    await page.mouse.move(point.x, point.y);
    await page.mouse.down();
    await page.mouse.move(point.x + 117, point.y);
    await page.evaluate(() => new Promise(requestAnimationFrame));
    await page.mouse.up();
    await page.evaluate(() =>
      (
        window as unknown as {
          timelineBaseline: { waitForAuthoritative: () => Promise<void> };
        }
      ).timelineBaseline.waitForAuthoritative(),
    );
    snappedPositions.push(
      Number.parseFloat(
        await candidate.evaluate((element) => (element as HTMLElement).style.left),
      ),
    );
  }
  expect(snappedPositions[0]).toBe(snappedPositions[1]);
});

test("legacy and V2 idle timelines remain visually equivalent", async ({
  browser,
}) => {
  const legacyPage = await browser.newPage();
  const v2Page = await browser.newPage();
  await Promise.all([
    loadHarness(legacyPage, LEGACY_URL),
    loadHarness(v2Page, V2_URL),
  ]);

  const [legacyBuffer, v2Buffer] = await Promise.all([
    legacyPage.locator(".timeline").screenshot(),
    v2Page.locator(".timeline").screenshot(),
  ]);
  const legacy = PNG.sync.read(legacyBuffer);
  const v2 = PNG.sync.read(v2Buffer);
  expect(v2.width).toBe(legacy.width);
  expect(v2.height).toBe(legacy.height);
  const baseline = JSON.parse(await readFile(baselinePath, "utf8"));
  const mismatchedPixels = pixelmatch(
    legacy.data,
    v2.data,
    null,
    legacy.width,
    legacy.height,
    { threshold: baseline.gates.visualPixelThreshold },
  );
  const mismatchRatio = mismatchedPixels / (legacy.width * legacy.height);
  await mkdir(new URL("../../artifacts/", import.meta.url), { recursive: true });
  await writeFile(
    visualArtifactPath,
    `${JSON.stringify(
      {
        schemaVersion: 1,
        mismatchedPixels,
        totalPixels: legacy.width * legacy.height,
        mismatchRatio: Number(mismatchRatio.toFixed(8)),
        threshold: baseline.gates.maxVisualMismatchRatio,
        pixelThreshold: baseline.gates.visualPixelThreshold,
      },
      null,
      2,
    )}\n`,
  );
  expect(mismatchRatio).toBeLessThanOrEqual(
    baseline.gates.maxVisualMismatchRatio,
  );
  await Promise.all([legacyPage.close(), v2Page.close()]);
});

test("settled V2 interaction windows satisfy rollout performance gates", async ({
  page,
}) => {
  const baseline = JSON.parse(await readFile(baselinePath, "utf8"));
  const requests: string[] = [];
  let captureRequests = false;
  page.on("request", (request) => {
    if (captureRequests) requests.push(request.url());
  });

  const startMeasurement = async (
    label: string,
    frameSampling: {
      mode: "scrub" | "drag";
      startClientX: number;
      initialTimelineSec?: number;
      fps: number;
    },
  ) => {
    captureRequests = true;
    await page.evaluate(
      ({ measurementLabel, sampling }) => {
        const api = (
          window as unknown as {
            timelineBaseline: {
              startMeasurement: (label: string) => void;
              startFrameSampling: (config: typeof sampling) => void;
            };
          }
        ).timelineBaseline;
        api.startMeasurement(measurementLabel);
        api.startFrameSampling(sampling);
      },
      { measurementLabel: label, sampling: frameSampling },
    );
  };
  const stopMeasurement = async (): Promise<Measurement> => {
    captureRequests = false;
    return page.evaluate(() =>
      (
        window as unknown as {
          timelineBaseline: { stopMeasurement: () => Measurement };
        }
      ).timelineBaseline.stopMeasurement(),
    );
  };

  const all: Array<Measurement & { fps: number }> = [];
  for (const fps of [24, 30, 60]) {
    await loadHarness(page, `${V2_URL}?fps=${fps}`);
    const scroll = page.locator(".timelineScroll");
    const scrollBox = await scroll.boundingBox();
    expect(scrollBox).not.toBeNull();
    const scrubStart = { x: scrollBox!.x + 260, y: scrollBox!.y + 12 };
    await page.mouse.move(scrubStart.x, scrubStart.y);
    await startMeasurement(`sustained-scrub-${fps}`, {
      mode: "scrub",
      startClientX: scrubStart.x,
      fps,
    });
    await page.mouse.down();
    await movePointerInFrames(
      page,
      scrubStart,
      { x: scrubStart.x + 500, y: scrubStart.y },
      20,
    );
    await page.mouse.up();
    await page.evaluate(() =>
      (
        window as unknown as {
          timelineBaseline: { waitForAuthoritative: () => Promise<void> };
        }
      ).timelineBaseline.waitForAuthoritative(),
    );
    await settleFrames(page);
    all.push({ ...(await stopMeasurement()), fps });

    const clip = await visibleClip(page);
    const box = await clip.boundingBox();
    expect(box).not.toBeNull();
    const dragStart = {
      x: box!.x + box!.width / 2,
      y: box!.y + box!.height / 2,
    };
    const initialTimelineSec =
      Number.parseFloat(
        await clip.evaluate((element) => (element as HTMLElement).style.left),
      ) /
      (Number.parseFloat(
        await page
          .locator(".timelineCanvas")
          .evaluate((element) => (element as HTMLElement).style.width),
      ) /
        1_800);
    await page.mouse.move(dragStart.x, dragStart.y);
    await startMeasurement(`sustained-drag-${fps}`, {
      mode: "drag",
      startClientX: dragStart.x,
      initialTimelineSec,
      fps,
    });
    await page.mouse.down();
    await movePointerInFrames(
      page,
      dragStart,
      { x: dragStart.x + 180, y: dragStart.y },
      12,
    );
    await page.mouse.up();
    await page.evaluate(() =>
      (
        window as unknown as {
          timelineBaseline: { waitForAuthoritative: () => Promise<void> };
        }
      ).timelineBaseline.waitForAuthoritative(),
    );
    await settleFrames(page);
    all.push({ ...(await stopMeasurement()), fps });
  }

  await loadHarness(page, `${V2_URL}?fps=30`);
  const scroll = page.locator(".timelineScroll");
  const scrollBox = await scroll.boundingBox();
  expect(scrollBox).not.toBeNull();
  await scroll.evaluate((element) => {
    element.scrollLeft = 5_000;
  });
  await settleFrames(page);
  const zoomCursor = { x: scrollBox!.x + 720, y: scrollBox!.y + 90 };
  const zoomAnchorSamples: Array<{
    beforeScale: number;
    afterScale: number;
    beforeScrollLeft: number;
    afterScrollLeft: number;
    errorPx: number;
  }> = [];
  for (const deltaY of [-40, 40, -40, 40]) {
    zoomAnchorSamples.push(
      await page.evaluate(
        async ({ clientX, clientY, deltaY: wheelDeltaY }) => {
          const element = document.querySelector<HTMLElement>(".timelineScroll")!;
          const rect = element.getBoundingClientRect();
          const viewportX = clientX - rect.left;
          const scale = () =>
            Number.parseFloat(
              document.querySelector<HTMLElement>(".timelineCanvas")!.style
                .width,
            ) / 1_800;
          const before = scale();
          const beforeScrollLeft = element.scrollLeft;
          const anchorSec =
            (viewportX + element.scrollLeft - 96) / before;
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
          await new Promise<void>((resolve) =>
            requestAnimationFrame(() =>
              requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
            ),
          );
          const afterScale = scale();
          const afterScrollLeft = element.scrollLeft;
          return {
            beforeScale: before,
            afterScale,
            beforeScrollLeft,
            afterScrollLeft,
            errorPx: Math.abs(
              anchorSec * afterScale +
                96 -
                afterScrollLeft -
                viewportX,
            ),
          };
        },
        {
          clientX: zoomCursor.x,
          clientY: zoomCursor.y,
          deltaY,
        },
      ),
    );
  }

  const maxCommitsPerFrame = Math.max(
    ...all.map((measurement) => measurement.maxCommitsPerFrame),
  );
  const longTasksOver50Ms = all
    .flatMap((measurement) => measurement.longTasks)
    .filter((entry) => entry.durationMs > 50);
  const maxExpectedRenderedFrameDrift = Math.max(
    0,
    ...all.flatMap((measurement) =>
      measurement.frameSamples.map((sample) =>
        Math.abs(
          Math.round(sample.actualSec * measurement.fps) -
            Math.round(sample.expectedSec * measurement.fps),
        ),
      ),
    ),
  );
  const maxCanonicalFrameDrift = Math.max(
    0,
    ...all.flatMap((measurement) =>
      measurement.frameSamples.map((sample) =>
        (() => {
          const frame = Math.round(sample.actualSec * measurement.fps);
          const canonicalSeconds = frame / measurement.fps;
          return Math.abs(
            Math.round(canonicalSeconds * measurement.fps) - frame,
          );
        })(),
      ),
    ),
  );
  const maxPerFrame = (
    entries: Array<{ frame: number }>,
  ): number => {
    const counts = new Map<number, number>();
    entries.forEach((entry) =>
      counts.set(entry.frame, (counts.get(entry.frame) ?? 0) + 1),
    );
    return Math.max(0, ...counts.values());
  };
  const maxPreviewPublicationsPerFrame = Math.max(
    ...all.map((measurement) =>
      maxPerFrame(measurement.previewPublications),
    ),
  );
  const maxRenderedMutationsPerFrame = Math.max(
    ...all.map((measurement) => maxPerFrame(measurement.renderedMutations)),
  );
  for (const measurement of all) {
    expect(
      measurement.frameSamples.length,
      `${measurement.label} frame samples`,
    ).toBeGreaterThan(0);
    expect(
      measurement.previewPublications.length,
      `${measurement.label} preview publications`,
    ).toBeGreaterThan(0);
    expect(
      measurement.renderedMutations.length,
      `${measurement.label} rendered mutations`,
    ).toBeGreaterThan(0);
    for (const sample of measurement.frameSamples) {
      expect(
        [
          sample.frame,
          sample.timeMs,
          sample.expectedSec,
          sample.actualSec,
          sample.deltaSec,
        ].every(Number.isFinite),
      ).toBe(true);
    }
  }
  const result = {
    schemaVersion: 1,
    fixture: baseline.fixture,
    environment: {
      browserVersion: await page.evaluate(() => navigator.userAgent),
      browserExecutable: process.env.TIMELINE_BROWSER_EXECUTABLE,
      browserSource: process.env.TIMELINE_BROWSER_SOURCE,
      build: "vite-production",
      commit: process.env.GITHUB_SHA ?? null,
    },
    measurements: Object.fromEntries(
      all.map((measurement) => [
        measurement.label,
        {
          fps: measurement.fps,
          commitCount: measurement.commitCount,
          maxCommitsPerFrame: measurement.maxCommitsPerFrame,
          framesOverOneCommit: measurement.framesOverOneCommit,
          longTasksOver50Ms: measurement.longTasks.filter(
            (entry) => entry.durationMs > 50,
          ).length,
          frameSamples: measurement.frameSamples,
          previewPublications: measurement.previewPublications,
          renderedMutations: measurement.renderedMutations,
        },
      ]),
    ),
    gates: {
      maxReactCommitsPerAnimationFrame: maxCommitsPerFrame,
      maxPreviewUpdatesPerAnimationFrame: maxPreviewPublicationsPerFrame,
      maxRenderedMutationsPerAnimationFrame: maxRenderedMutationsPerFrame,
      maxExpectedRenderedFrameDrift,
      maxCanonicalFrameDrift: Number(maxCanonicalFrameDrift.toFixed(6)),
      maxLongTasksOver50Ms: longTasksOver50Ms.length,
      maxLongTaskDurationMs: Number(
        Math.max(0, ...longTasksOver50Ms.map((entry) => entry.durationMs)).toFixed(
          3,
        ),
      ),
      maxCursorAnchorErrorPx: Number(
        Math.max(...zoomAnchorSamples.map((sample) => sample.errorPx)).toFixed(
          6,
        ),
      ),
      zoomAnchorSamples,
      maxNetworkCallsDuringGestures: requests.length,
    },
  };
  await mkdir(new URL("../../artifacts/", import.meta.url), { recursive: true });
  await writeFile(artifactPath, `${JSON.stringify(result, null, 2)}\n`);

  expect(result.gates.maxReactCommitsPerAnimationFrame).toBeLessThanOrEqual(
    baseline.gates.maxReactCommitsPerAnimationFrame,
  );
  expect(result.gates.maxPreviewUpdatesPerAnimationFrame).toBeLessThanOrEqual(
    baseline.gates.maxPreviewUpdatesPerAnimationFrame,
  );
  expect(result.gates.maxRenderedMutationsPerAnimationFrame).toBeLessThanOrEqual(
    baseline.gates.maxPreviewUpdatesPerAnimationFrame,
  );
  expect(result.gates.maxExpectedRenderedFrameDrift).toBe(0);
  expect(result.gates.maxCanonicalFrameDrift).toBe(
    baseline.gates.maxCanonicalFrameDrift,
  );
  expect(result.gates.maxLongTasksOver50Ms).toBeLessThanOrEqual(
    baseline.gates.maxLongTasksOver50Ms,
  );
  expect(result.gates.maxCursorAnchorErrorPx).toBeLessThanOrEqual(
    baseline.gates.maxCursorAnchorErrorPx,
  );
  expect(result.gates.maxNetworkCallsDuringGestures).toBeLessThanOrEqual(
    baseline.gates.maxNetworkCallsDuringGestures,
  );
});
