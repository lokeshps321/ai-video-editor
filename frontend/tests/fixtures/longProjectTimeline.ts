import type {
  TimelineCaptionBlock,
  TimelineLane,
  TimelineProps,
} from "../../src/components/Timeline";
import type { Clip, TranscriptWord } from "../../src/types";

const DURATION_SEC = 30 * 60;
const PRIMARY_CLIP_DURATION_SEC = 12;
const PRIMARY_CLIP_COUNT = DURATION_SEC / PRIMARY_CLIP_DURATION_SEC;
const SECONDARY_CLIP_COUNT = 45;
const BROLL_CLIP_COUNT = 120;
const TRANSCRIPT_WORD_COUNT = 4_500;
const CAPTION_BLOCK_COUNT = 600;
const WAVEFORM_PEAK_COUNT = 9_000;

const WORDS = [
  "today",
  "we",
  "explore",
  "a",
  "practical",
  "workflow",
  "for",
  "editing",
  "long",
  "form",
  "video",
  "with",
  "clear",
  "examples",
  "and",
  "useful",
  "context",
] as const;

export type LongProjectFixtureData = Pick<
  TimelineProps,
  | "words"
  | "timelineLanes"
  | "assetUrlById"
  | "assetNameById"
  | "assetDurationById"
  | "overlayClips"
  | "captionBlocks"
  | "durationSec"
  | "currentTimeSec"
  | "deletedWordIds"
  | "selectedWordIds"
  | "activeWordId"
  | "waveformPeaks"
  | "selectedLaneClipId"
  | "selectedBrollClipId"
  | "selectedCaptionId"
  | "lockedLaneIds"
>;

export type LongProjectFixture = {
  metadata: {
    name: string;
    durationMinutes: number;
    durationSec: number;
    laneCount: number;
    laneClipCount: number;
    primaryVideoClipCount: number;
    secondaryVideoClipCount: number;
    audioClipCount: number;
    transcriptWordCount: number;
    captionBlockCount: number;
    brollClipCount: number;
    waveformPeakCount: number;
    filmstripRequestsEnabled: boolean;
  };
  timelineProps: LongProjectFixtureData;
};

function createClip(
  id: string,
  assetId: string,
  timelineStartSec: number,
  durationSec: number,
  opacity = 1,
): Clip {
  return {
    id,
    asset_id: assetId,
    start_sec: 0,
    end_sec: durationSec,
    timeline_start_sec: timelineStartSec,
    speed: 1,
    broll_opacity: opacity,
    transform: {
      crop: null,
      crop_keyframes: [],
      scale: null,
      rotate: 0,
      flip: null,
    },
    adjustments: {
      brightness: 0,
      contrast: 0,
      saturation: 0,
      exposure: 0,
      temperature: 0,
      preset: null,
    },
    audio: {
      volume: 1,
      fade_in_sec: 0,
      fade_out_sec: 0,
      mute: false,
      keyframes: [],
    },
    transition: null,
    text_overlays: [],
  };
}

function createPrimaryClips(prefix: string): Clip[] {
  return Array.from({ length: PRIMARY_CLIP_COUNT }, (_, index) =>
    createClip(
      `${prefix}-clip-${index}`,
      `${prefix}-asset-${index}`,
      index * PRIMARY_CLIP_DURATION_SEC,
      PRIMARY_CLIP_DURATION_SEC,
    ),
  );
}

function createSecondaryClips(): Clip[] {
  return Array.from({ length: SECONDARY_CLIP_COUNT }, (_, index) => {
    const timelineStartSec = index * 40 + 5;
    const durationSec = index % 3 === 0 ? 7.5 : 5;
    return createClip(
      `v2-clip-${index}`,
      `v2-asset-${index}`,
      timelineStartSec,
      durationSec,
    );
  });
}

function createBrollClips(): Clip[] {
  return Array.from({ length: BROLL_CLIP_COUNT }, (_, index) => {
    const timelineStartSec = index * 15 + 4;
    const durationSec = index % 4 === 0 ? 7 : 5.5;
    return createClip(
      `broll-clip-${index}`,
      `broll-asset-${index}`,
      timelineStartSec,
      durationSec,
      0.72 + (index % 4) * 0.08,
    );
  });
}

function createWords(): TranscriptWord[] {
  const wordStepSec = DURATION_SEC / TRANSCRIPT_WORD_COUNT;
  return Array.from({ length: TRANSCRIPT_WORD_COUNT }, (_, index) => {
    const startSec = index * wordStepSec;
    const speakerIndex = Math.floor(startSec / 45) % 2;
    return {
      id: `word-${index}`,
      text: WORDS[index % WORDS.length],
      start_sec: Number(startSec.toFixed(3)),
      end_sec: Number((startSec + wordStepSec * 0.8).toFixed(3)),
      confidence: index % 29 === 0 ? 0.74 : 0.96,
      quality_score: index % 29 === 0 ? 0.7 : 0.95,
      quality_label: index % 29 === 0 ? "weak" : "trusted",
      source_pass: index % 29 === 0 ? "retry" : "primary",
      speaker_id: `speaker-${speakerIndex + 1}`,
      speaker_label: speakerIndex === 0 ? "Host" : "Guest",
    };
  });
}

function createCaptionBlocks(): TimelineCaptionBlock[] {
  const captionStepSec = DURATION_SEC / CAPTION_BLOCK_COUNT;
  return Array.from({ length: CAPTION_BLOCK_COUNT }, (_, index) => {
    const startSec = index * captionStepSec + 0.12;
    const primaryClipIndex = Math.floor(startSec / PRIMARY_CLIP_DURATION_SEC);
    const clipTimelineStartSec =
      primaryClipIndex * PRIMARY_CLIP_DURATION_SEC;
    const firstWordIndex = index * 7;
    const text = Array.from(
      { length: 7 },
      (_, wordOffset) => WORDS[(firstWordIndex + wordOffset) % WORDS.length],
    ).join(" ");
    return {
      id: `caption-${index}`,
      clipId: `v1-clip-${primaryClipIndex}`,
      laneId: "video-primary",
      laneLabel: "Video 1",
      text,
      style: index % 8 === 0 ? "highlight" : "default",
      startSec: Number(startSec.toFixed(3)),
      durationSec: 2.65,
      clipTimelineStartSec,
      clipSourceDurationSec: PRIMARY_CLIP_DURATION_SEC,
      clipSpeed: 1,
    };
  });
}

function createWaveformPeaks(): number[] {
  return Array.from({ length: WAVEFORM_PEAK_COUNT }, (_, index) => {
    const speechEnvelope = 0.35 + 0.35 * Math.abs(Math.sin(index * 0.071));
    const detail = 0.2 * Math.abs(Math.sin(index * 0.317));
    return Number(Math.min(1, speechEnvelope + detail).toFixed(4));
  });
}

/**
 * Deterministic, in-memory data for timeline unit and browser performance tests.
 *
 * Filmstrip URL entries are deliberately omitted. Timeline currently uses their
 * presence to render thumbnail images, which would create backend requests and
 * make gesture traces non-hermetic. Tests dedicated to filmstrips can populate
 * `assetUrlById` and intercept thumbnail requests explicitly.
 */
export function createLongProjectTimelineFixture(): LongProjectFixture {
  const primaryVideoClips = createPrimaryClips("v1");
  const secondaryVideoClips = createSecondaryClips();
  const audioClips = createPrimaryClips("a1");
  const overlayClips = createBrollClips();

  const timelineLanes: TimelineLane[] = [
    {
      id: "video-primary",
      label: "Video 1",
      kind: "video",
      clips: primaryVideoClips,
      volume: 1,
    },
    {
      id: "video-secondary",
      label: "Video 2",
      kind: "video",
      clips: secondaryVideoClips,
      volume: 1,
    },
    {
      id: "audio-dialogue",
      label: "Dialogue",
      kind: "audio",
      clips: audioClips,
      volume: 0.92,
    },
  ];

  const assetNameById = new Map<string, string>();
  const assetDurationById = new Map<string, number | null>();
  for (const clip of [...primaryVideoClips, ...secondaryVideoClips, ...audioClips]) {
    assetNameById.set(clip.asset_id, `${clip.asset_id}.mp4`);
    assetDurationById.set(clip.asset_id, clip.end_sec + 20);
  }

  return {
    metadata: {
      name: "30-minute interview with cutaways",
      durationMinutes: 30,
      durationSec: DURATION_SEC,
      laneCount: timelineLanes.length,
      laneClipCount:
        primaryVideoClips.length +
        secondaryVideoClips.length +
        audioClips.length,
      primaryVideoClipCount: primaryVideoClips.length,
      secondaryVideoClipCount: secondaryVideoClips.length,
      audioClipCount: audioClips.length,
      transcriptWordCount: TRANSCRIPT_WORD_COUNT,
      captionBlockCount: CAPTION_BLOCK_COUNT,
      brollClipCount: overlayClips.length,
      waveformPeakCount: WAVEFORM_PEAK_COUNT,
      filmstripRequestsEnabled: false,
    },
    timelineProps: {
      words: createWords(),
      timelineLanes,
      assetUrlById: new Map(),
      assetNameById,
      assetDurationById,
      overlayClips,
      captionBlocks: createCaptionBlocks(),
      durationSec: DURATION_SEC,
      currentTimeSec: DURATION_SEC * 0.42,
      deletedWordIds: new Set(["word-220", "word-221", "word-222"]),
      selectedWordIds: new Set(),
      activeWordId: "word-1890",
      waveformPeaks: createWaveformPeaks(),
      selectedLaneClipId: null,
      selectedBrollClipId: null,
      selectedCaptionId: null,
      lockedLaneIds: new Set(["video-secondary"]),
    },
  };
}
