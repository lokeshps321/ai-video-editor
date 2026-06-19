import {
  Captions,
  Clapperboard,
  Crop,
  Download,
  Scissors,
  Sparkles,
  type LucideIcon,
} from "lucide-react";

import type { VibeAction } from "../types";

export const LOW_CONFIDENCE_THRESHOLD = 0.6;
export const LOW_CONFIDENCE_WARN_RATIO = 0.18;
export const LOW_CONFIDENCE_WARN_MIN_COUNT = 30;

export const FILLER_SINGLE_WORDS_CONSERVATIVE = new Set([
  "um", "uh", "uhm", "umm", "hmm", "hm", "mm",
  "ah", "er", "erm", "eh", "huh", "mhm",
]);

export const FILLER_SINGLE_WORDS_AGGRESSIVE = new Set([
  "like", "basically", "literally", "actually", "right",
]);

export const ENABLE_AGGRESSIVE_FILLER_SINGLE_WORDS = ["1", "true", "yes", "on"].includes(
  String(import.meta.env.VITE_FILLER_AGGRESSIVE_SINGLE_WORDS ?? "false").trim().toLowerCase()
);

export const FILLER_MULTI_WORD_PHRASES: ReadonlyArray<ReadonlyArray<string>> = [
  ["you", "know"],
  ["i", "mean"],
  ["sort", "of"],
  ["kind", "of"],
  ["so", "yeah"],
];

export const TRANSCRIPT_LANGUAGE_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "auto", label: "Language: Auto" },
  { value: "en", label: "English" },
  { value: "kn", label: "Kannada" },
  { value: "hi", label: "Hindi" },
  { value: "ta", label: "Tamil" },
  { value: "te", label: "Telugu" },
  { value: "ml", label: "Malayalam" },
  { value: "mr", label: "Marathi" },
  { value: "bn", label: "Bengali" },
  { value: "gu", label: "Gujarati" },
  { value: "pa", label: "Punjabi" },
  { value: "or", label: "Odia" },
  { value: "ur", label: "Urdu" },
];

export const TRANSCRIPT_MODE_OPTIONS: ReadonlyArray<{ value: "auto" | "speech" | "song"; label: string }> = [
  { value: "auto", label: "Mode: Auto" },
  { value: "speech", label: "Speech" },
  { value: "song", label: "Song" },
];

export type CaptionStylePreset = {
  id: string;
  name: string;
  desc: string;
  color: string;
  preview_words: [string, string, string];
  preview_class: string;
  config: {
    font_name: string;
    font_size: number;
    primary_color: string;
    highlight_color: string;
    outline_color: string;
    outline_width: number;
    shadow: number;
    alignment: number;
    margin_v: number;
    render_style?: string;
  };
};

export const CAPTION_STYLE_PRESETS: ReadonlyArray<CaptionStylePreset> = [
  {
    id: "basic_white",
    name: "Basic White",
    desc: "Classic white captions",
    color: "#ffffff",
    preview_words: ["Clean", "clear", "subtitles"],
    preview_class: "basic",
    config: {
      font_name: "Arial-Bold",
      font_size: 26,
      primary_color: "&H00FFFFFF",
      highlight_color: "&H0000DDFF",
      outline_color: "&H00000000",
      outline_width: 3,
      shadow: 2,
      alignment: 2,
      margin_v: 60,
    },
  },
  {
    id: "hormozi_green",
    name: "Hormozi Green",
    desc: "Bold and punchy (viral retention)",
    color: "#00FF00",
    preview_words: ["This", "HOOKS", "fast"],
    preview_class: "hormozi",
    config: {
      font_name: "Arial-Black",
      font_size: 28,
      primary_color: "&H00FFFFFF",
      highlight_color: "&H0000FF00",
      outline_color: "&H00000000",
      outline_width: 3,
      shadow: 2,
      alignment: 2,
      margin_v: 140,
    },
  },
  {
    id: "shorts_viral",
    name: "MrBeast Yellow",
    desc: "High-contrast YouTube Shorts style",
    color: "#FFD600",
    preview_words: ["Watch", "THIS", "part"],
    preview_class: "viral",
    config: {
      font_name: "Impact",
      font_size: 30,
      primary_color: "&H00FFFFFF",
      highlight_color: "&H0000FFFF",
      outline_color: "&H00000000",
      outline_width: 3,
      shadow: 1,
      alignment: 2,
      margin_v: 140,
    },
  },
  {
    id: "neon_gamer",
    name: "Cyberpunk Red",
    desc: "Aggressive neon red highlight",
    color: "#FF003C",
    preview_words: ["Level", "UP", "now"],
    preview_class: "neon",
    config: {
      font_name: "Arial-Bold",
      font_size: 26,
      primary_color: "&H00FFFFFF",
      highlight_color: "&H003C00FF",
      outline_color: "&H00000000",
      outline_width: 3,
      shadow: 2,
      alignment: 2,
      margin_v: 110,
    },
  },
  {
    id: "pop_cyan",
    name: "Electric Cyan",
    desc: "Bright blue/cyan pop-out effect",
    color: "#00FFFF",
    preview_words: ["Fresh", "POP", "energy"],
    preview_class: "cyan",
    config: {
      font_name: "Arial-Bold",
      font_size: 26,
      primary_color: "&H00FFFFFF",
      highlight_color: "&H00FFFF00",
      outline_color: "&H00000000",
      outline_width: 3,
      shadow: 2,
      alignment: 2,
      margin_v: 110,
    },
  },
  {
    id: "minimalist",
    name: "Minimalist Soft",
    desc: "Aesthetic pastel subtitles",
    color: "#F48FB1",
    preview_words: ["soft", "story", "moment"],
    preview_class: "minimal",
    config: {
      font_name: "Arial-Bold",
      font_size: 24,
      primary_color: "&H00FFFFFF",
      highlight_color: "&H00D4B2FF",
      outline_color: "&H00444444",
      outline_width: 1,
      shadow: 0,
      alignment: 2,
      margin_v: 60,
    },
  },
  {
    id: "desi_gold",
    name: "Desi Gold",
    desc: "Warm gold accents — made for Indian creators",
    color: "#FFD700",
    preview_words: ["Namaste", "DESI", "vibes"],
    preview_class: "desi",
    config: {
      font_name: "Noto-Sans-Devanagari-Bold",
      font_size: 28,
      primary_color: "&H00FFFFFF",
      highlight_color: "&H0000D7FF",
      outline_color: "&H00000000",
      outline_width: 3,
      shadow: 2,
      alignment: 2,
      margin_v: 120,
    },
  },
  {
    id: "indic_clean",
    name: "Indic Clean",
    desc: "Clean Indic-script captions with native font rendering",
    color: "#80DEEA",
    preview_words: ["clear", "script", "clean"],
    preview_class: "indic",
    config: {
      font_name: "Noto-Sans-Kannada-Bold",
      font_size: 26,
      primary_color: "&H00FFFFFF",
      highlight_color: "&H00EADE80",
      outline_color: "&H00222222",
      outline_width: 2,
      shadow: 1,
      alignment: 2,
      margin_v: 80,
    },
  },
];

export const CAPTION_STYLE_CONFIG_BY_ID = CAPTION_STYLE_PRESETS.reduce<Record<string, CaptionStylePreset["config"]>>(
  (acc, style) => {
    acc[style.id] = style.config;
    return acc;
  },
  {}
);

export type FeatureTabId = "broll_studio" | "ai_actions" | "captions" | "export";

export const FEATURE_TAB_ITEMS: ReadonlyArray<{ id: FeatureTabId; label: string; icon: LucideIcon }> = [
  { id: "broll_studio", label: "B-roll", icon: Clapperboard },
  { id: "ai_actions", label: "AI Tools", icon: Sparkles },
  { id: "captions", label: "Captions", icon: Captions },
  { id: "export", label: "Export", icon: Download },
];

export const AI_ACTION_ITEMS: ReadonlyArray<{
  action: VibeAction;
  label: string;
  desc: string;
  icon: LucideIcon;
  primary?: boolean;
}> = [
  {
    action: "auto_cut_pauses",
    label: "Auto Cut Pauses",
    desc: "Remove dead air, awkward pauses, and filler gaps.",
    icon: Scissors,
  },
  {
    action: "trim_start_end",
    label: "Trim Start & End",
    desc: "Tighten the intro and outro around spoken content.",
    icon: Crop,
  },
  {
    action: "add_subtitles",
    label: "Add Captions",
    desc: "Generate styled captions from the current transcript.",
    icon: Captions,
    primary: true,
  },
];
