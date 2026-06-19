import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { motion, type Variants } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import { SignedIn, SignedOut, UserButton } from "@clerk/clerk-react";
import {
  AudioWaveform,
  Captions,
  ChevronRight,
  Globe2,
  Languages,
  Layers,
  Menu,
  MessageSquareText,
  Play,
  Scissors,
  Smartphone,
  Sparkles,
  UploadCloud,
  X,
  Zap,
} from "lucide-react";
import { setPendingUploadFile } from "./lib/pendingUpload";
import { BRAND } from "./config/brand";
import HeroVisual from "./HeroVisual";
import DemoStorySection from "./DemoStorySection";
import BrandLogo from "./components/BrandLogo";

import "./landing.css";

const heroEase = [0.16, 1, 0.3, 1] as const;
const sectionEase = [0.215, 0.61, 0.355, 1] as const;

const sectionReveal: Variants = {
  hidden: { opacity: 0, y: 26 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.65,
      ease: sectionEase,
    },
  },
};

const cardReveal: Variants = {
  hidden: { opacity: 0, y: 18 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.55,
      ease: sectionEase,
    },
  },
};

const gridReveal: Variants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.08,
    },
  },
};

const navSections = [
  { href: "#demo", label: "Demo" },
  { href: "#workflow", label: "Workflow" },
];

const tickerItems = [
  "Transcript-first cuts",
  "AI captions",
  "9:16 exports",
  "B-roll studio",
  "Indian language ASR",
  "Hook-ready timelines",
  "Filler removal",
  "Creator presets",
];

const workflow = [
  {
    icon: UploadCloud,
    label: "Upload",
    title: "Drop your talking-head clip",
    text: "Upload or drag a video. ClipMind opens the same transcript-first workspace you see in the editor.",
  },
  {
    icon: MessageSquareText,
    label: "Transcript",
    title: "Edit by spoken text",
    text: "Generate word-level transcript, select phrases, and delete filler by text — not razor blades.",
  },
  {
    icon: Zap,
    label: "Quick Edit",
    title: "One-click cut + captions",
    text: "Transcribe, auto-cut pauses, and apply captions in one pass. B-roll stays optional in the studio drawer.",
  },
  {
    icon: Smartphone,
    label: "Export",
    title: "Ship vertical",
    text: "Render 9:16 for Reels, Shorts, and TikTok from the same timeline.",
  },
];

const features = [
  {
    icon: Scissors,
    title: "Cut by text",
    text: "Trim video by editing the transcript, then fine-tune the timeline.",
  },
  {
    icon: Captions,
    title: "Caption engine",
    text: "Generate short-form captions with style presets and language-aware text.",
  },
  {
    icon: Layers,
    title: "B-roll planner",
    text: "Turn transcript moments into suggested cutaways and visual beats.",
  },
  {
    icon: Languages,
    title: "Bharat-ready ASR",
    text: "Work across all 12 supported Indian languages with language-aware transcription and captions.",
  },
];

function TickerRibbon() {
  return (
    <section className="ticker-strip" aria-label="ClipMind capabilities">
      <div className="ticker-track" aria-hidden="true">
        {[0, 1].map((copy) => (
          <div className="ticker-content" key={copy}>
            {tickerItems.map((item) => (
              <span className="ticker-item" key={`${copy}-${item}`}>
                <Zap size={16} />
                {item}
              </span>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  text,
}: {
  icon: typeof Scissors;
  title: string;
  text: string;
}) {
  return (
    <motion.article className="feature-card" variants={cardReveal}>
      <div className="feature-icon">
        <Icon size={22} />
      </div>
      <h3>{title}</h3>
      <p>{text}</p>
    </motion.article>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const [heroDragOver, setHeroDragOver] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const dropInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.title = BRAND.landingDocumentTitle;
  }, []);

  useEffect(() => {
    if (!mobileMenuOpen) return;
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setMobileMenuOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [mobileMenuOpen]);

  const handleFile = (file?: File) => {
    if (!file?.type.startsWith("video/")) return;
    setPendingUploadFile(file);
    navigate("/editor");
  };

  const openFilePicker = () => dropInputRef.current?.click();

  const handleDropKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    openFilePicker();
  };

  return (
    <div className="landing-page">
      <div className="page-texture" aria-hidden="true" />

      <nav className="glass-nav">
        <div className="nav-container">
          <BrandLogo variant="nav" />
          <div className="nav-actions">
            <div
              id="landing-nav-links"
              className={`nav-section-links${mobileMenuOpen ? " is-open" : ""}`}
            >
              {navSections.map((section) => (
                <a
                  key={section.href}
                  href={section.href}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  {section.label}
                </a>
              ))}
            </div>
            <SignedOut>
              <Link
                to="/sign-in"
                className="btn-secondary-small"
                onClick={() => setMobileMenuOpen(false)}
              >
                Sign In
              </Link>
              <Link
                to="/sign-up"
                className="btn-primary-small"
                onClick={() => setMobileMenuOpen(false)}
              >
                Get Started
                <ChevronRight size={16} />
              </Link>
            </SignedOut>
            <SignedIn>
              <UserButton afterSignOutUrl="/" />
              <Link
                to="/editor"
                className="btn-primary-small"
                onClick={() => setMobileMenuOpen(false)}
              >
                Launch Editor
                <ChevronRight size={16} />
              </Link>
            </SignedIn>
            <button
              type="button"
              className="nav-menu-toggle"
              aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
              aria-expanded={mobileMenuOpen}
              aria-controls="landing-nav-links"
              onClick={() => setMobileMenuOpen((open) => !open)}
            >
              {mobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
            </button>
          </div>
        </div>
      </nav>

      <main>
        <section className="hero-section">
          <div className="hero-shell">
            <motion.div
              className="hero-copy"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: heroEase }}
            >
              <span className="hero-badge">
                <Sparkles size={16} />
                AI editing studio for Reels, Shorts, and TikToks
              </span>
              <h1>ClipMind</h1>
              <p className="hero-kicker">
                Turn raw talking-head videos into captioned vertical shorts
                without wrestling a traditional timeline.
              </p>

              <div className="hero-actions">
                <Link to="/editor" className="btn-glow">
                  Start editing
                  <ChevronRight size={18} />
                </Link>
                <a href="#demo" className="btn-secondary">
                  <Play size={17} fill="currentColor" />
                  Watch flow
                </a>
              </div>

              <div
                className={`hero-dropzone ${heroDragOver ? "dragover" : ""}`}
                role="button"
                tabIndex={0}
                onKeyDown={handleDropKey}
                onDragOver={(event) => {
                  event.preventDefault();
                  setHeroDragOver(true);
                }}
                onDragLeave={() => setHeroDragOver(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setHeroDragOver(false);
                  handleFile(event.dataTransfer.files[0]);
                }}
                onClick={openFilePicker}
              >
                <input
                  ref={dropInputRef}
                  type="file"
                  accept="video/*"
                  className="sr-only"
                  onChange={(event) => handleFile(event.target.files?.[0])}
                />
                <UploadCloud size={19} />
                <span>Drop a video or browse</span>
              </div>

              <div className="hero-stats" aria-label="ClipMind highlights">
                <span>
                  <strong>{BRAND.languageCount}</strong> Indian languages
                </span>
                <span className="hero-stat hero-stat-export">
                  <strong>9:16</strong> export-first
                </span>
                <span>
                  <strong>Text</strong> driven cuts
                </span>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 36, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.9, delay: 0.12, ease: heroEase }}
            >
              <HeroVisual />
            </motion.div>
          </div>
        </section>

        <TickerRibbon />

        <section
          className="social-proof-strip"
          aria-label="Creator social proof"
        >
          <blockquote className="social-proof-quote">
            <p>{BRAND.socialProofQuote}</p>
            <footer>{BRAND.socialProofMetric}</footer>
          </blockquote>
        </section>

        <DemoStorySection />

        <motion.section
          id="workflow"
          className="workflow-section"
          variants={sectionReveal}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.22 }}
          transition={{ duration: 0.55, ease: heroEase }}
        >
          <div className="workflow-header">
            <span className="section-eyebrow">
              <AudioWaveform size={16} /> Creator workflow
            </span>
            <h2>From raw speech to publishable short.</h2>
          </div>
          <motion.div
            className="workflow-grid"
            variants={gridReveal}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.24 }}
          >
            {workflow.map((step, index) => {
              const Icon = step.icon;
              return (
                <motion.article
                  className="workflow-card"
                  key={step.label}
                  variants={cardReveal}
                >
                  <span className="step-number">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <div className="workflow-icon">
                    <Icon size={22} />
                  </div>
                  <span className="workflow-label">{step.label}</span>
                  <h3>{step.title}</h3>
                  <p>{step.text}</p>
                </motion.article>
              );
            })}
          </motion.div>
        </motion.section>

        <motion.section
          className="language-section"
          variants={sectionReveal}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.55, ease: heroEase }}
        >
          <motion.div className="language-card" variants={cardReveal}>
            <div>
              <span className="section-eyebrow">
                <Globe2 size={16} /> Built for Bharat
              </span>
              <h2>Speak naturally. Edit clearly.</h2>
              <p>{BRAND.heroSubtitle}</p>
            </div>
            <div className="language-grid">
              {BRAND.supportedLanguages.map((lang) => (
                <span key={lang}>{lang}</span>
              ))}
            </div>
          </motion.div>
        </motion.section>

        <motion.section
          className="features-section"
          variants={sectionReveal}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.55, ease: heroEase }}
        >
          <div className="section-copy compact">
            <span className="section-eyebrow">
              <Layers size={16} /> Studio engine
            </span>
            <h2>{BRAND.featuresSectionTitle}</h2>
          </div>
          <motion.div
            className="features-grid"
            variants={gridReveal}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.24 }}
          >
            {features.map((feature) => (
              <FeatureCard key={feature.title} {...feature} />
            ))}
          </motion.div>
        </motion.section>

        <motion.section
          className="final-cta"
          variants={sectionReveal}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.26 }}
          transition={{ duration: 0.55, ease: heroEase }}
        >
          <motion.div className="final-cta-inner" variants={cardReveal}>
            <span className="section-eyebrow">
              <Sparkles size={16} /> Start with a clip
            </span>
            <h2>Create like a pro.</h2>
            <p>
              Upload one talking-head video and turn the strongest moments into
              vertical shorts.
            </p>
            <Link to="/editor" className="btn-glow">
              Enter the editor
              <ChevronRight size={18} />
            </Link>
          </motion.div>
        </motion.section>
      </main>

      <footer className="glass-footer">
        <div className="container footer-content">
          <BrandLogo variant="footer" />
        </div>
      </footer>
    </div>
  );
}
