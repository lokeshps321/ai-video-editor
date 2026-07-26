import { useRef } from "react";
import {
    motion,
    useReducedMotion,
    useScroll,
    useSpring,
    useTransform,
    type MotionValue,
} from "framer-motion";
import type { LucideIcon } from "lucide-react";
import {
    BadgeCheck,
    Captions,
    Download,
    Scissors,
    Smartphone,
    Sparkles,
    UploadCloud,
    Wand2,
    Zap,
} from "lucide-react";

const storySteps = [
    { id: "upload", label: "Upload", icon: UploadCloud, at: 0.08 },
    { id: "transcript", label: "Transcript", icon: Wand2, at: 0.32 },
    { id: "quick_edit", label: "Quick Edit", icon: Zap, at: 0.58 },
    { id: "export", label: "Export", icon: Smartphone, at: 0.82 },
];

const quickEditStages = [
    { label: "Transcribing", at: 0.38 },
    { label: "Cutting pauses", at: 0.52 },
    { label: "Adding captions", at: 0.66 },
];

const aiActions = [
    { icon: Scissors, text: "Auto Cut Pauses", at: 0.44 },
    { icon: Captions, text: "Add Captions", at: 0.58 },
    { icon: Smartphone, text: "1080×1920 render", at: 0.72 },
];

const springConfig = {
    stiffness: 120,
    damping: 28,
    mass: 0.35,
    restDelta: 0.001,
} as const;

function StoryStep({
    label,
    icon: Icon,
    at,
    progress,
}: {
    label: string;
    icon: LucideIcon;
    at: number;
    progress: MotionValue<number>;
}) {
    const opacity = useTransform(progress, [at - 0.02, at + 0.12], [0.45, 1]);
    const scale = useTransform(progress, [at - 0.02, at + 0.12], [0.85, 1.05]);
    const active = useTransform(progress, [at, at + 0.08], [0, 1]);

    return (
        <motion.div className="demo-story-step" style={{ opacity }}>
            <motion.span
                className="demo-story-step-dot"
                style={{
                    scale,
                    // MotionValue CSS var for GPU-friendly border intensity.
                    ["--demo-step-active" as string]: active,
                }}
            >
                <Icon size={14} />
            </motion.span>
            <span>{label}</span>
        </motion.div>
    );
}

function QuickEditStage({
    label,
    at,
    progress,
}: {
    label: string;
    at: number;
    progress: MotionValue<number>;
}) {
    const opacity = useTransform(progress, [at - 0.08, at + 0.02], [0.2, 1]);
    const x = useTransform(progress, [at - 0.08, at + 0.02], [-10, 0]);

    return (
        <motion.span className="demo-quick-stage" style={{ opacity, x }}>
            {label}
        </motion.span>
    );
}

function ActionChipReveal({
    icon: Icon,
    text,
    at,
    progress,
}: {
    icon: LucideIcon;
    text: string;
    at: number;
    progress: MotionValue<number>;
}) {
    const opacity = useTransform(progress, [at - 0.06, at + 0.04], [0.15, 1]);
    const x = useTransform(progress, [at - 0.06, at + 0.04], [-16, 0]);

    return (
        <motion.div className="edit-chip" style={{ opacity, x }}>
            <Icon size={15} /> {text}
        </motion.div>
    );
}

function TranscriptWord({
    children,
    variant,
    at,
    progress,
}: {
    children: string;
    variant?: "filler" | "selected" | "active";
    at: number;
    progress: MotionValue<number>;
}) {
    const opacity = useTransform(progress, [at - 0.04, at + 0.06], [0.35, 1]);
    const scale = useTransform(progress, [at - 0.04, at + 0.06], [0.96, 1]);

    return (
        <motion.span
            className={`demo-word${variant ? ` demo-word-${variant}` : ""}`}
            style={{ opacity, scale }}
        >
            {children}
        </motion.span>
    );
}

export default function DemoStorySection() {
    const boardRef = useRef<HTMLDivElement>(null);
    const reduceMotion = useReducedMotion();

    const { scrollYProgress } = useScroll({
        target: boardRef,
        offset: ["start 0.85", "end 0.25"],
    });

    // Smooth scroll scrubbing so Chrome doesn't update transforms 1:1 with
    // every wheel tick (feels stuttery). Look stays the same.
    const progress = useSpring(scrollYProgress, springConfig);
    const drivenProgress = reduceMotion ? scrollYProgress : progress;

    const progressScale = useTransform(
        drivenProgress,
        [0, 0.28, 0.52, 0.72, 1],
        [0.06, 0.34, 0.62, 0.88, 1],
    );

    const toolbarScale = useTransform(drivenProgress, [0, 0.15], [0.97, 1]);
    const checkOpacity = useTransform(drivenProgress, [0.75, 0.92], [0, 1]);
    const checkScale = useTransform(drivenProgress, [0.75, 0.92], [0.6, 1]);
    const quickEditGlow = useTransform(drivenProgress, [0.28, 0.55], [0, 1]);

    const transcriptOpacity = useTransform(drivenProgress, [0.12, 0.34], [0.25, 1]);
    const transcriptY = useTransform(drivenProgress, [0.12, 0.34], [36, 0]);

    const editsOpacity = useTransform(drivenProgress, [0.32, 0.54], [0.25, 1]);
    const editsY = useTransform(drivenProgress, [0.32, 0.54], [36, 0]);

    const outputOpacity = useTransform(drivenProgress, [0.58, 0.82], [0.2, 1]);
    const outputY = useTransform(drivenProgress, [0.58, 0.82], [48, 0]);
    const outputScale = useTransform(drivenProgress, [0.62, 0.88], [0.88, 1]);
    const outputGlow = useTransform(drivenProgress, [0.72, 0.95], [0, 1]);

    const railScale = useTransform(drivenProgress, [0, 1], [0, 1]);

    return (
        <section id="demo" className="story-section">
            <motion.div
                className="section-copy"
                initial={{ opacity: 0, y: 28 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            >
                <span className="section-eyebrow">
                    <Zap size={16} /> Same flow as the editor
                </span>
                <h2>Upload, transcript, Quick Edit — then publish.</h2>
                <p>
                    This is the real ClipMind path: drop a talking-head clip, generate the transcript,
                    hit Quick Edit for cut + captions, and export 9:16. No separate prompt box — you
                    edit by text inside the studio.
                </p>
            </motion.div>

            <div
                className={`demo-story-shell${reduceMotion ? " demo-story-complete" : ""}`}
                ref={boardRef}
            >
                <div className="demo-story-rail" aria-hidden="true">
                    <div className="demo-story-rail-track">
                        <motion.span
                            className="demo-story-rail-fill"
                            style={{ scaleX: railScale }}
                        />
                    </div>
                    <div className="demo-story-steps">
                        {storySteps.map((step) => (
                            <StoryStep
                                key={step.id}
                                label={step.label}
                                icon={step.icon}
                                at={step.at}
                                progress={drivenProgress}
                            />
                        ))}
                    </div>
                </div>

                <div className="demo-board demo-editor-frame">
                    <motion.div className="demo-editor-toolbar" style={{ scale: toolbarScale }}>
                        <span className="demo-toolbar-btn">
                            <UploadCloud size={15} />
                            Upload Video
                        </span>
                        <span className="demo-toolbar-btn demo-toolbar-btn-quick">
                            <motion.span
                                className="demo-toolbar-btn-quick-glow"
                                style={{ opacity: quickEditGlow }}
                                aria-hidden="true"
                            />
                            <Zap size={15} />
                            Quick Edit
                            <motion.span
                                className="command-check demo-toolbar-check"
                                style={{ opacity: checkOpacity, scale: checkScale }}
                            >
                                <BadgeCheck size={16} />
                            </motion.span>
                        </span>
                        <span className="demo-toolbar-btn demo-toolbar-btn-muted">
                            <Download size={15} />
                            Export
                        </span>
                    </motion.div>

                    <div className="demo-quick-pipeline">
                        <span className="demo-pipeline-label">
                            <Sparkles size={14} />
                            Quick Edit pipeline
                        </span>
                        <div className="demo-quick-stages">
                            {quickEditStages.map((stage) => (
                                <QuickEditStage
                                    key={stage.label}
                                    label={stage.label}
                                    at={stage.at}
                                    progress={drivenProgress}
                                />
                            ))}
                        </div>
                        <div className="command-progress demo-pipeline-progress">
                            <motion.span style={{ scaleX: progressScale }} />
                        </div>
                    </div>

                    <div className="demo-columns">
                        <motion.div
                            className="demo-transcript demo-column-panel"
                            style={{
                                opacity: transcriptOpacity,
                                y: transcriptY,
                            }}
                        >
                            <span className="panel-label">Transcript Panel</span>
                            <p className="demo-transcript-line">
                                <TranscriptWord progress={drivenProgress} at={0.18} variant="filler">
                                    uh
                                </TranscriptWord>{" "}
                                <TranscriptWord progress={drivenProgress} at={0.2}>
                                    so basically
                                </TranscriptWord>{" "}
                                <TranscriptWord progress={drivenProgress} at={0.24} variant="selected">
                                    the real reason creators lose retention
                                </TranscriptWord>{" "}
                                <TranscriptWord progress={drivenProgress} at={0.28}>
                                    is the first five seconds.
                                </TranscriptWord>
                            </p>
                            <p className="demo-transcript-line">
                                <TranscriptWord progress={drivenProgress} at={0.3} variant="active">
                                    Show the payoff first,
                                </TranscriptWord>{" "}
                                <TranscriptWord progress={drivenProgress} at={0.32}>
                                    then explain the setup.
                                </TranscriptWord>
                            </p>
                            <p className="demo-transcript-line">
                                <TranscriptWord progress={drivenProgress} at={0.34} variant="filler">
                                    you know like
                                </TranscriptWord>{" "}
                                <TranscriptWord progress={drivenProgress} at={0.36}>
                                    captions should move with the beat.
                                </TranscriptWord>
                            </p>
                            <span className="demo-transcript-hint">
                                Select words in the editor — delete text to cut video.
                            </span>
                        </motion.div>

                        <motion.div
                            className="demo-edits demo-column-panel"
                            style={{ opacity: editsOpacity, y: editsY }}
                        >
                            <span className="panel-label">AI Tools · Quick Edit</span>
                            {aiActions.map((item) => (
                                <ActionChipReveal
                                    key={item.text}
                                    icon={item.icon}
                                    text={item.text}
                                    at={item.at}
                                    progress={drivenProgress}
                                />
                            ))}
                            <span className="demo-transcript-hint">
                                Same actions as the AI Tools drawer in the editor.
                            </span>
                        </motion.div>

                        <motion.div
                            className="demo-output demo-column-panel"
                            style={{
                                opacity: outputOpacity,
                                y: outputY,
                                scale: outputScale,
                            }}
                        >
                            <motion.div
                                className="demo-output-glow"
                                style={{ opacity: outputGlow }}
                                aria-hidden="true"
                            />
                            <div className="mini-phone">
                                <div className="mini-video">
                                    <span>RETENTION STARTS HERE</span>
                                </div>
                            </div>
                            <motion.strong style={{ opacity: outputGlow }}>
                                9:16 export ready
                            </motion.strong>
                        </motion.div>
                    </div>
                </div>
            </div>
        </section>
    );
}
