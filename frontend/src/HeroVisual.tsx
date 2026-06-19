import { useEffect, useRef, useState } from "react";
import {
    motion,
    useMotionValue,
    useReducedMotion,
    useSpring,
    useTransform,
} from "framer-motion";
import { Play, Smartphone, Zap } from "lucide-react";

const WAVEFORM_BARS = 18;

function useFinePointer() {
    const [finePointer, setFinePointer] = useState(true);

    useEffect(() => {
        const media = window.matchMedia("(hover: hover) and (pointer: fine)");
        const update = () => setFinePointer(media.matches);
        update();
        media.addEventListener("change", update);
        return () => media.removeEventListener("change", update);
    }, []);

    return finePointer;
}

export default function HeroVisual() {
    const containerRef = useRef<HTMLDivElement>(null);
    const frameRef = useRef<number | null>(null);
    const pendingPointerRef = useRef<{ x: number; y: number } | null>(null);
    const [inView, setInView] = useState(true);

    const reduceMotion = useReducedMotion();
    const finePointer = useFinePointer();
    const parallaxEnabled = !reduceMotion && finePointer;

    useEffect(() => {
        const node = containerRef.current;
        if (!node) return;

        const observer = new IntersectionObserver(
            ([entry]) => setInView(entry.isIntersecting),
            { rootMargin: "80px 0px", threshold: 0 },
        );
        observer.observe(node);
        return () => observer.disconnect();
    }, []);

    const mouseX = useMotionValue(0);
    const mouseY = useMotionValue(0);
    const rotateY = useSpring(useTransform(mouseX, [-0.5, 0.5], [7, -7]), {
        stiffness: 120,
        damping: 22,
    });
    const rotateX = useSpring(useTransform(mouseY, [-0.5, 0.5], [-5, 5]), {
        stiffness: 120,
        damping: 22,
    });

    const studioX = useSpring(useTransform(mouseX, [-0.5, 0.5], [-12, 12]), {
        stiffness: 90,
        damping: 20,
    });
    const phoneY = useSpring(useTransform(mouseY, [-0.5, 0.5], [10, -10]), {
        stiffness: 90,
        damping: 20,
    });
    const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!parallaxEnabled) return;

        const bounds = event.currentTarget.getBoundingClientRect();
        pendingPointerRef.current = {
            x: (event.clientX - bounds.left) / bounds.width - 0.5,
            y: (event.clientY - bounds.top) / bounds.height - 0.5,
        };

        if (frameRef.current !== null) return;
        frameRef.current = requestAnimationFrame(() => {
            frameRef.current = null;
            const pending = pendingPointerRef.current;
            if (!pending) return;
            mouseX.set(pending.x);
            mouseY.set(pending.y);
        });
    };

    const handlePointerLeave = () => {
        if (!parallaxEnabled) return;
        pendingPointerRef.current = null;
        mouseX.set(0);
        mouseY.set(0);
    };

    const heroClassName = [
        "hero-visual",
        parallaxEnabled ? "hero-visual-parallax" : "",
        inView ? "" : "hero-visual-paused",
    ]
        .filter(Boolean)
        .join(" ");

    return (
        <div
            ref={containerRef}
            className={heroClassName}
            aria-label="ClipMind editing preview"
            onPointerMove={handlePointerMove}
            onPointerLeave={handlePointerLeave}
        >
            <motion.div
                className="hero-visual-stage"
                style={
                    parallaxEnabled
                        ? { rotateX, rotateY, transformPerspective: 1200 }
                        : undefined
                }
            >
                <motion.div
                    className="desktop-studio hero-depth-layer"
                    style={parallaxEnabled ? { x: studioX } : undefined}
                >
                    <div className="studio-topbar">
                        <span className="window-dot cyan" />
                        <span className="window-dot amber" />
                        <span className="window-dot rose" />
                        <span className="studio-title">Short-Form Timeline</span>
                    </div>

                    <div className="studio-grid">
                        <div className="preview-panel">
                            <div className="preview-scene">
                                <div className="preview-subject" />
                                <span className="caption-burn">THIS PART HITS</span>
                                <span className="play-pill">
                                    <Play size={14} fill="currentColor" /> 00:18
                                </span>
                            </div>
                        </div>

                        <div className="transcript-panel">
                            <span className="panel-label">Transcript</span>
                            <p>
                                <span>We should</span> <mark>cut the intro</mark>{" "}
                                <span>and jump straight into the result...</span>
                            </p>
                            <p>
                                <span>Then add captions in</span> <mark>Kannada + English</mark>
                            </p>
                            <p>
                                <span>Drop b-roll when I mention growth.</span>
                            </p>
                        </div>

                        <div className="timeline-panel">
                            <div className="timeline-ruler">
                                <span>00:00</span>
                                <span>00:15</span>
                                <span>00:30</span>
                                <span>00:45</span>
                            </div>
                            <div className="track video-track">
                                <span className="clip clip-a" />
                                <span className="clip clip-b" />
                                <span className="clip clip-c" />
                            </div>
                            <div className="track caption-track">
                                <span className="clip caption-a" />
                                <span className="clip caption-b" />
                            </div>
                            <div className="waveform">
                                {Array.from({ length: WAVEFORM_BARS }).map((_, i) => (
                                    <span
                                        key={i}
                                        className="waveform-bar"
                                        style={{
                                            height: `${18 + Math.abs(Math.sin(i * 0.7)) * 58}%`,
                                            animationDelay: `${i * 0.05}s`,
                                        }}
                                    />
                                ))}
                            </div>
                            <span className="hero-visual-playhead" aria-hidden="true" />
                        </div>
                    </div>
                </motion.div>

                <motion.div
                    className="phone-export hero-depth-layer"
                    style={parallaxEnabled ? { y: phoneY } : undefined}
                >
                    <div className="floating-panel p2 floating-panel-ready">
                        <div className="floating-panel-header">
                            <Smartphone size={15} className="floating-panel-icon floating-panel-icon-violet" />
                            <span className="floating-panel-label">9:16 export ready</span>
                        </div>
                    </div>
                    <div className="phone-screen">
                        <span className="short-badge">9:16</span>
                        <div className="phone-subject" />
                        <div className="phone-caption">CREATE LIKE A PRO</div>
                        <div className="phone-actions">
                            <span />
                            <span />
                            <span />
                        </div>
                    </div>
                </motion.div>

            </motion.div>

            <motion.div className="ai-command-card">
                <div className="command-icon command-icon-quick">
                    <Zap size={17} />
                </div>
                <div>
                    <span>Quick Edit</span>
                    <strong>Transcript cut + captions in one tap</strong>
                </div>
            </motion.div>

            <div className="hero-float-cards" aria-hidden="true">
                <div className="floating-panel p1">
                    <div className="floating-panel-header">
                        <Zap size={18} className="floating-panel-icon floating-panel-icon-electric" />
                        <span className="floating-panel-label">Cut + Captions...</span>
                    </div>
                    <div className="floating-panel-progress-track">
                        <div className="floating-panel-progress-fill" />
                    </div>
                </div>
            </div>
        </div>
    );
}
