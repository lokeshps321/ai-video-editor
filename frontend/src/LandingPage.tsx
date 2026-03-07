import { useEffect } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Clapperboard, AudioWaveform, Zap, Layers, Sparkles, Wand2, TerminalSquare } from "lucide-react";
import { BRAND } from "./config/brand";

import "./landing.css";

function TickerRibbon() {
    const items = [
        "Native AI Timeline", "Generative B-Roll", "Magnetic Captions", "Transcript Cut",
        "Viewport Render", "Zero-Latency Caching", "Exact 9:16 Export", "Creator-First Timeline"
    ];

    return (
        <section className="ticker-section">
            <div className="ticker-track" aria-hidden="true">
                {[0, 1].map((copy) => (
                    <div key={copy} className="ticker-content">
                        {items.map((item, i) => (
                            <div key={`${copy}-${i}`} className="ticker-item">
                                <Zap size={20} />
                                {item}
                            </div>
                        ))}
                    </div>
                ))}
            </div>
        </section>
    );
}

function BentoCard({ className, icon: Icon, title, desc, delay, children }: any) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.8, delay, ease: [0.16, 1, 0.3, 1] }}
            className={`bento-card ${className}`}
        >
            <div className="bento-icon-wrapper">
                <Icon className="bento-icon" size={28} />
            </div>

            {children}

            <div className="bento-content">
                <h3 className="bento-title">{title}</h3>
                <p className="bento-desc">{desc}</p>
            </div>
        </motion.div>
    );
}

export default function LandingPage() {
    useEffect(() => {
        document.title = BRAND.landingDocumentTitle;
    }, []);

    return (
        <div className="landing-page dark-theme">
            <div className="noise-overlay" style={{ zIndex: -1 }}></div>
            <div className="mesh-glow-1"></div>
            <div className="mesh-glow-2"></div>

            <nav className="glass-nav">
                <div className="nav-container">
                    <div className="nav-logo">
                        <Zap className="nav-logo-icon" size={26} />
                        <span className="nav-logo-text">{BRAND.lead}{BRAND.accent ? <><span> </span><span className="text-gradient">{BRAND.accent}</span></> : null}</span>
                    </div>
                    <div className="nav-links">
                        <a href="#features">Features</a>
                        <a href="#demo">Story</a>
                        <Link to="/editor" className="btn-primary-small">Launch Editor</Link>
                    </div>
                </div>
            </nav>

            <main>
                <section className="hero-section">
                    <motion.div className="hero-content">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            className="badge-wrapper"
                        >
                            <span className="hero-badge"><Sparkles size={16} /> Enter the next generation</span>
                        </motion.div>

                        <motion.h1
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
                            className="hero-title"
                        >
                            Produce video at the <br /> speed of <span className="text-gradient-accent">thought.</span>
                        </motion.h1>

                        <motion.p
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
                            className="hero-subtitle"
                        >
                            {BRAND.productName} is a cinematic timeline powered by reasoning AI. Draft entire videos from text, generate contextual B-roll instantly, and refine natively in a studio-grade editor.
                        </motion.p>

                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
                        >
                            <Link to="/editor" className="btn-glow">
                                Start Creating
                                <div className="btn-glow-effect"></div>
                            </Link>
                        </motion.div>
                    </motion.div>

                    {/* Highly Interactive 3D Editor Mockup */}
                    <div className="hero-mockup-container">
                        <motion.div
                            className="hero-mockup-wrapper"
                            initial={{ opacity: 0, y: 100 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 1.5, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
                            style={{ height: "400px", display: "flex", justifyContent: "center", alignItems: "center" }}
                        >
                            {/* Empty space that the 3D model floats through */}


                            {/* Floating Parallax Layers representing AI tools */}
                            <div className="tilted-ui-layer">
                                <div className="floating-panel p1">
                                    <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "10px" }}>
                                        <Wand2 size={18} style={{ color: "var(--accent-electric)" }} />
                                        <span style={{ fontSize: "14px", fontWeight: "600" }}>Generating B-Roll...</span>
                                    </div>
                                    <div style={{ height: "4px", background: "rgba(255,255,255,0.1)", borderRadius: "2px", overflow: "hidden" }}>
                                        <div style={{ width: "72%", height: "100%", background: "var(--accent-electric)" }} />
                                    </div>
                                </div>

                                <div className="floating-panel p2">
                                    <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                                        <TerminalSquare size={18} style={{ color: "var(--accent-violet)" }} />
                                        <span style={{ fontSize: "14px", fontWeight: "600" }}>Transcript Sync OK</span>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                </section>

                <TickerRibbon />

                <section id="features" className="bento-section">
                    <div className="container">
                        <motion.div className="section-header">
                            <h2 className="section-title">A Studio in your Browser</h2>
                            <p className="section-desc">We reimagined the non-linear editor around conversational and generative workflows.</p>
                        </motion.div>

                        <div className="bento-grid">
                            <BentoCard
                                className="wide" icon={Layers} delay={0}
                                title="Generative B-Roll Matrix"
                                desc="ClipMind reads your transcript, infers context, and automatically fetches or generates perfect cinematic B-roll to fit exactly on your timeline. All in a single click."
                            >
                                <div className="bento-visual">
                                    {/* Simulated graphic or image for wide bento */}
                                    <div style={{ width: "100%", height: "100%", background: "linear-gradient(90deg, transparent, rgba(46, 88, 255, 0.2))", borderRadius: "20px" }} />
                                </div>
                            </BentoCard>

                            <BentoCard
                                className="square" icon={Clapperboard} delay={0.2}
                                title="Edit by Transcript"
                                desc="Cut the text. The video follows. A true hybrid workflow where the script drives the timeline completely."
                            >
                            </BentoCard>

                            <BentoCard
                                className="square" icon={AudioWaveform} delay={0.3}
                                title="Magnetic Captions"
                                desc="One-click typography. Styles ranging from modern minimal to high-retention neon outlines."
                            >
                            </BentoCard>

                            <BentoCard
                                className="tall" icon={Wand2} delay={0.4}
                                title="Agentic Rendering"
                                desc="Behind the scenes, distributed agents pre-fetch assets, re-frame ratios, and transcode locally so your flow state is never interrupted."
                            >
                            </BentoCard>
                        </div>
                    </div>
                </section>

                <section className="statement-section">
                    <motion.div
                        initial={{ opacity: 0, y: 50, scale: 0.95 }}
                        whileInView={{ opacity: 1, y: 0, scale: 1 }}
                        viewport={{ once: true, margin: "-100px" }}
                        transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                        className="big-statement-box"
                    >
                        <h2>Create like a Pro. <br /> <span className="text-outline">Direct like an Agent.</span></h2>
                        <Link to="/editor" className="btn-glow" style={{ marginTop: "20px" }}>
                            Enter the Timeline
                            <div className="btn-glow-effect"></div>
                        </Link>
                    </motion.div>
                </section>
            </main>

            <footer className="glass-footer">
                <div className="container footer-content">
                    <div className="footer-brand">
                        <div className="nav-logo">
                            <Zap className="nav-logo-icon" size={24} />
                            <span className="nav-logo-text">{BRAND.productName}</span>
                        </div>
                        <p className="footer-sub">{BRAND.footerTagline}</p>
                    </div>
                    <div className="footer-links">
                        <span>© 2026 {BRAND.productName}</span>
                        <a href="#">Showcase</a>
                        <a href="#">Documentation</a>
                        <a href="#">Privacy</a>
                    </div>
                </div>
            </footer>
        </div>
    );
}
