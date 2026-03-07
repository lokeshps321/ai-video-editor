import re

with open("frontend/src/styles.css", "r") as f:
    css = f.read()

# Make background roam more dramatic
old_bg = """@keyframes bgRoam {
  0% {
    transform: scale(1) translate(0, 0);
  }

  100% {
    transform: scale(1.1) translate(-2%, 2%);
  }
}"""
new_bg = """@keyframes bgRoam {
  0% {
    transform: scale(1) translate(0, 0);
    filter: hue-rotate(0deg);
  }
  50% {
    transform: scale(1.15) translate(-2%, 3%);
    filter: hue-rotate(15deg);
  }
  100% {
    transform: scale(1.05) translate(2%, -2%);
    filter: hue-rotate(-15deg);
  }
}"""
css = css.replace(old_bg, new_bg)

# Enhance transcript words
old_word = """.word {
  border: 1px solid transparent;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.04);
  color: var(--ink);
  font-size: 0.92rem;
  margin: 0;
  min-height: auto;
  padding: 4px 8px;
  cursor: pointer;
  transition: all 0.12s ease;
  line-height: 1.4;
}

.word:hover {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.12);
}"""

new_word = """.word {
  border: 1px solid transparent;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.03);
  color: var(--ink);
  font-size: 0.95rem;
  margin: 0;
  min-height: auto;
  padding: 4px 8px;
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1); /* Springy transition */
  line-height: 1.4;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.word:hover {
  background: rgba(255, 255, 255, 0.1);
  border-color: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px) scale(1.05);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  z-index: 2;
  position: relative;
}"""
css = css.replace(old_word, new_word)

# Enhance action cards
old_card = """.actionCard:not(:disabled):hover {
  background: rgba(139, 92, 246, 0.12);
  border-color: rgba(139, 92, 246, 0.35);
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(139, 92, 246, 0.15);
}"""

new_card = """.actionCard:not(:disabled):hover {
  background: linear-gradient(135deg, rgba(255, 46, 147, 0.15), rgba(139, 92, 246, 0.1));
  border-color: var(--accent);
  transform: translateY(-4px) scale(1.02);
  box-shadow: 0 10px 25px var(--accent-glow), inset 0 0 10px rgba(255, 255, 255, 0.1);
}"""
css = css.replace(old_card, new_card)

# Enhance actionCardPrimary hover
old_primary_card = """.actionCardPrimary:not(:disabled):hover {
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(0, 210, 211, 0.18));
  border-color: rgba(139, 92, 246, 0.5);
  box-shadow: 0 8px 24px rgba(139, 92, 246, 0.2);
}"""
new_primary_card = """.actionCardPrimary:not(:disabled):hover {
  background: linear-gradient(135deg, rgba(255, 46, 147, 0.3), rgba(0, 242, 254, 0.2));
  border-color: var(--accent);
  box-shadow: 0 12px 30px var(--accent-glow);
  transform: translateY(-4px) scale(1.02);
}"""
css = css.replace(old_primary_card, new_primary_card)

# Add slide up animations to panels
slide_up = """
.panel {
  animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
}

@keyframes slideUpFade {
  0% { opacity: 0; transform: translateY(30px); }
  100% { opacity: 1; transform: translateY(0); }
}

.featureTabs {
  border-radius: 100px !important;
  padding: 6px !important;
  background: rgba(0,0,0,0.4) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
}

.featureTab {
  border-radius: 100px !important;
}

.featureTab.active {
  background: var(--accent) !important;
  color: white !important;
  border-color: var(--accent) !important;
  box-shadow: 0 4px 15px var(--accent-glow) !important;
}
"""
css = css + "\n" + slide_up

with open("frontend/src/styles.css", "w") as f:
    f.write(css)
