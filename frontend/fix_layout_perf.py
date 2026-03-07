import re

with open("frontend/src/styles.css", "r") as f:
    css = f.read()

# 1. Update Colors for a stunning, modern creator tool look (deep purple/blue + neon cyan)
new_vars = """:root {
  --bg-gradient: linear-gradient(135deg, #0b0914 0%, #16122b 50%, #0d0a18 100%);
  --bg: #0b0914;
  --panel: rgba(22, 20, 35, 0.85);
  --panel-blur: 12px; /* Reduced for better scrolling performance */
  --panel-hover: rgba(35, 32, 55, 0.95);
  --ink: #f0f0f5;
  --muted: #a09eb5;
  --accent: #00f2fe;
  --accent-secondary: #4facfe;
  --accent-tertiary: #ff0844;
  --accent-glow: rgba(0, 242, 254, 0.4);
  --accent-ink: #ffffff;
  --danger: #ff4757;
  --danger-soft: rgba(255, 71, 87, 0.15);
  --success: #2ed573;
  --success-glow: rgba(46, 213, 115, 0.2);
  --select: rgba(0, 242, 254, 0.2);
  --active: #ffb142;
  --active-glow: rgba(255, 177, 66, 0.3);
  --filler: rgba(255, 165, 2, 0.15);
  --filler-border: #ffa502;
  --search: rgba(255, 8, 68, 0.25);
  --search-active: rgba(255, 8, 68, 0.6);
  --border: rgba(255, 255, 255, 0.08);
  --border-light: rgba(255, 255, 255, 0.04);
  --shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
  --radius: 20px;
  --radius-sm: 10px;
}"""
css = re.sub(r':root\s*\{[^}]+\}', new_vars, css, count=1)

# 2. Fix the collision/merge issue by giving the sticky preview block a solid, stylized background
old_preview_block = """.workspacePreviewBlock {
  position: sticky;
  top: 12px;
  z-index: 5;
  background: transparent;
  border: none;
  box-shadow: none;
  padding: 0;
}"""

new_preview_block = """.workspacePreviewBlock {
  position: sticky;
  top: -18px; /* Sticks flush to the top padding of the card */
  z-index: 20; /* High z-index to stay above feature tabs */
  background: linear-gradient(180deg, #181528 80%, rgba(24, 21, 40, 0.95) 100%);
  padding: 18px 18px 16px 18px;
  margin: -18px -18px 14px -18px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: var(--radius) var(--radius) 0 0;
  box-shadow: 0 15px 30px -10px rgba(0,0,0,0.7);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
}"""
css = css.replace(old_preview_block, new_preview_block)

# 3. Optimize background animation (remove expensive hue-rotate that causes lag)
old_bg_anim = """@keyframes bgRoam {
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
new_bg_anim = """@keyframes bgRoam {
  0% {
    transform: scale(1) translate(0, 0);
    opacity: 0.8;
  }
  50% {
    transform: scale(1.1) translate(-1%, 2%);
    opacity: 1;
  }
  100% {
    transform: scale(1.05) translate(1%, -1%);
    opacity: 0.9;
  }
}"""
if old_bg_anim in css:
    css = css.replace(old_bg_anim, new_bg_anim)

# 4. Remove 'background-attachment: fixed' from body as it tanks scroll performance with backdrop-filters
css = css.replace("background-attachment: fixed;", "/* removed fixed attachment for performance */")

# 5. Simplify .word transition (spring animation on hundreds of words can be laggy)
old_word_transition = "transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1); /* Springy transition */"
new_word_transition = "transition: transform 0.15s ease, background 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;"
css = css.replace(old_word_transition, new_word_transition)

with open("frontend/src/styles.css", "w") as f:
    f.write(css)
