import re

with open("frontend/src/styles.css", "r") as f:
    css = f.read()

# Update CSS Variables
new_vars = """:root {
  --bg-gradient: linear-gradient(135deg, #050505 0%, #0d0b14 50%, #07040a 100%);
  --bg: #050505;
  --panel: rgba(20, 20, 25, 0.6);
  --panel-blur: 24px;
  --panel-hover: rgba(30, 30, 40, 0.8);
  --ink: #f8f9fa;
  --muted: #a1a3b5;
  --accent: #ff2e93;
  --accent-secondary: #8b5cf6;
  --accent-tertiary: #00f2fe;
  --accent-glow: rgba(255, 46, 147, 0.4);
  --accent-ink: #ffffff;
  --danger: #ff4757;
  --danger-soft: rgba(255, 71, 87, 0.15);
  --success: #2ed573;
  --success-glow: rgba(46, 213, 115, 0.2);
  --select: rgba(255, 46, 147, 0.2);
  --active: #ffa502;
  --active-glow: rgba(255, 165, 2, 0.3);
  --filler: rgba(255, 165, 2, 0.15);
  --filler-border: #ffa502;
  --search: rgba(0, 242, 254, 0.25);
  --search-active: rgba(0, 242, 254, 0.6);
  --border: rgba(255, 255, 255, 0.08);
  --border-light: rgba(255, 255, 255, 0.03);
  --shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
  --radius: 20px;
  --radius-sm: 10px;
}"""

css = re.sub(r':root\s*\{[^}]+\}', new_vars, css)

# Update Hero section for more pop
hero_replacements = {
    "background: linear-gradient(to right, #ffffff, #a78bfa, #34d399);": "background: linear-gradient(to right, #ffffff, var(--accent), var(--accent-secondary), var(--accent-tertiary));",
    "box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);": "box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), 0 0 0 1px var(--border);",
    "background: linear-gradient(135deg, var(--accent), #6366f1);": "background: linear-gradient(135deg, var(--accent), var(--accent-secondary));",
}

for old, new in hero_replacements.items():
    css = css.replace(old, new)
    
# Improve buttons globally
button_enhancement = """
button {
  transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}
button:active {
  transform: scale(0.95) !important;
}
"""
css = css + "\n" + button_enhancement

# Write back
with open("frontend/src/styles.css", "w") as f:
    f.write(css)
