import re

with open("frontend/src/styles.css", "r") as f:
    css = f.read()

new_heading_style = "\nh2, h3 {\n  background: linear-gradient(135deg, #ffffff 0%, var(--muted) 100%);\n  -webkit-background-clip: text;\n  -webkit-text-fill-color: transparent;\n  background-clip: text;\n  letter-spacing: -0.02em;\n}\n"
css += new_heading_style

css = css.replace("width: min(1720px, 97vw);", "width: min(1800px, 98vw);")
css = css.replace("padding: 18px 0 34px;", "padding: 24px 0 40px;")

old_card_style = ".card {\n  background: var(--panel);\n  border: 1px solid var(--border);\n  border-radius: var(--radius);\n  box-shadow: var(--shadow);\n  backdrop-filter: blur(var(--panel-blur));\n  -webkit-backdrop-filter: blur(var(--panel-blur));\n}"
new_card_style = ".card {\n  background: linear-gradient(180deg, rgba(20,20,25,0.7) 0%, rgba(15,15,20,0.85) 100%);\n  border: 1px solid rgba(255, 255, 255, 0.05);\n  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06), 0 20px 40px -10px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);\n  border-radius: var(--radius);\n  backdrop-filter: blur(var(--panel-blur));\n  -webkit-backdrop-filter: blur(var(--panel-blur));\n  transition: transform 0.3s ease, box-shadow 0.3s ease;\n}\n\n.card:hover {\n  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05), 0 25px 50px -12px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.15);\n}"
css = css.replace(old_card_style, new_card_style)

custom_scrollbar = "\n::-webkit-scrollbar {\n  width: 8px;\n  height: 8px;\n}\n::-webkit-scrollbar-track {\n  background: rgba(0, 0, 0, 0.2);\n  border-radius: 4px;\n}\n::-webkit-scrollbar-thumb {\n  background: rgba(255, 255, 255, 0.1);\n  border-radius: 4px;\n}\n::-webkit-scrollbar-thumb:hover {\n  background: var(--accent);\n}\n"
css += custom_scrollbar

primary_btn_old = "button.primaryBtn {\n  background: var(--accent);\n  color: white;\n  border-color: transparent;\n  font-weight: 600;\n}\n\nbutton.primaryBtn:not(:disabled):hover {\n  background: #7c3aed;\n  box-shadow: 0 4px 15px var(--accent-glow);\n}"
primary_btn_new = "button.primaryBtn {\n  background: linear-gradient(135deg, var(--accent), var(--accent-secondary));\n  color: white;\n  border-color: transparent;\n  font-weight: 700;\n  text-transform: uppercase;\n  letter-spacing: 0.05em;\n  font-size: 0.85rem;\n  padding: 0 20px;\n}\n\nbutton.primaryBtn:not(:disabled):hover {\n  background: linear-gradient(135deg, var(--accent-secondary), var(--accent));\n  box-shadow: 0 8px 25px var(--accent-glow);\n  transform: translateY(-2px) scale(1.02) !important;\n}"
css = css.replace(primary_btn_old, primary_btn_new)

with open("frontend/src/styles.css", "w") as f:
    f.write(css)