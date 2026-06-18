import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0A0A14') # Deep Navy background to match PPT
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

CYAN = '#00FFFF'
MAGENTA = '#BC13FE'
WHITE = '#FFFFFF'
BOX_BG = '#141428'

def draw_node(x, y, w, h, text, border=CYAN):
    # Base box
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.8,rounding_size=1.5",
                         edgecolor=border, facecolor=BOX_BG, linewidth=2.5, zorder=2)
    ax.add_patch(box)
    
    # Outer glow (hack with lower alpha box)
    glow = FancyBboxPatch((x - w/2 - 0.5, y - h/2 - 0.5), w + 1, h + 1, boxstyle="round,pad=0.8,rounding_size=1.5",
                          edgecolor=border, facecolor='none', linewidth=5, alpha=0.2, zorder=1)
    ax.add_patch(glow)

    # Multi-line text handling
    lines = text.split('\n')
    y_start = y + (len(lines) - 1) * 1.5
    for i, line in enumerate(lines):
        bold = 'bold' if i == 0 else 'normal'
        fontsize = 14 if i == 0 else 12
        color = WHITE if i == 0 else '#AAAAAA'
        if line:
             ax.text(x, y_start - i*3.5, line, ha='center', va='center', color=color, 
                    fontsize=fontsize, fontweight=bold, fontfamily='sans-serif', zorder=3)

def draw_arrow(x1, y1, x2, y2, color=CYAN, offset=0):
    # offset accounts for box edge approximation
    if y1 > y2: # Arrow points down
        ax.annotate('', xy=(x2, y2+offset), xytext=(x1, y1-offset),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color=color, shrinkA=0, shrinkB=0), zorder=0)
    elif y1 < y2: # Arrow points up
        ax.annotate('', xy=(x2, y2-offset), xytext=(x1, y1+offset),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color=color, shrinkA=0, shrinkB=0), zorder=0)
    else: # Arrow points sideways
        dir_mod = -1 if x1 > x2 else 1
        ax.annotate('', xy=(x2-offset*dir_mod, y2), xytext=(x1+offset*dir_mod, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color=color, shrinkA=0, shrinkB=0), zorder=0)


# Define nodes
nodes = {
    'cam': (50, 90, 28, 6, "1. Webcam Input\n(1280x720 RGB Capture)", CYAN),
    'pre': (50, 75, 32, 6, "2. Image Preprocessing\n(Resize to 224x224 & Normalize)", CYAN),
    'mp':  (25, 55, 32, 7, "3A. MediaPipe Tracker\n(Extracts 33 Spatial Landmarks)", MAGENTA),
    'cnn': (75, 55, 32, 7, "3B. MobileNetV2 CNN\n(Predicts Yoga Pose Class)", MAGENTA),
    'calc':(25, 35, 32, 7, "4. Joint Angle Calculator\n(Applies EMA 0.10 Smoothing)", CYAN),
    'rule':(50, 20, 38, 7, "5. Biomechanical Rules Engine\n(Cross-references expected ideal angles)", MAGENTA),
    'hud': (25, 5, 28, 6, "6A. UI HUD Renderer\n(Skeletal Overlay Output)", CYAN),
    'tts': (75, 5, 28, 6, "6B. pyttsx3 Voice Engine\n(Speaks Priority Correction)", CYAN)
}

# Draw boxes
for k, v in nodes.items():
    draw_node(v[0], v[1], v[2], v[3], v[4], v[5])

# Draw arrows (approx box height offset = 4.5)
OFS = 4.0
draw_arrow(nodes['cam'][0], nodes['cam'][1], nodes['pre'][0], nodes['pre'][1], color=CYAN, offset=OFS)

# Split paths
ax.plot([50, 25], [nodes['pre'][1]-OFS, nodes['mp'][1]+OFS], color=CYAN, lw=2.5, zorder=0)
ax.plot([50, 75], [nodes['pre'][1]-OFS, nodes['cnn'][1]+OFS], color=CYAN, lw=2.5, zorder=0)

draw_arrow(nodes['mp'][0], nodes['mp'][1], nodes['calc'][0], nodes['calc'][1], color=MAGENTA, offset=OFS)

# Rejoin to Rules Engine
ax.plot([25, 50], [nodes['calc'][1]-OFS, nodes['rule'][1]+OFS], color=CYAN, lw=2.5, zorder=0)
ax.plot([75, 50], [nodes['cnn'][1]-OFS, nodes['rule'][1]+OFS], color=MAGENTA, lw=2.5, zorder=0)

# Final split out to UI and Voice
ax.plot([50, 25], [nodes['rule'][1]-OFS, nodes['hud'][1]+OFS], color=MAGENTA, lw=2.5, zorder=0)
ax.plot([50, 75], [nodes['rule'][1]-OFS, nodes['tts'][1]+OFS], color=MAGENTA, lw=2.5, zorder=0)

# Overall title
plt.text(50, 98, "YOGA AI: SYSTEM ARCHITECTURE WORKFLOW", ha='center', va='center', 
         color=WHITE, fontsize=20, fontweight='bold')

output_path = r"c:\Users\thota\OneDrive\Documents\Project\results\architecture_flowchart.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.2)
print("Flowchart generation complete.")
