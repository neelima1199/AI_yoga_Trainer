import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create Dark and Light Versions
def create_flowchart(theme="dark"):
    is_dark = (theme == "dark")
    bg_color = "#121212" if is_dark else "#FFFFFF"
    box_color = "#1E1E24" if is_dark else "#F0F0F5"
    text_color = "#FFFFFF" if is_dark else "#000000"
    line_color = "#00FFFF" if is_dark else "#0055FF"
    alt_line_color = "#BC13FE" if is_dark else "#FF00AA"
    
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=bg_color)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_node(x, y, w, h, text, border):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="square,pad=0",
                             edgecolor=border, facecolor=box_color, linewidth=2)
        ax.add_patch(box)
        
        lines = text.split('\n')
        y_start = y + (len(lines) - 1) * 1.5
        for i, line in enumerate(lines):
            bold = 'bold' if i == 0 else 'normal'
            fontsize = 14 if i == 0 else 12
            color = text_color if i == 0 else ("#AAAAAA" if is_dark else "#555555")
            if line:
                 ax.text(x, y_start - i*3.5, line, ha='center', va='center', color=color, 
                        fontsize=fontsize, fontweight=bold, fontfamily='sans-serif')

    def draw_arrow(x1, y1, x2, y2, color, offset=0):
        if y1 > y2: # points down
            ax.annotate('', xy=(x2, y2+offset), xytext=(x1, y1-offset),
                        arrowprops=dict(arrowstyle="->", lw=2, color=color))
        elif y1 < y2: # points up
            ax.annotate('', xy=(x2, y2-offset), xytext=(x1, y1+offset),
                        arrowprops=dict(arrowstyle="->", lw=2, color=color))
        else: # points sideways
            dir_mod = -1 if x1 > x2 else 1
            ax.annotate('', xy=(x2-offset*dir_mod, y2), xytext=(x1+offset*dir_mod, y1),
                        arrowprops=dict(arrowstyle="->", lw=2, color=color))

    nodes = {
        'cam': (50, 90, 28, 6, "1. Webcam Input\n(1280x720 Capture)", line_color),
        'pre': (50, 75, 32, 6, "2. Image Preprocessing\n(Resize & Normalize)", line_color),
        'mp':  (25, 55, 32, 7, "3A. MediaPipe Tracker\n(Extracts 33 Landmarks)", alt_line_color),
        'cnn': (75, 55, 32, 7, "3B. MobileNetV2 CNN\n(Pose Classification)", alt_line_color),
        'calc':(25, 35, 32, 7, "4. Angle Calculator\n(EMA Smoothing)", line_color),
        'rule':(50, 20, 38, 7, "5. Biomechanical Rules Engine\n(Ideal Angle Validation)", alt_line_color),
        'hud': (25, 5, 28, 6, "6A. UI HUD Renderer\n(Skeletal Overlay)", line_color),
        'tts': (75, 5, 28, 6, "6B. Voice Engine\n(Speaks Correction)", line_color)
    }

    for k, v in nodes.items():
        draw_node(v[0], v[1], v[2], v[3], v[4], v[5])

    OFS = 4.0
    draw_arrow(nodes['cam'][0], nodes['cam'][1], nodes['pre'][0], nodes['pre'][1], line_color, OFS)
    ax.plot([50, 25], [nodes['pre'][1]-OFS, nodes['mp'][1]+OFS], color=line_color, lw=2)
    ax.plot([50, 75], [nodes['pre'][1]-OFS, nodes['cnn'][1]+OFS], color=line_color, lw=2)
    draw_arrow(nodes['mp'][0], nodes['mp'][1], nodes['calc'][0], nodes['calc'][1], alt_line_color, OFS)
    ax.plot([25, 50], [nodes['calc'][1]-OFS, nodes['rule'][1]+OFS], color=line_color, lw=2)
    ax.plot([75, 50], [nodes['cnn'][1]-OFS, nodes['rule'][1]+OFS], color=alt_line_color, lw=2)
    ax.plot([50, 25], [nodes['rule'][1]-OFS, nodes['hud'][1]+OFS], color=alt_line_color, lw=2)
    ax.plot([50, 75], [nodes['rule'][1]-OFS, nodes['tts'][1]+OFS], color=alt_line_color, lw=2)

    plt.text(50, 98, "SYSTEM ARCHITECTURE", ha='center', va='center', 
             color=text_color, fontsize=18, fontweight='bold')

    output_path = rf"c:\Users\thota\OneDrive\Documents\Project\results\simple_flowchart_{theme}.png"
    # Using solid facecolor to completely avoid transparency issues in Canva
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
    print(f"Saved: {output_path}")
    plt.close()

create_flowchart("dark")
create_flowchart("light")
