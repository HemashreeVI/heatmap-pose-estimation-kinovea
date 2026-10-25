"""Generate docs/assets/preview.png, the README banner / GitHub social preview (1280x640).

Photo: COCO val2017 image 000000000785 (Flickr photo 6795644157), licensed
CC BY 2.0. Keypoints are the COCO human annotations; each joint is drawn
as the Gaussian heatmap a heatmap model like HRNet is trained to predict.
"""
import os.path as osp

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import patheffects as pe

ASSETS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "docs", "assets")
W, H = 1280, 640
CYAN, ORANGE, RED, GREEN, MUTED = "#4FC3F7", "#FFB74D", "#FF5A5F", "#69F0AE", "#9AA5B8"
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
KP = np.array([[367, 81], [374, 73], [360, 75], [386, 78], [356, 81], [399, 108], [358, 129], [433, 142],
               [341, 159], [449, 165], [309, 178], [424, 203], [393, 214], [429, 294], [367, 273],
               [466, 362], [396, 341]], float)
BONES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
         (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

# ---- photo panel: crop around the skier and scale up
img = cv2.cvtColor(cv2.imread(osp.join(ASSETS, "coco_000000000785.jpg")), cv2.COLOR_BGR2RGB).astype(float) / 255
x0, y0, x1, y1 = 200, 35, 620, 415
crop = img[y0:y1, x0:x1]
PX, PY, PW = 676, 44, 560
s = PW / (x1 - x0)
PH = int(round((y1 - y0) * s))
crop = cv2.resize(crop, (PW, PH), interpolation=cv2.INTER_CUBIC)
kp = (KP - [x0, y0]) * s

# darken + cool the photo a little so the heatmaps glow
photo = crop * 0.68 + np.array([0.01, 0.03, 0.08])
ys, xs = np.mgrid[0:PH, 0:PW]
sigma = 2 * (346.68 * 1.25 / 64) * s          # sigma = 2 heatmap px, in panel pixels
heat = np.zeros((PH, PW))
for x, y in kp:
    heat = np.maximum(heat, np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2)))
colour = plt.get_cmap("turbo")(heat)[..., :3]
alpha = (np.clip(heat, 0, 1) ** 0.8 * 0.85)[..., None]
photo = photo * (1 - alpha) + colour * alpha

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
gy, gx = np.mgrid[0:H, 0:W]
t = np.clip(gx / W * 0.6 + gy / H * 0.4, 0, 1)[..., None]
bg = np.array([8, 13, 30]) / 255 * (1 - t) + np.array([18, 32, 64]) / 255 * t
ax.imshow(bg, extent=[0, W, H, 0], aspect="auto", zorder=0)
for v in range(0, W, 40):
    ax.plot([v, v], [0, H], color="white", alpha=0.025, lw=1, zorder=0)
for v in range(0, H, 40):
    ax.plot([0, W], [v, v], color="white", alpha=0.025, lw=1, zorder=0)

# rounded photo panel
im = ax.imshow(np.clip(photo, 0, 1), extent=[PX, PX + PW, PY + PH, PY], zorder=2)
clip = FancyBboxPatch((PX, PY), PW, PH, boxstyle="round,pad=0,rounding_size=18", transform=ax.transData)
im.set_clip_path(clip)
ax.add_patch(FancyBboxPatch((PX, PY), PW, PH, boxstyle="round,pad=0,rounding_size=18", fill=False,
                            ec="#2A4A7A", lw=2, zorder=5))

# skeleton
P = kp + [PX, PY]
def bone_colour(a, b):
    na, nb = NAMES[a], NAMES[b]
    if na.startswith("left") and nb.startswith("left"):
        return ORANGE
    if na.startswith("right") and nb.startswith("right"):
        return CYAN
    return "#E8E8E8"
for a, b in BONES:
    ax.plot(*zip(P[a], P[b]), color=bone_colour(a, b), lw=3, solid_capstyle="round", zorder=3,
            path_effects=[pe.Stroke(linewidth=6, foreground="black", alpha=0.35), pe.Normal()])
ax.scatter(P[:, 0], P[:, 1], s=26, color="white", edgecolor="black", linewidth=0.8, zorder=4)

# knee angle on the left knee
hip, knee, ank = P[11], P[13], P[15]
a1 = np.arctan2(*(hip - knee)[::-1]); a2 = np.arctan2(*(ank - knee)[::-1])
v1, v2 = hip - knee, ank - knee
ang = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))
if abs(a2 - a1) > np.pi:
    a2 += 2 * np.pi * (1 if a2 < a1 else -1)
th = np.linspace(a1, a2, 40)
ax.plot(knee[0] + 30 * np.cos(th), knee[1] + 30 * np.sin(th), color=GREEN, lw=3, zorder=5)
ax.text(knee[0] + 44, knee[1] - 6, "knee angle θ", color=GREEN, fontsize=13, weight="bold", va="center", zorder=6,
        path_effects=[pe.Stroke(linewidth=4, foreground="#0B1328"), pe.Normal()])

# per-joint heatmap tiles (inset, bottom-left of the photo)
tiles = [("nose", 0), ("l. wrist", 9), ("r. elbow", 8), ("l. knee", 13), ("r. ankle", 16)]
TX, TY, TW, TH_ = PX + PW - 5 * 54 - 10, PY + 46, 46, 62
ax.add_patch(FancyBboxPatch((TX - 10, TY - 30), len(tiles) * (TW + 8) + 12, TH_ + 52,
                            boxstyle="round,pad=0,rounding_size=10", fc="#0E1A33", ec="#2A4A7A", lw=1.2, alpha=0.9, zorder=6))
ax.text(TX - 2, TY - 15, "per-joint heatmaps (64×48)", color="white", fontsize=10, weight="bold", va="center", zorder=7)
hy, hx = np.mgrid[0:64, 0:48]
for i, (label, j) in enumerate(tiles):
    jitter = [(-3, 4), (5, -2), (-4, -5), (2, 6), (6, 1)][i]
    m = np.exp(-((hx - 24 - jitter[0]) ** 2 + (hy - 32 - jitter[1]) ** 2) / (2 * 3.0 ** 2))
    x = TX + i * (TW + 8)
    ax.imshow(m, cmap="turbo", vmin=0, vmax=1, extent=[x, x + TW, TY + TH_, TY], zorder=7)
    ax.text(x + TW / 2, TY + TH_ + 10, label, color=MUTED, fontsize=8.5, ha="center", va="center", zorder=7)

ax.text(PX + PW - 12, PY + PH - 12, "Photo: COCO / Flickr #6795644157, CC BY 2.0", color="#E0E6F0", alpha=0.85,
        fontsize=8.5, ha="right", va="bottom", zorder=7,
        path_effects=[pe.Stroke(linewidth=3, foreground="#000000", alpha=0.5), pe.Normal()])

# ---- left column text
L = 64
ax.add_patch(FancyBboxPatch((L, 56), 330, 32, boxstyle="round,pad=0,rounding_size=16", fc="#12305A", ec=CYAN, lw=1.2, zorder=6))
ax.text(L + 165, 72, "RESEARCH INTERNSHIP · VIT CHENNAI", color=CYAN, fontsize=11.5, weight="bold", ha="center", va="center", zorder=7)
ax.text(L, 140, "Human Pose", color="white", fontsize=50, weight="bold", va="center", zorder=7)
ax.text(L, 205, "Estimation", color="white", fontsize=50, weight="bold", va="center", zorder=7)
ax.text(L, 270, "using Heatmaps", color=CYAN, fontsize=50, weight="bold", va="center", zorder=7)
ax.text(L, 330, "17 body joints with MMPose (HRNet) · joint-confidence\nheatmap analysis · validated against Kinovea",
        color="#CBD5E0", fontsize=15, va="top", linespacing=1.55, zorder=7)
x, y = L, 418
for c in ["MMPose", "HRNet", "PyTorch", "OpenCV", "Matplotlib", "Kinovea"]:
    w = len(c) * 10.5 + 30
    if x + w > 600:
        x, y = L, y + 50
    ax.add_patch(FancyBboxPatch((x, y), w, 36, boxstyle="round,pad=0,rounding_size=18", fc="#132A4A", ec="#3D6DA8", lw=1.3, zorder=6))
    ax.text(x + w / 2, y + 18.5, c, color="white", fontsize=13, weight="bold", ha="center", va="center", zorder=7)
    x += w + 12

# ---- bottom pipeline strip
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()
def width(txt):
    bb = txt.get_window_extent(renderer)
    return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
y = 580
x = L
steps = [("Video", MUTED), ("HRNet", CYAN), ("Heatmaps", CYAN), ("Reliability flags", RED), ("Kinovea validation", GREEN)]
for i, (s_, col) in enumerate(steps):
    tx = ax.text(x, y, s_, color=col, fontsize=13, weight="bold", va="center", zorder=7)
    x += width(tx) + 12
    if i < len(steps) - 1:
        ta = ax.text(x, y, "→", color="#5A6B85", fontsize=15, va="center", zorder=7)
        x += width(ta) + 12
ax.text(L, 612, "github.com/HemashreeVI/heatmap-pose-estimation-kinovea", color="#6B7A92",
        fontsize=11, va="center", family="monospace", zorder=7)

fig.savefig(osp.join(ASSETS, "preview.png"), dpi=100)
print("wrote", osp.join(ASSETS, "preview.png"))
