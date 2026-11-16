import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Ellipse, FancyArrowPatch, Arc

OUT = sys.argv[1]
rng = np.random.default_rng(11)
BG = "#0b1220"
CARD = "#111b2e"
EDGE = "#25415f"
ACC = "#5cc8ff"
ICON = "#dfe8f5"; BODY = "#1c2f4a"; WHEEL = "#0b1220"; CITYC = "#7fd0ff"; CF = ("#14304f", "#0e2640", "#1d4a78")
TXT = "#e8eef7"
MUTED = "#8fa0b5"
cmap = LinearSegmentedColormap.from_list(
    "h", ["#0b3d6e", "#1479b8", "#12b5b0", "#6fdc8c", "#e9e26b", "#ff9f5a", "#ff5a7a"])

# ---------------- scene ----------------
def make_scene(nbx=8, nby=5, bw=16.0, street=5.0):
    pitch = bw + street
    blds = []
    for i in range(nbx):
        for j in range(nby):
            ox, oy = i * pitch, j * pitch
            # split block in 1-2 buildings
            n = 2 if rng.random() < 0.45 else 1
            for k in range(n):
                if rng.random() < 0.12:
                    continue
                w = bw / n - 1.2 if n == 2 else rng.uniform(8, bw - 2)
                d = rng.uniform(7, bw - 2)
                x0 = ox + (k * bw / n if n == 2 else rng.uniform(0, bw - w))
                y0 = oy + rng.uniform(0, bw - d)
                h = rng.gamma(2.0, 3.0) + 4
                if rng.random() < 0.06:
                    h += 16
                gable = h < 14 and rng.random() < 0.6
                blds.append((x0, y0, w, d, h, gable))
    W = nbx * pitch; D = nby * pitch
    return blds, W, D, pitch, bw

def box_pts(b, dens):
    x0, y0, w, d, h, gable = b
    P = []
    # walls
    for (a, c, fixed, other) in [(x0, x0 + w, "y", y0), (x0, x0 + w, "y", y0 + d),
                                 (y0, y0 + d, "x", x0), (y0, y0 + d, "x", x0 + w)]:
        n = int((c - a) * h * 7 * dens)
        t = rng.uniform(a, c, n); z = rng.uniform(0, h, n); o = np.full(n, other)
        P.append(np.c_[t, o, z] if fixed == "y" else np.c_[o, t, z])
    n = int(w * d * 12 * dens)
    u = rng.uniform(0, 1, n); v = rng.uniform(0, 1, n)
    if gable:
        rh = min(w, d) * 0.28
        z = h + rh * (1 - np.abs(2 * u - 1))     # ridge along y
        P.append(np.c_[x0 + u * w, y0 + v * d, z])
    else:
        P.append(np.c_[x0 + u * w, y0 + v * d, np.full(n, h) + rng.normal(0, 0.04, n)])
    return np.vstack(P)

def scene_points(blds, W, D, pitch, bw, dens=1.0):
    P = []
    n = int(W * D * 5 * dens)
    gx, gy = rng.uniform(0, W, n), rng.uniform(0, D, n)
    gz = 0.9 * np.sin(gx / 37.0) + 0.7 * np.cos(gy / 23.0) + rng.normal(0, 0.05, n)
    P.append(np.c_[gx, gy, gz])
    for b in blds:
        P.append(box_pts(b, dens))
    # street trees
    for _ in range(260):
        tx = rng.uniform(0, W); ty = rng.uniform(0, D)
        mx, my = tx % pitch, ty % pitch
        if mx < bw and my < bw:
            continue
        if any(b[0] - 1.5 < tx < b[0] + b[2] + 1.5 and b[1] - 1.5 < ty < b[1] + b[3] + 1.5 for b in blds):
            continue
        r = rng.uniform(1.1, 2.0); m = int(120 * dens)
        v = rng.normal(size=(m, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        v[:, 2] = np.abs(v[:, 2])
        P.append(np.c_[tx + v[:, 0] * r, ty + v[:, 1] * r, 2.0 + v[:, 2] * r * 1.2])
    return np.vstack(P)

def proj(P, cx, cy, yaw, tilt):
    x, y, z = P[:, 0] - cx, P[:, 1] - cy, P[:, 2]
    xr = x * np.cos(yaw) - y * np.sin(yaw)
    yr = x * np.sin(yaw) + y * np.cos(yaw)
    return xr, yr * tilt + z * 0.95, yr

# ---------------- icons ----------------
def drone(ax, x, y, s=1.0, c=None, z=20):
    c = c or (CITYC if 'city_icon' in 'def drone(' else ICON)
    ax.add_patch(Ellipse((x, y), 46 * s, 16 * s, fc=c, ec="none", zorder=z))
    for dx, dy in [(-44, 12), (44, 12), (-36, -8), (36, -8)]:
        ax.plot([x, x + dx * s], [y, y + dy * s], color=c, lw=2.2 * s, zorder=z)
        ax.add_patch(Ellipse((x + dx * s, y + dy * s + 4 * s), 34 * s, 6 * s, fc=ACC, ec="none", alpha=0.85, zorder=z))
    ax.add_patch(Circle((x, y - 12 * s), 6 * s, fc="#223a57", ec=c, lw=1.2, zorder=z))

def satellite(ax, x, y, s=1.0, ang=-25, c=None, z=20):
    c = c or (CITYC if 'city_icon' in 'def satellite(' else ICON)
    r = np.deg2rad(ang)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    def poly(pts, fc, ec="none", lw=0):
        p = (R @ (np.array(pts) * s).T).T + [x, y]
        ax.add_patch(Polygon(p, closed=True, fc=fc, ec=ec, lw=lw, zorder=z))
    poly([(-12, -9), (12, -9), (12, 9), (-12, 9)], "#e3c46a")
    for sgn in (-1, 1):
        for k in range(2):
            x0 = sgn * (16 + k * 21) if sgn > 0 else sgn * (16 + k * 21) - 19
            poly([(x0, -10), (x0 + 19, -10), (x0 + 19, 10), (x0, 10)], "#2c5b9a", ec=c, lw=0.8)
    poly([(-2, -9), (2, -9), (3, -16), (-3, -16)], c)

def car(ax, x, y, s=1.0, c=None, z=20):
    c = c or (CITYC if 'city_icon' in 'def car(' else ICON)
    ax.add_patch(FancyBboxPatch((x - 52 * s, y - 6 * s), 104 * s, 22 * s, boxstyle="round,pad=0,rounding_size=%f" % (8 * s),
                                fc=BODY, ec=c, lw=1.6, zorder=z))
    ax.add_patch(Polygon([(x - 30 * s, y + 16 * s), (x - 18 * s, y + 32 * s), (x + 22 * s, y + 32 * s), (x + 38 * s, y + 16 * s)],
                         closed=True, fc=BODY, ec=c, lw=1.6, zorder=z))
    for dx in (-30, 30):
        ax.add_patch(Circle((x + dx * s, y - 6 * s), 9 * s, fc=WHEEL, ec=c, lw=1.6, zorder=z + 1))
    ax.add_patch(FancyBboxPatch((x - 6 * s, y + 32 * s), 12 * s, 7 * s, boxstyle="round,pad=0,rounding_size=2", fc=ACC, ec="none", zorder=z))
    for k, rr in enumerate((12, 20, 28)):
        ax.add_patch(Arc((x, y + 36 * s), rr * 2 * s, rr * 1.4 * s, theta1=30, theta2=150, color=ACC, lw=1.3, alpha=0.9 - 0.2 * k, zorder=z))

def robot(ax, x, y, s=1.0, c=None, z=20):
    c = c or (CITYC if 'city_icon' in 'def robot(' else ICON)
    ax.add_patch(FancyBboxPatch((x - 30 * s, y), 60 * s, 26 * s, boxstyle="round,pad=0,rounding_size=%f" % (6 * s), fc=BODY, ec=c, lw=1.6, zorder=z))
    for dx in (-26, 26):
        ax.add_patch(Circle((x + dx * s, y - 2 * s), 11 * s, fc=WHEEL, ec=c, lw=1.6, zorder=z + 1))
    ax.plot([x, x + 14 * s, x + 30 * s], [y + 26 * s, y + 52 * s, y + 46 * s], color=c, lw=2.6 * s, zorder=z)
    ax.add_patch(Circle((x + 32 * s, y + 46 * s), 5 * s, fc=ACC, ec="none", zorder=z))
    ax.add_patch(Circle((x - 12 * s, y + 13 * s), 4 * s, fc=ACC, ec="none", zorder=z + 1))

def city_icon(ax, x, y, s=1.0, c=None, z=20):
    c = c or (CITYC if 'city_icon' in 'def city_icon(' else ICON)
    def cube(cx, cy, w, h, d=10):
        fr = [(cx, cy), (cx + w, cy), (cx + w, cy + h), (cx, cy + h)]
        tp = [(cx, cy + h), (cx + w, cy + h), (cx + w + d, cy + h + d * 0.6), (cx + d, cy + h + d * 0.6)]
        sd = [(cx + w, cy), (cx + w + d, cy + d * 0.6), (cx + w + d, cy + h + d * 0.6), (cx + w, cy + h)]
        for pts, f in ((fr, CF[0]), (sd, CF[1]), (tp, CF[2])):
            ax.add_patch(Polygon([(x + px * s, y + py * s) for px, py in pts], closed=True, fc=f, ec=c, lw=1.2, zorder=z))
    cube(-46, 0, 26, 34); cube(-14, -6, 28, 58); cube(20, 0, 24, 26)

def sensor_card(ax, x, y, w, h, title, sub, icon, icon_s=0.9):
    icon(ax, x + 90, y + h / 2 - 4, icon_s)
    ax.text(x + 175, y + h / 2 + 12, title, color=TXT, fontsize=13.5, fontweight="bold", va="center", zorder=5)
    ax.text(x + 175, y + h / 2 - 16, sub, color=MUTED, fontsize=10, va="center", zorder=5)

# ================= BANNER =================
blds, W, D, pitch, bw = make_scene()
P = scene_points(blds, W, D, pitch, bw)
fig = plt.figure(figsize=(16, 4.4), dpi=150, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 2400); ax.set_ylim(0, 660); ax.axis("off"); ax.set_facecolor(BG)
# soft glow
yy, xx = np.mgrid[0:660, 0:2400]
glow = np.exp(-(((xx - 1650) / 700) ** 2 + ((yy - 250) / 260) ** 2))
img = np.zeros((660, 2400, 4)); img[..., :3] = matplotlib.colors.to_rgb("#143a63"); img[..., 3] = glow * 0.55
ax.imshow(img, extent=[0, 2400, 0, 660], origin="lower", zorder=0, aspect="auto")

sx, sy, dp = proj(P, P[:, 0].mean(), P[:, 1].mean(), 0.62, 0.40)
tw = 1250; th = 330
sc = min(tw / (sx.max() - sx.min()), th / (sy.max() - sy.min()))
X = (sx - sx.min()) * sc + 1060
Y = (sy - sy.min()) * sc + 40
o = np.argsort(-dp)
c = cmap(np.clip(P[o, 2] / 26, 0, 1)); c[:, 3] = 0.92
# darken ground
gmask = P[o, 2] < 1.6
c[gmask] = matplotlib.colors.to_rgba("#134c7a", 0.55)
ax.scatter(X[o], Y[o], s=1.3, c=c, linewidths=0, rasterized=True, zorder=3)

# scan beams
tx, ty = 1650, 190
satellite(ax, 2230, 585, 1.5, -22)
ax.add_patch(Polygon([(2218, 560), (tx - 220, ty + 20), (tx + 260, ty - 10)], closed=True, fc="#5cc8ff", alpha=0.07, ec="none", zorder=1))
ax.plot([2218, tx], [560, ty + 10], ls=(0, (5, 4)), color=ACC, lw=1.2, alpha=0.8, zorder=15)
drone(ax, 1470, 560, 1.25)
ax.add_patch(Polygon([(1470, 540), (1290, 250), (1600, 235)], closed=True, fc="#5cc8ff", alpha=0.09, ec="none", zorder=1))
ax.plot([1470, 1450], [540, 270], ls=(0, (5, 4)), color=ACC, lw=1.2, alpha=0.8, zorder=15)

ax.text(90, 395, "Abbas Salehitangrizi", color="white", fontsize=44, fontweight="bold", va="center", zorder=30)
ax.text(94, 300, "Geospatial AI  ·  3D Computer Vision  ·  LiDAR Mapping", color="#8fd3ff", fontsize=19, va="center", zorder=30)
ax.text(94, 235, "Turning point clouds and imagery into 3D models of cities", color=MUTED, fontsize=15, va="center", zorder=30)
for k, (lab) in enumerate(["Remote Sensing", "Photogrammetry", "Deep Learning"]):
    wlab = 40 + len(lab) * 14.5
    xs = 94 + sum(40 + len(l) * 14.5 + 16 for l in ["Remote Sensing", "Photogrammetry", "Deep Learning"][:k])
    ax.add_patch(FancyBboxPatch((xs, 130), wlab, 36, boxstyle="round,pad=0,rounding_size=18", fc="none", ec="#3b6fa0", lw=1.3, zorder=30))
    ax.text(xs + wlab / 2, 148, lab, color="#bcd6ee", fontsize=12.5, ha="center", va="center", zorder=31)
fig.savefig(f"{OUT}/banner.png", facecolor=BG)
plt.close(fig)

# ================= CONCEPT DIAGRAM =================
BG = "#ffffff"; CARD = "#f3f7fb"; EDGE = "#c3d3e4"; ACC = "#1f7fc4"; TXT = "#14233a"; MUTED = "#5a6c82"
ICON = "#1d3550"; BODY = "#dbe8f5"; WHEEL = "#ffffff"; CITYC = "#1f6fb0"; CF = ("#dbe9f7", "#c2d8ee", "#eef5fc")

fig = plt.figure(figsize=(17.8, 7.6), dpi=125, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1780); ax.set_ylim(140, 900); ax.axis("off")
ax.text(890, 862, "From sensors to 3D understanding", color=TXT, fontsize=28, fontweight="bold", ha="center", va="center")
ax.text(890, 822, "Multi-modal geospatial data  →  deep learning  →  real-world applications", color=MUTED, fontsize=14, ha="center", va="center")

# column headers
for x, t in [(150, "SENSING"), (800, "AI & FUSION"), (1510, "APPLICATIONS")]:
    ax.text(x, 760, t, color=ACC, fontsize=13, fontweight="bold", ha="center", va="center")
    ax.plot([x - 90, x + 90], [738, 738], color=ACC, lw=1.2, alpha=0.6)

# sensors
cw, ch = 390, 150
sensor_card(ax, 20, 200, cw, ch, "Satellite imagery", "Optical · SAR · multispectral", lambda a, x, y, s: satellite(a, x, y, s * 1.1, -20))
sensor_card(ax, 20, 380, cw, ch, "UAV photogrammetry", "Multi-view images · 3D mesh", lambda a, x, y, s: drone(a, x, y, s * 0.8))
sensor_card(ax, 20, 560, cw, ch, "LiDAR scanning", "Airborne & mobile point clouds", lambda a, x, y, s: car(a, x, y - 10, s * 0.8))

# data layers (center): point cloud + optical imagery
from matplotlib.transforms import Affine2D
b2, W2, D2, p2, bw2 = make_scene(4, 3)
P2 = scene_points(b2, W2, D2, p2, bw2, dens=0.9)
YAW, TILT = 0.62, 0.42
sx2, sy2, dp2 = proj(P2, P2[:, 0].mean(), P2[:, 1].mean(), YAW, TILT)
sc2 = min(290 / (sx2.max() - sx2.min()), 330 / (sy2.max() - sy2.min()))
CLOUD_Y, IMG_Y = 600, 345
X2 = (sx2 - sx2.min()) * sc2 + 490; Y2 = (sy2 - sy2.min()) * sc2 + (CLOUD_Y - (sy2.max() - sy2.min()) * sc2 / 2)
o2 = np.argsort(-dp2)
ax.scatter(X2[o2], Y2[o2], s=1.6, c=cmap(np.clip(P2[o2, 2] / 26, 0, 1)), linewidths=0, zorder=4, rasterized=True)
ax.text(635, 712, "Point cloud", color=TXT, fontsize=14, fontweight="bold", ha="center", va="center", zorder=5)

def make_ortho(blds, W, D, bw, pitch, res=0.2):
    r2 = np.random.default_rng(5)
    nx, ny = int(W / res), int(D / res)
    xs = (np.arange(nx) + 0.5) * res; ys = (np.arange(ny) + 0.5) * res
    XX, YY = np.meshgrid(xs, ys)
    img = np.zeros((ny, nx, 3))
    img[:] = matplotlib.colors.to_rgb("#a9b88f")
    street = ((XX % pitch) >= bw) | ((YY % pitch) >= bw)
    img[street] = matplotlib.colors.to_rgb("#7b838d")
    img += r2.normal(0, 0.018, img.shape)
    roofs = ["#c9744f", "#a9aeb6", "#6f7b8a", "#d9d3c5", "#8a5a48", "#4f6f90", "#b9b2a4"]
    for (x0, y0, w, d, h, gable) in blds:
        sh = (XX >= x0 + 1.4) & (XX <= x0 + w + 1.4) & (YY >= y0 - 1.4) & (YY <= y0 + d - 1.4)
        img[sh] *= 0.72
    for (x0, y0, w, d, h, gable) in blds:
        m = (XX >= x0) & (XX <= x0 + w) & (YY >= y0) & (YY <= y0 + d)
        col = np.array(matplotlib.colors.to_rgb(roofs[r2.integers(len(roofs))]))
        img[m] = col
        if gable:
            left = m & (XX < x0 + w / 2); right = m & (XX >= x0 + w / 2)
            img[left] = np.clip(col * 1.12, 0, 1); img[right] = col * 0.82
    for _ in range(140):
        tx, ty = r2.uniform(0, W), r2.uniform(0, D)
        if any(b[0] - 0.5 < tx < b[0] + b[2] + 0.5 and b[1] - 0.5 < ty < b[1] + b[3] + 0.5 for b in blds):
            continue
        rr = r2.uniform(1.0, 1.9)
        m = (XX - tx) ** 2 + (YY - ty) ** 2 < rr ** 2
        img[m] = np.array(matplotlib.colors.to_rgb("#4f8a47")) * r2.uniform(0.8, 1.15)
    return np.clip(img, 0, 1)

ortho = make_ortho(b2, W2, D2, bw2, p2)
cxm, cym = P2[:, 0].mean(), P2[:, 1].mean()
c0, s0 = np.cos(YAW), np.sin(YAW)
def aff(yoff):
    return np.array([[sc2 * c0, -sc2 * s0, sc2 * (-c0 * cxm + s0 * cym - sx2.min()) + 490],
                     [TILT * sc2 * s0, TILT * sc2 * c0, TILT * sc2 * (-s0 * cxm - c0 * cym) - sc2 * sy2.min() + yoff],
                     [0, 0, 1]])
corners = np.array([[0, 0, 1], [W2, 0, 1], [0, D2, 1], [W2, D2, 1]]).T
cy_ = (aff(0) @ corners)[1]
M = aff(IMG_Y - (cy_.min() + cy_.max()) / 2)
im = ax.imshow(ortho, extent=(0, W2, 0, D2), origin="lower", interpolation="bilinear", aspect="auto", zorder=3)
im.set_transform(Affine2D(M) + ax.transData)
poly = (M @ corners)[:2].T[[0, 1, 3, 2]]
ax.add_patch(Polygon(poly, closed=True, fc="none", ec=ACC, lw=1.4, zorder=4))
ax.set_xlim(0, 1780); ax.set_ylim(140, 900)
ax.text(635, 262, "Satellite / UAV imagery", color=TXT, fontsize=14, fontweight="bold", ha="center", va="center", zorder=5)

# neural net
layers = [4, 6, 6, 3]
lx = [880, 960, 1040, 1120]
pos = []
for L, n in enumerate(layers):
    ys = np.linspace(560 - (n - 4) * 0 , 560 - 280, n) if False else np.linspace(600, 330, n)
    pos.append([(lx[L], y) for y in ys])
ax.text(1000, 665, "Deep neural network", color=TXT, fontsize=14, fontweight="bold", ha="center", va="center", zorder=5)
for L in range(len(layers) - 1):
    for (x1, y1) in pos[L]:
        for (x2, y2) in pos[L + 1]:
            ax.plot([x1, x2], [y1, y2], color="#3d7fb8", lw=0.7, alpha=0.5, zorder=3)
for L, pts in enumerate(pos):
    for (x, y) in pts:
        ax.add_patch(Circle((x, y), 13, fc=["#1479b8", "#12b5b0", "#6fdc8c", "#e9e26b"][L], ec="white", lw=1.0, zorder=4))
ax.text(1000, 290, "segmentation · detection · reconstruction", color=MUTED, fontsize=10.5, ha="center", va="center", zorder=5)

# application icons
def sprout(ax, x, y, s=1.0, c=None, z=20):
    c = c or ICON
    ax.plot([x - 40 * s, x + 40 * s], [y - 20 * s] * 2, color=c, lw=2.2, zorder=z)
    for dx in (-26, 26):
        ax.plot([x + dx * s, x + dx * s], [y - 20 * s, y - 4 * s], color="#4f9e4a", lw=2.0, zorder=z)
        ax.add_patch(Ellipse((x + dx * s, y - 2 * s), 8 * s, 10 * s, fc="#6fdc8c", ec="#4f9e4a", lw=1.0, zorder=z))
    ax.plot([x, x], [y - 20 * s, y + 16 * s], color="#3f8a3a", lw=2.6, zorder=z)
    ax.add_patch(Ellipse((x - 13 * s, y + 8 * s), 28 * s, 12 * s, angle=28, fc="#6fdc8c", ec="#3f8a3a", lw=1.2, zorder=z))
    ax.add_patch(Ellipse((x + 13 * s, y + 20 * s), 28 * s, 12 * s, angle=-28, fc="#8be8a0", ec="#3f8a3a", lw=1.2, zorder=z))

def flood(ax, x, y, s=1.0, c=None, z=20):
    c = c or ICON
    ax.add_patch(Polygon([(x - 26 * s, y + 2 * s), (x, y + 28 * s), (x + 26 * s, y + 2 * s)], closed=True, fc=BODY, ec=c, lw=1.6, zorder=z))
    ax.add_patch(FancyBboxPatch((x - 20 * s, y - 20 * s), 40 * s, 22 * s, boxstyle="square,pad=0", fc=BODY, ec=c, lw=1.6, zorder=z))
    xs = np.linspace(x - 44 * s, x + 44 * s, 80)
    top = y - 8 * s + 3 * s * np.sin((xs - x) / (7 * s))
    ax.fill_between(xs, y - 26 * s, top, color=ACC, alpha=0.35, zorder=z + 1, lw=0)
    ax.plot(xs, top, color=ACC, lw=2.0, zorder=z + 2)
    ax.plot(xs, top - 9 * s, color=ACC, lw=1.4, alpha=0.7, zorder=z + 2)

def marine(ax, x, y, s=1.0, c=None, z=20):
    c = c or ICON
    xs = np.linspace(x - 44 * s, x + 44 * s, 80)
    ax.plot(xs, y + 6 * s + 2.5 * s * np.sin((xs - x) / (6 * s)), color=ACC, lw=2.0, zorder=z + 2)
    sea = y - 22 * s + 7 * s * np.sin((xs - x) / (13 * s) + 1)
    ax.fill_between(xs, y - 30 * s, sea, color="#8fb8d9", alpha=0.8, zorder=z, lw=0)
    ax.plot(xs, sea, color=c, lw=1.6, zorder=z + 1)
    ax.add_patch(Polygon([(x, y + 30 * s), (x - 16 * s, y - 20 * s), (x + 16 * s, y - 20 * s)], closed=True, fc="#6fdc8c", alpha=0.3, ec="none", zorder=z + 1))
    ax.plot([x, x], [y + 30 * s, y - 20 * s], color="#3f8a3a", lw=1.4, ls=(0, (3, 2)), zorder=z + 2)
    ax.add_patch(FancyBboxPatch((x - 7 * s, y + 28 * s), 14 * s, 6 * s, boxstyle="round,pad=0,rounding_size=2", fc=c, ec="none", zorder=z + 2))

# applications
def app_row(cy, title, sub, icon, s):
    icon(ax, 1290, cy - 5, s)
    ax.text(1350, cy + 13, title, color=TXT, fontsize=13.5, fontweight="bold", va="center", zorder=5)
    ax.text(1350, cy - 19, sub, color=MUTED, fontsize=10, va="center", zorder=5)
APPS = [
    (690, "3D urban modeling", "Façades · roofs · windows", lambda a, x, y, s: city_icon(a, x, y - 15, s), 0.8),
    (595, "Autonomous driving", "HD maps · scene understanding", lambda a, x, y, s: car(a, x, y - 12, s), 0.6),
    (500, "Robotics", "Mapping · navigation", lambda a, x, y, s: robot(a, x, y - 20, s), 0.8),
    (405, "Precision agriculture", "Crop & field monitoring", lambda a, x, y, s: sprout(a, x, y, s), 0.9),
    (310, "Flood & disaster response", "Damage assessment · flood mapping", lambda a, x, y, s: flood(a, x, y, s), 0.9),
    (215, "Coastal & marine mapping", "Bathymetric LiDAR · habitats", lambda a, x, y, s: marine(a, x, y, s), 0.9),
]
for (cy, t, sb, ic, sz) in APPS:
    app_row(cy, t, sb, ic, sz)

# arrows
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, color=ACC, lw=1.8, zorder=6,
                                 connectionstyle="arc3,rad=0"))
IMG_C, CLD_C = IMG_Y, CLOUD_Y - 40
for (y, tgt) in [(635, CLD_C), (455, CLD_C), (455, IMG_C), (275, IMG_C)]:
    arrow(440, y, 478, tgt)
arrow(802, CLD_C, 848, 470)
arrow(802, IMG_C, 848, 470)
for (cy, *_r) in APPS:
    arrow(1152, 470, 1198, cy)

fig.savefig(f"{OUT}/interests.png", facecolor=BG)
print("ok")
