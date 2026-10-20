#!/usr/bin/env python3
"""
Model Town Park, Lahore -- a pixel-art day in the park.

One continuous side-scrolling world, rendered at 270x480 and upscaled 4x
(nearest neighbour) to 1080x1920. The camera glides through the park while
the day passes:

    Main Gate (dawn) -> clay jogging track & the little hills (morning)
    -> sundial lawn with picnics and play area (afternoon)
    -> the boating lake and bridge (sunset into a lamp-lit night)

Pure numpy; frames are piped into ffmpeg. A soft music-box soundtrack with
birdsong by day and crickets by night is synthesised alongside.

    python3 model_town_park.py out.mp4
    python3 model_town_park.py --preview 2,12,22,40 OUTDIR
"""
import math
import os
import subprocess
import sys
import wave

import numpy as np

W, H, SCALE, FPS = 270, 570, 4, 30
SH, OFF = 480, 90          # screen height; world rows above OFF are cropped
DUR = 50.0
NF = int(DUR * FPS)
WW = 1410                  # world width
PAR = 0.55                 # far tree-line parallax
PW = int(WW * PAR) + W + 8
WY0, WY1 = 318, 408        # lake surface rows
LAKE_X0 = 1118
TRACK_Y0, TRACK_Y1, TRACK_X1 = 374, 388, 700
PATH_Y0, PATH_Y1 = 466, 478
SUNDIAL = (905, 402)
SWING_X, SLIDE_X = 780, 816

STOPS = [(0, 0), (5.5, 0), (9.5, 380), (16, 380), (20, 770), (27, 770), (31, 1140), (DUR, 1140)]
FG_PAR = 1.25               # framing foreground trees


def C(r, g, b):
    return np.array([r, g, b], dtype=float) / 255.0


def hash01(*a):
    x = math.sin(sum(v * k for v, k in zip(a, (12.9898, 78.233, 37.719, 4.581)))) * 43758.5453
    return x - math.floor(x)


BAY4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
BAY = np.tile(BAY4, (H // 4 + 1, W // 4 + 1))[:H, :W]
YY, XX = np.mgrid[0:H, 0:W]


def kf(T, keys):
    if T <= keys[0][0]:
        return np.asarray(keys[0][1], float)
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if T <= t1:
            u = (T - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)
            return np.asarray(v0, float) * (1 - u) + np.asarray(v1, float) * u
    return np.asarray(keys[-1][1], float)


def cam_at(T):
    return int(round(float(kf(T, STOPS))))


# ---------------------------------------------------------------- time of day
SKY_KEYS = [
    (0, [(84, 96, 168), (196, 156, 180), (252, 196, 160)]),
    (4.5, [(60, 128, 220), (122, 180, 240), (196, 226, 250)]),
    (12, [(40, 112, 222), (90, 166, 240), (164, 214, 250)]),
    (24, [(44, 112, 216), (98, 166, 236), (176, 214, 246)]),
    (28.5, [(62, 100, 192), (154, 150, 204), (255, 202, 134)]),
    (32, [(50, 50, 122), (184, 92, 122), (255, 142, 82)]),
    (35, [(24, 26, 72), (72, 46, 104), (164, 82, 104)]),
    (38, [(9, 13, 36), (20, 26, 62), (42, 42, 88)]),
    (DUR, [(9, 13, 36), (20, 26, 62), (42, 42, 88)]),
]
AMB_KEYS = [(0, (0.78, 0.74, 0.84)), (4.5, (1, 0.98, 0.95)), (12, (1, 1, 1)), (24, (1, 0.97, 0.92)),
            (28.5, (1.0, 0.87, 0.72)), (32, (0.86, 0.62, 0.56)), (35, (0.46, 0.42, 0.62)),
            (38, (0.27, 0.31, 0.5)), (DUR, (0.27, 0.31, 0.5))]
CLOUD_KEYS = [
    (0, [(255, 222, 220), (232, 196, 206), (196, 168, 196)]),
    (4.5, [(255, 255, 255), (236, 242, 250), (200, 214, 236)]),
    (24, [(255, 255, 255), (236, 242, 250), (200, 214, 236)]),
    (28.5, [(255, 240, 214), (246, 212, 182), (214, 170, 160)]),
    (32, [(255, 192, 150), (240, 140, 130), (170, 90, 120)]),
    (35, [(150, 92, 124), (110, 72, 112), (70, 52, 92)]),
    (38, [(58, 62, 98), (42, 46, 82), (30, 34, 64)]),
    (DUR, [(58, 62, 98), (42, 46, 82), (30, 34, 64)]),
]


def lights_at(T):
    return float(np.clip((T - 32.4) / 3.6, 0, 1))


def night_at(T):
    return float(kf(T, [(0, 0.15), (3, 0), (34, 0), (38, 1), (DUR, 1)]))


# ---------------------------------------------------------------- layers
class Layer:
    """Albedo + emissive paint buffer. `ox` shifts world x into this buffer."""

    def __init__(self, w=W, ox=0):
        self.w, self.ox = w, ox
        self.rgb = np.zeros((H, w, 3))
        self.a = np.zeros((H, w), bool)
        self.em = np.zeros((H, w, 3))
        self.thr = np.full((H, w), -1.0)
        self.twa = np.zeros((H, w))
        self.ph = np.zeros((H, w))

    def put(self, ys, xs, col, em=None, thr=-1.0, twa=0.0, ph=0.0):
        ys = np.rint(np.asarray(ys, float)).astype(int).ravel()
        xs = np.rint(np.asarray(xs, float)).astype(int).ravel() - self.ox
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < self.w)
        if not ok.any():
            return
        col = np.asarray(col, float)
        col = col[ok] if col.ndim == 2 else col
        ys, xs = ys[ok], xs[ok]
        self.rgb[ys, xs] = col
        self.a[ys, xs] = True
        if em is None:
            self.em[ys, xs] = 0
            self.thr[ys, xs] = -1
            self.twa[ys, xs] = 0
            return
        em = np.asarray(em, float)
        self.em[ys, xs] = em[ok] if em.ndim == 2 else em
        for arr, v in ((self.thr, thr), (self.twa, twa), (self.ph, ph)):
            v = np.asarray(v, float)
            arr[ys, xs] = v[ok] if v.ndim == 1 else v

    def rect(self, x0, y0, x1, y1, col, **k):
        x0, x1 = int(round(x0)), int(round(x1))
        y0, y1 = max(int(round(y0)), 0), min(int(round(y1)), H)
        if x0 < x1 and y0 < y1:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            self.put(ys, xs, col, **k)

    def px(self, x, y, col, **k):
        self.put([y], [x], col, **k)

    def line(self, x0, y0, x1, y1, col, **k):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        self.put(np.linspace(y0, y1, n), np.linspace(x0, x1, n), col, **k)

    def mask(self, x0, y0, m, col, **k):
        ys, xs = np.nonzero(m)
        self.put(ys + y0, xs + x0, col, **k)

    def disk(self, cx, cy, r, col, **k):
        R = int(math.ceil(r)) + 1
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        self.mask(int(round(cx)) - R, int(round(cy)) - R, xx * xx + yy * yy <= r * r + 0.3, col, **k)

    def ellipse(self, cx, cy, rx, ry, col, **k):
        X, Y = int(math.ceil(rx)) + 1, int(math.ceil(ry)) + 1
        yy, xx = np.mgrid[-Y:Y + 1, -X:X + 1]
        m = (xx / max(rx, 0.1)) ** 2 + (yy / max(ry, 0.1)) ** 2 <= 1.0
        self.mask(int(round(cx)) - X, int(round(cy)) - Y, m, col, **k)

    def blit(self, spr, x0, y0, flip=False, em_scale=1.0):
        xs = spr['w'] - 1 - spr['xs'] if flip else spr['xs']
        self.put(spr['ys'] + y0, xs + x0, spr['col'], em=spr['em'] * em_scale)

    def text(self, x, y, s, font, col, **k):
        gw = len(next(iter(font.values()))[0])
        for i, ch in enumerate(s):
            g = font.get(ch)
            if g is not None:
                m = np.array([[c == '#' for c in row] for row in g])
                self.mask(x + i * (gw + 1), y, m, col, **k)


def composite(base, em, L, amb, lt, T, x0=0):
    """Composite layer L (cropped at buffer column x0) onto the screen."""
    if L.w == W and x0 == 0:
        ys, xs = np.nonzero(L.a)
        wx = xs
    else:
        ys, xs = np.nonzero(L.a[:, x0:x0 + W])
        wx = xs + x0
    if not len(ys):
        return
    base[ys, xs] = L.rgb[ys, wx] * amb
    f = np.clip((lt - L.thr[ys, wx]) / 0.08, 0, 1)
    tw = L.twa[ys, wx]
    f = f * (1 - tw * 0.5 * (1 + np.sin(2 * np.pi * (T * 1.1 + L.ph[ys, wx]))))
    em[ys, xs] = L.em[ys, wx] * f[:, None]


def shifted(m, dy, dx):
    out = np.zeros_like(m)
    h, w = m.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        m[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def shade3(m, d=2):
    s = np.where(m, 2, 0)
    s[m & ~shifted(m, -d, -d)] = 1
    s[m & ~shifted(m, d, d)] = 3
    return s


def compile_sprite(rows, pal):
    w = max(len(r) for r in rows)
    ys, xs, col, em = [], [], [], []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in pal:
                c, e = pal[ch]
                ys.append(y), xs.append(x), col.append(c)
                em.append(np.zeros(3) if e is None else e)
    return dict(ys=np.array(ys), xs=np.array(xs), col=np.array(col).reshape(-1, 3),
                em=np.array(em).reshape(-1, 3), w=w, h=len(rows))


def blur(a, r):
    k = 2 * r + 1
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode='edge'), axis=ax)
        a = (np.take(c, np.arange(k, c.shape[ax]), axis=ax) - np.take(c, np.arange(0, c.shape[ax] - k), axis=ax)) / k
    return a


# ---------------------------------------------------------------- fonts
FONT3 = {
    'A': ['.#.', '#.#', '###', '#.#', '#.#'], 'D': ['##.', '#.#', '#.#', '#.#', '##.'],
    'E': ['###', '#..', '##.', '#..', '###'], 'H': ['#.#', '#.#', '###', '#.#', '#.#'],
    'K': ['#.#', '#.#', '##.', '#.#', '#.#'], 'L': ['#..', '#..', '#..', '#..', '###'],
    'M': ['#.#', '###', '###', '#.#', '#.#'], 'N': ['###', '#.#', '#.#', '#.#', '#.#'],
    'O': ['###', '#.#', '#.#', '#.#', '###'], 'P': ['##.', '#.#', '##.', '#..', '#..'],
    'R': ['##.', '#.#', '##.', '#.#', '#.#'], 'T': ['###', '.#.', '.#.', '.#.', '.#.'],
    'W': ['#.#', '#.#', '#.#', '###', '#.#'], ' ': ['...'] * 5,
}
FONT5 = {
    'M': ['#...#', '##.##', '#.#.#', '#.#.#', '#...#', '#...#', '#...#'],
    'O': ['.###.', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
    'D': ['####.', '#...#', '#...#', '#...#', '#...#', '#...#', '####.'],
    'E': ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
    'L': ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
    'T': ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
    'W': ['#...#', '#...#', '#...#', '#.#.#', '#.#.#', '##.##', '#...#'],
    'N': ['#...#', '##..#', '#.#.#', '#..##', '#...#', '#...#', '#...#'],
    'P': ['####.', '#...#', '#...#', '####.', '#....', '#....', '#....'],
    'A': ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
    'R': ['####.', '#...#', '#...#', '####.', '#.#..', '#..#.', '#...#'],
    'K': ['#...#', '#..#.', '#.#..', '##...', '#.#..', '#..#.', '#...#'],
    ' ': ['.....'] * 7,
}

# ---------------------------------------------------------------- palette
GRASS = [C(78, 150, 70), C(70, 140, 64)]
GRASS_BACK = [C(96, 160, 84), C(88, 152, 78)]
CLAY, CLAY_D, CLAY_L, CLAY_WET = C(194, 116, 76), C(168, 98, 64), C(214, 140, 96), C(132, 74, 50)
TILE, TILE_D, KERB = C(196, 186, 170), C(166, 156, 142), C(118, 112, 106)
LEAF = [C(30, 88, 50), C(44, 116, 60), C(80, 154, 76)]
TRUNK = C(98, 70, 50)
IRON = C(36, 72, 50)
LAMP_WARM = C(255, 214, 150)
STONE, STONE_D, STONE_L = C(202, 196, 184), C(160, 154, 144), C(226, 222, 212)
BRONZE, BRONZE_D = C(178, 128, 66), C(120, 82, 42)


def tree(L, cx, by, r, rng, pal=LEAF, trunk_h=None, kind='round'):
    """A park tree: trunk + shaded canopy built from a few overlapping blobs."""
    th = trunk_h if trunk_h is not None else int(r * 0.9)
    L.rect(cx - 1, by - th - 4, cx + 2, by, TRUNK)
    L.px(cx - 1, by - 1, TRUNK * 0.8)
    L.px(cx + 2, by - 1, TRUNK * 0.8)
    R = int(r * 1.9) + 3
    yy, xx = np.mgrid[-2 * R:R // 2, -R:R + 1]
    cy = -th - r * 0.75
    m = np.zeros(yy.shape, bool)
    if kind == 'tall':     # eucalyptus-ish: narrow and tall
        for k in range(5):
            ry = r * (0.55 + 0.1 * rng.random())
            m |= ((xx - rng.uniform(-r * 0.3, r * 0.3)) / (r * 0.55)) ** 2 + \
                 ((yy - (cy - k * r * 0.45)) / ry) ** 2 <= 1
    else:
        m |= xx ** 2 + (yy - cy) ** 2 <= r * r
        for k in range(4):
            ox, oy = rng.uniform(-r * 0.8, r * 0.8), rng.uniform(-r * 0.6, r * 0.2)
            rr = r * rng.uniform(0.5, 0.75)
            m |= (xx - ox) ** 2 + (yy - cy - oy) ** 2 <= rr * rr
    s = shade3(m, 2)
    noise = rng.random(m.shape)
    ox, oy = cx - R, by - 2 * R
    for v, col in ((1, pal[0]), (2, pal[1]), (3, pal[2])):
        L.mask(ox, oy, s == v, col)
    L.mask(ox, oy, m & (noise < 0.08) & (s >= 2), pal[0])
    L.mask(ox, oy, m & (noise > 0.94) & (s == 2), pal[2])


def lamp_post(L, x, by, h=36, small=False, thr=0.2):
    """Black park lamp with a lantern head."""
    if small:
        L.rect(x, by - h, x + 1, by, C(40, 40, 44))
        L.rect(x - 1, by - h - 2, x + 2, by - h, C(240, 226, 190), em=LAMP_WARM, thr=thr)
        return
    L.rect(x - 1, by - 3, x + 2, by, C(30, 34, 32))
    L.rect(x, by - h, x + 1, by - 3, C(36, 40, 38))
    L.rect(x - 2, by - h - 1, x + 3, by - h, C(30, 34, 32))
    L.rect(x - 2, by - h - 6, x + 3, by - h - 1, C(30, 34, 32))
    L.rect(x - 1, by - h - 5, x + 2, by - h - 1, C(236, 224, 196), em=LAMP_WARM, thr=thr)
    L.rect(x - 1, by - h - 7, x + 2, by - h - 6, C(30, 34, 32))
    L.px(x, by - h - 8, C(30, 34, 32))


def bench(L, x, by):
    wood, wood_d, leg = C(150, 98, 56), C(118, 76, 44), C(40, 40, 44)
    L.rect(x, by - 9, x + 16, by - 8, wood)
    L.rect(x, by - 7, x + 16, by - 6, wood_d)
    L.rect(x, by - 4, x + 16, by - 3, wood)
    for lx in (x + 1, x + 14):
        L.rect(lx, by - 9, lx + 1, by, leg)


# ================================================================= WORLD BUILD
def build_far():
    """Distant ring of park trees (parallax)."""
    L = Layer(PW)
    rng = np.random.default_rng(5)
    haze = [C(70, 118, 96), C(84, 136, 104), C(110, 160, 118)]
    m = np.zeros((H, PW), bool)
    yy, xx = np.mgrid[0:H, 0:PW]
    x = -10
    while x < PW + 10:
        r = rng.uniform(10, 20)
        top = rng.uniform(236, 258)
        if rng.random() < 0.18:   # tall eucalyptus crown
            for k in range(4):
                m |= ((xx - x) / (r * 0.5)) ** 2 + ((yy - (top - 6 - k * 9)) / (r * 0.6)) ** 2 <= 1
        m |= (xx - x) ** 2 + (yy - top) ** 2 <= r * r
        x += r * rng.uniform(0.9, 1.4)
    m[262:300] = True
    s = shade3(m, 2)
    for v, col in ((1, haze[0]), (2, haze[1]), (3, haze[2])):
        L.mask(0, 0, s == v, col)
    noise = rng.random((H, PW))
    L.mask(0, 0, m & (noise < 0.06) & (s == 2), haze[0])
    return L


def build_back():
    L = Layer(WW)
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:H, 0:WW]
    # manicured lawn with perspective mowing stripes
    for y in range(296, PATH_Y0):
        band = int(260 / max(y - 286, 1)) % 2
        pal = GRASS_BACK if y < 360 else GRASS
        L.rect(0, y, WW, y + 1, pal[band])
    tuft = rng.random((PATH_Y0 - 296, WW))
    L.mask(0, 296, tuft < 0.025, C(58, 124, 56))
    L.mask(0, 296, tuft > 0.985, C(120, 182, 96))

    # mid-ground trees around the park ring (not over the lake)
    for x in list(range(-10, LAKE_X0 - 20, 46)):
        if 70 <= x <= 200:          # keep the gate clear
            continue
        r = rng.uniform(18, 27)
        tree(L, x + rng.integers(-10, 10), int(rng.integers(306, 324)), r, rng,
             kind='tall' if rng.random() < 0.25 else 'round', trunk_h=int(rng.integers(14, 24)))

    # ---- B: the little hills
    for cx, hgt, hw in ((436, 30, 56), (524, 44, 66), (612, 26, 46)):
        m = (np.abs(xx - cx) < hw) & (yy > TRACK_Y0 - hgt * np.cos(np.clip((xx - cx) / hw, -1, 1) * np.pi / 2) ** 2) \
            & (yy < TRACK_Y0 + 2)
        sub = m[:, cx - hw - 2:cx + hw + 2]
        s = shade3(sub, 3)
        for v, col in ((1, C(66, 132, 62)), (2, C(90, 160, 78)), (3, C(124, 186, 98))):
            L.mask(cx - hw - 2, 0, s == v, col)
        L.mask(cx - hw - 2, 0, sub & (rng.random(sub.shape) < 0.03), C(70, 136, 64))
    tree(L, 540, 336, 9, rng, trunk_h=8)
    for fx in range(470, 600, 7):
        L.px(fx + rng.integers(0, 4), 370 + rng.integers(0, 3), [C(250, 220, 80), C(250, 250, 250)][fx % 2])

    # ---- the clay jogging track
    L.rect(0, TRACK_Y0, TRACK_X1, TRACK_Y1, CLAY)
    L.rect(0, TRACK_Y0, TRACK_X1, TRACK_Y0 + 1, CLAY_D)
    L.rect(0, TRACK_Y1 - 1, TRACK_X1, TRACK_Y1, CLAY_D)
    tex = rng.random((TRACK_Y1 - TRACK_Y0 - 2, TRACK_X1))
    L.mask(0, TRACK_Y0 + 1, tex < 0.06, CLAY_D)
    L.mask(0, TRACK_Y0 + 1, tex > 0.95, CLAY_L)
    for y in range(TRACK_Y0, TRACK_Y1):     # rounded end where it bends away
        e = TRACK_X1 - int(10 * (1 - ((y - TRACK_Y0) / (TRACK_Y1 - TRACK_Y0) - 0.5) ** 2 * 4) ** 0.5)
        L.rect(e, y, TRACK_X1, y + 1, GRASS_BACK[0])

    # ---- C: the sundial in its round flower bed
    sx, sy = SUNDIAL
    flowers = [C(232, 56, 70), C(250, 206, 60), C(236, 110, 196), C(250, 250, 244), C(250, 140, 50)]
    for i, a in enumerate(np.linspace(0, 2 * np.pi, 300, endpoint=False)):
        for rr in (54, 56, 58):
            fx, fy = sx + rr * math.cos(a), sy + rr * 0.19 * math.sin(a)
            L.px(fx, fy, flowers[(i + rr) % 5] if rng.random() < 0.75 else C(46, 118, 56))
    L.ellipse(sx, sy, 51, 9.2, C(186, 176, 158))      # paving ring
    L.ellipse(sx, sy, 48, 8.4, C(206, 198, 182))
    for k in range(0, 360, 30):                       # radiating paving joints
        a = math.radians(k)
        L.line(sx + 32 * math.cos(a), sy + 5.6 * math.sin(a), sx + 47 * math.cos(a), sy + 8.2 * math.sin(a),
               C(186, 176, 158))
    L.ellipse(sx, sy + 1, 30, 6.0, STONE_D)           # steps
    L.ellipse(sx, sy, 30, 5.6, STONE)
    L.ellipse(sx, sy - 3, 22, 4.6, STONE_D)
    L.ellipse(sx, sy - 4, 22, 4.3, STONE_L)
    L.rect(sx - 6, sy - 27, sx + 7, sy - 4, STONE)    # pedestal
    L.rect(sx + 3, sy - 27, sx + 7, sy - 4, STONE_D)
    L.rect(sx - 6, sy - 27, sx - 4, sy - 4, STONE_L)
    for y in (sy - 20, sy - 12):
        L.rect(sx - 6, y, sx + 7, y + 1, STONE_D)
    L.rect(sx - 9, sy - 30, sx + 10, sy - 27, STONE_L)
    L.rect(sx - 9, sy - 28, sx + 10, sy - 27, STONE_D)
    L.ellipse(sx, sy - 32, 18, 4.6, BRONZE_D)         # dial plate
    L.ellipse(sx, sy - 33, 18, 4.3, BRONZE)
    L.ellipse(sx, sy - 33, 14, 3.2, C(196, 146, 80))
    for k in range(13):
        a = -np.pi * k / 12
        L.px(sx + 16 * math.cos(a), sy - 33 + 3.7 * math.sin(a), BRONZE_D)
        L.px(sx + 16 * math.cos(a + np.pi), sy - 33 + 3.7 * math.sin(a + np.pi), BRONZE_D)
    L.line(sx - 2, sy - 33, sx + 10, sy - 46, BRONZE_D)  # gnomon
    L.line(sx - 1, sy - 33, sx + 10, sy - 45, C(214, 164, 90))
    L.line(sx + 10, sy - 46, sx + 10, sy - 33, BRONZE_D)
    L.line(sx, sy - 33, sx + 9, sy - 33, BRONZE_D)

    # ---- C: play area (swing frame + slide; animated parts drawn live)
    sw = SWING_X
    for dx in (0, 24):
        L.line(sw + dx - 5, 393, sw + dx, 368, C(220, 70, 60))
        L.line(sw + dx + 5, 393, sw + dx, 368, C(220, 70, 60))
    L.rect(sw - 1, 367, sw + 26, 369, C(250, 196, 60))
    L.ellipse(sw + 12, 394, 18, 2, C(196, 170, 120))
    # slide
    sl = SLIDE_X
    L.rect(sl, 368, sl + 2, 394, C(80, 140, 220))
    L.rect(sl + 7, 368, sl + 9, 394, C(80, 140, 220))
    for y in range(371, 394, 4):
        L.rect(sl, y, sl + 9, y + 1, C(230, 230, 236))
    L.rect(sl - 1, 366, sl + 10, 369, C(250, 196, 60))
    L.line(sl + 10, 368, sl + 30, 392, C(240, 80, 70))
    L.line(sl + 10, 369, sl + 30, 393, C(200, 56, 50))
    L.line(sl + 11, 368, sl + 31, 392, C(250, 130, 110))

    # benches by the walking track
    for bx in (300, 660, 1030):
        bench(L, bx, PATH_Y0 - 1)

    # ---- D: the boating lake
    water = np.zeros((H, WW), bool)
    for y in range(WY0, WY1):
        edge = LAKE_X0 + int(12 * math.sin((y - WY0) / (WY1 - WY0) * np.pi) ** 0.7) - 10
        water[y, edge:] = True
    # far shore promenade
    L.rect(LAKE_X0 - 30, 296, WW, WY0, GRASS_BACK[0])
    L.rect(LAKE_X0 - 30, 310, WW, 314, C(200, 190, 172))
    L.rect(LAKE_X0 - 30, 314, WW, WY0, C(150, 140, 126))
    for x in range(LAKE_X0 + 6, WW, 30):
        if not (1206 <= x <= 1336):
            lamp_post(L, x, 310, h=12, small=True, thr=0.1 + 0.4 * hash01(x))
    for x in range(LAKE_X0 - 6, WW, 22):
        if not (1200 <= x <= 1340):
            tree(L, x + rng.integers(-5, 6), 298 + int(rng.integers(0, 6)), rng.uniform(9, 14), rng,
                 kind='tall' if rng.random() < 0.3 else 'round', trunk_h=int(rng.integers(6, 11)))
    # gazebo on the far shore
    gx = 1376
    L.rect(gx - 12, 306, gx + 13, 309, C(214, 204, 186))
    for px_ in (gx - 10, gx - 3, gx + 4, gx + 11):
        L.rect(px_, 292, px_ + 2, 306, C(240, 234, 220))
    L.rect(gx - 8, 293, gx + 10, 306, C(70, 46, 34), em=LAMP_WARM * 0.75, thr=0.2)
    for px_ in (gx - 10, gx - 3, gx + 4, gx + 11):
        L.rect(px_, 292, px_ + 2, 306, C(240, 234, 220))
    for i, y in enumerate(range(280, 292)):
        hw = 2 + i * 1.3
        L.rect(gx + 1 - hw, y, gx + 1 + hw, y + 1, C(176, 60, 50) if i % 3 else C(144, 46, 40))
    L.rect(gx - 15, 291, gx + 17, 293, C(240, 234, 220))
    L.line(gx + 1, 279, gx + 1, 275, C(232, 182, 60))
    # the bridge across the lake
    bx0, bx1, deck = 1206, 1336, 300
    spans = [(bx0 + 6, 1250), (1254, 1288), (1292, bx1 - 6)]
    for x in range(bx0, bx1 + 1):
        u = (x - bx0) / (bx1 - bx0)
        top = deck - 8 * math.sin(np.pi * u)
        L.rect(x, top, x + 1, WY0, C(226, 220, 206))
        L.px(x, top, C(244, 240, 230))
        L.px(x, top + 3, C(196, 188, 174))
        L.px(x, top - 6, C(244, 240, 230))            # hand rail
        if x % 5 == 0:
            L.rect(x, top - 6, x + 1, top, C(244, 240, 230))
        if x % 4 == 2:                                 # fairy lights along the rail
            L.px(x, top - 7, C(240, 236, 220), em=[C(255, 210, 120), C(255, 240, 200), C(255, 170, 90)][x % 3],
                 thr=0.3, twa=0.5, ph=hash01(x, 9))
        for a0, a1 in spans:                          # arched openings onto the far water
            if a0 <= x <= a1:
                v = (x - a0) / (a1 - a0)
                rise = (WY0 - top - 6) * math.sin(np.pi * v) ** 0.55
                ub = WY0 - rise
                L.rect(x, ub, x + 1, WY0, C(52, 92, 136))
                L.px(x, ub - 1, C(186, 178, 164))
                if int(ub) + 3 < WY0 and x % 7 == 3:
                    L.px(x, ub + 3, C(80, 124, 166))
    for a0, a1 in spans[:-1]:                         # piers
        L.rect(a1 + 1, WY0 - 3, a1 + 4, WY0, C(196, 188, 174))
    for u in (0.0, 0.5, 1.0):
        x = int(bx0 + u * (bx1 - bx0))
        top = deck - 8 * math.sin(np.pi * u)
        lamp_post(L, x, top - 1, h=12, small=True, thr=0.15)
    # near bank
    L.rect(LAKE_X0 - 20, WY1, WW, PATH_Y0, GRASS[1])
    L.rect(LAKE_X0 - 20, WY1, WW, WY1 + 2, C(150, 140, 126))
    for y in range(WY0, WY1):
        e = int(np.argmax(water[y])) if water[y].any() else WW
        L.rect(e - 2, y, e, y + 1, C(150, 140, 126))
    # jetty for the pedal boats
    L.rect(1146, WY1 - 6, 1186, WY1 - 3, C(150, 104, 62))
    L.rect(1146, WY1 - 3, 1186, WY1 - 2, C(110, 74, 44))
    for x in range(1148, 1186, 9):
        L.rect(x, WY1 - 3, x + 1, WY1 + 1, C(96, 64, 40))
    L.mask(0, 0, water, C(70, 120, 170))
    for bx in (1236, 1338):
        bench(L, bx, 431)
    return L, water


def build_mid():
    """Things in front of the jogging track: Main Gate, boundary railing."""
    L = Layer(WW)
    gx0, gx1 = 74, 198
    # boundary railing on a brick plinth
    for x0, x1 in ((-2, gx0), (gx1, 300)):
        L.rect(x0, 404, x1, 412, C(170, 84, 60))
        L.rect(x0, 404, x1, 405, C(196, 110, 80))
        for x in range(x0, x1, 4):
            L.rect(x, 384, x + 1, 404, IRON)
            L.px(x, 383, C(220, 190, 80))
        L.rect(x0, 388, x1, 389, IRON)
        L.rect(x0, 399, x1, 400, IRON)
        for x in range(x0 + 10, x1 - 4, 30):
            L.rect(x - 1, 380, x + 2, 404, C(170, 84, 60))
            L.rect(x - 2, 379, x + 3, 381, C(220, 210, 190))
    # gate pillars
    PT = 300
    for px_ in (gx0, gx1 - 16):
        L.rect(px_, PT, px_ + 16, 412, C(226, 216, 196))
        L.rect(px_ + 12, PT, px_ + 16, 412, C(196, 186, 168))
        L.rect(px_, PT, px_ + 2, 412, C(240, 234, 222))
        L.rect(px_ + 3, PT + 12, px_ + 13, 402, C(176, 88, 62))
        for y in range(PT + 15, 402, 4):
            L.rect(px_ + 3, y, px_ + 13, y + 1, C(150, 72, 52))
            L.rect(px_ + 3 + (5 if (y // 4) % 2 else 0), y + 1, px_ + 4 + (5 if (y // 4) % 2 else 0), y + 4,
                   C(150, 72, 52))
        L.rect(px_ - 2, 402, px_ + 18, 412, C(214, 204, 186))
        L.rect(px_ - 2, 402, px_ + 18, 403, C(240, 234, 222))
        L.rect(px_ - 2, PT - 3, px_ + 18, PT + 1, C(240, 234, 220))
        L.rect(px_ - 1, PT + 1, px_ + 17, PT + 2, C(196, 186, 168))
        L.rect(px_ + 4, PT - 13, px_ + 12, PT - 3, C(30, 34, 32))
        L.rect(px_ + 5, PT - 12, px_ + 11, PT - 4, C(236, 224, 196), em=LAMP_WARM, thr=0.0)
        L.rect(px_ + 7, PT - 12, px_ + 9, PT - 4, C(30, 34, 32))
        L.rect(px_ + 3, PT - 15, px_ + 13, PT - 13, C(30, 34, 32))
        L.px(px_ + 8, PT - 16, C(30, 34, 32))
        L.ellipse(px_ + 8, 414, 12, 2.4, C(140, 98, 70))
        for i in range(12):
            L.px(px_ - 3 + i * 2, 412 - (i % 2), [C(232, 56, 70), C(250, 206, 60), C(250, 250, 244)][i % 3])
    # name board
    bx0, bx1 = gx0 + 16, gx1 - 16
    L.rect(bx0, 310, bx1, 330, C(28, 88, 60))
    L.rect(bx0, 310, bx1, 312, C(220, 190, 90))
    L.rect(bx0, 328, bx1, 330, C(220, 190, 90))
    L.rect(bx0, 330, bx1, 331, C(20, 60, 40))
    L.text(92, 316, 'MODEL TOWN PARK', FONT5, C(250, 240, 210), em=C(255, 236, 190) * 0.5, thr=0.0)
    # open iron gate leaves, folded back against the pillars
    for x0, d in ((gx0 + 16, 1), (gx1 - 23, -1)):
        for i in range(7):
            x = x0 + i if d > 0 else x0 + 6 - i
            L.rect(x, 366 + i, x + 1, 412, IRON)
            L.px(x, 365 + i, C(220, 190, 80))
        L.rect(x0, 380, x0 + 7, 381, C(220, 190, 80))
        L.rect(x0, 396, x0 + 7, 397, IRON)
    # entrance plaza (replaces the lawn in front of the gate)
    L.rect(-2, 412, 300, PATH_Y0, C(176, 168, 154))
    for y in range(416, PATH_Y0, 6):
        L.rect(-2, y, 300, y + 1, C(156, 148, 136))
    for x in range(0, 300, 10):
        L.rect(x, 412, x + 1, PATH_Y0, C(160, 152, 140))
    # round planters on the gate plaza
    for cx in (34, 240):
        L.ellipse(cx, 446, 16, 4.2, C(200, 192, 178))
        L.ellipse(cx, 445, 14, 3.4, C(110, 76, 50))
        for k in range(22):
            a = k / 22 * 2 * np.pi
            L.rect(cx + 11 * math.cos(a), 444 + 2.5 * math.sin(a) - 1, cx + 11 * math.cos(a) + 2,
                   444 + 2.5 * math.sin(a) + 1, [C(232, 56, 70), C(250, 206, 60), C(236, 110, 196)][k % 3])
        L.ellipse(cx, 440, 5, 5, LEAF[1])
        L.ellipse(cx - 1, 438, 3, 3, LEAF[2])
    # bushes that tuck the jogging track's far end away
    rng = np.random.default_rng(8)
    for x in (TRACK_X1 - 6, TRACK_X1 + 6):
        m = np.zeros((20, 26), bool)
        yy, xx = np.mgrid[0:20, 0:26]
        for k in range(4):
            m |= (xx - rng.uniform(6, 20)) ** 2 + (yy - rng.uniform(8, 14)) ** 2 <= rng.uniform(30, 55)
        m[18:] = False
        s = shade3(m, 2)
        for v, col in ((1, LEAF[0]), (2, LEAF[1]), (3, LEAF[2])):
            L.mask(x - 13, 372, s == v, col)
    return L


def build_front():
    L = Layer(WW)
    rng = np.random.default_rng(13)
    L.rect(0, PATH_Y0, WW, PATH_Y1, TILE)
    for x in range(0, WW, 7):
        L.rect(x, PATH_Y0, x + 1, PATH_Y1, TILE_D)
    L.rect(0, PATH_Y0 + 6, WW, PATH_Y0 + 7, TILE_D)
    L.rect(0, PATH_Y0, WW, PATH_Y0 + 1, KERB)
    L.rect(0, PATH_Y1, WW, PATH_Y1 + 2, KERB)
    for y in range(PATH_Y1 + 2, H):
        L.rect(0, y, WW, y + 1, GRASS[(y // 5) % 2] * 0.92)
    tex = rng.random((H - PATH_Y1 - 2, WW))
    L.mask(0, PATH_Y1 + 2, tex < 0.05, C(52, 112, 52))
    L.mask(0, PATH_Y1 + 2, tex > 0.985, C(250, 236, 120))
    L.mask(0, PATH_Y1 + 2, (tex > 0.975) & (tex <= 0.985), C(250, 250, 250))
    # flower beds along the walk
    for x0 in range(30, WW, 140):
        for x in range(x0, x0 + 46):
            for y in range(PATH_Y1 + 5, PATH_Y1 + 10):
                if rng.random() < 0.6:
                    L.px(x, y, [C(232, 56, 70), C(250, 206, 60), C(236, 110, 196), C(250, 140, 50)][(x // 3 + y) % 4])
                else:
                    L.px(x, y, C(40, 110, 50))
    hedge = [C(30, 84, 44), C(46, 116, 56), C(86, 156, 74)]
    blooms = [C(236, 70, 120), C(250, 250, 244), C(250, 200, 60), C(220, 50, 70), C(244, 140, 200)]
    x = -20
    while x < WW + 30:
        ry = rng.uniform(14, 22)
        leaf_cluster(L, x, H - ry * 0.6, rng.uniform(18, 28), ry, rng, hedge)
        col = blooms[rng.integers(len(blooms))]
        for k in range(int(rng.integers(5, 11))):
            bx, by = x + rng.uniform(-18, 18), H - rng.uniform(4, ry * 1.4)
            L.rect(bx, by, bx + 2, by + 2, col)
            L.px(bx + 1, by, col * 0.75 + 0.25)
        x += rng.uniform(22, 40)
    lamps = [20, 250, 360, 480, 600, 720, 840, 960, 1080, 1170, 1290, 1400]
    for x in lamps:
        lamp_post(L, x, PATH_Y0 + 1, h=46, thr=0.05 + 0.25 * hash01(x, 2))
    return L, lamps


def leaf_cluster(L, cx, cy, rx, ry, rng, pal):
    """A hanging mass of foliage (foreground framing)."""
    X, Y = int(rx) + 4, int(ry) + 10
    yy, xx = np.mgrid[-Y:Y + 1, -X:X + 1]
    m = np.zeros(yy.shape, bool)
    for k in range(9):
        ox, oy = rng.uniform(-rx * 0.8, rx * 0.8), rng.uniform(-ry * 0.6, ry * 0.5)
        r = rng.uniform(0.35, 0.6) * min(rx, ry * 1.6)
        m |= ((xx - ox) / 1.25) ** 2 + (yy - oy) ** 2 <= r * r
    # drooping tips
    for k in range(int(rx // 3)):
        x0 = int(rng.uniform(-rx * 0.9, rx * 0.9))
        col = np.nonzero(m[:, x0 + X])[0]
        if len(col):
            ln = int(rng.integers(2, 8))
            m[col.max():min(col.max() + ln, 2 * Y), x0 + X] = True
    s = shade3(m, 2)
    noise = rng.random(m.shape)
    for v, col in ((1, pal[0]), (2, pal[1]), (3, pal[2])):
        L.mask(int(cx) - X, int(cy) - Y, s == v, col)
    L.mask(int(cx) - X, int(cy) - Y, m & (noise < 0.1), pal[0])
    L.mask(int(cx) - X, int(cy) - Y, m & (noise > 0.93) & (s >= 2), pal[2])


FG_W = int(1140 * FG_PAR) + W + 80
FG_LEAF = [C(20, 58, 36), C(32, 84, 46), C(58, 118, 58)]


def build_fg():
    """Big near trees: canopy framing the top at each stop, trunks that wipe past between stops."""
    L = Layer(FG_W)
    rng = np.random.default_rng(77)
    stops = [s[1] for s in STOPS[::2]]
    for k, c in enumerate(stops):
        o = int(c * FG_PAR)
        top = OFF
        leaf_cluster(L, o + 6, top + 14, 46, 18, rng, FG_LEAF)
        leaf_cluster(L, o + 30, top + 4, 40, 12, rng, FG_LEAF)
        leaf_cluster(L, o + 262, top + 18, 44, 20, rng, FG_LEAF)
        leaf_cluster(L, o + 238, top + 4, 40, 12, rng, FG_LEAF)
    for a, b in zip(stops, stops[1:]):
        o = int((a + b) / 2 * FG_PAR) + W // 2
        bark, bark_d, bark_l = C(70, 50, 38), C(48, 34, 28), C(98, 72, 52)
        for y in range(OFF, H):
            u = (y - OFF) / (H - OFF)
            hw = 9 + int(5 * max(0, u - 0.85) / 0.15 * 1.6)
            L.rect(o - hw, y, o + hw, y + 1, bark)
            L.rect(o - hw, y, o - hw + 3, y + 1, bark_l)
            L.rect(o + hw - 4, y, o + hw, y + 1, bark_d)
            if (y * 7 + o) % 11 < 2:
                L.rect(o - hw + 4, y, o + rng.integers(-3, 4), y + 1, bark_d)
        leaf_cluster(L, o, OFF + 20, 70, 28, rng, FG_LEAF)
        leaf_cluster(L, o - 40, OFF + 6, 50, 16, rng, FG_LEAF)
        leaf_cluster(L, o + 44, OFF + 10, 50, 18, rng, FG_LEAF)
        # bushes at its foot
        leaf_cluster(L, o - 18, H - 8, 30, 10, rng, [C(30, 84, 44), C(46, 116, 56), C(86, 156, 74)])
        leaf_cluster(L, o + 20, H - 6, 26, 9, rng, [C(30, 84, 44), C(46, 116, 56), C(86, 156, 74)])
    return L


def lamp_pools(lamps):
    m = np.zeros((H, WW))
    yy, xx = np.mgrid[0:H, 0:WW]
    for x in lamps:
        d = ((xx - x) / 34.0) ** 2 + ((yy - (PATH_Y0 + 5)) / 14.0) ** 2
        m += np.where(d < 1, 0.16 * (1 - d), 0)
        cone = (yy > PATH_Y0 - 50) & (yy < PATH_Y0) & (np.abs(xx - x) < 2 + (yy - (PATH_Y0 - 50)) * 0.45)
        m += cone * 0.025
    for x in (1236 + 8, 1338 + 8):   # glow onto the near bank
        d = ((xx - x) / 40.0) ** 2 + ((yy - 432) / 14.0) ** 2
        m += np.where(d < 1, 0.08 * (1 - d), 0)
    return np.floor(m * 48) / 48


# ================================================================= PEOPLE
SKINS = [C(214, 168, 128), C(196, 146, 106), C(176, 124, 88), C(150, 104, 72)]
KAMEEZ = [C(244, 242, 234), C(150, 196, 230), C(170, 214, 170), C(222, 204, 170), C(150, 50, 60),
          C(54, 58, 72), C(240, 200, 90), C(120, 90, 160), C(236, 120, 150)]
DUPATTA = [C(240, 110, 160), C(250, 200, 70), C(220, 60, 70), C(150, 110, 210), C(80, 190, 180)]

ADULT = {
    'man': ["..HH.", ".HHSS", ".HSS.", "..S..", ".KKK.", "KKKKK", "KKKKK", "SKKKS", ".KKK.", ".KKK.", ".PPP."],
    'woman': [".DDD.", "DDSSS", "DDSS.", "DDS..", "DKKK.", "DKKKK", "KKKKK", "SKKKS", ".KKK.", ".KKK.", ".KKK."],
    'jogger': [".ccc.", ".HSSc", ".HSS.", "..S..", ".TTT.", "TTTTT", "STTTS", ".TTT.", ".TTT.", ".PPP.", ".PPP."],
}
LEGS = [[".P.P.", "F...F"], ["..P..", "..FF."]]
KID = [".HH.", ".SS.", "KKKK", "SKKS", ".KK.", ".PP."]
KID_LEGS = [["P..P", "F..F"], [".PP.", ".FF."]]
SMALL = [".H.", ".S.", "KKK", "KKK", "SKS", ".P."]
SMALL_LEGS = [["P.P", "F.F"], [".P.", ".F."]]
SIT = [".HH..", ".HSS.", "..S..", ".KKK.", "KKKKK", "SKKKS", "PPPPP", "FPPP."]
SIT_W = [".DD..", "DDSS.", "DDS..", "DKKK.", "KKKKK", "SKKKS", "PPPPP", "FPPP."]


def palette(rng, kind):
    sk = SKINS[rng.integers(len(SKINS))]
    k = KAMEEZ[rng.integers(len(KAMEEZ))]
    pal = {'S': (sk, None), 'K': (k, None), 'F': (C(60, 44, 36), None),
           'H': ((C(170, 170, 170) if rng.random() < 0.25 else C(30, 26, 24)), None),
           'P': ((k if rng.random() < 0.6 else C(240, 238, 230)) * 0.92, None),
           'D': (DUPATTA[rng.integers(len(DUPATTA))], None),
           'T': ([C(40, 60, 120), C(160, 30, 40), C(40, 40, 46), C(30, 110, 90)][rng.integers(4)], None),
           'c': (C(240, 240, 240), None)}
    if kind == 'jogger':
        pal['P'] = (C(40, 40, 50), None)
        pal['F'] = (C(240, 240, 240), None)
    return pal


def person(rng, kind):
    pal = palette(rng, kind)
    if kind in ('kid', 'kidg'):
        base = list(KID)
        if kind == 'kidg':
            base[0] = ".DD."
        return [compile_sprite(base + l, pal) for l in KID_LEGS]
    if kind == 'small':
        return [compile_sprite(SMALL + l, pal) for l in SMALL_LEGS]
    if kind in ('sit', 'sitw'):
        return [compile_sprite(SIT if kind == 'sit' else SIT_W, pal)]
    return [compile_sprite(ADULT[kind] + l, pal) for l in LEGS]


def exercise_frames(rng):
    pal = palette(rng, 'man')
    down = ADULT['man'] + LEGS[0][:0] + [".P.P.", "F...F"]
    up = ["S.H.S", "SHHSS", "SHSS.", ".S.S.", "..S..", ".KKK.", ".KKK.", ".KKK.", ".KKK.", ".KKK.", ".PPP.",
          ".P.P.", "F...F"]
    return [compile_sprite(down, pal), compile_sprite(up, pal)]


# ================================================================= SCENE
class Scene:
    def __init__(self):
        self.far = build_far()
        self.back, self.water = build_back()
        self.mid = build_mid()
        self.front, lamps = build_front()
        self.pools = lamp_pools(lamps)
        self.fg = build_fg()
        rng = np.random.default_rng(42)
        s = np.clip((YY - OFF) / (270 - OFF), 0, 1) ** 0.9
        self.NQ = 18
        self.skyq = np.clip(np.floor(s * self.NQ + BAY), 0, self.NQ).astype(int)
        self.clouds = self._clouds()
        self.stars = [(int(rng.integers(0, W)), int(rng.integers(OFF, 240)), rng.uniform(0.25, 1.0),
                       rng.uniform(0, 6.28), rng.uniform(1.2, 3.5)) for _ in range(130)]
        # walkers on the tiled walking track, grouped around each stop
        self.walkers = []
        kinds_by_seg = [('man', 'jogger', 'woman', 'man', 'jogger'), ('man', 'woman', 'jogger', 'woman'),
                        ('woman', 'man', 'kid', 'woman', 'kidg'), ('man', 'woman', 'woman', 'man', 'kid')]
        for seg, kinds in enumerate(kinds_by_seg):
            cam = (0, 380, 760, 1140)[seg]
            for i, k in enumerate(kinds):
                d = 1 if rng.random() < 0.5 else -1
                v = {'jogger': 26, 'kid': 20, 'kidg': 18}.get(k, 11) * rng.uniform(0.85, 1.15)
                self.walkers.append(dict(fr=person(rng, k), seg=seg, x0=cam - 30 + rng.uniform(0, 330),
                                         span=330, cam=cam, v=v * d, y=PATH_Y0 + 9 + int(rng.integers(0, 3)),
                                         ph=rng.uniform(0, 2)))
        # pair of strollers for the night by the lake (walk together)
        self.couple = [person(rng, 'man'), person(rng, 'woman')]
        self.joggers = [dict(fr=person(rng, 'small'), x0=rng.uniform(0, TRACK_X1), v=rng.uniform(16, 26) * (
            1 if i % 2 else -1), y=TRACK_Y0 + 9 + (i % 3)) for i in range(14)]
        self.exercise = [exercise_frames(rng) for _ in range(5)]
        self.picnic = [person(rng, 'sit')[0], person(rng, 'sitw')[0], person(rng, 'sitw')[0], person(rng, 'sit')[0]]
        self.picnic2 = [person(rng, 'sit')[0], person(rng, 'sitw')[0]]
        self.kids = [person(rng, 'kid') for _ in range(6)] + [person(rng, 'kidg') for _ in range(3)]
        self.hill_folk = [person(rng, 'sit')[0], person(rng, 'sitw')[0]]
        self.bench_folk = [person(rng, 'sit')[0], person(rng, 'sitw')[0], person(rng, 'sit')[0]]
        self.boaters = [person(rng, 'small') for _ in range(6)]

    def _clouds(self):
        rng = np.random.default_rng(21)
        out = []
        for (w, h, y, v) in [(54, 15, 44, 1.6), (40, 12, 104, 2.6), (66, 17, 150, 1.2), (34, 10, 20, 3.0),
                             (46, 13, 196, 2.0)]:
            m = np.zeros((h + 2, w + 2), bool)
            yy, xx = np.mgrid[0:h + 2, 0:w + 2]
            n = int(rng.integers(4, 7))
            for j in range(n):
                cx = (j + 0.5) / n * w + rng.uniform(-3, 3)
                r = rng.uniform(0.35, 0.62) * h * (1.25 if 0 < j < n - 1 else 0.85)
                m |= (xx - cx) ** 2 + (yy - (h - r * 0.55)) ** 2 <= r * r
            m[h:, :] = False
            s = shade3(m, 2)
            s[m & (yy >= h - 3)] = 1
            out.append(dict(s=s, x0=rng.uniform(0, W), y=y + OFF + 10, v=v, w=w + 2))
        return out

    # ------------------------------------------------------------ sky
    def sky(self, base, em, T, cam, amb, nf):
        top, mid, hor = [np.array(c) / 255 for c in kf(T, SKY_KEYS)]
        lut = np.zeros((self.NQ + 1, 3))
        for q in range(self.NQ + 1):
            s = q / self.NQ
            lut[q] = top + (mid - top) * (s / 0.5) if s < 0.5 else mid + (hor - mid) * ((s - 0.5) / 0.5)
        base[:] = lut[self.skyq]
        em[:] = 0
        if nf > 0:
            for x, y, b, ph, w in self.stars:
                v = b * nf * (0.6 + 0.4 * math.sin(T * w + ph))
                em[y, x] += np.array([0.85, 0.9, 1.0]) * v
        # sun travels east -> west over the day
        s = float(np.clip(T / 33.0, 0, 1))
        sx, sy = 26 + 220 * s, 256 - 128 * math.sin(np.pi * s)
        if sy < 262:
            sc = kf(T, [(0, (255, 214, 170)), (5, (255, 248, 222)), (24, (255, 246, 212)), (29, (255, 214, 130)),
                        (32, (255, 150, 70))]) / 255
            m = (XX - sx) ** 2 + (YY - sy) ** 2 <= 81.5
            base[m] = sc
            em[m] = sc * 0.35
        self.sun = (sx, sy, s)
        # clouds
        hl, md, sh = [np.array(c) / 255 for c in kf(T, CLOUD_KEYS)]
        for cl in self.clouds:
            x = int(round((cl['x0'] + cl['v'] * T - cam * 0.12) % (W + cl['w'] + 20))) - cl['w'] - 10
            ys, xs = np.nonzero(cl['s'])
            vals = cl['s'][ys, xs]
            ys, xs = ys + cl['y'], xs + x
            ok = (xs >= 0) & (xs < W)
            base[ys[ok], xs[ok]] = np.array([sh, md, hl])[vals[ok] - 1]
            em[ys[ok], xs[ok]] = 0
        # moon (full, rises at dusk)
        ma = float(np.clip((T - 33.5) / 3, 0, 1))
        if ma > 0:
            mx, my = 62, int(round(178 - 30 * ma))
            m = (XX - mx) ** 2 + (YY - my) ** 2 <= 56
            keep = m & (BAY < ma)
            base[keep] = C(248, 244, 222)
            crater = keep & (((XX - mx + 2) ** 2 + (YY - my + 1) ** 2 <= 4) | ((XX - mx - 3) ** 2 + (YY - my - 2) ** 2 <= 2))
            base[crater] = C(222, 216, 196)
            em[keep] += C(255, 246, 214) * 0.3
            self.moon = (mx, my, ma)
        else:
            self.moon = None
        # green parakeets in the morning, a flock of birds at sunset
        for (t0, t1, n, y0, col, d) in ((1.0, 9.0, 5, 200, C(70, 190, 80), 1), (28.5, 35.0, 7, 190, C(40, 30, 50), -1)):
            if t0 < T < t1:
                for i in range(n):
                    u = (T - t0) / (t1 - t0)
                    bx = (-20 + u * (W + 40)) if d > 0 else (W + 20 - u * (W + 40))
                    bx += abs(i - n // 2) * 7 * -d + i * 2
                    by = y0 + abs(i - n // 2) * 4 + 6 * math.sin(T * 2 + i)
                    flap = (int(T * 8) + i) % 2
                    pts = ((-1, 0), (0, 1), (1, 0)) if flap else ((-1, 1), (0, 1), (1, 1), (-2, 0), (2, 0))
                    for dx, dy in pts + ((-2 * d, 1),):
                        x, y = int(round(bx + dx)), int(round(by + dy))
                        if 0 <= x < W and 0 <= y < H:
                            base[y, x] = col * amb

    # ------------------------------------------------------------ actors
    def back_actors(self, L, T, cam, lt, shadows):
        # joggers on the clay track
        for j in self.joggers:
            x = (j['x0'] + j['v'] * T) % (TRACK_X1 + 40) - 20
            if x > TRACK_X1 - 14:
                continue
            fr = j['fr'][int(T * 6 + j['x0']) % 2]
            L.blit(fr, int(round(x)), j['y'] - fr['h'], flip=j['v'] < 0)
        # A: an early walker stretching by the gate
        if cam < 300:
            st = self.exercise[0][int(T * 1.2) % 2]
            L.blit(st, 46, 437 - st['h'])
            shadows.append((48, 437, 4))
        # B: the water bowser watering the clay track
        tx = self.tanker_x(T)
        if -60 < tx - cam < W + 60:
            self.draw_tanker(L, tx, T)
        # B: morning exercise circle
        for i, fr in enumerate(self.exercise):
            x = 452 + i * 13
            y = 424 + (i % 2) * 4
            f = fr[int(T * 1.6 + (0 if i else 0.5)) % 2] if i else fr[int(T * 1.6) % 2]
            L.blit(f, x, y - f['h'], flip=i % 2 == 1)
            shadows.append((x + 2, y, 4))
        # B: kids on the little hills, family sitting on top
        L.blit(self.hill_folk[0], 516, 334 - 8)
        L.blit(self.hill_folk[1], 523, 334 - 8, flip=True)
        for i, (cx, hw, hgt) in enumerate(((436, 56, 30), (524, 66, 44), (612, 46, 26))):
            u = (T * (0.18 + 0.05 * i) + i * 0.3) % 1.0
            ux = (u * 2 - 1) * 0.85
            x = cx + ux * hw
            y = TRACK_Y0 - hgt * math.cos(ux * np.pi / 2) ** 2
            fr = self.kids[i][int(T * 7) % 2]
            L.blit(fr, int(round(x)) - 2, int(round(y)) - fr['h'], flip=False)
        # C: swing (pendulum) and slide
        th = 0.7 * math.sin(2 * np.pi * T / 2.2)
        for k, (sx, kid) in enumerate(((SWING_X, self.kids[3]), (SWING_X + 24, self.kids[6]))):
            thk = th if k == 0 else -0.5 * math.sin(2 * np.pi * T / 2.6 + 1)
            ex, ey = sx + 20 * math.sin(thk), 368 + 20 * math.cos(thk)
            L.line(sx, 368, ex, ey, C(150, 150, 160))
            L.rect(ex - 2, ey, ex + 3, ey + 1, C(80, 60, 40))
            f = kid[0]
            L.blit(f, int(round(ex)) - 2, int(round(ey)) - f['h'] + 3)
        c = T % 4.0
        kid = self.kids[7]
        if c < 2.0:
            x, y = SLIDE_X + 4, 393 - (c / 2.0) * 25
        elif c < 2.7:
            u = (c - 2.0) / 0.7
            x, y = SLIDE_X + 10 + 20 * u * u, 368 + 24 * u * u
        else:
            u = (c - 2.7) / 1.3
            x, y = SLIDE_X + 30 - 24 * u, 393
        f = kid[int(T * 6) % 2]
        L.blit(f, int(round(x)) - 2, int(round(y)) - f['h'] + (2 if 2.0 <= c < 2.7 else 0), flip=c >= 2.7)
        # C: sundial shadow, opposite the sun
        sx, sy, s = self.sun
        if s < 0.97 and T < 33:
            phi = -np.pi * s
            dx_, dy_ = SUNDIAL
            L.line(dx_, dy_ - 33, dx_ + 15 * math.cos(phi), dy_ - 33 + 3.4 * math.sin(phi), BRONZE_D * 0.6)
            L.line(dx_, dy_ - 32, dx_ + 14 * math.cos(phi), dy_ - 32 + 3.2 * math.sin(phi), BRONZE_D * 0.75)
            gl = 18 + 30 * (1 - math.sin(np.pi * s))
            for k in range(int(gl)):
                u = k / gl
                gx = dx_ + (k + 10) * math.cos(phi)
                gy = dy_ + 4 * math.sin(phi) * u + 1
                shadows.append((gx, gy, max(5 - u * 3, 1)))
        # C: picnics
        L.rect(784, 426, 826, 431, C(190, 40, 50))
        for x in range(786, 826, 4):
            L.rect(x, 427, x + 2, 428, C(240, 200, 80))
            L.px(x + 1, 429, C(40, 90, 150))
        L.rect(800, 422, 805, 427, C(196, 110, 50))     # handi
        L.rect(800, 421, 805, 422, C(150, 80, 40))
        L.rect(808, 420, 810, 427, C(230, 230, 236))   # thermos
        L.rect(811, 425, 815, 426, C(244, 244, 240))
        for i, p in enumerate(self.picnic):
            L.blit(p, 786 + i * 10 - (4 if i > 1 else 0) + (6 if i > 1 else 0), 428 - p['h'], flip=i >= 2)
        L.rect(986, 432, 1014, 436, C(60, 120, 190))
        for x in range(988, 1014, 5):
            L.rect(x, 433, x + 2, 434, C(250, 250, 250))
        for i, p in enumerate(self.picnic2):
            L.blit(p, 990 + i * 12, 434 - p['h'], flip=i == 1)
        # kid running circles round the picnic
        a = T * 1.6
        x, y = 806 + 30 * math.cos(a), 432 + 5 * math.sin(a)
        f = self.kids[4][int(T * 8) % 2]
        L.blit(f, int(round(x)), int(round(y)) - f['h'], flip=math.sin(a) > 0)
        # C: tape-ball cricket
        c = T % 3.6
        bx0, bat = 966, 1020
        L.rect(bat + 6, 404, bat + 7, 414, C(240, 230, 200))
        L.rect(bat + 8, 404, bat + 9, 414, C(240, 230, 200))
        L.rect(bat + 6, 403, bat + 9, 404, C(240, 230, 200))
        bowler = self.kids[0][int(T * 8) % 2 if c < 1.0 else 0]
        bxp = bx0 + min(c, 1.0) * 14
        L.blit(bowler, int(bxp), 414 - bowler['h'])
        bk = self.kids[1][0]
        L.blit(bk, bat, 414 - bk['h'], flip=True)
        swing = 1.4 < c < 1.7
        L.line(bat, 409, bat - (4 if swing else 1), 409 - (2 if swing else 5), C(220, 190, 130))
        if 1.0 <= c < 1.5:
            u = (c - 1.0) / 0.5
            ball = (bxp + 4 + (bat - bxp - 4) * u, 404 + 8 * abs(math.sin(u * np.pi * 1.2)))
        elif 1.5 <= c < 3.2:
            u = (c - 1.5) / 1.7
            ball = (bat - u * 70, 406 - 40 * math.sin(np.pi * u))
        else:
            ball = None
        if ball:
            L.px(ball[0], ball[1], C(250, 230, 60))
        fielder = self.kids[2][int(T * 8) % 2]
        fx = 948 + 10 * math.sin(T * 0.9)
        L.blit(fielder, int(fx), 418 - fielder['h'], flip=True)
        for x in (976, 1022, 950):
            shadows.append((x, 414, 3))
        # pigeons pecking near the picnic
        for i in range(5):
            px_ = 846 + i * 6 + int(2 * math.sin(T * 0.7 + i))
            peck = (int(T * 3 + i * 0.7) % 3) == 0
            L.rect(px_, 433, px_ + 3, 435, C(130, 136, 150))
            L.px(px_ + 3, 433 + (1 if peck else 0), C(96, 100, 120))
            L.px(px_ + 3, 432 + (1 if peck else 0), C(96, 100, 120))
        # ice-cream cart
        cx = 878
        L.rect(cx, 422, cx + 14, 432, C(244, 244, 244))
        L.rect(cx, 425, cx + 14, 427, C(220, 50, 60))
        L.ellipse(cx + 3, 433, 2, 2, C(40, 40, 44))
        L.ellipse(cx + 11, 433, 2, 2, C(40, 40, 44))
        L.line(cx + 7, 422, cx + 7, 410, C(120, 120, 120))
        for i in range(-7, 8):
            L.rect(cx + 7 + i, 409 + abs(i) // 3, cx + 8 + i, 411 + abs(i) // 3,
                   [C(250, 220, 70), C(220, 60, 80)][(i + 7) // 3 % 2])
        vend = self.walkers[0]['fr'][0]
        L.blit(vend, cx + 15, 434 - vend['h'])
        # D: folks on the lakeside benches
        L.blit(self.bench_folk[0], 1238, 429 - 8)
        L.blit(self.bench_folk[1], 1245, 429 - 8)
        L.blit(self.bench_folk[2], 1343, 429 - 8)
        L.rect(1348, 420, 1352, 424, C(236, 232, 214))    # newspaper

    def tanker_x(self, T):
        return 700 - (T - 4.0) * 30

    def draw_tanker(self, L, x, T):
        y = TRACK_Y1 + 1
        # tractor (facing left)
        L.rect(x, y - 10, x + 14, y - 4, C(200, 40, 40))
        L.rect(x + 3, y - 12, x + 4, y - 18, C(50, 50, 54))
        L.rect(x + 8, y - 16, x + 13, y - 10, C(60, 60, 64))
        L.rect(x + 9, y - 15, x + 12, y - 11, C(160, 200, 230))
        L.disk(x + 11, y - 3, 3.8, C(30, 30, 32))
        L.disk(x + 11, y - 3, 1.5, C(200, 160, 60))
        L.disk(x + 2, y - 1, 2, C(30, 30, 32))
        L.px(x + 10, y - 13, C(200, 150, 110))
        # water tank trolley behind it
        tx = x + 20
        L.rect(x + 14, y - 7, tx, y - 6, C(60, 60, 64))
        L.rect(tx, y - 13, tx + 24, y - 4, C(70, 110, 150))
        L.rect(tx, y - 13, tx + 24, y - 12, C(120, 160, 200))
        L.rect(tx, y - 5, tx + 24, y - 4, C(50, 80, 110))
        L.rect(tx + 18, y - 15, tx + 22, y - 13, C(50, 80, 110))
        L.disk(tx + 12, y - 2, 2.6, C(30, 30, 32))
        L.px(tx + 12, y - 2, C(150, 150, 150))
        L.rect(tx + 24, y - 6, tx + 26, y - 5, C(60, 60, 64))
        # spray fan from the rear sprinkler
        for k in range(30):
            u = hash01(k, int(T * 20))
            a = np.pi * (-0.45 + 0.9 * u)
            r = 2 + ((T * 5 + k * 0.37) % 1.0) * 10
            L.px(tx + 26 + r * math.cos(a), y - 5 - r * math.sin(a) * 0.7 + r * 0.4, C(200, 230, 250))

    def front_actors(self, L, T, cam, lt, shadows):
        nightish = T > 36
        for w in self.walkers:
            if abs(w['cam'] - cam) > 400:
                continue
            x = w['cam'] - 30 + (w['x0'] - w['cam'] + 30 + w['v'] * T) % w['span']
            fr = w['fr'][int(T * 4 + w['ph']) % 2]
            if nightish and w['seg'] == 3 and w['fr'] is self.walkers[-1]['fr']:
                continue
            L.blit(fr, int(round(x)), w['y'] - fr['h'], flip=w['v'] < 0)
            shadows.append((x + 2, w['y'], 4))
        # butterflies over the hedge by day
        if T < 30:
            for i in range(10):
                wx = cam + (hash01(i, 5) * 300 + T * (6 + 4 * hash01(i, 6))) % 300 - 15
                y = H - 34 - 14 * hash01(i, 8) + 5 * math.sin(T * 2.3 + i)
                col = [C(250, 220, 70), C(250, 250, 250), C(240, 140, 60), C(150, 190, 250)][i % 4]
                if int(T * 9 + i) % 2:
                    L.px(wx - 1, y, col)
                    L.px(wx + 1, y, col)
                else:
                    L.px(wx - 1, y - 1, col)
                    L.px(wx + 1, y - 1, col)
                L.px(wx, y, C(40, 30, 30))
        # an evening stroll: a couple walking together by the lake
        if T > 30:
            x = 1440 - (T - 30) * 9
            for k, fr in enumerate(self.couple):
                f = fr[int(T * 3.5 + k) % 2]
                L.blit(f, int(round(x + k * 6)), PATH_Y0 + 10 - f['h'], flip=True)
                shadows.append((x + k * 6 + 2, PATH_Y0 + 10, 4))

    # ------------------------------------------------------------ water
    def water_pass(self, base, em, T, cam, lt, nf):
        wm = self.water[:, cam:cam + W]
        if not wm.any():
            return
        top_col = kf(T, SKY_KEYS)[1] / 255
        deep = np.array([0.16, 0.30, 0.42]) * (0.4 + 0.6 * (1 - nf)) + np.array([0.02, 0.03, 0.08]) * nf
        rows = np.arange(WY0, WY1)
        d = rows - WY0
        for i, y in enumerate(rows):
            m = wm[y]
            if not m.any():
                continue
            src = WY0 - 1 - d[i]
            amp = 0.4 + d[i] * 0.045
            dx = int(round(amp * math.sin(y * 0.85 + T * 2.2) + 0.6 * math.sin(y * 2.3 - T * 3.1)))
            cols = np.clip(np.arange(W) + dx, 0, W - 1)
            refl = base[src, cols]
            k = 0.5 + 0.18 * (d[i] / (WY1 - WY0))
            colr = refl * (1 - k) + deep * k
            # pixel-art ripple dashes
            rip = ((np.arange(W) + cam + int(T * 6) * (1 if y % 2 else -1) + y * 7) % 23) < 3
            rip &= (y % 3 == 0)
            colr = np.where(rip[:, None], colr * 0.88 + top_col * 0.12, colr)
            base[y, m] = colr[m]
            e = np.maximum(em[src, cols], np.maximum(em[min(src + 1, WY0 - 1), cols], em[src - 1, cols]) * 0.7) * 0.8
            brk = 0.55 + 0.45 * np.sign(np.sin((np.arange(W) + cam) * 0.9 + y * 1.7 + T * 4))
            em[y, m] = (e * brk[:, None])[m]
        # moonlight glade
        if self.moon is not None:
            mx, my, ma = self.moon
            for y in rows:
                if y % 2:
                    continue
                hw = 1 + (y - WY0) * 0.07
                c = mx + 2.5 * math.sin(y * 0.7 + T * 1.5)
                ln = 1 + int(hash01(y, int(T * 4)) * 3)
                x0 = int(c - hw * hash01(y, 7, int(T * 3)))
                xs = np.arange(x0, x0 + ln)
                xs = xs[(xs >= 0) & (xs < W)]
                xs = xs[wm[y, xs]]
                em[y, xs] += C(255, 246, 214) * 0.28 * ma

    def water_actors(self, L, R, T, cam, lt, nf):
        """Boats and birds on the lake. R receives their mirrored reflections."""
        def put_with_reflection(draw, x, y):
            draw(L, x, y, False)
            draw(R, x, y, True)

        night = float(np.clip((T - 36) / 3, 0, 1))
        boats = [
            (1200, 352, 26, 0.10, 0, C(60, 120, 220), C(250, 200, 50)),
            (1300, 372, 34, 0.07, 1.5, C(230, 60, 60), C(250, 250, 250)),
            (1260, 392, 22, 0.12, 3.0, C(250, 190, 40), C(60, 160, 120)),
        ]
        for i, (cx, cy, rx, w, ph, hull, roof) in enumerate(boats):
            moving = 1 - night if i else 1 - night * 0.6
            a = ph + T * w * 2 * np.pi * 0.25 * (0.3 + 0.7 * moving)
            x = cx + rx * math.cos(a)
            y = cy + 4 * math.sin(a)
            if night > 0 and i:
                dock = (1150 + i * 16, WY1 - 4)
                x = x * (1 - night) + dock[0] * night
                y = y * (1 - night) + dock[1] * night
            y += 0.6 * math.sin(T * 2 + i)
            facing = -math.sin(a) < 0 if night < 1 or i == 0 else False
            boaters = self.boaters[2 * i:2 * i + 2] if (i == 0 or night < 0.9) else []

            def draw(Lr, x, y, mirror, hull=hull, roof=roof, facing=facing, boaters=boaters, i=i):
                xi, yi = int(round(x)), int(round(y))
                sgn = -1 if mirror else 1
                def row(r, x0, x1, col, **k):
                    yy = yi + sgn * r if not mirror else yi + 1 - r
                    Lr.rect(xi + x0, yy, xi + x1, yy + 1, col, **k)
                for r in range(0, 4):
                    row(-r, -7 + (1 if r == 0 else 0), 8 - (1 if r == 0 else 0), hull * (0.75 if r == 0 else 1.0))
                row(-4, -6, 7, C(240, 240, 236))
                for k, b in enumerate(boaters):
                    f = b[0]
                    for rr in range(3):
                        c = f['col'][min(rr, len(f['col']) - 1)] if False else None
                    row(-5, -3 + k * 4, -2 + k * 4, C(200, 150, 110))
                    row(-6, -3 + k * 4, -2 + k * 4, C(30, 26, 24))
                for r in (7, 8):
                    row(-r, -6, 7, roof)
                for xx in (-6, 6):
                    for r in (5, 6):
                        row(-r, xx, xx + 1, C(200, 200, 200))
                if i == 0:      # a little lantern on the evening boat
                    row(-6, 7, 8, C(250, 230, 180), em=LAMP_WARM * lt * 0.9)
                # wake
                if not mirror:
                    for k in range(1, 6):
                        Lr.px(xi + (-8 - k * 2 if facing else 8 + k * 2), yi + 1 + (k % 2), C(220, 236, 246))
            put_with_reflection(draw, x, y)

        swans = [(1170, 340, 0.06, 0), (1188, 346, 0.06, 0.4), (1330, 388, 0.05, 2), (1390, 356, 0.05, 4)]
        for i, (cx, cy, w, ph) in enumerate(swans):
            a = ph + T * w * 2 * np.pi * (1 - 0.7 * night)
            x = cx + 18 * math.cos(a)
            y = cy + 2 * math.sin(a)
            right = math.sin(a) < 0

            def draw(Lr, x, y, mirror, right=right, i=i):
                xi, yi = int(round(x)), int(round(y))
                wht = C(250, 250, 246) if i < 3 else C(250, 250, 246)
                pts = [(-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-3, -1), (-2, -1), (-1, -1), (0, -1),
                       (1, -1), (-4, -2), (-3, -2), (-2, -2), (1, -2), (2, -3), (2, -4), (2, -5), (3, -5)]
                for dx, dy in pts:
                    dx = dx if right else -dx
                    yy = yi + dy if not mirror else yi + 1 - dy
                    col = C(250, 160, 60) if (dx, dy) in ((3, -5), (-3, -5)) else (wht if dy > -2 or abs(dx) == 2 else wht * 0.92)
                    Lr.px(xi + dx, yy, col)
                if not mirror:
                    Lr.px(xi + (-5 if right else 5), yi + 1, C(220, 236, 246))
                    Lr.px(xi + (-7 if right else 7), yi + 1, C(200, 224, 240))
            put_with_reflection(draw, x, y)
        # ducks
        for i in range(5):
            a = T * 0.35 + i * 1.3
            x = 1220 + 60 * math.cos(a * 0.5 + i) + i * 30
            y = 362 + 14 * math.sin(a * 0.3 + i * 2)
            right = math.sin(a * 0.5 + i) < 0

            def draw(Lr, x, y, mirror, right=right, i=i):
                xi, yi = int(round(x)), int(round(y))
                body = C(150, 110, 80) if i % 2 else C(240, 236, 226)
                for dx, dy, col in ((-2, 0, body), (-1, 0, body), (0, 0, body), (1, 0, body), (-1, -1, body),
                                    (0, -1, body), (1, -1, C(40, 110, 70) if i % 2 else body),
                                    (1, -2, C(40, 110, 70) if i % 2 else body), (2, -2, C(240, 170, 50))):
                    dx = dx if right else -dx
                    Lr.px(xi + dx, yi + dy if not mirror else yi + 1 - dy, col)
            put_with_reflection(draw, x, y)

    # ------------------------------------------------------------ frame
    def render(self, T):
        amb = kf(T, AMB_KEYS)
        lt = lights_at(T)
        nf = night_at(T)
        cam = cam_at(T)
        base = np.zeros((H, W, 3))
        em = np.zeros((H, W, 3))
        self.sky(base, em, T, cam, amb, nf)
        composite(base, em, self.far, amb, lt, T, x0=int(cam * PAR))
        composite(base, em, self.back, amb, lt, T, x0=cam)

        # the water bowser leaves the clay track dark and wet, drying slowly
        tx = self.tanker_x(T) + 46
        cols = np.arange(W) + cam
        tpass = 4.0 + (700 + 46 - cols) / 30.0
        wet = np.where((cols > tx) & (cols < TRACK_X1), np.clip(1 - (T - tpass) / 16.0, 0, 1), 0)
        if wet.any():
            ys = slice(TRACK_Y0 + 1, TRACK_Y1 - 1)
            q = np.floor(wet * 3) / 3
            seg = base[ys]
            base[ys] = seg * (1 - 0.35 * q[None, :, None]) + CLAY_WET * amb * 0.35 * q[None, :, None]
            glint = (((cols + np.arange(TRACK_Y0 + 1, TRACK_Y1 - 1)[:, None] * 5) % 17) == 0) & (q[None, :] > 0.6)
            base[ys][glint] = base[ys][glint] * 0.6 + 0.4 * np.array([0.7, 0.8, 0.9]) * amb

        shadows = []
        backL = Layer(W, ox=cam)
        self.back_actors(backL, T, cam, lt, shadows)
        self._shadows(base, shadows, cam)
        composite(base, em, backL, amb, lt, T)
        composite(base, em, self.mid, amb, lt, T, x0=cam)

        self.water_pass(base, em, T, cam, lt, nf)
        wL, wR = Layer(W, ox=cam), Layer(W, ox=cam)
        self.water_actors(wL, wR, T, cam, lt, nf)
        # reflections: mirrored sprites blended into the water
        ys, xs = np.nonzero(wR.a & self.water[:, cam:cam + W])
        if len(ys):
            ripple = np.rint(0.8 * np.sin(ys * 1.3 + T * 3)).astype(int)
            xs2 = np.clip(xs + ripple, 0, W - 1)
            base[ys, xs2] = base[ys, xs2] * 0.55 + wR.rgb[ys, xs] * amb * 0.45
            em[ys, xs2] += wR.em[ys, xs] * 0.5
        composite(base, em, wL, amb, lt, T)

        composite(base, em, self.front, amb, lt, T, x0=cam)
        shadows = []
        frontL = Layer(W, ox=cam)
        self.front_actors(frontL, T, cam, lt, shadows)
        self._shadows(base, shadows, cam)
        composite(base, em, frontL, amb, lt, T)
        composite(base, em, self.fg, amb, lt, T, x0=int(cam * FG_PAR))

        em += self.pools[:, cam:cam + W, None] * LAMP_WARM * lt
        self.fireflies(em, T, cam, nf)
        self.mist(base, T)
        self.title(base, em, T)

        g = blur(blur(em, 2), 2) * 0.9 + blur(blur(em, 7), 7) * 0.45
        g = np.floor(g * 24) / 24
        out = np.clip(base + em + g, 0, 1)
        fade = min(1.0, T / 0.8, (DUR - T) / 1.4)
        out *= np.floor(max(fade, 0) * 16) / 16
        return (out[OFF:OFF + SH] * 255 + 0.5).astype(np.uint8)

    def _shadows(self, base, shadows, cam):
        for (x, y, r) in shadows:
            xi, yi = int(round(x)) - cam, int(round(y))
            r = int(round(r))
            x0, x1 = max(xi - r, 0), min(xi + r + 1, W)
            if x0 < x1 and 0 <= yi < H:
                base[yi, x0:x1] *= 0.78

    def fireflies(self, em, T, cam, nf):
        if nf < 0.3 or cam < 1000:
            return
        for i in range(34):
            wx = 1140 + hash01(i, 1) * 280 + 10 * math.sin(T * 0.5 + i)
            y = 340 + hash01(i, 2) * 200 + 6 * math.sin(T * 0.8 + i * 1.7)
            b = max(0.0, math.sin(T * (1.3 + hash01(i, 3)) + i * 2.1)) ** 3 * nf
            x = int(round(wx)) - cam
            if 0 <= x < W and 0 <= int(y) < H:
                em[int(y), x] += C(210, 255, 120) * b

    def mist(self, base, T):
        a = float(np.clip(1 - T / 6.5, 0, 1)) * 0.4
        if a <= 0:
            return
        prof = np.clip(1 - np.abs(YY - 330) / 70, 0, 1) * a
        m = BAY < prof
        base[m] = base[m] * 0.7 + C(236, 226, 236) * 0.3

    def title(self, base, em, T):
        a = float(np.clip((T - 42.5) / 2.0, 0, 1))
        if a <= 0:
            return
        L = Layer(W)
        L.text(91, 196, 'MODEL TOWN PARK', FONT5, C(252, 240, 214), em=C(255, 236, 196) * 0.55)
        L.text(124, 210, 'LAHORE', FONT3, C(236, 220, 190), em=C(255, 226, 180) * 0.35)
        ys, xs = np.nonzero(L.a & (BAY < a))
        base[ys, xs] = L.rgb[ys, xs]
        em[ys, xs] += L.em[ys, xs]


# ================================================================= AUDIO
def synth_audio(path, sr=44100):
    n = int(DUR * sr)
    mix = np.zeros(n)
    rng = np.random.default_rng(3)
    bpm = 80
    beat = 60 / bpm
    bar = beat * 4

    def midi(m):
        return 440 * 2 ** ((m - 69) / 12)

    def bell(f, t0, dur, vol):
        i0 = int(t0 * sr)
        m = min(int(dur * sr), n - i0)
        if m <= 0:
            return
        t = np.arange(m) / sr
        w = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t * 6) \
            + 0.12 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t * 10)
        mix[i0:i0 + m] += w * np.minimum(t / 0.004, 1) * np.exp(-t * 2.6) * vol

    def soft(f, t0, dur, vol):
        i0 = int(t0 * sr)
        m = min(int(dur * sr), n - i0)
        if m <= 0:
            return
        t = np.arange(m) / sr
        ph = 2 * np.pi * f * t + 0.004 * f / 5.0 * np.sin(2 * np.pi * 5 * t) * (t > 0.2)
        w = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
        env = np.minimum(t / 0.03, 1) * np.exp(-t * 0.9) * np.clip((dur - t) / 0.08, 0, 1)
        mix[i0:i0 + m] += w * env * vol

    chords = [[65, 69, 72, 77], [62, 65, 69, 74], [58, 62, 65, 70], [60, 64, 67, 72]]  # F Dm Bb C
    roots = [41, 38, 46, 48]
    nbars = int(math.ceil(DUR / bar))
    for b in range(nbars):
        ch = chords[b % 4]
        for e, idx in enumerate([0, 1, 2, 3, 2, 1, 2, 1]):
            bell(midi(ch[idx] + 12), b * bar + e * beat / 2, 1.6, 0.045)
        soft(midi(roots[b % 4]), b * bar, bar * 0.98, 0.10)
        soft(midi(roots[b % 4] + 7), b * bar + 2 * beat, bar * 0.5, 0.04)
    phrase1 = [[(0, 69, 1), (1, 72, 1), (2, 77, 1.5), (3.5, 76, .5)], [(0, 74, 1.5), (1.5, 72, .5), (2, 69, 2)],
               [(0, 70, 1), (1, 74, 1), (2, 77, 1), (3, 74, 1)], [(0, 72, 1.5), (1.5, 67, .5), (2, 72, 2)]]
    phrase2 = [[(0, 69, 1), (1, 72, 1), (2, 77, 1), (3, 81, 1)], [(0, 79, 1.5), (1.5, 77, .5), (2, 74, 2)],
               [(0, 77, 1), (1, 74, 1), (2, 70, 1), (3, 74, 1)], [(0, 76, 1.5), (1.5, 74, .5), (2, 72, 2)]]
    plan = {1: phrase1, 2: phrase1, 3: phrase1, 4: phrase1, 5: phrase2, 6: phrase2, 7: phrase2, 8: phrase2,
            10: phrase1, 11: phrase1, 12: phrase1, 13: phrase1, 14: phrase2, 15: phrase2}
    for b in range(nbars):
        if b in plan:
            for st, m, d in plan[b][b % 4]:
                soft(midi(m), b * bar + st * beat, d * beat * 0.95, 0.075)
    soft(midi(77), 16 * bar, 3 * beat, 0.06)
    # birdsong by day
    t = 0.3
    while t < 31:
        for k in range(int(rng.integers(2, 5))):
            m = int(0.07 * sr)
            i0 = int((t + k * 0.09) * sr)
            tt = np.arange(m) / sr
            f0 = rng.uniform(2600, 3800)
            f = f0 + rng.uniform(-900, 900) * tt / 0.07
            ph = 2 * np.pi * np.cumsum(f) / sr
            vol = 0.035 * (1 - max(0, t - 26) / 5)
            mix[i0:i0 + m] += np.sin(ph) * np.sin(np.pi * tt / 0.07) * vol
        t += rng.uniform(0.5, 1.6)
    # crickets by night
    i0 = int(35 * sr)
    tt = np.arange(n - i0) / sr
    pulse = (np.sin(2 * np.pi * 14 * tt) > 0.3) * (np.sin(2 * np.pi * 0.45 * tt) > -0.2)
    cr = np.sin(2 * np.pi * 4700 * tt) * pulse * 0.012 * np.clip(tt / 3, 0, 1)
    mix[i0:] += cr
    t = np.arange(n) / sr
    mix *= np.minimum(t / 1.0, 1) * np.clip((DUR - t) / 2.5, 0, 1)
    mix = np.tanh(mix * 1.5) * 0.8
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wv:
        wv.setnchannels(1)
        wv.setsampwidth(2)
        wv.setframerate(sr)
        wv.writeframes(pcm.tobytes())


# ================================================================= MAIN
def main():
    args = sys.argv[1:]
    scene = Scene()
    if args and args[0] == '--preview':
        from PIL import Image
        outdir = args[2] if len(args) > 2 else '.'
        for s in args[1].split(','):
            img = scene.render(float(s))
            Image.fromarray(img).resize((W * SCALE, SH * SCALE), Image.NEAREST).save(
                os.path.join(outdir, f'mtp_{float(s):05.1f}.png'))
        return
    out = args[0] if args else 'model_town_park_pixel.mp4'
    wav = os.path.splitext(out)[0] + '.wav'
    synth_audio(wav)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{SH}',
           '-r', str(FPS), '-i', '-', '-i', wav,
           '-vf', f'scale={W * SCALE}:{SH * SCALE}:flags=neighbor,format=yuv420p',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-tune', 'animation',
           '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(NF):
        p.stdin.write(scene.render(i / FPS).tobytes())
        if i % 150 == 0:
            print(f'frame {i}/{NF}', flush=True)
    p.stdin.close()
    p.wait()
    os.remove(wav)
    print('wrote', out)


if __name__ == '__main__':
    main()
