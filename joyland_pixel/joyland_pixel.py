#!/usr/bin/env python3
"""
Joyland, Fortress Stadium, Lahore -- pixel-art animation.

Renders a 270x480 pixel scene, upscaled 4x (nearest neighbour) to 1080x1920,
that runs from a bright afternoon through sunset into a lit-up night with
fireworks. Pure numpy; frames are piped straight into ffmpeg. A small
chiptune soundtrack (with booms synced to the fireworks) is synthesised too.

    python3 joyland_pixel.py out.mp4            # full render
    python3 joyland_pixel.py --preview 2,11,18  # PNG stills at those seconds
"""
import math
import os
import subprocess
import sys
import wave

import numpy as np

W, H, SCALE, FPS = 270, 480, 4, 30
DUR = 26.0
NF = int(DUR * FPS)
HORIZON = 300


def C(r, g, b):
    return np.array([r, g, b], dtype=float) / 255.0


def hsv(h, s=1.0, v=1.0):
    h = (h % 1.0) * 6
    i = int(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6])


def hash01(*a):
    x = math.sin(sum(v * k for v, k in zip(a, (12.9898, 78.233, 37.719, 4.581)))) * 43758.5453
    return x - math.floor(x)


BAY4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
BAY = np.tile(BAY4, (H // 4 + 1, W // 4 + 1))[:H, :W]
YY, XX = np.mgrid[0:H, 0:W]


def kf(T, keys):
    """Smooth keyframe interpolation; values may be scalars or arrays."""
    if T <= keys[0][0]:
        return np.asarray(keys[0][1], float)
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if T <= t1:
            u = (T - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)
            return np.asarray(v0, float) * (1 - u) + np.asarray(v1, float) * u
    return np.asarray(keys[-1][1], float)


# ---------------------------------------------------------------- time of day
DAY = [(38, 104, 214), (76, 150, 236), (150, 206, 248)]
SKY_KEYS = [
    (0, DAY), (4, DAY),
    (8, [(52, 92, 190), (122, 140, 212), (252, 198, 132)]),
    (10.5, [(48, 46, 118), (172, 82, 122), (252, 130, 72)]),
    (12.5, [(20, 22, 64), (62, 40, 98), (150, 62, 92)]),
    (14.5, [(8, 10, 30), (16, 18, 52), (44, 32, 80)]),
    (DUR, [(8, 10, 30), (16, 18, 52), (44, 32, 80)]),
]
AMB_KEYS = [(0, (1, 1, 1)), (4, (1, 1, 1)), (8, (1.0, 0.9, 0.76)), (10.5, (0.88, 0.62, 0.56)),
            (12.5, (0.48, 0.42, 0.62)), (14.5, (0.27, 0.29, 0.47)), (DUR, (0.27, 0.29, 0.47))]
CLOUD_KEYS = [
    (0, [(255, 255, 255), (234, 241, 250), (196, 212, 236)]),
    (4, [(255, 255, 255), (234, 241, 250), (196, 212, 236)]),
    (8, [(255, 240, 214), (246, 212, 182), (214, 170, 160)]),
    (10.5, [(255, 192, 150), (240, 140, 130), (170, 90, 120)]),
    (12.5, [(150, 90, 122), (110, 70, 110), (70, 50, 90)]),
    (14.5, [(58, 60, 96), (42, 44, 80), (30, 32, 62)]),
    (DUR, [(58, 60, 96), (42, 44, 80), (30, 32, 62)]),
]


def lights_at(T):
    return float(np.clip((T - 9.6) / 3.2, 0, 1))


def night_at(T):
    return float(kf(T, [(0, 0), (11, 0), (14.5, 1), (DUR, 1)]))


# ---------------------------------------------------------------- layers
class Layer:
    """Albedo + emissive paint buffer. Emissive pixels carry a switch-on
    threshold (against the global 'lights' level) and an optional twinkle."""

    def __init__(self):
        self.rgb = np.zeros((H, W, 3))
        self.a = np.zeros((H, W), bool)
        self.em = np.zeros((H, W, 3))
        self.thr = np.full((H, W), -1.0)
        self.twa = np.zeros((H, W))
        self.ph = np.zeros((H, W))

    def put(self, ys, xs, col, em=None, thr=-1.0, twa=0.0, ph=0.0):
        ys = np.rint(np.asarray(ys, float)).astype(int).ravel()
        xs = np.rint(np.asarray(xs, float)).astype(int).ravel()
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
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
        x0, x1 = max(int(x0), 0), min(int(x1), W)
        y0, y1 = max(int(y0), 0), min(int(y1), H)
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

    def blit(self, spr, x0, y0, flip=False, em_scale=1.0):
        xs = spr['w'] - 1 - spr['xs'] if flip else spr['xs']
        self.put(spr['ys'] + y0, xs + x0, spr['col'], em=spr['em'] * em_scale)

    def text(self, x, y, s, font, col, **k):
        gw = len(next(iter(font.values()))[0])
        for i, ch in enumerate(s):
            g = font.get(ch)
            if g is None:
                continue
            m = np.array([[c == '#' for c in row] for row in g])
            self.mask(x + i * (gw + 1), y, m, col, **k)


class Static:
    """A pre-rendered Layer flattened to its painted pixels."""

    def __init__(self, L, uplight=None):
        self.idx = np.nonzero(L.a)
        self.rgb, self.em = L.rgb[self.idx], L.em[self.idx]
        self.thr, self.twa, self.ph = L.thr[self.idx], L.twa[self.idx], L.ph[self.idx]
        self.up = None if uplight is None else uplight[self.idx]


def composite(base, em, L, amb, lights, T, up_col=None):
    if isinstance(L, Layer):
        idx = np.nonzero(L.a)
        rgb, e, thr, twa, ph, up = L.rgb[idx], L.em[idx], L.thr[idx], L.twa[idx], L.ph[idx], None
    else:
        idx, rgb, e, thr, twa, ph, up = L.idx, L.rgb, L.em, L.thr, L.twa, L.ph, L.up
    lit = rgb * amb
    if up is not None and up_col is not None:
        lit = lit + rgb * up[:, None] * up_col
    base[idx] = lit
    f = np.clip((lights - thr) / 0.08, 0, 1)
    f = f * (1 - twa * 0.5 * (1 + np.sin(2 * np.pi * (T * 1.3 + ph))))
    em[idx] = e * f[:, None]


def shifted(m, dy, dx):
    """out[y, x] = m[y - dy, x - dx] (zero-filled)."""
    out = np.zeros_like(m)
    h, w = m.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        m[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def shade3(m, d=2):
    """0 empty, 1 shadow (lower right), 2 mid, 3 highlight (upper left)."""
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
    return dict(ys=np.array(ys), xs=np.array(xs), col=np.array(col), em=np.array(em), w=w, h=len(rows))


# ---------------------------------------------------------------- fonts
FONT3 = {
    'A': ['.#.', '#.#', '###', '#.#', '#.#'], 'D': ['##.', '#.#', '#.#', '#.#', '##.'],
    'E': ['###', '#..', '##.', '#..', '###'], 'F': ['###', '#..', '##.', '#..', '#..'],
    'I': ['###', '.#.', '.#.', '.#.', '###'], 'M': ['#.#', '###', '###', '#.#', '#.#'],
    'O': ['###', '#.#', '#.#', '#.#', '###'], 'R': ['##.', '#.#', '##.', '#.#', '#.#'],
    'S': ['.##', '#..', '.#.', '..#', '##.'], 'T': ['###', '.#.', '.#.', '.#.', '.#.'],
    'U': ['#.#', '#.#', '#.#', '#.#', '###'], 'L': ['#..', '#..', '#..', '#..', '###'],
    'H': ['#.#', '#.#', '###', '#.#', '#.#'], ' ': ['...'] * 5,
}
FONT5 = {
    'J': ['..###', '...#.', '...#.', '...#.', '#..#.', '#..#.', '.##..'],
    'O': ['.###.', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
    'Y': ['#...#', '#...#', '.#.#.', '..#..', '..#..', '..#..', '..#..'],
    'L': ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
    'A': ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
    'N': ['#...#', '##..#', '#.#.#', '#..##', '#...#', '#...#', '#...#'],
    'D': ['####.', '#...#', '#...#', '#...#', '#...#', '#...#', '####.'],
}

# ---------------------------------------------------------------- palette
BRICK, BRICK_D, BRICK_L, BRICK_DD = C(172, 78, 52), C(154, 67, 46), C(190, 94, 62), C(128, 54, 40)
CREAM, CREAM_D, SLIT = C(238, 216, 174), C(206, 178, 136), C(52, 24, 20)
WHITE_DOME, DOME_SH, GOLD = C(244, 240, 230), C(206, 200, 192), C(232, 182, 60)
WARM = C(255, 168, 88)
STEEL, STEEL_D = C(232, 234, 242), C(168, 172, 190)
FAIRY = [C(255, 90, 90), C(255, 210, 80), C(110, 220, 255), C(140, 255, 140), C(255, 130, 230)]


def arch_mask(w, h, k=0.78):
    """Pointed (Mughal-ish) arch of width w and total height h."""
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    xc = (w - 1) / 2
    a = w / 2
    r = k * w
    head = math.sqrt(max(r * r - (r - a) ** 2, 0))
    spring = head
    below = (ys >= spring) & (np.abs(xs - xc) <= a - 0.5)
    lc, rc = xc - a + r, xc + a - r
    d1 = np.hypot(xs - lc, ys - spring)
    d2 = np.hypot(xs - rc, ys - spring)
    above = (ys < spring) & (d1 <= r) & (d2 <= r)
    return below | above


def dome_mask(w, h):
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    xc = (w - 1) / 2
    u = (xs - xc) / (w / 2)
    v = (h - 0.5 - ys) / h
    # onion-ish: bulge low, taper to a point
    return (v >= 0) & (np.abs(u) <= np.sqrt(np.clip(1 - v, 0, 1)) * (1.0 + 0.25 * np.sin(np.pi * v)) * (1 - v) ** 0.15)


# ================================================================= SCENE BUILD
def build_city():
    L = Layer()
    rng = np.random.default_rng(7)
    x = -4
    tone_flip = 0
    while x < W:
        w = int(rng.integers(7, 19))
        h = int(rng.integers(12, 40))
        tone_flip ^= 1
        col = C(128, 146, 180) * (0.94 if tone_flip else 1.03) * rng.uniform(0.97, 1.03)
        L.rect(x, HORIZON - h, x + w, HORIZON, col)
        if rng.random() < 0.35:
            L.rect(x + 2, HORIZON - h - 3, x + 5, HORIZON - h, col * 0.92)
        if rng.random() < 0.25:
            L.line(x + w - 3, HORIZON - h - 1, x + w - 3, HORIZON - h - 7, col * 0.9)
        for wy in range(HORIZON - h + 3, HORIZON - 2, 4):
            for wx in range(x + 2, x + w - 1, 3):
                if rng.random() < 0.5:
                    L.px(wx, wy, col * 0.85, em=C(255, 206, 130) * rng.uniform(0.45, 0.9),
                         thr=rng.uniform(0.02, 0.95))
        x += w + int(rng.integers(-2, 2))

    # Minar-e-Pakistan on the skyline
    mx = 78
    mc = C(160, 172, 200)
    for i, (y0, y1, hw) in enumerate([(296, 300, 12), (292, 296, 9), (289, 292, 6)]):
        L.rect(mx - hw, y0, mx + hw + 1, y1, mc * (0.92 + 0.04 * i))
    for y in range(240, 289):
        hw = 3 if y > 270 else 2 if y > 250 else 1
        L.rect(mx - hw, y, mx + hw + 1, y + 1, mc)
        L.px(mx + hw, y, mc * 0.85)
    L.rect(mx - 3, 247, mx + 4, 249, mc * 1.05)
    L.disk(mx, 236, 2.4, mc * 1.05)
    L.line(mx, 233, mx, 228, mc)
    L.px(mx, 227, C(200, 40, 40), em=C(255, 40, 40), thr=0.3, twa=1.0, ph=0.0)
    return Static(L)


def build_fort():
    L = Layer()
    rng = np.random.default_rng(3)
    WT, WB = 300, 372

    def bricks(x0, x1, y0, y1, base, mortar):
        L.rect(x0, y0, x1, y1, base)
        for c, y in enumerate(range(y0 + 3, y1, 4)):
            L.rect(x0, y, x1, y + 1, mortar)
            off = 0 if c % 2 else 5
            for x in range(x0 + off, x1, 10):
                L.rect(x, y + 1, x + 1, y + 4, mortar)

    def fairy(x, y, i):
        L.px(x, y, C(90, 90, 90), em=FAIRY[i % len(FAIRY)], thr=0.3 + 0.1 * hash01(x, y), twa=0.7,
             ph=hash01(x, 3 * y))

    def merlons(x0, x1, ytop, step, mw, col):
        for i, x in enumerate(range(x0, x1, step)):
            L.rect(x, ytop, min(x + mw, x1), ytop + 8, col)
            L.rect(x, ytop, min(x + mw, x1), ytop + 2, CREAM)
            fairy(x + mw // 2, ytop - 1, i)

    # main curtain wall
    bricks(0, W, WT, WB, BRICK, BRICK_D)
    merlons(0, W, WT - 8, 10, 6, BRICK)
    L.rect(0, WT, W, WT + 3, CREAM)
    L.rect(0, WT + 3, W, WT + 4, BRICK_DD)
    L.rect(0, WB - 8, W, WB, CREAM_D)
    L.rect(0, WB - 8, W, WB - 7, CREAM)

    # recessed arched panels with glowing slits
    am = arch_mask(12, 36)
    am_out = arch_mask(14, 37)
    for cx in [50, 72, 196, 218]:
        L.mask(cx - 7, 322, am_out, CREAM_D)
        L.mask(cx - 6, 323, am, BRICK_DD)
        L.rect(cx - 1, 334, cx + 1, 348, SLIT, em=WARM * 0.85, thr=0.15 + 0.3 * hash01(cx))

    # bastions
    for cx in (17, 253):
        x0, x1, top = cx - 18, cx + 18, 262
        bricks(x0, x1, top, WB - 8, BRICK, BRICK_D)
        for x in range(x0, x1):
            rel = (x - cx) / 18
            if rel < -0.55:
                L.rect(x, top, x + 1, WB - 8, BRICK_L)
            elif rel > 0.5:
                L.rect(x, top, x + 1, WB - 8, BRICK_DD)
        L.rect(x0 - 2, top - 5, x1 + 2, top, CREAM)
        for x in range(x0 - 1, x1 + 2, 3):
            L.px(x, top, SLIT)
        merlons(x0 - 2, x1 + 2, top - 13, 8, 5, BRICK)
        # chhatri (domed kiosk)
        L.rect(cx - 7, top - 15, cx + 8, top - 13, CREAM)
        for px_ in (cx - 6, cx - 2, cx + 3, cx + 7):
            L.rect(px_, top - 23, px_ + 1, top - 15, CREAM)
        L.rect(cx - 5, top - 23, cx + 7, top - 15, SLIT, em=WARM * 0.7, thr=0.25)
        for px_ in (cx - 6, cx - 2, cx + 3, cx + 7):
            L.rect(px_, top - 23, px_ + 1, top - 15, CREAM)
        L.rect(cx - 8, top - 25, cx + 9, top - 23, CREAM)
        dm = dome_mask(13, 9)
        sh = np.zeros_like(dm)
        sh[:, 8:] = True
        L.mask(cx - 6, top - 34, dm & ~sh, WHITE_DOME)
        L.mask(cx - 6, top - 34, dm & sh, DOME_SH)
        L.line(cx, top - 35, cx, top - 38, GOLD)
        for wy in (284, 326):
            L.rect(cx - 1, wy, cx + 1, wy + 7, SLIT, em=WARM * 0.8, thr=0.3)

    # gatehouse
    GX0, GX1, GT = 98, 172, 250
    bricks(GX0, GX1, GT, WB - 8, BRICK_L, BRICK)
    merlons(GX0, GX1, GT - 7, 8, 5, BRICK_L)
    L.rect(GX0, GT, GX1, GT + 3, CREAM)
    for tx in (GX0 - 4, GX1 - 8):
        bricks(tx, tx + 12, 236, WB - 8, BRICK, BRICK_D)
        L.rect(tx, 236, tx + 3, WB - 8, BRICK_L)
        L.rect(tx + 9, 236, tx + 12, WB - 8, BRICK_DD)
        for by in (236, 262, 300):
            L.rect(tx - 1, by, tx + 13, by + 2, CREAM)
        dm = dome_mask(12, 10)
        sh = np.zeros_like(dm)
        sh[:, 7:] = True
        L.mask(tx, 226, dm & ~sh, WHITE_DOME)
        L.mask(tx, 226, dm & sh, DOME_SH)
        L.line(tx + 5, 225, tx + 5, 221, GOLD)
        L.rect(tx + 5, 270, tx + 7, 280, SLIT, em=WARM * 0.8, thr=0.2)
        L.rect(tx + 5, 316, tx + 7, 326, SLIT, em=WARM * 0.8, thr=0.25)
    # jharokha windows
    wm, wo = arch_mask(9, 15), arch_mask(11, 16)
    for cx in (116, 135, 154):
        L.mask(cx - 5, 262, wo, CREAM)
        L.mask(cx - 4, 263, wm, SLIT, em=WARM * 0.75, thr=0.2 + 0.2 * hash01(cx, 1))
        L.rect(cx - 5, 278, cx + 6, 280, CREAM)
    # plaque
    L.rect(101, 300, 170, 312, CREAM)
    L.rect(101, 311, 170, 312, CREAM_D)
    L.text(104, 304, 'FORTRESS STADIUM', FONT3, BRICK_DD)
    # main gate
    gm_o, gm = arch_mask(36, 54), arch_mask(30, 50)
    L.mask(135 - 18, WB - 54, gm_o, CREAM)
    L.mask(135 - 15, WB - 50, gm, C(40, 20, 18), em=C(255, 150, 80) * 0.5, thr=0.15)
    L.rect(124, WB - 4, 147, WB, C(70, 40, 30), em=C(255, 180, 100) * 0.6, thr=0.15)
    for sx in (112, 158):
        L.rect(sx, 338, sx + 2, 342, C(40, 30, 20), em=C(255, 205, 120), thr=0.12)
        L.px(sx, 337, GOLD)

    # grass, hedges, trees, palms
    L.rect(0, WB, W, 380, C(62, 132, 60))
    for i in range(160):
        x, y = int(rng.integers(0, W)), int(rng.integers(WB + 1, 379))
        L.px(x, y, [C(230, 60, 80), C(250, 210, 60), C(250, 140, 200), C(250, 250, 250), C(44, 108, 48)][i % 5])
    hm = np.zeros((16, W), bool)
    for x in range(W):
        top = 6 + int(round(1.6 * math.sin(x * 0.55) + 1.2 * math.sin(x * 0.23 + 1)))
        hm[top:, x] = True
    hm[:, 114:157] = False
    hs = shade3(hm, 1)
    L.mask(0, 362, hs == 1, C(26, 82, 46))
    L.mask(0, 362, hs == 2, C(38, 108, 58))
    L.mask(0, 362, hs == 3, C(70, 146, 72))
    # path to gate
    for y in range(WB, 380):
        w = 12 + (y - WB)
        L.rect(135 - w, y, 135 + w, y + 1, C(196, 184, 164))

    def tree(cx, by, r):
        L.rect(cx - 1, by - r, cx + 1, by, C(96, 66, 44))
        R = r + 3
        yy, xx = np.mgrid[-2 * R:R, -R:R + 1]
        m = (xx ** 2 + (yy + r) ** 2 <= r * r) | ((xx + r * 0.6) ** 2 + (yy + r * 0.6) ** 2 <= (r * 0.7) ** 2) \
            | ((xx - r * 0.6) ** 2 + (yy + r * 0.6) ** 2 <= (r * 0.7) ** 2)
        s = shade3(m, 2)
        ox, oy = cx - R, by - r - 2 * R + r
        for v, col in ((1, C(28, 86, 48)), (2, C(44, 120, 62)), (3, C(84, 160, 78))):
            L.mask(ox, oy, s == v, col)
        sp = m & (np.random.default_rng(cx).random(m.shape) < 0.12)
        L.mask(ox, oy, sp, C(30, 92, 50))

    for cx, r in ((48, 9), (84, 7), (192, 8), (224, 10)):
        tree(cx, 376, r)

    def palm(bx, by, lean):
        ty = by - 46
        for i in range(47):
            y = by - i
            x = bx + lean * (i / 46) ** 2
            L.px(x, y, C(128, 94, 62) if (i // 2) % 2 else C(104, 76, 50))
            L.px(x + 1, y, C(92, 66, 44))
        tx = bx + lean
        for ang in (-2.7, -2.2, -1.6, -1.0, -0.45, 0.1, 0.65):
            for j in range(13):
                d = j
                x = tx + math.cos(ang) * d
                y = ty + math.sin(ang) * d * 0.6 + 0.045 * d * d
                L.px(x, y, C(52, 140, 66) if j % 3 else C(36, 106, 52))
                if j > 3:
                    L.px(x, y + 1, C(36, 106, 52))
        L.disk(tx, ty + 1, 1.5, C(110, 80, 40))

    palm(91, 378, 3)
    palm(180, 378, -3)

    # kerbs, road, pavement
    def kerb(y):
        for x in range(0, W, 6):
            L.rect(x, y, x + 6, y + 3, C(242, 198, 44) if (x // 6) % 2 == 0 else C(32, 32, 36))
        L.rect(0, y, W, y + 1, C(250, 236, 180))
        for x in range(0, W, 12):
            L.rect(x + 6, y, x + 12, y + 1, C(80, 80, 86))

    L.rect(0, 380, W, 438, C(74, 74, 82))
    noise = np.random.default_rng(5).random((58, W))
    L.mask(0, 380, noise < 0.06, C(86, 86, 94))
    L.mask(0, 380, noise > 0.95, C(62, 62, 70))
    for x in range(0, W, 20):
        L.rect(x, 408, x + 11, 410, C(226, 226, 214))
    L.rect(0, 383, W, 384, C(150, 150, 140))
    kerb(380)
    kerb(437)
    L.rect(0, 440, W, H, C(160, 152, 142))
    for y in range(444, H, 9):
        L.rect(0, y, W, y + 1, C(138, 130, 122))
        off = 0 if (y // 9) % 2 else 9
        for x in range(off, W, 18):
            L.rect(x, y + 1, x + 1, y + 9, C(138, 130, 122))
    L.rect(0, 440, W, 442, C(118, 112, 106))

    # warm floodlighting on the fortress at night
    up = np.zeros((H, W))
    fort_rows = (YY >= 220) & (YY < WB)
    up[fort_rows] = np.clip((YY[fort_rows] - 220) / 152, 0, 1) ** 1.4 * 0.75
    up[(XX >= 96) & (XX < 174) & fort_rows] *= 1.25
    return Static(L, uplight=up)


def catenary(L, x0, y0, x1, y1, sag, col):
    n = int(abs(x1 - x0)) + 1
    xs = np.linspace(x0, x1, n)
    u = np.linspace(0, 1, n)
    ys = y0 + (y1 - y0) * u + sag * 4 * u * (1 - u)
    L.put(ys, xs, col)
    return xs, ys


def build_foreground():
    L = Layer()
    POLE, POLE_L = C(70, 68, 74), C(104, 102, 108)
    wire = C(36, 36, 42)
    poles = (30, 206)
    for x in poles:
        L.rect(x, 314, x + 3, H, POLE)
        L.rect(x, 314, x + 1, H, POLE_L)
        L.rect(x - 11, 320, x + 14, 322, POLE)
        L.rect(x - 7, 330, x + 10, 331, POLE)
        for ix in (x - 10, x - 4, x + 7, x + 13):
            L.px(ix, 319, C(210, 206, 190))
        # street lamp
        L.line(x + 3, 342, x + 13, 338, POLE)
        L.rect(x + 11, 337, x + 19, 340, C(50, 50, 56))
        L.rect(x + 12, 340, x + 18, 341, C(200, 196, 170), em=C(255, 226, 170), thr=0.12)
    L.rect(poles[1] + 3, 352, poles[1] + 11, 364, C(120, 124, 130))
    L.rect(poles[1] + 3, 352, poles[1] + 11, 353, C(150, 154, 160))
    L.rect(poles[1] + 5, 356, poles[1] + 9, 357, C(220, 180, 40))
    a, b = poles
    bunting_cols = [C(220, 40, 60), C(250, 200, 40), C(40, 160, 220), C(60, 180, 80), C(244, 244, 244)]
    for (oa, ob, sag) in ((-10, -10, 13), (-4, -4, 15), (7, 7, 14), (13, 13, 12)):
        catenary(L, -6, 324, a + oa, 319, 5, wire)
        xs, ys = catenary(L, a + oa, 319, b + ob, 319, sag, wire)
        catenary(L, b + ob, 319, W + 6, 326, 6, wire)
        if oa == -4:
            for i in range(4, len(xs) - 4, 6):
                c = bunting_cols[(i // 6) % 5]
                x, y = xs[i], ys[i]
                L.rect(x - 1, y + 1, x + 2, y + 2, c)
                L.rect(x - 1 + 0.4, y + 2, x + 1, y + 3, c)
                L.px(x, y + 3, c)
    catenary(L, a + 3, 331, b, 333, 18, wire)
    return Static(L)


def lamp_cones():
    m = np.zeros((H, W))
    for x in (30, 206):
        lx, ly = x + 15, 341
        dy = YY - ly
        hw = 3 + dy * 0.42
        inside = (dy > 0) & (np.abs(XX - lx) < hw)
        m += np.where(inside, 0.035, 0)
        pool = ((XX - lx) / 46.0) ** 2 + ((YY - 440) / 30.0) ** 2
        m += np.where(pool < 1, 0.13 * (1 - pool), 0)
    return np.floor(m * 40) / 40


# ---------------------------------------------------------------- sky things
def make_clouds():
    rng = np.random.default_rng(21)
    out = []
    specs = [(54, 16, 40, 2.2), (40, 12, 108, 3.4), (70, 18, 150, 1.6), (34, 10, 22, 4.2), (46, 13, 196, 2.8)]
    for i, (w, h, y, v) in enumerate(specs):
        m = np.zeros((h + 2, w + 2), bool)
        yy, xx = np.mgrid[0:h + 2, 0:w + 2]
        n = int(rng.integers(4, 7))
        for j in range(n):
            cx = (j + 0.5) / n * w + rng.uniform(-3, 3)
            r = rng.uniform(0.35, 0.62) * h * (1.25 if 0 < j < n - 1 else 0.85)
            m |= (xx - cx) ** 2 + (yy - (h - r * 0.55)) ** 2 <= r * r
        m[h:, :] = False
        s = shade3(m, 2)
        s[(m) & (yy >= h - 3)] = 1
        out.append(dict(s=s, x0=rng.uniform(0, W), y=y, v=v, w=w + 2))
    return out


STAR_RNG = np.random.default_rng(99)
STARS = [(int(STAR_RNG.integers(0, W)), int(STAR_RNG.integers(0, 250)), STAR_RNG.uniform(0.25, 1.0),
          STAR_RNG.uniform(0, 6.28), STAR_RNG.uniform(1.5, 4.0)) for _ in range(150)]


def make_fireworks():
    rng = np.random.default_rng(11)
    pal = [C(255, 80, 90), C(255, 205, 70), C(90, 205, 255), C(130, 255, 130), C(255, 120, 250),
           C(255, 255, 255), C(255, 150, 60), C(180, 140, 255)]
    times = []
    t = 13.8
    while t < 21.8:
        times.append((t, False))
        t += rng.uniform(0.42, 0.85)
    for k in range(9):
        times.append((21.9 + k * 0.17, True))
    fws = []
    for i, (tl, big) in enumerate(times):
        kind = rng.choice(['peony', 'peony', 'ring', 'willow', 'two']) if not big else \
            rng.choice(['peony', 'willow', 'ring'])
        n = int(rng.integers(46, 70)) + (30 if big else 0)
        spd = rng.uniform(62, 86) * (1.25 if big else 1.0)
        if kind == 'ring':
            ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
            tilt = rng.uniform(0, np.pi)
            ex, ey = np.cos(ph), np.sin(ph) * rng.uniform(0.35, 0.7)
            vx = (ex * np.cos(tilt) - ey * np.sin(tilt)) * spd
            vy = (ex * np.sin(tilt) + ey * np.cos(tilt)) * spd
        else:
            z = rng.uniform(-1, 1, n)
            ph = rng.uniform(0, 2 * np.pi, n)
            rr = np.sqrt(1 - z * z) * rng.uniform(0.85, 1.0, n)
            vx, vy = rr * np.cos(ph) * spd, rr * np.sin(ph) * spd
        c1 = pal[rng.integers(len(pal))]
        c2 = pal[rng.integers(len(pal))]
        if kind == 'willow':
            cols = np.tile(C(255, 190, 90), (n, 1))
            drag, grav, life, trail = 1.4, 34, rng.uniform(2.4, 2.9, n), 6
        else:
            cols = np.array([c1 if (kind != 'two' or j % 2) else c2 for j in range(n)])
            drag, grav, life, trail = 2.4, 18, rng.uniform(1.3, 2.0, n), 3
        fws.append(dict(tl=tl, rise=rng.uniform(0.75, 1.05), x=rng.uniform(24, 246),
                        by=rng.uniform(46, 150) - (12 if big else 0), vx=vx, vy=vy, cols=cols,
                        drag=drag, grav=grav, life=life, trail=trail, crackle=rng.random() < 0.4,
                        seed=i, flash=c1))
    return fws


def draw_fireworks(em, fws, T):
    pts_y, pts_x, pts_c = [], [], []
    sky_flash = np.zeros(3)
    for f in fws:
        tb = f['tl'] + f['rise']
        if T < f['tl'] or T > tb + 3.2:
            continue
        if T < tb:
            u = (T - f['tl']) / f['rise']
            for k in range(6):
                uu = max(u - k * 0.025, 0)
                y = HORIZON + (f['by'] - HORIZON) * (1 - (1 - uu) ** 2)
                x = f['x'] + 1.5 * math.sin(uu * 9 + f['seed'])
                b = 1.0 if k == 0 else 0.5 * (1 - k / 6) * (0.5 + hash01(k, T * 30))
                pts_y.append([y]), pts_x.append([x]), pts_c.append(np.array([[1.0, 0.85, 0.6]]) * b)
            continue
        tau = T - tb
        if tau < 0.12:
            r = 4.5 * (1 - tau / 0.12) + 1
            yy, xx = np.mgrid[-6:7, -6:7]
            m = xx * xx + yy * yy <= r * r
            pts_y.append((yy[m] + f['by']).ravel()), pts_x.append((xx[m] + f['x']).ravel())
            pts_c.append(np.tile(np.array([1.0, 0.95, 0.85]) * 1.4, (m.sum(), 1)))
        if tau < 0.3:
            sky_flash += f['flash'] * 0.035 * (1 - tau / 0.3)
        for k in range(f['trail']):
            ts = tau - k * 0.045
            if ts < 0:
                break
            d = (1 - math.exp(-f['drag'] * ts)) / f['drag']
            x = f['x'] + f['vx'] * d
            y = f['by'] + f['vy'] * d + 0.5 * f['grav'] * ts * ts
            b = np.clip(1 - ts / f['life'], 0, 1) ** 1.5 * (1.0 if k == 0 else 0.6 * (1 - k / f['trail']))
            if f['crackle'] and k == 0:
                fr = int(T * FPS)
                flick = np.array([hash01(f['seed'], j, fr) for j in range(len(x))])
                b = b * np.where((ts / f['life'] > 0.55) & (flick < 0.5), 0.15, 1.0) * (1 + 0.6 * (flick > 0.85))
            col = f['cols'] * b[:, None]
            if ts < 0.15:
                col = col + (1 - ts / 0.15) * 0.3
            pts_y.append(y), pts_x.append(x), pts_c.append(col)
    if pts_y:
        ys = np.rint(np.concatenate(pts_y)).astype(int)
        xs = np.rint(np.concatenate(pts_x)).astype(int)
        cs = np.concatenate(pts_c)
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        np.add.at(em, (ys[ok], xs[ok]), cs[ok])
    return sky_flash


# ---------------------------------------------------------------- rides
HX, HY, RO, RI, NG = 176, 186, 74, 68, 16
GONDOLA_COLS = [C(232, 64, 72), C(250, 196, 50), C(60, 140, 230), C(70, 190, 100), C(240, 110, 200),
                C(250, 140, 50), C(130, 100, 230), C(60, 200, 210)]


def draw_wheel(L, T, lt):
    rot = 2 * np.pi * T / 42.0
    # legs + bracing
    for lx in (132, 220):
        L.line(HX, HY, lx, HORIZON, STEEL_D)
        L.line(HX + 1, HY, lx + 1, HORIZON, STEEL)
    for yb in (232, 262, 290):
        u = (yb - HY) / (HORIZON - HY)
        L.line(HX + (132 - HX) * u, yb, HX + (220 - HX) * u, yb, STEEL_D)
    # rims
    for r, col in ((RO, STEEL), (RI, C(226, 72, 84))):
        a = np.linspace(0, 2 * np.pi, int(2 * np.pi * r * 1.6), endpoint=False)
        L.put(HY + r * np.sin(a), HX + r * np.cos(a), col)
    angs = rot + 2 * np.pi * np.arange(NG) / NG
    for i, a in enumerate(angs):
        a2 = a + np.pi / NG
        L.line(HX, HY, HX + RI * math.cos(a), HY + RI * math.sin(a), C(206, 210, 224))
        L.line(HX + RI * math.cos(a), HY + RI * math.sin(a), HX + RO * math.cos(a2), HY + RO * math.sin(a2), STEEL_D)
        L.line(HX + RO * math.cos(a2), HY + RO * math.sin(a2), HX + RI * math.cos(a + 2 * np.pi / NG),
               HY + RI * math.sin(a + 2 * np.pi / NG), STEEL_D)
    # bulbs
    mode = int(T / 3.0) % 3
    nb = 48
    for j in range(nb):
        a = rot + 2 * np.pi * j / nb + np.pi / nb
        if mode == 0:
            c = hsv(j / nb - T * 0.35)
        elif mode == 1:
            c = C(255, 90, 200) if (j + int(T * 5)) % 2 else C(90, 230, 255)
        else:
            c = C(255, 240, 200) * (0.25 + 0.75 * (hash01(j, int(T * 12)) > 0.45))
        L.px(HX + (RO + 1) * math.cos(a), HY + (RO + 1) * math.sin(a), C(245, 245, 235), em=c * lt)
    for i, a in enumerate(angs):
        for r in (18, 32, 46, 60):
            b = 0.5 + 0.5 * math.sin(T * 7 - r * 0.16 + (i % 2) * math.pi * (mode == 1))
            L.px(HX + r * math.cos(a), HY + r * math.sin(a), C(220, 220, 220),
                 em=hsv(0.12 + 0.5 * (mode == 2) + i / NG * (mode == 0)) * b * lt)
    L.disk(HX, HY, 4.2, C(84, 86, 98))
    L.disk(HX, HY, 2, C(236, 236, 244), em=C(255, 255, 255) * 0.8 * lt)
    # gondolas
    for i, a in enumerate(angs):
        x, y = HX + RO * math.cos(a), HY + RO * math.sin(a)
        x, y = int(round(x)), int(round(y))
        col = GONDOLA_COLS[i % len(GONDOLA_COLS)]
        L.line(x, y, x, y + 2, C(80, 80, 90))
        L.rect(x - 2, y + 2, x + 3, y + 3, col * 0.75)
        L.rect(x - 3, y + 3, x + 4, y + 4, col * 0.85)
        L.rect(x - 3, y + 4, x + 4, y + 7, col)
        L.rect(x - 2, y + 5, x, y + 6, C(150, 200, 232), em=C(255, 220, 150) * 0.8 * lt)
        L.rect(x + 1, y + 5, x + 3, y + 6, C(150, 200, 232), em=C(255, 220, 150) * 0.8 * lt)
        L.rect(x - 2, y + 7, x + 3, y + 8, col * 0.7)


def make_hull():
    w, h = 44, 15
    rgb = np.zeros((h, w, 3))
    a = np.zeros((h, w), bool)
    em = np.zeros((h, w, 3))
    rng = np.random.default_rng(4)
    for r in range(5, 15):
        k = (r - 5) / 9
        x0 = 2 + int(round(k * k * 8))
        x1 = w - 3 - int(round(k * k * 8))
        a[r, x0:x1 + 1] = True
        if r <= 6:
            c = C(196, 42, 46)
        elif r == 7:
            c = C(244, 196, 64)
        elif r == 14:
            c = C(80, 44, 28)
        else:
            c = C(140, 82, 44) if r % 2 else C(122, 70, 38)
        rgb[r, x0:x1 + 1] = c
    # bow / stern peaks
    for i in range(5):
        for x in range(0, 4 - i // 2):
            a[i, x + i // 2] = True
            rgb[i, x + i // 2] = C(196, 42, 46)
            a[i, w - 1 - x - i // 2] = True
            rgb[i, w - 1 - x - i // 2] = C(196, 42, 46)
    a[0, 0] = a[0, w - 1] = True
    rgb[0, 0] = rgb[0, w - 1] = C(244, 196, 64)
    # riders
    for x in range(6, w - 6, 2):
        a[3, x] = True
        rgb[3, x] = C(200, 150, 110)
        a[2, x] = True
        rgb[2, x] = C(30, 26, 26) if rng.random() < 0.7 else C(220, 60, 90)
    # gunwale with bulbs
    a[4, 3:w - 3] = True
    rgb[4, 3:w - 3] = C(244, 196, 64)
    for x in range(4, w - 4, 3):
        em[4, x] = FAIRY[(x // 3) % len(FAIRY)]
        rgb[4, x] = C(250, 250, 240)
    return dict(rgb=rgb, a=a, em=em, ax=21.5, ay=4)


HULL = make_hull()
PIV = (52, 204)


def draw_ship(L, T, lt):
    px_, py_ = PIV
    for lx in (22, 82):
        L.line(px_, py_, lx, HORIZON, C(196, 48, 52))
        L.line(px_ + 1, py_, lx + 1, HORIZON, C(150, 34, 40))
    L.line(30, 262, 74, 262, C(150, 34, 40))
    th = 0.85 * math.sin(2 * np.pi * T / 4.6)
    la = 44
    ax, ay = px_ + la * math.sin(th), py_ + la * math.cos(th)
    ux, uy = math.cos(th), -math.sin(th)  # hull long axis
    for off in (-9, 9):
        L.line(px_, py_, ax + ux * off, ay + uy * off, C(244, 200, 50))
    # rotate hull sprite
    R = 26
    oys, oxs = np.mgrid[int(ay) - R:int(ay) + R, int(ax) - R:int(ax) + R]
    dx, dy = oxs - ax, oys - ay
    c, s = math.cos(th), math.sin(th)
    sx = HULL['ax'] + dx * c - dy * s
    sy = HULL['ay'] + dx * s + dy * c
    sxi, syi = np.floor(sx).astype(int), np.floor(sy).astype(int)
    hh, ww = HULL['a'].shape
    ok = (sxi >= 0) & (sxi < ww) & (syi >= 0) & (syi < hh)
    ok[ok] = HULL['a'][syi[ok], sxi[ok]]
    L.put(oys[ok], oxs[ok], HULL['rgb'][syi[ok], sxi[ok]], em=HULL['em'][syi[ok], sxi[ok]] * lt)
    L.disk(px_, py_, 2.2, C(244, 200, 50))
    L.px(px_, py_, C(90, 60, 30))


def drop_y(T):
    c = T % 7.0
    top, bot = 158, 288
    if c < 3.2:
        u = c / 3.2
        return bot + (top - bot) * (u * u * (3 - 2 * u))
    if c < 4.4:
        return top
    if c < 5.0:
        u = (c - 4.4) / 0.6
        return top + (bot - top) * u * u
    u = (c - 5.0) / 2.0
    return bot - 6 * math.exp(-6 * u) * abs(math.sin(u * 9))


def draw_tower(L, T, lt):
    tx = 260
    L.rect(tx - 2, 150, tx, HORIZON, STEEL)
    L.rect(tx, 150, tx + 2, HORIZON, STEEL_D)
    for y in range(152, HORIZON, 6):
        L.px(tx - 2, y, STEEL_D)
    cy = drop_y(T)
    falling = 4.4 <= T % 7.0 < 5.0
    for i, y in enumerate(range(156, 292, 7)):
        b = 1.0 if falling else (0.2 + 0.8 * (((y - cy) // 7 + int(T * 8)) % 4 == 0))
        L.px(tx + 1, y, C(230, 230, 230), em=(C(255, 60, 80) if falling else C(255, 230, 120)) * b * lt)
    L.rect(tx - 6, 142, tx + 6, 150, C(212, 52, 64))
    L.rect(tx - 6, 142, tx + 6, 143, C(240, 110, 110))
    L.line(tx, 141, tx, 134, STEEL_D)
    L.px(tx, 133, C(200, 40, 40), em=C(255, 40, 40) * (1.0 if int(T * 1.5) % 2 else 0.1))
    y = int(round(cy))
    L.rect(tx - 8, y, tx + 8, y + 4, C(250, 196, 50))
    L.rect(tx - 8, y + 3, tx + 8, y + 4, C(60, 120, 220))
    for x in range(tx - 7, tx + 8, 2):
        L.px(x, y - 1, C(200, 150, 110))
        L.px(x, y - 2, C(30, 26, 26))
        L.px(x, y + 4, C(60, 60, 100))
        L.px(x, y + 5 + (1 if falling else 0), C(40, 40, 60))
    L.put([y + 1] * 4, [tx - 7, tx - 3, tx + 3, tx + 7], C(250, 250, 240), em=FAIRY[int(T * 4) % 5] * lt)


# ---------------------------------------------------------------- sign
SIGN_COLS = [C(255, 70, 90), C(255, 150, 50), C(255, 220, 60), C(90, 230, 110), C(60, 200, 255),
             C(130, 120, 255), C(255, 100, 230)]


def draw_sign(L, T, lt):
    L.line(116, 236, 116, 244, C(60, 60, 70))
    L.line(154, 236, 154, 244, C(60, 60, 70))
    L.rect(108, 221, 163, 237, C(34, 38, 82))
    L.rect(109, 222, 162, 236, C(26, 28, 62))
    per = [(x, 221) for x in range(108, 163, 3)] + [(x, 236) for x in range(108, 163, 3)] + \
          [(108, y) for y in range(224, 236, 3)] + [(162, y) for y in range(224, 236, 3)]
    for i, (x, y) in enumerate(per):
        on = ((i + int(T * 10)) % 3 == 0)
        L.px(x, y, C(240, 220, 160), em=C(255, 210, 90) * (1.0 if on else 0.25) * lt)
    for i, ch in enumerate('JOYLAND'):
        # neon flickers into life one letter at a time, then the odd stutter
        t_on = 10.6 + i * 0.16
        if T < t_on:
            level = 0.0
        elif T < t_on + 0.7:
            level = 1.0 if hash01(i, int(T * FPS)) > 0.45 else 0.08
        else:
            level = 0.15 if (hash01(i, int(T * 6)) > 0.97) else 1.0
        g = FONT5[ch]
        m = np.array([[c == '#' for c in row] for row in g])
        col = SIGN_COLS[i]
        L.mask(114 + i * 6, 226, m, col * 0.9 + 0.1, em=col * 1.25 * level)


# ---------------------------------------------------------------- vehicles & people
K_, WH_, GL_ = C(28, 28, 32), C(150, 150, 160), C(130, 172, 204)
HEAD_L, TAIL_L = (C(255, 244, 210), C(1.0, 0.95, 0.8)), (C(170, 30, 30), C(1.0, 0.18, 0.12))

RICKSHAW = [
    "......KKKKKKKKKKKK....",
    ".....KKKKKKKKKKKKKK...",
    ".....KK.HH....HH.KK...",
    ".....KK.SS....SS..KK..",
    ".....KK.RR....rr...K..",
    ".....GGGRRGGGGrrGGGGG.",
    "....TGGGGGGGGGGGGGGGGL",
    ".....YYYYYYYYYYYYYYYYG",
    ".....GGGGGGGGGGGGGGGG.",
    "......gggggggggggggg..",
    "......KKK.......KKK...",
    ".....KWWWK.....KWWWK..",
    "......KKK.......KKK...",
]
BIKE = [
    ".......HH.......",
    "......HHSS......",
    "......HSSS......",
    ".......SS.......",
    "......RRRR......",
    ".....RRRRRSS....",
    ".....RRRR..SSK..",
    ".....PPPPP..K...",
    "...TBBPPPPBBBK..",
    "...BBBBBBBBBBBBL",
    "..KKK.KWWK..KKK.",
    ".KWWWK.KK..KWWWK",
    "..KKK.......KKK.",
]
CAR = [
    "........BBBBBBBBB.........",
    ".......BCCCCBCCCCCB.......",
    "......BCCCCCBCCCCCCB......",
    "..BBBBBBBBBBBBBBBBBBBBBB..",
    ".TBBBBBBBBBBBBBBBBBBBBBBBL",
    ".BBBBBBBBBBBBBBBBBBBBBBBBB",
    ".bbbbbbbbbbbbbbbbbbbbbbbbb",
    ".bbKKKbbbbbbbbbbbbbKKKbbb.",
    "...KWWWK...........KWWWK..",
    "....KKK.............KKK...",
]


def truck_rows():
    w, h = 40, 19
    g = [['.'] * w for _ in range(h)]

    def fill(x0, x1, y0, y1, ch):
        for y in range(y0, y1):
            for x in range(x0, x1):
                g[y][x] = ch
    # crown above cab
    for x, ch in zip(range(29, 37), 'MOYGNYOM'):
        g[0][x] = ch
    fill(28, 38, 1, 2, 'Y')
    for x in range(29, 38, 2):
        g[1][x] = 'M'
    fill(27, 39, 2, 3, 'O')
    fill(27, 39, 3, 4, 'N')
    # cargo box
    fill(0, 26, 2, 15, 'N')
    fill(1, 25, 3, 4, 'Y')
    for y in range(4, 7):
        for x in range(1, 25):
            g[y][x] = 'MOGR'[((x - 1) // 4 + (y > 5)) % 4]
    fill(1, 25, 7, 8, 'Y')
    for x in range(1, 25):
        g[8][x] = 'w' if x % 2 else 'R'
        g[9][x] = 'R' if x % 2 else 'w'
    fill(1, 25, 10, 13, 'G')
    for x in range(3, 25, 5):
        g[11][x] = 'Y'
    fill(1, 25, 13, 14, 'Y')
    # cab
    fill(26, 39, 4, 15, 'O')
    fill(26, 39, 4, 5, 'N')
    fill(28, 38, 5, 9, 'C')
    fill(28, 38, 9, 10, 'N')
    for x in range(27, 39, 2):
        g[12][x] = 'Y'
    g[11][39] = 'L'
    g[12][0] = 'T'
    fill(26, 40, 15, 16, 'w')
    fill(0, 26, 15, 16, 'K')
    for wx in (3, 11, 30):
        for dy, row in enumerate([".KKKK.", "KWWWWK", "KWWWWK", ".KKKK."]):
            for dx, ch in enumerate(row):
                if ch != '.':
                    g[15 + dy][wx + dx] = ch
    for x in range(18, 26, 2):
        g[16][x] = 'Y'
    return [''.join(r) for r in g]


def vehicle_sprite(kind, rng):
    base = {'K': (K_, None), 'W': (WH_, None), 'w': (C(236, 236, 240), None), 'C': (GL_, None),
            'L': HEAD_L, 'T': TAIL_L, 'S': (C(200, 150, 110), None), 'H': (C(30, 25, 25), None),
            'P': (C(60, 60, 90), None), 'Y': (C(245, 200, 40), None)}
    shirts = [C(240, 240, 236), C(70, 120, 200), C(200, 60, 60), C(80, 160, 110), C(230, 200, 90), C(150, 90, 170)]
    pal = dict(base)
    pal['R'] = (shirts[rng.integers(len(shirts))], None)
    pal['r'] = (shirts[rng.integers(len(shirts))], None)
    if kind == 'rickshaw':
        green = rng.random() < 0.7
        pal['G'] = (C(30, 150, 80) if green else C(50, 110, 200), None)
        pal['g'] = (C(20, 100, 55) if green else C(34, 76, 140), None)
        rows = RICKSHAW
    elif kind == 'bike':
        pal['B'] = ([(C(204, 30, 40), None), (C(40, 40, 46), None), (C(30, 70, 160), None)])[rng.integers(3)]
        if rng.random() < 0.3:
            pal['H'] = (C(230, 230, 230), None)  # helmet
        rows = BIKE
    elif kind == 'car':
        b, bd = [(C(232, 232, 236), C(186, 186, 196)), (C(180, 184, 194), C(138, 142, 152)),
                 (C(200, 40, 40), C(148, 28, 30)), (C(50, 50, 56), C(30, 30, 34)),
                 (C(40, 82, 170), C(28, 56, 120))][rng.integers(5)]
        pal['B'], pal['b'] = (b, None), (bd, None)
        rows = CAR
    else:
        pal.update({'M': (C(220, 40, 140), None), 'O': (C(245, 140, 30), None), 'G': (C(40, 160, 80), None),
                    'N': (C(40, 80, 200), None), 'R': (C(214, 40, 40), None)})
        rows = truck_rows()
    return compile_sprite(rows, pal)


def make_traffic():
    rng = np.random.default_rng(17)
    speeds = {'bike': (44, 58), 'car': (36, 48), 'rickshaw': (28, 36), 'truck': (24, 28)}
    out = []
    for lane, (ybot, d) in enumerate(((405, -1), (434, 1))):
        kinds = ['truck', 'bike', 'car', 'bike', 'rickshaw', 'car', 'bike', 'rickshaw', 'bike', 'car']
        rng.shuffle(kinds)
        period = W + 140
        for i, k in enumerate(kinds):
            spr = vehicle_sprite(k, rng)
            out.append(dict(spr=spr, y=ybot + int(rng.integers(0, 3)) + (2 if k == 'bike' else 0) - spr['h'],
                            v=rng.uniform(*speeds[k]) * d, x0=i * period / len(kinds) + rng.uniform(-8, 8),
                            period=period, flip=d < 0, lane=lane))
    return out


def ped_sprite(rng, frame, scarf):
    legs = [".PP.", ".PP.", ".PP.", ".KK."] if frame == 0 else [".PP.", "P..P", "P..P", "K..K"]
    rows = [("rrr." if scarf else ".HH."), ".SS.", "RRRR", "RRRR", "SRRS", ".RR.", ".RR."] + legs
    shirts = [C(240, 236, 226), C(90, 150, 210), C(200, 70, 90), C(90, 160, 120), C(220, 180, 80)]
    pal = {'H': (C(30, 25, 25), None), 'S': (C(196, 148, 108), None), 'K': (K_, None),
           'R': (shirts[rng.integers(5)], None), 'r': (C(220, 70, 120), None), 'P': (C(230, 226, 214), None)}
    return pal, rows


def make_people():
    out = []
    rng = np.random.default_rng(31)
    for i in range(5):
        st = int(rng.integers(1 << 30))
        scarf = i % 2 == 1
        frames = []
        for f in range(2):
            pal, rows = ped_sprite(np.random.default_rng(st), f, scarf)
            frames.append(compile_sprite(rows, pal))
        d = 1 if i % 2 else -1
        out.append(dict(frames=frames, y=458 + int(rng.integers(0, 12)), v=rng.uniform(9, 15) * d,
                        x0=rng.uniform(0, W), flip=d < 0))
    return out


# ================================================================= RENDER
def blur(a, r):
    k = 2 * r + 1
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode='edge'), axis=ax)
        a = (np.take(c, np.arange(k, c.shape[ax]), axis=ax) - np.take(c, np.arange(0, c.shape[ax] - k), axis=ax)) / k
    return a


class Scene:
    def __init__(self):
        self.city = build_city()
        self.fort = build_fort()
        self.fg = build_foreground()
        self.cones = lamp_cones()
        self.clouds = make_clouds()
        self.fws = make_fireworks()
        self.traffic = make_traffic()
        self.people = make_people()
        s = np.clip(YY / 272, 0, 1) ** 0.9
        self.NQ = 18
        self.skyq = np.clip(np.floor(s * self.NQ + BAY), 0, self.NQ).astype(int)
        self.beam = []
        for ox in (66, 214):
            dx, dy = XX - ox, (HORIZON - 4) - YY
            self.beam.append((np.arctan2(dx, np.maximum(dy, 0.01)), np.hypot(dx, dy), dy > 0))

    def sky(self, base, em, T, amb, nightf):
        top, mid, hor = [np.array(c) / 255 for c in kf(T, SKY_KEYS)]
        lut = np.zeros((self.NQ + 1, 3))
        for q in range(self.NQ + 1):
            s = q / self.NQ
            lut[q] = top + (mid - top) * (s / 0.5) if s < 0.5 else mid + (hor - mid) * ((s - 0.5) / 0.5)
        base[:] = lut[self.skyq]
        em[:] = 0
        # stars
        if nightf > 0:
            for x, y, b, ph, w in STARS:
                v = b * nightf * (0.6 + 0.4 * math.sin(T * w + ph))
                em[y, x] += np.array([0.85, 0.9, 1.0]) * v
                if b > 0.9:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if 0 <= x + dx < W:
                            em[y + dy, x + dx] += np.array([0.6, 0.7, 1.0]) * v * 0.35
        # moon
        ma = float(np.clip((T - 12.5) / 2.0, 0, 1))
        if ma > 0:
            yy, xx = np.mgrid[-9:10, -9:10]
            m = (xx * xx + yy * yy <= 50) & ~((xx - 3) ** 2 + (yy + 2) ** 2 <= 38)
            ys, xs = np.nonzero(m)
            ys, xs = ys + 58 - 9, xs + 222 - 9
            keep = BAY[ys, xs] < ma
            base[ys[keep], xs[keep]] = C(248, 242, 216)
            em[ys[keep], xs[keep]] += C(255, 245, 210) * 0.25
        # sun
        sx, sy = kf(T, [(0, (58, 92)), (4, (58, 104)), (8, (62, 196)), (10.5, (66, 284)), (12, (68, 330))])
        sc = kf(T, [(0, (255, 250, 224)), (5, (255, 246, 210)), (8, (255, 220, 140)), (10.5, (255, 150, 70))]) / 255
        r = 9
        m = (XX - sx) ** 2 + (YY - sy) ** 2 <= r * r + 0.5
        base[m] = sc
        em[m] = sc * 0.35
        if T > 7:
            # pixel bands across the setting sun
            band = m & ((YY - int(sy)) % 5 == 3) & (YY > sy)
            base[band] = lut[np.minimum(self.skyq[band] + 1, self.NQ)]
            em[band] = 0
        # searchlights
        sl = float(np.clip((T - 12.3) / 1.5, 0, 1))
        if sl > 0:
            for i, (ang, dist, up) in enumerate(self.beam):
                a0 = (0.42 if i == 0 else -0.42) * math.sin(T * 0.45 + i * 1.7) + (-0.18 if i == 0 else 0.18)
                inb = up & (np.abs(ang - a0) < 0.045)
                v = np.floor(np.clip(0.11 * (1 - dist / 380), 0, 1) * sl * 50) / 50
                em[inb] += np.array([0.8, 0.86, 1.0]) * v[inb][:, None]
        # clouds
        hl, md, sh = [np.array(c) / 255 for c in kf(T, CLOUD_KEYS)]
        for cl in self.clouds:
            x = int(round((cl['x0'] + cl['v'] * T) % (W + cl['w'] + 20))) - cl['w'] - 10
            s = cl['s']
            ys, xs = np.nonzero(s)
            vals = s[ys, xs]
            ys, xs = ys + cl['y'], xs + x
            ok = (xs >= 0) & (xs < W)
            ys, xs, vals = ys[ok], xs[ok], vals[ok]
            base[ys, xs] = np.array([sh, md, hl])[vals - 1]
            em[ys, xs] = 0
        # kites over the afternoon
        reel = float(kf(T, [(0, 0), (6.5, 0), (10.5, 1)]))
        ka = 1.0
        if reel < 1:
            for i, (kx, ky, ca, cb, anchor) in enumerate([
                    (150, 62, C(230, 40, 70), C(250, 210, 50), (-30, 360)),
                    (232, 92, C(30, 140, 70), C(250, 250, 250), (300, 360)),
                    (104, 112, C(60, 110, 230), C(244, 120, 40), (-30, 360))]):
                x = kx + 4 * math.sin(T * 0.9 + i * 2)
                y = ky + 3 * math.sin(T * 1.3 + i) + reel ** 1.6 * (330 - ky)
                x = x + (anchor[0] - x) * 0.25 * reel
                xi, yi = int(round(x)), int(round(y))
                n = int(max(abs(anchor[0] - x), abs(anchor[1] - y)))
                for j in range(0, n, 2):
                    u = j / n
                    px_ = int(round(x + (anchor[0] - x) * u))
                    py_ = int(round(y + 3 + (anchor[1] - y) * u + 18 * u * (1 - u)))
                    if 0 <= px_ < W and 0 <= py_ < H:
                        base[py_, px_] = base[py_, px_] * 0.6 + 0.4 * C(240, 240, 240) * amb
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        if abs(dx) + abs(dy) <= 4 and BAY[(yi + dy) % H, (xi + dx) % W] < ka:
                            c = ca if dy < 0 else cb
                            if dx == 0 or dy == 0:
                                c = c * 0.8
                            if 0 <= yi + dy < H and 0 <= xi + dx < W:
                                base[yi + dy, xi + dx] = c * amb
                for j, (dx, dy) in enumerate(((0, 5), (1, 6), (0, 7), (-1, 8))):
                    if 0 <= xi + dx < W and BAY[yi + dy, xi + dx] < ka:
                        base[yi + dy, xi + dx] = (ca if j % 2 else cb) * amb
        # birds at sunset
        if 6.5 < T < 13:
            for i in range(7):
                bx = W + 10 - (T - 6.5) * 46 + abs(i - 3) * 7 + i * 1.5
                by = 138 + abs(i - 3) * 4 - (T - 6.5) * 3
                flap = (int(T * 8) + i) % 2
                pts = ((-1, 0), (0, 1), (1, 0)) if flap else ((-1, 1), (0, 1), (1, 1), (-2, 0), (2, 0))
                for dx, dy in pts:
                    x, y = int(round(bx + dx)), int(round(by + dy))
                    if 0 <= x < W and 0 <= y < H:
                        base[y, x] = C(40, 30, 50)
        flash = draw_fireworks(em, self.fws, T)
        em[:HORIZON] += flash * (1 - YY[:HORIZON, :, None] / HORIZON * 0.5)

    def render(self, T):
        amb = kf(T, AMB_KEYS)
        lt = lights_at(T)
        nf = night_at(T)
        base = np.zeros((H, W, 3))
        em = np.zeros((H, W, 3))
        self.sky(base, em, T, amb, nf)
        composite(base, em, self.city, amb, lt, T)

        rides = Layer()
        draw_tower(rides, T, lt)
        draw_ship(rides, T, lt)
        draw_wheel(rides, T, lt)
        composite(base, em, rides, amb, lt, T)

        composite(base, em, self.fort, amb, lt, T, up_col=C(255, 150, 80) * 0.55 * lt)
        sign = Layer()
        draw_sign(sign, T, lt)
        composite(base, em, sign, amb, lt, T)

        cars = Layer()
        beams = np.zeros((H, W))
        for v in sorted(self.traffic, key=lambda v: v['y'] + v['spr']['h']):
            x = int(round((v['x0'] + v['v'] * T) % v['period'])) - 70
            cars.blit(v['spr'], x, v['y'], flip=v['flip'], em_scale=lt)
            if lt > 0:
                spr = v['spr']
                li = np.nonzero(np.all(spr['em'] == HEAD_L[1], axis=1))[0]
                for j in li:
                    hx = (spr['w'] - 1 - spr['xs'][j]) if v['flip'] else spr['xs'][j]
                    hx, hy = hx + x, spr['ys'][j] + v['y']
                    d = -1 if v['flip'] else 1
                    for k in range(1, 30):
                        hw = 1 + k // 6
                        xx = hx + d * k
                        if 0 <= xx < W:
                            beams[max(hy - hw, 0):hy + hw + 2, xx] += 0.2 * (1 - k / 30)
        composite(base, em, cars, amb, lt, T)
        em += np.floor(np.minimum(beams, 0.3) * 30)[:, :, None] / 30 * C(255, 240, 200) * lt

        ppl = Layer()
        for p in self.people:
            x = (p['x0'] + p['v'] * T) % (W + 20) - 10
            fr = p['frames'][int(T * 4 + p['x0']) % 2]
            ppl.blit(fr, int(round(x)), p['y'] - fr['h'], flip=p['flip'])
        # balloon seller
        bx, by = 150, 470
        pal, rows = ped_sprite(np.random.default_rng(2), 0, False)
        ppl.blit(compile_sprite(rows, pal), bx, by - 11)
        for i, (dx, dy, col) in enumerate(((-5, -24, C(240, 60, 80)), (0, -27, C(250, 210, 60)),
                                           (5, -24, C(70, 170, 240)), (-2, -20, C(120, 220, 120)),
                                           (3, -20, C(240, 120, 220)), (8, -18, C(250, 150, 60)))):
            yb = by - 11 + dy + int(round(math.sin(T * 2.2 + i)))
            ppl.line(bx + 3, by - 7, bx + 1 + dx, yb + 2, C(210, 210, 210))
            ppl.disk(bx + 1 + dx, yb, 2.7, col)
            ppl.px(bx + dx, yb - 1, col * 0.5 + 0.5)
            ppl.px(bx + 1 + dx, yb + 3, col * 0.8)
        composite(base, em, ppl, amb, lt, T)

        composite(base, em, self.fg, amb, lt, T)
        em += self.cones[:, :, None] * C(255, 214, 150) * lt

        g = blur(blur(em, 2), 2) * 0.9 + blur(blur(em, 7), 7) * 0.45
        g = np.floor(g * 24) / 24
        out = np.clip(base + em + g, 0, 1)
        return (out * 255 + 0.5).astype(np.uint8)


# ================================================================= AUDIO
def synth_audio(path, fws, sr=44100):
    n = int(DUR * sr)
    mix = np.zeros(n)
    bpm = 96
    beat = 60 / bpm
    bar = beat * 4
    rng = np.random.default_rng(1)

    def note(f, t0, dur, vol, kind='sq', duty=0.25, decay=3.0):
        i0 = int(t0 * sr)
        m = min(int(dur * sr), n - i0)
        if m <= 0:
            return
        t = np.arange(m) / sr
        vib = 1 + (0.004 * np.sin(2 * np.pi * 5.5 * t) * (t > 0.15) if kind == 'lead' else 0)
        ph = np.cumsum(np.full(m, f) * vib) / sr % 1
        if kind == 'tri':
            w = 4 * np.abs(ph - 0.5) - 1
        else:
            w = np.where(ph < (0.5 if kind == 'lead' else duty), 1.0, -1.0)
        env = np.minimum(t / 0.005, 1) * np.exp(-t * decay) * np.minimum((dur - t) / 0.02, 1)
        mix[i0:i0 + m] += w * env * vol

    def midi(m):
        return 440 * 2 ** ((m - 69) / 12)

    chords = [[57, 60, 64, 69], [53, 57, 60, 65], [55, 60, 64, 67], [55, 59, 62, 67]]
    roots = [45, 41, 48, 43]
    nbars = int(math.ceil(DUR / bar))
    for b in range(nbars):
        ch = chords[b % 4]
        for e, idx in enumerate([0, 1, 2, 3, 2, 1, 2, 3]):
            note(midi(ch[idx] + 12), b * bar + e * beat / 2, beat / 2, 0.05, 'sq', 0.125, 6)
        for k in range(4):
            note(midi(roots[b % 4]), b * bar + k * beat, beat * 0.9, 0.16, 'tri', decay=1.5)
    melA = [[(0, 76, 1), (1, 74, .5), (1.5, 72, .5), (2, 69, 2)],
            [(0, 72, 1), (1, 69, .5), (1.5, 72, .5), (2, 77, 1.5), (3.5, 76, .5)],
            [(0, 76, 1), (1, 79, 1), (2, 76, 1), (3, 74, 1)],
            [(0, 74, 1.5), (1.5, 71, .5), (2, 67, 2)]]
    melB = [[(0, 81, 1), (1, 79, .5), (1.5, 76, .5), (2, 72, 1), (3, 76, 1)],
            [(0, 77, 1), (1, 76, .5), (1.5, 74, .5), (2, 72, 2)],
            [(0, 72, 1), (1, 74, 1), (2, 76, 1), (3, 79, 1)],
            [(0, 79, 1), (1, 77, .5), (1.5, 76, .5), (2, 74, 1), (3, 71, 1)]]
    for b in range(2, 10):
        for st, m, d in (melA if b < 6 else melB)[(b - 2) % 4]:
            note(midi(m), b * bar + st * beat, d * beat * 0.95, 0.07, 'lead', decay=1.2)
    note(midi(69), 10 * bar, 4 * beat, 0.08, 'lead', decay=0.6)
    # drums from bar 6 (nightfall)
    for b in range(6, 10):
        for k in range(8):
            t0 = b * bar + k * beat / 2
            i0 = int(t0 * sr)
            if k % 4 == 0:
                m = int(0.25 * sr)
                t = np.arange(m) / sr
                mix[i0:i0 + m] += np.sin(2 * np.pi * (50 + 90 * np.exp(-t * 30)) * t) * np.exp(-t * 14) * 0.28
            if k % 4 == 2:
                m = int(0.18 * sr)
                t = np.arange(m) / sr
                mix[i0:i0 + m] += rng.uniform(-1, 1, m) * np.exp(-t * 22) * 0.09
            m = int(0.04 * sr)
            mix[i0:i0 + m] += rng.uniform(-1, 1, m) * np.exp(-np.arange(m) / sr * 90) * 0.035
    # fireworks: soft launch whistles + booms on burst
    for f in fws:
        t0 = f['tl']
        m = int(f['rise'] * sr)
        i0 = int(t0 * sr)
        if i0 + m < n:
            t = np.arange(m) / sr
            fr = 900 + 1400 * t / f['rise']
            mix[i0:i0 + m] += np.sin(2 * np.pi * np.cumsum(fr) / sr) * 0.012 * np.minimum(t / 0.1, 1)
        tb = f['tl'] + f['rise']
        i0 = int(tb * sr)
        m = min(int(1.4 * sr), n - i0)
        if m <= 0:
            continue
        t = np.arange(m) / sr
        nz = rng.uniform(-1, 1, m)
        k = np.ones(60) / 60
        low = np.convolve(np.convolve(nz, k, 'same'), k, 'same') * 6
        mix[i0:i0 + m] += low * np.exp(-t * 5) * 0.32
        if f['crackle'] or f['trail'] > 3:
            cr = (rng.random(m) < 0.004) * rng.uniform(-1, 1, m)
            cr = np.convolve(cr, np.exp(-np.arange(80) / 12), 'same')
            mix[i0:i0 + m] += cr * np.clip((t - 0.4) / 0.3, 0, 1) * np.exp(-t * 1.6) * 0.07
    t = np.arange(n) / sr
    mix *= np.minimum(t / 0.6, 1) * np.clip((DUR - t) / 1.8, 0, 1)
    mix = np.tanh(mix * 1.4) * 0.8
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
            Image.fromarray(img).resize((W * SCALE, H * SCALE), Image.NEAREST).save(
                os.path.join(outdir, f'preview_{float(s):05.1f}.png'))
        return
    out = args[0] if args else 'joyland_fortress_pixel.mp4'
    wav = os.path.splitext(out)[0] + '.wav'
    synth_audio(wav, scene.fws)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-i', wav,
           '-vf', f'scale={W * SCALE}:{H * SCALE}:flags=neighbor,format=yuv420p',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-tune', 'animation',
           '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(NF):
        p.stdin.write(scene.render(i / FPS).tobytes())
        if i % 60 == 0:
            print(f'frame {i}/{NF}', flush=True)
    p.stdin.close()
    p.wait()
    os.remove(wav)
    print('wrote', out)


if __name__ == '__main__':
    main()
