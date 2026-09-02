"""Procedural textures for scene decoration.

Generates (cached on disk, all purely visual, no physics impact):
- assets/sky/skybox.png       cube-map sky (gradient + clouds + sun glow) laid out
                              in a 3x4 grid for MuJoCo gridsize="3 4" gridlayout="..U.LFRB.D.."
- assets/textures/court_*.png grayscale speckle/blotch textures tinted by material rgba
"""

import os

import numpy as np

import imageio.v2 as imageio

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

FACE = 512

ZENITH = np.array([0.13, 0.31, 0.65])
HORIZON = np.array([0.72, 0.80, 0.87])
GROUND_HAZE = np.array([0.28, 0.35, 0.29])
NADIR = np.array([0.13, 0.17, 0.14])
CLOUD = np.array([0.99, 0.99, 1.00])
SUN_WARM = np.array([1.00, 0.80, 0.52])

SUN_U, SUN_V = 0.66, 0.36


def _bilinear_resize(a, h, w):
    hh, ww = a.shape
    yi = np.linspace(0.0, hh - 1.0, h)
    xi = np.linspace(0.0, ww - 1.0, w)
    y0 = np.floor(yi).astype(int)
    y1 = np.minimum(y0 + 1, hh - 1)
    x0 = np.floor(xi).astype(int)
    x1 = np.minimum(x0 + 1, ww - 1)
    wy = (yi - y0)[:, None]
    wx = (xi - x0)[None, :]
    return (a[np.ix_(y0, x0)] * (1.0 - wy) * (1.0 - wx)
            + a[np.ix_(y0, x1)] * (1.0 - wy) * wx
            + a[np.ix_(y1, x0)] * wy * (1.0 - wx)
            + a[np.ix_(y1, x1)] * wy * wx)


def _fbm(h, w, seed, octaves=5, freq=(3.0, 6.0)):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w))
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        gh = max(2, int(freq[0] * (2 ** o)))
        gw = max(2, int(freq[1] * (2 ** o)))
        acc += amp * _bilinear_resize(rng.random((gh, gw)), h, w)
        total += amp
        amp *= 0.55
    return acc / total


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _side_strip(seed):
    """Continuous 4-face horizontal strip (h, 4w, 3) for the L,F,R,B faces."""
    h = FACE
    w4 = FACE * 4
    n = _fbm(h, w4, seed, octaves=5, freq=(2.5, 5.0))
    v = np.linspace(0.0, 1.0, h)[:, None]
    u = np.tile(np.linspace(0.0, 1.0, w4), (h, 1))
    window = (_smoothstep(0.02, 0.11, v) * (1.0 - _smoothstep(0.40, 0.50, v)))
    alpha = np.clip((n - 0.47) * 3.6, 0.0, 1.0) ** 0.9 * window
    t_up = np.clip(v / 0.5, 0.0, 1.0)
    sky = ZENITH[None, None, :] + (HORIZON - ZENITH)[None, None, :] * (t_up ** 1.4)[:, :, None]
    t_dn = np.clip((v - 0.5) / 0.5, 0.0, 1.0)
    gnd = GROUND_HAZE[None, None, :] + (NADIR - GROUND_HAZE)[None, None, :] * (t_dn ** 0.8)[:, :, None]
    col = np.where((v < 0.5)[:, :, None], sky, gnd) * np.ones((h, w4, 3))
    warm = 0.30 * np.exp(-(((u - (1.0 + SUN_U) / 4.0) ** 2) + ((v - SUN_V) ** 2)) / 0.16)
    col = col + warm[:, :, None] * np.array([0.30, 0.12, -0.05])[None, None, :]
    cloud_col = CLOUD[None, None, :] + (SUN_WARM * 0.25 - CLOUD * 0.25)[None, None, :] * \
        np.exp(-(((u - (1.0 + SUN_U) / 4.0) ** 2) + ((v - SUN_V) ** 2)) / 0.22)[:, :, None]
    col = col * (1.0 - alpha[:, :, None]) + cloud_col * alpha[:, :, None]
    return np.clip(col, 0.0, 1.0)


def _add_sun(face):
    h = FACE
    u = np.tile(np.linspace(0.0, 1.0, FACE), (h, 1))
    v = np.linspace(0.0, 1.0, h)[:, None] * np.ones((1, FACE))
    d = np.sqrt((u - SUN_U) ** 2 + (v - SUN_V) ** 2)
    above = 1.0 - _smoothstep(SUN_V + 0.02, 0.5, v)
    glow = (0.35 * np.exp(-(d / 0.12) ** 2) + 0.75 * np.exp(-(d / 0.045) ** 2)) * above
    face = face + glow[:, :, None] * SUN_WARM[None, None, :]
    face[(d < 0.020) & (v <= 0.5)] = np.array([1.30, 1.22, 1.02])
    return np.clip(face, 0.0, 1.0)


def build_skybox():
    strip = _side_strip(11)
    faces = {"L": strip[:, 0:FACE], "F": _add_sun(strip[:, FACE:2 * FACE]),
             "R": strip[:, 2 * FACE:3 * FACE], "B": strip[:, 3 * FACE:4 * FACE]}
    zenith = np.tile(ZENITH, (FACE, FACE, 1))
    cloud_up = _fbm(FACE, FACE, 11, octaves=5, freq=(2.5, 5.0))
    uu = np.linspace(0.0, 1.0, FACE)[None, :]
    vv = np.linspace(0.0, 1.0, FACE)[:, None]
    edge = np.clip(uu * 4.0, 0, 1) * np.clip((1.0 - uu) * 4.0, 0, 1) * \
        np.clip(vv * 4.0, 0, 1) * np.clip((1.0 - vv) * 4.0, 0, 1)
    a_up = (np.clip((cloud_up - 0.47) * 3.6, 0, 1) ** 0.9) * np.minimum(edge, 1.0) * 0.85
    zenith = zenith * (1.0 - a_up[:, :, None]) + (CLOUD * 0.97)[None, None, :] * a_up[:, :, None]
    nadir = np.tile(NADIR, (FACE, FACE, 1))
    faces["U"] = zenith
    faces["D"] = nadir
    grid = np.empty((FACE * 3, FACE * 4, 3))
    layout = [[".", ".", "U", "."], ["L", "F", "R", "B"], [".", "D", ".", "."]]
    for r in range(3):
        for c in range(4):
            key = layout[r][c]
            if key == ".":
                continue
            grid[r * FACE:(r + 1) * FACE, c * FACE:(c + 1) * FACE] = faces[key]
    return (np.clip(grid, 0.0, 1.0) * 255.0).astype(np.uint8)


def ensure_skybox():
    out_dir = os.path.join(ASSETS_DIR, "sky")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "skybox.png")
    if not os.path.exists(path):
        imageio.imwrite(path, build_skybox())
    return path


def build_surface_texture(seed, base=0.82, speckle=0.055, blotch=0.05, size=512):
    rng = np.random.default_rng(seed)
    grain = rng.random((size, size))
    blot = _fbm(size, size, seed + 1, octaves=4, freq=(4.0, 4.0))
    g = base + speckle * (grain - 0.5) * 2.0 + blotch * (blot - 0.5) * 2.0
    return (np.clip(g, 0.55, 0.97) * 255.0).astype(np.uint8)


def ensure_surface_texture(name, seed, **kw):
    out_dir = os.path.join(ASSETS_DIR, "textures")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        imageio.imwrite(path, build_surface_texture(seed, **kw))
    return path


if __name__ == "__main__":
    print(ensure_skybox())
    print(ensure_surface_texture("court_surface.png", 3))
    print(ensure_surface_texture("court_apron.png", 5))
    print(ensure_surface_texture("court_ground.png", 9))
