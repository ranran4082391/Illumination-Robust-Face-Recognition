"""Physics-based face relighting via Spherical Harmonics (SH, L=2) + fixed
ellipsoid normal template.

Why SH instead of a learned relighting net: no extra onnx face-relight model
was reliably available offline on this box; Ramamoorthi 2001 diffuse irradiance
on a fixed ellipsoid normal map gives natural shadow relief across the nose
bridge / cheekbones / eye sockets for aligned frontal faces (LFW crop), which
is exactly the upgrade from the previous hard-mask "yin-yang" split.

Normal template
---------------
112x112, centered, anisotropic ellipsoid (slightly taller than wide):
    nx = x
    ny = y * 1.15
    r2 = nx^2 + ny^2
    inside: nz = sqrt(max(0, 1 - r2)) * 0.85   (upper hemisphere)
    outside (hair/background halo): clamped to shallow horizontal normals
Normals are L2-normalised afterwards.

SH irradiance (Ramamoorthi 2001)
--------------------------------
Real SH basis Y_k (k=0..8) evaluated on each pixel normal n.
Environment radiance L_lm of a delta point light of intensity I along
direction l is L_lm = I * Y_lm(l). Irradiance at n:
    E(n) = sum_k A_l(k) * L_k * Y_k(n)
with A_0=pi, A_1=2pi/3, A_2=pi/4.
A weak fill light from the mirrored opposite direction + ambient floor keep
the shadow side from clipping to black.

Public API
----------
    make_normal_map(size=112) -> (nx, ny, nz) float32 arrays
    sample_light(rng) -> dict with direction, intensity, ambient, prompt text
    relight(img_rgb_uint8, light) -> uint8 image
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 112

# ---------------------------------------------------------------------------
# Normal template (built once, cached)
# ---------------------------------------------------------------------------
_NORMAL_CACHE: dict[str, np.ndarray] = {}


def make_normal_map(size: int = SIZE) -> np.ndarray:
    """Return (size, size, 3) unit normals pointing out of the ellipsoid."""
    key = f"n{size}"
    if key in _NORMAL_CACHE:
        return _NORMAL_CACHE[key]

    ax = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    ay = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    gx, gy = np.meshgrid(ax, ay)  # note: meshgrid with default ij-like indexing
    # We want row=y, col=x. meshgrid(ax, ay) -> gx varies along axis1, gy along axis0.
    # That's exactly what we want: gx[i,j] = ax[j], gy[i,j] = ay[i].

    nx = gx.copy()
    ny = gy * 1.15  # ellipsoid taller than wide
    r2 = nx * nx + ny * ny
    inside = r2 <= 1.0
    nz = np.zeros_like(nx)
    nz[inside] = np.sqrt(np.clip(1.0 - r2[inside], 0.0, 1.0)) * 0.85
    # outside the silhouette: shallow horizontal normals (hair / background)
    nz[~inside] = 0.05
    # slight vertical asymmetry: forehead bulges a touch more than chin
    nz = nz * (1.0 - 0.15 * gy)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    _NORMAL_CACHE[key] = n.astype(np.float32)
    return _NORMAL_CACHE[key]


# ---------------------------------------------------------------------------
# SH basis (Ramamoorthi 2001, 9 real basis functions)
# ---------------------------------------------------------------------------
def sh_basis(n: np.ndarray) -> np.ndarray:
    """n: (...,3) unit normals. Returns (...,9) basis evaluations Y00..Y22."""
    nx = n[..., 0]
    ny = n[..., 1]
    nz = n[..., 2]
    Y = np.stack(
        [
            0.282095 * np.ones_like(nx),               # Y00
            0.488603 * ny,                             # Y1,-1
            0.488603 * nz,                             # Y1,0
            0.488603 * nx,                             # Y1,1
            1.092548 * nx * ny,                        # Y2,-2
            1.092548 * ny * nz,                        # Y2,-1
            0.315392 * (3.0 * nz * nz - 1.0),         # Y2,0
            1.092548 * nx * nz,                        # Y2,1
            0.546274 * (nx * nx - ny * ny),           # Y2,2
        ],
        axis=-1,
    )
    return Y.astype(np.float32)


# irradiance convolution factors A_l for l=0,1,2
_A = np.array(
    [np.pi, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3,
     np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4],
    dtype=np.float32,
)


def _sh_coeffs_for_direction(lx: float, ly: float, lz: float) -> np.ndarray:
    """Radiance SH coefficients L_k for a delta light of unit intensity along l."""
    n = np.array([[[lx, ly, lz]]], dtype=np.float32)
    return sh_basis(n)[0, 0]  # (9,)


# ---------------------------------------------------------------------------
# Light sampling
# ---------------------------------------------------------------------------
def _hash_seed(name: str) -> int:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def sample_light(seed_name: str, variant: int) -> dict:
    """Deterministic per (image name, variant) light setup.

    Horizontal component is biased large so the split is a half-face relight;
    vertical tilt randomises upper/lower key. Returns dict with direction,
    intensity, ambient, and an English prompt string for the Qwen perceptor.
    """
    rng = np.random.default_rng(_hash_seed(f"{seed_name}#{variant}"))

    # horizontal key: strong |lx| for half-face lighting
    side = rng.choice([-1.0, 1.0])
    lx = side * rng.uniform(0.55, 0.92)
    # vertical tilt: mostly upper key, sometimes lower
    ly = rng.uniform(-0.45, 0.55)
    lz = np.sqrt(max(0.05, 1.0 - lx * lx - ly * ly))
    l = np.array([lx, ly, lz], dtype=np.float32)
    l /= np.linalg.norm(l)

    intensity = float(rng.uniform(1.05, 1.6))
    ambient = float(rng.uniform(0.06, 0.16))
    # weak fill from the opposite horizontal side, still upper hemisphere
    fill_l = np.array([-l[0] * 0.7, -l[1] * 0.3, abs(l[2])], dtype=np.float32)
    fill_l /= np.linalg.norm(fill_l)
    fill_intensity = float(rng.uniform(0.05, 0.12))

    # ---- natural-language prompt (discretised for a small Qwen cache) ----
    # Image frame: +x = right, +y = DOWN (gy=-1 at top row). l points TOWARD light.
    horiz = "right" if l[0] > 0 else "left"   # lx>0 -> light comes from image right
    opp = "left" if horiz == "right" else "right"
    if l[1] < -0.25:
        vert = "upper"
    elif l[1] > 0.25:
        vert = "lower"
    else:
        vert = "mid-level"
    strength = "strong" if abs(l[0]) > 0.75 else "moderate"
    if vert == "mid-level":
        prompt = (f"dominant light from the {horiz} side, {strength} side contrast, "
                  f"soft gradient shadows falling on the {opp} cheek and eye socket")
    else:
        prompt = (f"dominant light from the {vert}-{horiz}, {strength} side contrast, "
                  f"soft gradient shadows falling on the {opp} cheek and eye socket")

    return {
        "dir": l,
        "fill_dir": fill_l,
        "intensity": intensity,
        "fill_intensity": fill_intensity,
        "ambient": ambient,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Relighting
# ---------------------------------------------------------------------------
def relight(img_rgb_uint8: np.ndarray, light: dict, size: int = SIZE) -> np.ndarray:
    """Relight an RGB uint8 image (already 112x112) with the given light.

    Returns uint8 RGB. albedo = input image; outgoing = albedo * irradiance.
    """
    assert img_rgb_uint8.shape[:2] == (size, size), img_rgb_uint8.shape
    n = make_normal_map(size)                       # (H,W,3)
    Y = sh_basis(n)                                 # (H,W,9)

    L_main = _sh_coeffs_for_direction(*light["dir"]) * light["intensity"]
    L_fill = _sh_coeffs_for_direction(*light["fill_dir"]) * light["fill_intensity"]
    Lk = L_main + L_fill                            # (9,)

    # irradiance per pixel: E = sum_k A_k L_k Y_k(n)
    E = (Y * (_A[None, None, :] * Lk[None, None, :])).sum(axis=-1)  # (H,W)
    E = np.clip(E, 0.0, None)
    lighting = E + light["ambient"]                  # add ambient floor

    # Normalise so the 85th-percentile lit pixel maps to ~1.25x original.
    # This keeps exposure comparable to the orig while preserving the
    # lit/shadow ratio across the face (the whole image is not underexposed).
    p85 = float(np.percentile(lighting, 85)) + 1e-6
    lighting = lighting / p85 * 1.25

    albedo = img_rgb_uint8.astype(np.float32) / 255.0
    out = albedo * lighting[..., None]
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLI: generate relit variants for every orig image referenced in meta.json
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Quick sanity render saved to .tmp for visual inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = make_normal_map()
    print("normal map range:", n.min(), n.max(), "shape", n.shape)
    print("center normal:", n[56, 56], " edge normal:", n[5, 5])

    root = Path(__file__).resolve().parent.parent
    src = root / "data/train/00_Ariel_Sharon/Ariel_Sharon_0001_orig.jpg"
    img = np.array(Image.open(src).convert("RGB").resize((SIZE, SIZE)))

    rows = [img]
    for v in (1, 2, 3):
        light = sample_light("Ariel_Sharon_0001", v)
        out = relight(img, light)
        print(f"variant {v}: dir={light['dir']}, prompt='{light['prompt']}'")
        rows.append(out)
    combo = np.concatenate(rows, axis=1)
    Image.fromarray(combo).save(root / "figures/_relight_selftest.png")
    print("saved figures/_relight_selftest.png")


if __name__ == "__main__":
    _self_test()
