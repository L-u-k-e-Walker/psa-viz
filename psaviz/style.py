"""Shared look for both visuals: theme tokens, species colours, loading ramps.

Colours follow the dataviz reference palette (categorical slots 1-2, chart
surfaces and ink). A (heavy, extract) = orange, B (light, raffinate) = blue.
The blue/orange pair is documented as passing the all-pairs CVD and
normal-vision checks in both modes (the validator script needs node, which
is not installed here).
"""

from __future__ import annotations

import numpy as np

THEMES = {
    "white": dict(
        bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
        A="#eb6834", B="#2a78d6", bare="#d8d4cb", metal="#9a9890", glass="#b9c6d2",
        highlight="#f0efec",
    ),
    "dark": dict(
        bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
        A="#d95926", B="#3987e5", bare="#4a4945", metal="#6d6b66", glass="#5d6b78",
        highlight="#2c2c2a",
    ),
}

SPECIES_LABEL = {
    "A": "A  heavy · extract",
    "B": "B  light · raffinate",
}


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def rgb_hex(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


# --- OKLab (Björn Ottosson) for perceptually even ramps
def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gam(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def rgb_to_oklab(rgb):
    lms = _lin(rgb) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_rgb(lab):
    lms = (np.asarray(lab) @ np.linalg.inv(_M2).T) ** 3
    return np.clip(_gam(lms @ np.linalg.inv(_M1).T), 0, 1)


def ramp(c0, c1, n=256):
    """n colours from c0 to c1, interpolated in OKLab. Returns (n, 3) RGB."""
    a, b = rgb_to_oklab(hex_rgb(c0)), rgb_to_oklab(hex_rgb(c1))
    t = np.linspace(0, 1, n)[:, None]
    return oklab_to_rgb(a + (b - a) * t)


def darken(h, f=0.18):
    lab = rgb_to_oklab(hex_rgb(h))
    lab[0] *= 1 - f
    return rgb_hex(oklab_to_rgb(lab))


def loading_ramp(theme, species, n=256, deep=False):
    """Sequential one-hue ramp for a species' loading: bare sorbent -> species colour.

    ``deep`` ends on a darker step of the hue, so that molecules drawn in the
    species colour stay visible in front of fully loaded particles."""
    t = THEMES[theme]
    f = (0.30 if theme == "white" else 0.22) if deep else (0.08 if theme == "white" else 0.0)
    return ramp(t["bare"], darken(t[species], f), n)


def apply_ramp(r, x):
    """Map x in [0, 1] (array) through ramp r (n, 3)."""
    x = np.clip(np.asarray(x, float), 0, 1)
    idx = x * (len(r) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(r) - 1)
    w = (idx - i0)[..., None]
    return r[i0] * (1 - w) + r[i1] * w


def mpl_rc(theme):
    t = THEMES[theme]
    return {
        "font.family": "DejaVu Sans", "font.size": 15,
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.facecolor": t["bg"],
        "figure.facecolor": t["bg"], "savefig.facecolor": t["bg"], "axes.grid": True,
        "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlecolor": t["ink"],
        "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.frameon": False,
    }
