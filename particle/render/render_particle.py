"""Render the particle zoom: equilibrium- vs kinetic-selective particle (env viz).

usage:
  python render_particle.py --stills            key stills for both themes
  python render_particle.py --all --theme dark  every frame -> frames_<theme>/, then MP4 + GIF
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
from scene_particle import CUT_HALF, HERE, PHI_C, THETA_MAX, ParticleScene  # noqa: E402
from psaviz.cases import equilibrium_particle, kinetic_particle  # noqa: E402
from psaviz.style import SPECIES_LABEL, THEMES, loading_ramp, mpl_rc  # noqa: E402

W, H = 1920, 1080
TAU = 1.14  # s, A uptake half-time of the equilibrium particle (kinetic: 1.05 s)
TITLES = [("Equilibrium-selective", "A held more strongly · equal diffusion"),
          ("Kinetic-selective", "A diffuses 100x faster · equal affinity")]


class ParticleData:
    def __init__(self, times):
        self.t = np.asarray(times, float)
        tt = np.unique(np.concatenate([[0.0], self.t[self.t > 0], np.geomspace(1e-3, 3e4, 300)]))
        self.prof = []
        self.curves = []
        for f in (equilibrium_particle, kinetic_particle):
            p = f()
            X = p.solve(tt).y
            if hasattr(p, "eps"):
                _, q, _ = p.split(X)
                th = q / p.qs[:, None, None]
            else:
                th, _ = p.split(X)
            self.prof.append((p.g.rc / p.g.R, th))
            self.curves.append(p.mean_loading(X) / p.qs[:, None])
        self.tt = tt

    def at(self, t):
        j = int(np.argmin(np.abs(self.tt - t)))
        return [dict(r=r, thA=th[0, :, j], thB=th[1, :, j]) for r, th in self.prof], j


class Composer:
    def __init__(self, theme, scene: ParticleScene, data: ParticleData):
        self.theme, self.T, self.sc, self.d = theme, THEMES[theme], scene, data
        self.cx = [scene.project(c)[0] for c in scene.centres]
        self.ramps = {k: loading_ramp(theme, k) for k in ("A", "B")}
        self.lt = np.log10(np.maximum(data.tt, 1e-3) / TAU)

    def draw(self, img, t, j, out):
        T = self.T
        with plt.rc_context(mpl_rc(self.theme)):
            fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img, interpolation="lanczos")
            ax.set_axis_off()
            for k, (title, sub) in enumerate(TITLES):
                x = self.cx[k] / W
                fig.text(x, 1 - 36 / H, title, ha="center", va="top", fontsize=26, fontweight="bold", color=T["ink"])
                fig.text(x, 1 - 76 / H, sub, ha="center", va="top", fontsize=16, color=T["ink2"])
                self._tags(fig, k)
                self._uptake(fig, k, t, j)
            # clock
            ttxt = "0" if t <= 0 else (f"{t / TAU:.2g}" if t / TAU < 1000 else f"{t / TAU:.1e}".replace("e+0", "e"))
            fig.text(0.5, 1 - 40 / H, f"t / τ$_A$ = {ttxt}", ha="center", va="top", fontsize=19,
                     color=T["ink"], family="DejaVu Sans Mono")
            self._legend(fig)
            fig.text(24 / W, 18 / H, "Spherical diffusion models: macropore diffusion with near-local equilibrium (left), "
                     "Maxwell–Stefan micropore diffusion (right). Generic species, illustrative inputs; molecules not to scale.\n"
                     "Same feed step and clock for both: clean particles, gas switched to the column feed (30 % A) at t = 0. "
                     "τ$_A$ = 1.1 s, the A uptake half-time (about the same in both particles).",
                     fontsize=11.5, color=T["muted"], va="bottom", linespacing=1.45)
            fig.savefig(out, dpi=100)
            plt.close(fig)

    def _tags(self, fig, k):
        T = self.T
        c = self.sc.centres[k]
        for key, phi in (("A", PHI_C - CUT_HALF), ("B", PHI_C + CUT_HALF)):
            e = np.array([np.cos(phi), np.sin(phi), 0.0])
            x, y = self.sc.project(c + 1.08 * e + np.array([0, 0, -0.98]))
            fig.text(x / W, 1 - y / H, f"{key} loading", ha="center", va="center", fontsize=13.5, color=T["ink"],
                     bbox=dict(boxstyle="round,pad=0.25", fc=T["bg"], ec=T[key], lw=1.6, alpha=0.9))

    def _uptake(self, fig, k, t, j):
        T = self.T
        x0 = (self.cx[k] - 250) / W
        ax = fig.add_axes([x0, 140 / H, 480 / W, 140 / H])
        cur = self.d.curves[k]
        for i, key in enumerate(("A", "B")):
            ax.plot(self.lt, cur[i], color=T[key], lw=1.5, alpha=0.28)
            ax.plot(self.lt[:j + 1], cur[i, :j + 1], color=T[key], lw=2.6)
            ax.plot([self.lt[j]], [cur[i, j]], "o", ms=8, color=T[key], mec=T["bg"], mew=1.5)
            ax.text(self.lt[-1] + 0.08, cur[i, -1], f"$\\bar q_{key}$", color=T["ink"], fontsize=14, va="center")
        ax.axvline(self.lt[j], color=T["ink2"], lw=1.0, alpha=0.6)
        ax.set(xlim=(self.lt[1], self.lt[-1]), ylim=(0, THETA_MAX), yticks=[0, 0.2, 0.4],
               xlabel="log$_{10}$( t / τ$_A$ )")
        if k == 0:
            ax.set_ylabel("q / q$_s$", fontsize=13)
        ax.tick_params(length=0)

    def _legend(self, fig):
        T = self.T
        x0 = 0.5
        for i, s in enumerate(("A", "B")):
            y = (290 - i * 34) / H
            fig.lines.append(plt.Line2D([x0 - 120 / W], [y], marker="o", ms=11, color=T[s], mec=T["bg"],
                                        transform=fig.transFigure, figure=fig))
            fig.text(x0 - 100 / W, y, SPECIES_LABEL[s], fontsize=15, color=T["ink"], va="center")
        for i, s in enumerate(("A", "B")):
            ax = fig.add_axes([x0 - 100 / W, (178 - 30 * i) / H, 200 / W, 13 / H])
            ax.imshow(self.ramps[s][None], aspect="auto", extent=(0, THETA_MAX, 0, 1))
            ax.set_yticks([])
            ax.set_xticks([0, 0.2, 0.4] if i == 1 else [])
            ax.grid(False)
            ax.tick_params(length=0, labelsize=11.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.text(x0 - 108 / W, (184 - 30 * i) / H, s, ha="right", va="center", fontsize=13, color=T["ink2"])
        fig.text(x0, 200 / H, "face colour: loading  q / q$_s$", ha="center", va="bottom", fontsize=13,
                 color=T["ink2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--theme", default="dark", choices=list(THEMES))
    ap.add_argument("--t", type=float, nargs="*", default=[0.3, 3.0, 40.0, 3000.0])
    args = ap.parse_args()
    if args.stills:
        data = ParticleData(args.t)
        for theme in (["dark", "white"] if args.theme == "dark" else [args.theme]):
            sc = ParticleScene(theme, (W, H), sep=3.6, px_per_unit=225.0, y_shift_px=125)
            comp = Composer(theme, sc, data)
            for t in args.t:
                t0 = time.time()
                prof, j = data.at(t)
                for _ in range(10):
                    sc.bath.advance()
                sc.build(prof)
                tmp = HERE / "_tmp3d.png"
                sc.render(tmp)
                out = HERE / f"still_{theme}_t{t:g}.png"
                comp.draw(plt.imread(tmp), t, j, out)
                tmp.unlink()
                print(f"{out.name}  {time.time() - t0:.1f}s", flush=True)
            sc.close()


if __name__ == "__main__":
    main()
