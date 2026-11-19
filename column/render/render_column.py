"""Render the column visual: 3D bed (OVITO) + aligned profile panels (matplotlib). Env viz.

usage:
  python render_column.py --stills            4 key stills (one per step) for both themes
  python render_column.py --all --theme dark  every frame -> frames_<theme>/, then MP4 + GIF
  options: --size 1920x1080 --fps 30 --seconds 20 --workers 6
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle  # noqa: E402

import sys  # noqa: E402

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
from scene import HERE, BedScene, CycleData, GasDots  # noqa: E402
from psaviz.column import STEP_LABELS  # noqa: E402
from psaviz.style import SPECIES_LABEL, THEMES, hex_rgb, loading_ramp, mpl_rc  # noqa: E402

W, H = 1920, 1080
R3D = (700, 1080)  # 3D render size (left part of the frame)
X3D = 70  # left offset of the 3D render in the frame (px)
STEP_TEXT = {
    "RP": "feed enters the bottom, top closed;\npressure rises",
    "FEED": "A is held on the particles,\nB flows through as product",
    "BD": "pressure released through the bottom;\nA desorbs and leaves",
    "PURGE": "B from the product tank sweeps\nthe remaining A out of the bottom",
}
END_LABEL = {  # (bottom, top) flow annotations: (text, colour key or None)
    "RP": (("Feed in  (30 % A)", "ink"), ("closed", None)),
    "FEED": (("Feed in  (30 % A)", "ink"), ("Raffinate out:\nlight product, B", "B")),
    "BD": (("Extract out:\nheavy, A-rich (vent)", "A"), ("closed", None)),
    "PURGE": (("Extract out:\nheavy, A-rich (vent)", "A"), ("Purge in:\nB from product tank", "B")),
}


FPS, SECONDS = 30, 20
STEP_VIDEO_S = np.array([4.0, 6.0, 4.0, 6.0])  # user choice: stretch the pressure-changing steps


def time_map(frac_video, cyc):
    """Video time fraction [0, 1) -> cycle time (s). Each step is linear inside its video slot."""
    tv = frac_video * STEP_VIDEO_S.sum()
    edges = np.concatenate([[0.0], np.cumsum(STEP_VIDEO_S)])
    s = int(min(np.searchsorted(edges, tv, side="right") - 1, 3))
    return cyc.starts[s] + (tv - edges[s]) / STEP_VIDEO_S[s] * cyc.dur[s]


class Composer:
    def __init__(self, theme, cyc: CycleData, scene: BedScene):
        self.theme, self.T, self.cyc, self.sc = theme, THEMES[theme], cyc, scene
        self.pt_frac, self.pt_P = cyc.pressure_trace()
        # reference flows for arrow strength
        tt = np.linspace(0, cyc.t_cycle, 480)
        sts = [cyc.state(t) for t in tt]
        self.f_ref_bot = max(abs(s["flow_bottom"]) for s in sts)
        self.f_ref_top = max(abs(s["flow_top"]) for s in sts)
        p = scene.pack
        self.y_bed0 = scene.project([0, 0, 0])[1]
        self.y_bed1 = scene.project([0, 0, p.H])[1]
        self.pipe_bot = scene.project([0, -1.1, -scene.plenum - 4.0])
        self.pipe_top = scene.project([0, -1.1, p.H + scene.plenum + 4.0])
        self.ramp = loading_ramp(theme, "A", deep=True)

    def fig_xy(self, x, y_top):
        return (x + X3D) / W, 1 - y_top / H

    def draw(self, st, img, out):
        T, cyc = self.T, self.cyc
        with plt.rc_context(mpl_rc(self.theme)):
            fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
            ax = fig.add_axes([X3D / W, 0, R3D[0] / W, 1])
            ax.imshow(img, interpolation="lanczos")
            ax.set_axis_off()
            self._arrows(fig, st)
            # aligned profile panels
            b = 1 - self.y_bed0 / H
            h = (self.y_bed0 - self.y_bed1) / H
            z = cyc.z
            axg = fig.add_axes([845 / W, b, 270 / W, h])
            axg.plot(st["yA"], z, color=T["A"], lw=2.6, solid_capstyle="round")
            axg.plot(1 - st["yA"], z, color=T["B"], lw=2.6, solid_capstyle="round")
            axg.set(xlim=(0, 1), ylim=(0, 1), xlabel="mole fraction  y", ylabel="height  z / L",
                    xticks=[0.25, 0.5, 0.75, 1.0])
            axg.set_title("Gas phase", loc="left", pad=40)
            self._direct_label(axg, st["yA"], z, "$y_A$", T["A"])
            self._direct_label(axg, 1 - st["yA"], z, "$y_B$", T["B"])
            axq = fig.add_axes([1160 / W, b, 270 / W, h])
            axq.plot(st["qA"] / cyc.qs, z, color=T["A"], lw=2.6)
            axq.plot(st["qB"] / cyc.qs, z, color=T["B"], lw=2.6)
            axq.set(xlim=(0, 0.42), ylim=(0, 1), xlabel="loading  q / q$_s$", xticks=[0.1, 0.2, 0.3, 0.4])
            axq.set_yticklabels([])
            axq.set_title("Adsorbed phase", loc="left", pad=40)
            self._direct_label(axq, st["qA"] / cyc.qs, z, "$q_A$", T["A"])
            self._direct_label(axq, st["qB"] / cyc.qs, z, "$q_B$", T["B"])
            for a in (axg, axq):
                a.tick_params(length=0)
            self._steps(fig, st)
            self._pressure(fig, st)
            self._legend(fig)
            fig.text(24 / W, 1 - 34 / H, "PSA column", fontsize=26, fontweight="bold", color=T["ink"], va="top")
            fig.text(24 / W, 1 - 72 / H, "4-step cycle at cyclic steady state", fontsize=16, color=T["ink2"], va="top")
            fig.text(24 / W, 1 - 100 / H, f"t / t$_{{cycle}}$ = {st['frac']:.2f}", fontsize=16, color=T["ink2"],
                     va="top", family="DejaVu Sans Mono")
            fig.text(1480 / W, 22 / H, "Isothermal LDF column model (psa-sim physics)\n"
                     "at cyclic steady state; generic species and\nillustrative inputs. Bed geometry, particle size\n"
                     "and molecule speeds are not to scale.",
                     fontsize=11.5, color=T["muted"], va="bottom", linespacing=1.4)
            fig.savefig(out, dpi=100)
            plt.close(fig)

    def _direct_label(self, ax, x, z, text, color):
        ax.text(min(x[-1] + 0.03 * ax.get_xlim()[1], 0.93 * ax.get_xlim()[1]), 1.015, text, color=self.T["ink"],
                fontsize=15, va="bottom", ha="center")
        ax.plot([x[-1]], [1.0], "o", ms=7, color=color, clip_on=False, mec=self.T["bg"], mew=1.5)

    def _arrows(self, fig, st):
        T = self.T
        kind = st["kind"]
        for where, (xpix, ypix), flow, fref in (("bot", self.pipe_bot, st["flow_bottom"], self.f_ref_bot),
                                                 ("top", self.pipe_top, st["flow_top"], self.f_ref_top)):
            text, ckey = END_LABEL[kind][0 if where == "bot" else 1]
            x, y = self.fig_xy(xpix, ypix)
            strength = min(1.0, np.sqrt(abs(flow) / (0.5 * fref))) if fref > 0 else 0
            ax_x = x - 0.0
            if ckey is None or strength < 0.02:
                fig.text(x + 34 / W, y + (14 if where == "bot" else -14) / H, text if ckey else "closed",
                         color=T["muted"], fontsize=14, va="center")
                continue
            col = T[ckey] if ckey in ("A", "B") else T["ink2"]
            up = flow > 0
            L = 70 / H
            if where == "bot":
                y0 = y - 6 / H - (L if up else 0)
            else:
                y0 = y + 6 / H + (0 if up else L)
            dy = L if up else -L
            fig.patches.append(FancyArrow(ax_x, y0, 0, dy, width=0.012 * 0.6, head_width=0.012 * 1.4,
                                          head_length=0.024, length_includes_head=True, color=col,
                                          alpha=0.25 + 0.75 * strength, transform=fig.transFigure,
                                          figure=fig, lw=0))
            fig.text(x + 34 / W, y0 + dy / 2, text, color=T["ink"], fontsize=14, va="center", linespacing=1.3)

    def _steps(self, fig, st):
        T = self.T
        x0, y_top, w, hrow, gap = 1480 / W, 1 - 40 / H, 400 / W, 92 / H, 10 / H
        fig.text(x0, y_top + 4 / H, "Cycle steps", fontsize=17, fontweight="bold", color=T["ink"], va="bottom")
        for i, s in enumerate(self.cyc.case.steps):
            y = y_top - (i + 1) * hrow - i * gap + 0 / H
            active = i == st["step"]
            box = FancyBboxPatch((x0, y), w, hrow, boxstyle="round,pad=0,rounding_size=0.006",
                                 transform=fig.transFigure, figure=fig, lw=1.0,
                                 ec=T["axis"] if not active else T["ink2"],
                                 fc=T["highlight"] if active else T["bg"])
            fig.patches.append(box)
            if active:
                fig.patches.append(Rectangle((x0, y), w * st["step_frac"], 5 / H, transform=fig.transFigure,
                                             figure=fig, color=T["ink2"], lw=0))
            ink = T["ink"] if active else T["muted"]
            fig.text(x0 + 14 / W, y + hrow - 14 / H, f"{i + 1}  {STEP_LABELS[s.kind]}", fontsize=16,
                     fontweight="bold" if active else "normal", color=ink, va="top")
            fig.text(x0 + 36 / W, y + hrow - 42 / H, STEP_TEXT[s.kind], fontsize=12.5,
                     color=T["ink2"] if active else T["muted"], va="top", linespacing=1.3)

    def _pressure(self, fig, st):
        T = self.T
        ax = fig.add_axes([1530 / W, 380 / H, 340 / W, 165 / H])
        starts = self.cyc.starts / self.cyc.t_cycle
        for i in range(4):
            if i % 2 == 0:
                ax.axvspan(starts[i], starts[i + 1], color=T["highlight"], lw=0, zorder=0)
            ax.text(0.5 * (starts[i] + starts[i + 1]), 1.17, str(i + 1), ha="center", va="center",
                    fontsize=12, color=T["ink"] if i == st["step"] else T["muted"])
        ax.plot(self.pt_frac, self.pt_P, color=T["ink2"], lw=2.0)
        ax.plot([st["frac"]], [st["P_rel"]], "o", ms=10, color=T["ink"], mec=T["bg"], mew=2)
        ax.set(xlim=(0, 1), ylim=(0, 1.1), xlabel="t / t$_{cycle}$", yticks=[0.5, 1.0], xticks=[0, 0.5, 1])
        ax.set_title("Column pressure  P / P$_H$", loc="left", pad=22, fontsize=15)
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0)

    def _legend(self, fig):
        T = self.T
        x0, y0 = 1480 / W, 285 / H
        for i, s in enumerate(("A", "B")):
            y = y0 - i * 34 / H
            fig.lines.append(plt.Line2D([x0 + 10 / W], [y], marker="o", ms=11, color=T[s], mec=T["bg"],
                                        transform=fig.transFigure, figure=fig))
            fig.text(x0 + 30 / W, y, SPECIES_LABEL[s], fontsize=15, color=T["ink"], va="center")
        ax = fig.add_axes([x0 + 4 / W, 168 / H, 300 / W, 16 / H])
        ax.imshow(self.ramp[None, :, :], aspect="auto", extent=(0, 1, 0, 1))
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([])
        ax.grid(False)
        ax.tick_params(length=0, labelsize=12)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(x0, 194 / H, "Particle colour: A held,  q$_A$ / q$_{A,feed}$", fontsize=13.5, color=T["ink2"],
                 va="bottom")


def precompute_dots(cyc, path, n_frames):
    """Run the gas molecules through one warm-up cycle, then store one full cycle of frames."""
    from scene import Packing
    dots = GasDots(Packing())
    P_H_C = cyc.P_H * 1e5 / cyc.m.RT
    dots.reset(cyc.state(time_map(0.0, cyc)), P_H_C)
    pos, sp, off = [], [], [0]
    for f in range(2 * n_frames):
        st = cyc.state(time_map((f % n_frames) / n_frames, cyc))
        dots.advance(st, P_H_C, 1.0 / FPS, cyc.case.bed_void)
        if f >= n_frames:
            pos.append(dots.pos.astype(np.float32))
            sp.append(dots.sp.astype(np.int8))
            off.append(off[-1] + len(dots.pos))
    np.savez_compressed(path, pos=np.concatenate(pos), sp=np.concatenate(sp), off=np.array(off),
                        r_dot=dots.r_dot)


def worker(theme, a, b, n_frames, outdir):
    from types import SimpleNamespace
    cyc = CycleData()
    scene = BedScene(theme, R3D)
    comp = Composer(theme, cyc, scene)
    d = np.load(HERE / "frames" / "dots.npz")
    for f in range(a, b):
        out = outdir / f"f_{f:04d}.png"
        if out.exists():
            continue
        st = cyc.state(time_map(f / n_frames, cyc))
        o0, o1 = d["off"][f], d["off"][f + 1]
        dots = SimpleNamespace(pos=d["pos"][o0:o1].astype(float), sp=d["sp"][o0:o1].astype(int),
                               r_dot=float(d["r_dot"]))
        scene.set_state(st, cyc, dots)
        tmp = outdir / f"_3d_{f:04d}.png"
        scene.render(tmp)
        comp.draw(st, plt.imread(tmp), out)
        tmp.unlink()


def assemble(frames, stem, fps=FPS, gif_width=1920, gif_fps=15):
    mp4, gif = HERE / f"{stem}.mp4", HERE / f"{stem}.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                    "-movflags", "+faststart", str(mp4)], check=True)
    vf = (f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:"
          f"stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f_%04d.png"),
                    "-vf", vf, str(gif)], check=True)
    return mp4, gif


def render_one(theme, t_cycle_s, out, dots_state=None, cyc=None, scene=None, comp=None, dots=None):
    st = cyc.state(t_cycle_s)
    P_H_C = cyc.P_H * 1e5 / cyc.m.RT
    if dots is not None and dots_state is None:
        dots.reset(st, P_H_C)
    scene.set_state(st, cyc, dots)
    tmp = Path(str(out) + ".3d.png")
    scene.render(tmp)
    img = plt.imread(tmp)
    comp.draw(st, img, out)
    tmp.unlink()
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--theme", default="dark", choices=list(THEMES))
    ap.add_argument("--t", type=float, nargs="*", default=[3.0, 52.0, 62.0, 105.0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--worker", type=int, nargs=2, default=None)
    ap.add_argument("--gif-width", type=int, default=1920)
    args = ap.parse_args()
    n_frames = FPS * SECONDS
    outdir = HERE / "frames" / args.theme
    if args.worker is not None:
        worker(args.theme, *args.worker, n_frames, outdir)
        return
    if args.all:
        outdir.mkdir(parents=True, exist_ok=True)
        cyc = CycleData()
        dpath = HERE / "frames" / "dots.npz"
        if not dpath.exists():
            t0 = time.time()
            precompute_dots(cyc, dpath, n_frames)
            print(f"gas molecules precomputed ({time.time() - t0:.0f}s)", flush=True)
        t0 = time.time()
        edges = np.linspace(0, n_frames, args.workers + 1).astype(int)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        procs = [subprocess.Popen([sys.executable, "-I", __file__, "--theme", args.theme, "--worker", str(a), str(b)],
                                  env=env) for a, b in zip(edges[:-1], edges[1:])]
        codes = [p.wait() for p in procs]
        n_done = len(list(outdir.glob("f_*.png")))
        print(f"{args.theme}: {n_done}/{n_frames} frames in {time.time() - t0:.0f}s, worker exit codes {codes}", flush=True)
        if n_done == n_frames:
            mp4, gif = assemble(outdir, f"psa_column_{args.theme}", gif_width=args.gif_width)
            print(mp4.name, f"{mp4.stat().st_size / 1e6:.1f} MB", gif.name, f"{gif.stat().st_size / 1e6:.1f} MB")
        return
    cyc = CycleData()
    if args.stills:
        for theme in (["dark", "white"] if args.theme == "dark" else [args.theme]):
            scene = BedScene(theme, R3D)
            comp = Composer(theme, cyc, scene)
            dots = GasDots(scene.pack)
            for t in args.t:
                t0 = time.time()
                out = HERE / f"still_{theme}_t{int(round(t)):03d}.png"
                render_one(theme, t, out, cyc=cyc, scene=scene, comp=comp, dots=dots)
                print(f"{out.name}  {time.time() - t0:.1f}s", flush=True)
            scene.close()


if __name__ == "__main__":
    main()
