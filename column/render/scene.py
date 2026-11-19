"""Column visual: solver state per frame, gas dots, OVITO 3D bed (env viz).

* ``CycleData`` re-integrates one CSS cycle from ../run/css_state.npz and
  returns the column state at any time in the cycle.
* ``GasDots`` draws gas molecules as small spheres whose number density follows
  the local gas concentration (so the bed visibly fills on pressurization and
  empties on blowdown) and whose species follow the local y_A. They drift with
  the gas velocity (sped down for watchability); births/deaths per height bin
  keep the counts on target (adsorption removes gas, desorption adds it).
* ``BedScene`` is the OVITO scene: randomly packed spheres in a cut-away
  column, coloured by their A loading, glass wall, end caps and pipes.
Geometry units: particle diameter = 1 (packing.npz).
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from psaviz.column import ColumnCase, ColumnModel  # noqa: E402
from psaviz.isotherm import loading  # noqa: E402
from psaviz.style import THEMES, apply_ramp, hex_rgb, loading_ramp  # noqa: E402


# ---------------------------------------------------------------- solver data
class CycleData:
    def __init__(self):
        d = np.load(ROOT / "column" / "run" / "css_state.npz", allow_pickle=False)
        self.case = ColumnCase.from_dict(json.loads(str(d["case"])))
        self.m = ColumnModel(self.case)
        _, self.recs = self.m.run_cycle(d["x_start"])
        self.dur = np.array([s.duration for s in self.case.steps])
        self.starts = np.concatenate([[0.0], np.cumsum(self.dur)])
        self.t_cycle = float(self.starts[-1])
        self.P_H = float(self.m.column_pressure(self.recs[0]["x_end"]))
        self.qA_ref = float(loading(np.array(self.case.y_feed) * self.P_H, self.case.qs, self.case.b)[0])
        self.qs = float(self.case.qs[0])
        self.z = (np.arange(self.m.N) + 0.5) / self.m.N

    def state(self, t):
        """Column state at time t (s) in the cycle (0 <= t <= t_cycle)."""
        t = float(np.clip(t, 0.0, self.t_cycle))
        s = int(min(np.searchsorted(self.starts, t, side="right") - 1, len(self.dur) - 1))
        ts = t - self.starts[s]
        step = self.case.steps[s]
        X = self.m.sample(self.recs[s]["segs"], np.array([ts]))
        u, c, q, cR, C, _ = self.m.velocities(X, step.kind, step.cv)
        c, q, C, u = c[..., 0], q[..., 0], C[:, 0], u[:, 0]
        a = self.m.area
        return dict(
            t=t, frac=t / self.t_cycle, step=s, kind=step.kind, t_in_step=ts, step_frac=ts / step.duration,
            yA=c[0] / C, qA=q[0], qB=q[1], C=C, P=float(C.mean() * self.m.RT / 1e5),
            P_rel=float(C.mean() * self.m.RT / 1e5 / self.P_H), u=u,
            flow_bottom=float(u[0] * C[0] * a), flow_top=float(u[-1] * C[-1] * a),  # mol/s, + = upward
            p_tank=float(cR.sum() * self.m.RT / 1e5),
        )

    def pressure_trace(self, n=600):
        t = np.linspace(0, self.t_cycle, n)
        return t / self.t_cycle, np.array([self.state(tt)["P_rel"] for tt in t])


# ------------------------------------------------------------------ packing
class Packing:
    def __init__(self):
        d = np.load(HERE / "packing.npz")
        self.pos, self.r = d["pos"], float(d["radius"])
        self.R, self.H = float(d["R_col"]), float(d["H_bed"])
        self.tree = cKDTree(self.pos)


# ---------------------------------------------------------------- gas dots
class GasDots:
    """Gas molecules in a slab around the cut plane (y in [y0, y1])."""

    def __init__(self, pack: Packing, n_ref=1100, r_dot=0.16, y0=-0.7, y1=1.2, nbins=40, seed=3, cut_y=0.0):
        self.p, self.n_ref, self.r_dot = pack, n_ref, r_dot
        self.y0, self.y1, self.nbins = y0, y1, nbins
        self.rng = np.random.default_rng(seed)
        self.edges = np.linspace(0.0, pack.H, nbins + 1)
        self.pos = np.zeros((0, 3))
        self.sp = np.zeros(0, int)  # 0 = A, 1 = B
        # pool of void points to spawn from
        n = 400_000
        rho = (pack.R - 0.3) * np.sqrt(self.rng.random(n))
        ph = self.rng.random(n) * 2 * np.pi
        pts = np.c_[rho * np.cos(ph), rho * np.sin(ph), self.rng.random(n) * pack.H]
        pts = pts[(pts[:, 1] > y0) & (pts[:, 1] < y1)]
        shown = cKDTree(pack.pos[pack.pos[:, 1] > cut_y])  # only spheres left after the cutaway
        dist, _ = shown.query(pts)
        self.pool = pts[dist > pack.r + r_dot * 0.9]
        self.pool_bin = np.clip(np.digitize(self.pool[:, 2], self.edges) - 1, 0, nbins - 1)
        self.pool_by_bin = [np.nonzero(self.pool_bin == b)[0] for b in range(nbins)]

    def targets(self, st, P_H_C):
        """Target A/B counts per bin from the solver state."""
        zc = 0.5 * (self.edges[1:] + self.edges[:-1]) / self.p.H
        zz = (np.arange(len(st["C"])) + 0.5) / len(st["C"])
        C = np.interp(zc, zz, st["C"]) / P_H_C
        yA = np.interp(zc, zz, st["yA"])
        n_bin = self.n_ref * C / self.nbins
        return n_bin * yA, n_bin * (1 - yA)

    def _spawn(self, b, k, species):
        idx = self.rng.choice(self.pool_by_bin[b], size=k, replace=False)
        jitter = self.rng.normal(0, 0.05, (k, 3))
        return self.pool[idx] + jitter, np.full(k, species)

    def reset(self, st, P_H_C):
        self.pos, self.sp = np.zeros((0, 3)), np.zeros(0, int)
        self.match(st, P_H_C)

    def match(self, st, P_H_C, band=0.08):
        """Births/deaths only when a bin leaves a +-band around its target (no frame-to-frame flicker)."""
        tA, tB = self.targets(st, P_H_C)
        b = np.clip(np.digitize(self.pos[:, 2], self.edges) - 1, 0, self.nbins - 1)
        keep = np.ones(len(self.pos), bool)
        new_p, new_s = [], []
        for bb in range(self.nbins):
            for s, tgt in ((0, tA[bb]), (1, tB[bb])):
                tol = max(1.0, band * tgt)
                have = np.nonzero((b == bb) & (self.sp == s) & keep)[0]
                if len(have) > tgt + tol:
                    n_kill = len(have) - int(np.ceil(tgt + tol * 0.5))
                    keep[self.rng.choice(have, n_kill, replace=False)] = False
                elif len(have) < tgt - tol:
                    p, sp = self._spawn(bb, int(np.floor(tgt - tol * 0.5)) - len(have), s)
                    new_p.append(p)
                    new_s.append(sp)
        self.pos = np.concatenate([self.pos[keep]] + new_p) if new_p else self.pos[keep]
        self.sp = np.concatenate([self.sp[keep]] + new_s) if new_s else self.sp[keep]

    def advance(self, st, P_H_C, dt_video, eps, k=0.18, vmax=0.6, jitter=0.025):
        """Drift with the interstitial gas velocity, then re-match the targets.

        Video speed (bed lengths per video second) = vmax tanh(k u/eps / vmax), u in m/s for the
        1 m bed: watchable and monotonic in the real velocity, not to scale."""
        zz = np.linspace(0, 1, len(st["u"]))
        u = np.interp(self.pos[:, 2] / self.p.H, zz, st["u"]) / eps
        v = vmax * np.tanh(k * u / vmax)
        self.pos[:, 2] += v * dt_video * self.p.H
        self.pos += self.rng.normal(0, jitter, self.pos.shape)
        ok = (self.pos[:, 2] > 0) & (self.pos[:, 2] < self.p.H)
        r = np.hypot(self.pos[:, 0], self.pos[:, 1])
        ok &= (r < self.p.R - 0.2) & (self.pos[:, 1] > self.y0) & (self.pos[:, 1] < self.y1)
        self.pos, self.sp = self.pos[ok], self.sp[ok]
        self.match(st, P_H_C)


# --------------------------------------------------------------- OVITO scene
def cylinder_mesh(R, z0, z1, th0, th1, n=96, cap=False):
    th = np.linspace(th0, th1, n)
    v = np.concatenate([np.c_[R * np.cos(th), R * np.sin(th), np.full(n, z0)],
                        np.c_[R * np.cos(th), R * np.sin(th), np.full(n, z1)]])
    f = []
    for i in range(n - 1):
        f += [[i, i + 1, n + i + 1], [i, n + i + 1, n + i]]
    nrm = np.concatenate([np.c_[np.cos(th), np.sin(th), np.zeros(n)]] * 2)
    if cap:
        for z in (z0, z1):
            c = len(v)
            v = np.concatenate([v, [[0, 0, z]], np.c_[R * np.cos(th), R * np.sin(th), np.full(n, z)]])
            nrm = np.concatenate([nrm, np.tile([0, 0, 1 if z == z1 else -1], (n + 1, 1))])
            for i in range(n - 1):
                f.append([c, c + 1 + i, c + 2 + i])
    return np.asarray(v, float), np.asarray(f, int), np.asarray(nrm, float)


class BedScene:
    ELEV = np.radians(14.0)

    def __init__(self, theme, size=(700, 1080), cut_y=0.0, bed_px=660):
        from ovito.data import DataCollection
        from ovito.pipeline import Pipeline, StaticSource
        from ovito.vis import ParticlesVis, TachyonRenderer, TriangleMeshVis, Viewport

        self.theme, self.size = theme, size
        self.T = THEMES[theme]
        self.pack = Packing()
        p = self.pack
        self.keep = p.pos[:, 1] > cut_y
        self.bed_pos = p.pos[self.keep]
        rng = np.random.default_rng(11)
        self.jitter = 1.0 + rng.normal(0, 0.035, len(self.bed_pos))[:, None]
        self.ramp = loading_ramp(theme, "A", deep=True)
        # bed particles
        self.bed = DataCollection()
        pt = self.bed.create_particles(count=len(self.bed_pos))
        pt.create_property("Position", data=self.bed_pos)
        pt.create_property("Radius", data=np.full(len(self.bed_pos), p.r * 0.985))
        pt.create_property("Color", data=np.tile(hex_rgb(self.T["bare"]), (len(self.bed_pos), 1)))
        pt.vis = ParticlesVis(radius=0.5)
        # vessel: back half of the glass wall, cut-away caps, pipes
        R, H = p.R + 0.08, p.H
        self.plenum = 1.4
        z0, z1 = -self.plenum, H + self.plenum
        v, f, n = cylinder_mesh(R, z0, z1, 0.0, np.pi)
        m = self.bed.triangle_meshes.create("wall")
        m.set_vertices(v); m.set_faces(f); m.set_normals(n[f].reshape(-1, 3))
        m.vis = TriangleMeshVis(color=tuple(hex_rgb(self.T["glass"])), transparency=0.72,
                                highlight_edges=False, backface_culling=False)
        metal = tuple(hex_rgb(self.T["metal"]))
        for name, (zz0, zz1, rr, th0, th1, cap) in {
            "cap_bot": (z0 - 0.5, z0, R + 0.35, 0.0, np.pi, True),
            "cap_top": (z1, z1 + 0.5, R + 0.35, 0.0, np.pi, True),
            "pipe_bot": (z0 - 4.0, z0 - 0.5, 1.1, 0.0, 2 * np.pi, True),
            "pipe_top": (z1 + 0.5, z1 + 4.0, 1.1, 0.0, 2 * np.pi, True),
        }.items():
            v, f, n = cylinder_mesh(rr, zz0, zz1, th0, th1, cap=cap)
            mm = self.bed.triangle_meshes.create(name)
            mm.set_vertices(v); mm.set_faces(f)
            mm.vis = TriangleMeshVis(color=metal, transparency=0.0, highlight_edges=False)
        # distributor plates: thin translucent discs at the bed ends
        for name, zz in (("plate_bot", -0.15), ("plate_top", H + 0.15)):
            v, f, n = cylinder_mesh(R - 0.02, zz - 0.08, zz + 0.08, 0.0, np.pi, cap=True)
            mm = self.bed.triangle_meshes.create(name)
            mm.set_vertices(v); mm.set_faces(f)
            mm.vis = TriangleMeshVis(color=metal, transparency=0.55, highlight_edges=False)
        self.bed_pipe = Pipeline(source=StaticSource(data=self.bed))
        self.bed_pipe.add_to_scene()
        self.gas = DataCollection()
        self.gas_pipe = Pipeline(source=StaticSource(data=self.gas))
        self.gas_pipe.add_to_scene()
        # camera: orthographic, slightly from above, looking at the cut face
        self.target = np.array([0.0, 0.0, H / 2])
        d = np.array([0.0, np.cos(self.ELEV), -np.sin(self.ELEV)])
        self.dir = d
        self.up = np.array([0.0, np.sin(self.ELEV), np.cos(self.ELEV)])
        self.fov = size[1] / (2 * bed_px / (H * np.cos(self.ELEV)))  # bed spans bed_px pixels
        self.vp = Viewport(type=Viewport.Type.Ortho, camera_pos=tuple(self.target - 80 * d),
                           camera_dir=tuple(d), fov=self.fov)
        self.renderer = TachyonRenderer(ambient_occlusion=True, ambient_occlusion_brightness=0.85,
                                        ambient_occlusion_samples=12, antialiasing_samples=12,
                                        direct_light_intensity=0.95, shadows=False)

    def project(self, p):
        """World point -> pixel (x, y from top) in the rendered image."""
        W, Hh = self.size
        s = Hh / (2 * self.fov)
        rgt = np.cross(self.dir, self.up)
        q = np.asarray(p, float) - self.target
        return W / 2 + s * (q @ rgt), Hh / 2 - s * (q @ self.up)

    def set_state(self, st, cyc: CycleData, dots: GasDots | None):
        z = self.bed_pos[:, 2] / self.pack.H
        qA = np.interp(z, cyc.z, st["qA"]) / cyc.qA_ref
        col = np.clip(apply_ramp(self.ramp, qA) * self.jitter, 0, 1)
        dc = self.bed.clone()  # the pipeline shares the source data after a render; edit a fresh copy
        dc.particles_.create_property("Color", data=col)
        self.bed_pipe.source.data = dc
        from ovito.data import DataCollection
        g = DataCollection()
        if dots is not None and len(dots.pos):
            pt = g.create_particles(count=len(dots.pos))
            pt.create_property("Position", data=dots.pos)
            pt.create_property("Radius", data=np.full(len(dots.pos), dots.r_dot))
            cA, cB = hex_rgb(self.T["A"]), hex_rgb(self.T["B"])
            pt.create_property("Color", data=np.where(dots.sp[:, None] == 0, cA, cB))
        self.gas_pipe.source.data = g

    def close(self):
        self.bed_pipe.remove_from_scene()
        self.gas_pipe.remove_from_scene()

    def render(self, path):
        self.vp.render_image(size=self.size, filename=str(path), background=hex_rgb(self.T["bg"]),
                             alpha=True, renderer=self.renderer)
