"""Particle zoom: two cut-away spheres (OVITO meshes) with radial loading gradients (env viz).

Each particle is a sphere with a 90-degree "orange slice" removed toward the
camera, which exposes two half-disc faces through the centre:
  left face  = A loading q_A(r) / q_s   (orange ramp, + adsorbed A molecules)
  right face = B loading q_B(r) / q_s   (blue ramp,   + adsorbed B molecules)
Both faces use the same colour scale for both particles (theta from 0 to
THETA_MAX), so colours compare directly. Adsorbed molecules: fixed random
sites on each face; a site shows a molecule when its random number u < theta(r) / THETA_MAX,
so sites fill and empty smoothly as the loading changes (no flicker).
Gas molecules (30 % A) move around the particles by a random walk.
Units: particle radius = 1.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from psaviz.style import THEMES, apply_ramp, hex_rgb, loading_ramp  # noqa: E402

THETA_MAX = 0.42
CUT_HALF = np.radians(45.0)  # half-width of the removed wedge
PHI_C = -np.pi / 2  # wedge centre points to -y (toward the camera)


def sphere_mesh(n_lat=72, n_lon=200):
    """UV sphere (R = 1) without the wedge |phi - PHI_C| < CUT_HALF."""
    th = np.linspace(0, np.pi, n_lat)
    ph = np.linspace(PHI_C + CUT_HALF, PHI_C + 2 * np.pi - CUT_HALF, n_lon)
    T, P = np.meshgrid(th, ph, indexing="ij")
    v = np.c_[(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()]
    f = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            a, b, c, d = i * n_lon + j, i * n_lon + j + 1, (i + 1) * n_lon + j + 1, (i + 1) * n_lon + j
            f += [[a, d, c], [a, c, b]]
    return v, np.asarray(f)


def face_mesh(phi_f, n_r=80, n_psi=90):
    """Half disc through the z axis at azimuth phi_f. Returns vertices, faces, vertex radius."""
    e = np.array([np.cos(phi_f), np.sin(phi_f), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    r = np.linspace(0, 1, n_r)
    psi = np.linspace(0, np.pi, n_psi)
    Rg, Pg = np.meshgrid(r, psi, indexing="ij")
    v = (Rg * np.sin(Pg)).ravel()[:, None] * e + (Rg * np.cos(Pg)).ravel()[:, None] * z
    f = []
    for i in range(n_r - 1):
        for j in range(n_psi - 1):
            a, b, c, d = i * n_psi + j, i * n_psi + j + 1, (i + 1) * n_psi + j + 1, (i + 1) * n_psi + j
            f += [[a, d, c], [a, c, b]]
    return v, np.asarray(f), Rg.ravel(), e


def face_sites(phi_f, n=900, rng=None, rmol=0.022):
    """Random sites on a half disc (uniform per area), lifted off the face toward the open wedge."""
    rng = rng or np.random.default_rng(0)
    e = np.array([np.cos(phi_f), np.sin(phi_f), 0.0])
    nrm = np.cross(np.array([0, 0, 1.0]), e)
    if nrm @ np.array([np.cos(PHI_C), np.sin(PHI_C), 0]) < 0:
        nrm = -nrm
    r = np.sqrt(rng.random(n)) * (1 - 1.6 * rmol)
    psi = rng.random(n) * np.pi
    pos = (r * np.sin(psi))[:, None] * e + (r * np.cos(psi))[:, None] * np.array([0, 0, 1.0]) + 0.6 * rmol * nrm
    return pos, r, rng.random(n)


class GasBath:
    """Gas molecules in a box around both particles; reflect off particles, walls and the open wedges."""

    def __init__(self, centres, box, n=190, yA=0.30, r_mol=0.026, step=0.03, seed=5):
        self.c = np.asarray(centres, float)
        self.box = np.asarray(box, float)  # [[x0, x1], [y0, y1], [z0, z1]]
        self.r_mol, self.step = r_mol, step
        self.rng = np.random.default_rng(seed)
        self.pos = self._sample(n)
        self.sp = (self.rng.random(n) > yA).astype(int)  # 0 = A, 1 = B

    def _bad(self, p):
        bad = np.zeros(len(p), bool)
        for c in self.c:
            d = p - c
            r = np.linalg.norm(d, axis=1)
            bad |= r < 1.0 + self.r_mol + 0.01
            # keep the open wedge in front of each particle clear so the faces stay visible
            ph = np.arctan2(d[:, 1], d[:, 0])
            dph = np.abs((ph - PHI_C + np.pi) % (2 * np.pi) - np.pi)
            bad |= (dph < CUT_HALF + 0.12) & (r < 1.55) & (np.abs(d[:, 2]) < 1.15)
        for a in range(3):
            bad |= (p[:, a] < self.box[a, 0]) | (p[:, a] > self.box[a, 1])
        return bad

    def _sample(self, n):
        out = np.zeros((0, 3))
        while len(out) < n:
            p = self.box[:, 0] + self.rng.random((4 * n, 3)) * (self.box[:, 1] - self.box[:, 0])
            out = np.concatenate([out, p[~self._bad(p)]])
        return out[:n]

    def advance(self, k=1.0):
        new = self.pos + self.rng.normal(0, self.step * k, self.pos.shape)
        bad = self._bad(new)
        self.pos[~bad] = new[~bad]


class ParticleScene:
    ELEV = np.radians(16.0)

    def __init__(self, theme, size=(1920, 1080), sep=2.9, px_per_unit=290.0, y_shift_px=-60):
        from ovito.pipeline import Pipeline, StaticSource
        from ovito.vis import TachyonRenderer, Viewport
        from ovito.data import DataCollection

        self.theme, self.size, self.T = theme, size, THEMES[theme]
        self.centres = np.array([[-sep / 2 * 1.0, 0, 0], [sep / 2 * 1.0, 0, 0]])
        self.sph_v, self.sph_f = sphere_mesh()
        self.faces = {}
        for key, phi in (("A", PHI_C - CUT_HALF), ("B", PHI_C + CUT_HALF)):
            v, f, r, e = face_mesh(phi)
            self.faces[key] = (v, f, r)
        rng = np.random.default_rng(1)
        self.sites = {k: face_sites(phi, rng=rng) for k, phi in (("A", PHI_C - CUT_HALF), ("B", PHI_C + CUT_HALF))}
        self.ramps = {k: loading_ramp(theme, k) for k in ("A", "B")}
        self.pipe = Pipeline(source=StaticSource(data=DataCollection()))
        self.pipe.add_to_scene()
        d = np.array([0.0, np.cos(self.ELEV), -np.sin(self.ELEV)])
        self.dir, self.up = d, np.array([0.0, np.sin(self.ELEV), np.cos(self.ELEV)])
        self.target = np.array([0.0, 0.0, 0.0]) - self.up * (y_shift_px / px_per_unit)
        self.fov = size[1] / (2 * px_per_unit)
        self.vp = Viewport(type=Viewport.Type.Ortho, camera_pos=tuple(self.target - 50 * d), camera_dir=tuple(d),
                           fov=self.fov)
        self.renderer = TachyonRenderer(ambient_occlusion=True, ambient_occlusion_brightness=0.8,
                                        ambient_occlusion_samples=12, antialiasing_samples=12,
                                        direct_light_intensity=0.9, shadows=False)
        self.bath = GasBath(self.centres, [[-3.4, 3.4], [-1.3, 1.5], [-1.12, 1.12]])

    def project(self, p):
        W, H = self.size
        s = H / (2 * self.fov)
        rgt = np.cross(self.dir, self.up)
        q = np.asarray(p, float) - self.target
        return W / 2 + s * (q @ rgt), H / 2 - s * (q @ self.up)

    def build(self, prof):
        """prof: list (per particle) of dicts with r (N,), thA (N,), thB (N,)."""
        from ovito.data import DataCollection
        from ovito.vis import ParticlesVis, TriangleMeshVis

        T = self.T
        dc = DataCollection()
        bare = hex_rgb(T["bare"])
        pos, rad, col = [], [], []
        for k, (c, pr) in enumerate(zip(self.centres, prof)):
            m = dc.triangle_meshes.create(f"sphere{k}")
            m.set_vertices(self.sph_v + c)
            m.set_faces(self.sph_f)
            m.set_normals(self.sph_v[self.sph_f].reshape(-1, 3))
            # outer surface: bare sorbent, lightly tinted by the surface loading mix
            m.vis = TriangleMeshVis(color=tuple(bare), highlight_edges=False)
            for key in ("A", "B"):
                v, f, rv = self.faces[key]
                th = np.interp(rv, pr["r"], pr["th" + key])
                cols = apply_ramp(self.ramps[key], th / THETA_MAX)
                mm = dc.triangle_meshes.create(f"face{k}{key}")
                mm.set_vertices(v * 0.999 + c)
                mm.set_faces(f)
                mm.set_vertex_colors(np.c_[cols, np.ones(len(cols))])
                mm.vis = TriangleMeshVis(highlight_edges=False)
                sp, sr, su = self.sites[key]
                thr = np.interp(sr, pr["r"], pr["th" + key]) / THETA_MAX
                on = su < thr
                pos.append(sp[on] + c)
                rad.append(np.full(on.sum(), 0.022))
                col.append(np.tile(hex_rgb(T[key]), (on.sum(), 1)))
        b = self.bath
        pos.append(b.pos)
        rad.append(np.full(len(b.pos), b.r_mol))
        col.append(np.where(b.sp[:, None] == 0, hex_rgb(T["A"]), hex_rgb(T["B"])))
        pos, rad, col = np.concatenate(pos), np.concatenate(rad), np.concatenate(col)
        pt = dc.create_particles(count=len(pos))
        pt.create_property("Position", data=pos)
        pt.create_property("Radius", data=rad)
        pt.create_property("Color", data=col)
        pt.vis = ParticlesVis()
        self.pipe.source.data = dc

    def render(self, path):
        self.vp.render_image(size=self.size, filename=str(path), background=hex_rgb(self.T["bg"]),
                             alpha=False, renderer=self.renderer)

    def close(self):
        self.pipe.remove_from_scene()
