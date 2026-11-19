"""Single-column PSA model: isothermal, no axial pressure drop, LDF, extended Langmuir.

A compact rewrite of psa-sim's flow-controlled model (toPSAil ``DP0DT0``),
restricted to what one column running a 4-step Skarstrom cycle needs. Same
equations, written in SI units:

* Gas in cell n (superficial face velocities u, upwind face concentrations):
  eps dz dc_i/dt = u_n c_i(face n) - u_{n+1} c_i(face n+1) - (1-eps) rho_p dz dq_i/dt
* LDF: dq_i/dt = k_i (q_i* - q_i), extended Langmuir q_i*.
* No momentum balance: the pressure is uniform along the bed. The face
  velocities come from the overall balance, marched cell by cell:
  u_{n+1} = u_n - (1-eps) rho_p dz sum_i dq_i/dt / C_n - dz g,
  where g = 0 in constant-pressure steps (FEED, PURGE) and
  g = (u_0 - u_N - sum vads)/L in pressure-changing steps (RP, BD).
* Valves are linear: molar flow = Cv (P_upstream - P_downstream). In a
  constant-pressure step the column pressure is fixed, so the valve flow is a
  fixed flow; the other end is "free" (its flow follows from the balance).
* Raffinate (light-product) tank: well mixed, volume V, set pressure P_set.
  During FEED it first accumulates until it is back at P_set (psa-sim's
  "pre-fill"), then the net inflow is withdrawn as product. The purge draws
  from it, so its pressure dips during PURGE.

Steps: RP (pressurize with feed through the feed end, product end closed),
FEED (feed in at the feed end, light product out of the product end into the
tank), BD (counter-current blowdown: feed end vented, product end closed),
PURGE (counter-current purge with tank gas through the product end, feed end
vented). Gas leaving the feed end (BD + PURGE) is the heavy extract.

Index 0 = A (heavy, extract), index 1 = B (light, raffinate).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .isotherm import R_GAS, loading

KINDS = ("RP", "FEED", "BD", "PURGE")
STEP_LABELS = {
    "RP": "Pressurization (feed)",
    "FEED": "Adsorption (feed)",
    "BD": "Blowdown",
    "PURGE": "Purge",
}


@dataclass
class Step:
    kind: str
    duration: float  # s
    cv: float  # mol/(s bar), the step's one valve (feed end for RP/FEED/BD, product end for PURGE)


@dataclass
class ColumnCase:
    name: str = "generic_AB"
    length: float = 1.0  # m
    radius: float = 0.1  # m (inner)
    bed_void: float = 0.40
    pellet_density: float = 1100.0  # kg/m^3
    temperature: float = 298.15  # K
    qs: tuple = (3.0, 3.0)  # mol/kg (A, B)
    b: tuple = (0.60, 0.12)  # 1/bar
    k_ldf: tuple = (1.0, 2.5)  # 1/s
    y_feed: tuple = (0.30, 0.70)
    p_feed: float = 3.3  # bar, feed header (constant)
    p_vent: float = 1.0  # bar, vent at the feed end
    tank_volume: float = 1.0  # m^3
    tank_p_set: float = 2.9  # bar
    p_init: float = 1.0  # bar, initial column pressure (pure B, equilibrated)
    steps: list = field(default_factory=lambda: [
        Step("RP", 10.0, 1.1), Step("FEED", 50.0, 1.27),
        Step("BD", 10.0, 2.2), Step("PURGE", 50.0, 0.069)])
    n_cells: int = 100
    rtol: float = 1e-6
    atol_rel: float = 1e-9
    css_tol: float = 1e-7
    max_cycles: int = 400
    provenance: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["steps"] = [asdict(s) if not isinstance(s, dict) else s for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["steps"] = [Step(**s) for s in d["steps"]]
        for key in ("qs", "b", "k_ldf", "y_feed"):
            d[key] = tuple(d[key])
        return cls(**d)


class ColumnModel:
    """Method-of-lines model of one column + raffinate tank."""

    def __init__(self, case: ColumnCase):
        self.case = case
        N = self.N = int(case.n_cells)
        self.area = math.pi * case.radius**2
        self.dz = case.length / N
        self.eps = case.bed_void
        self.rho_s = (1.0 - case.bed_void) * case.pellet_density  # kg adsorbent per m^3 bed
        self.RT = R_GAS * case.temperature
        self.qs = np.asarray(case.qs, float)
        self.b = np.asarray(case.b, float)
        self.k = np.asarray(case.k_ldf, float)
        self.yf = np.asarray(case.y_feed, float)
        # layout: c (2N, component-major), q (2N), tank c (2), counters (8)
        self.s_c = slice(0, 2 * N)
        self.s_q = slice(2 * N, 4 * N)
        self.s_t = slice(4 * N, 4 * N + 2)
        self.s_cnt = slice(4 * N + 2, 4 * N + 10)
        self.n = 4 * N + 10
        self.c_ref = case.p_feed * 1e5 / self.RT
        self.n_ref = self.c_ref * self.eps * self.area * case.length
        atol = np.empty(self.n)
        atol[self.s_c] = case.atol_rel * self.c_ref
        atol[self.s_q] = case.atol_rel * float(self.qs.max())
        atol[self.s_t] = case.atol_rel * self.c_ref
        atol[self.s_cnt] = case.atol_rel * self.n_ref
        self.atol = atol
        self.n_rhs = 0

    # ------------------------------------------------------------ helpers
    def split(self, x):
        """Views (c[2,N], q[2,N], cR[2], counters[8]) of a state vector (or [n, K] block)."""
        N = self.N
        tail = x.shape[1:]
        return (x[self.s_c].reshape((2, N) + tail), x[self.s_q].reshape((2, N) + tail),
                x[self.s_t], x[self.s_cnt])

    def initial_state(self):
        c = self.case
        x = np.zeros(self.n)
        C0 = c.p_init * 1e5 / self.RT
        cc, qq, cR, _ = self.split(x)
        cc[1, :] = C0
        qq[:] = loading(np.array([0.0, c.p_init]), self.qs, self.b)[:, None]
        cR[1] = c.tank_p_set * 1e5 / self.RT
        return x

    def tank_pressure(self, x):
        return x[self.s_t].sum(0) * self.RT / 1e5

    def column_pressure(self, x):
        c, _, _, _ = self.split(x)
        return c.sum(0).mean(0) * self.RT / 1e5

    def inventory(self, x):
        """Moles of A and B held in column gas, adsorbent, and tank."""
        c, q, cR, _ = self.split(x)
        a, dz = self.area, self.dz
        return (self.eps * a * dz * c.sum(1) + self.rho_s * a * dz * q.sum(1)
                + self.case.tank_volume * cR)

    # ---------------------------------------------------------------- rhs
    def velocities(self, x, kind, cv):
        """Face superficial velocities u[N+1, K] and helper arrays for state block x[n, K]."""
        N, dz, eps = self.N, self.dz, self.eps
        c, q, cR, _ = self.split(x)
        C = c.sum(0)
        p = c * (self.RT / 1e5)
        bp = self.b[:, None, None] * np.maximum(p, 0.0)
        qstar = self.qs[:, None, None] * bp / (1.0 + bp.sum(0))
        dq = self.k[:, None, None] * (qstar - q)
        vads = self.rho_s * dz * dq.sum(0) / C  # m/s per cell
        P0 = C[0] * self.RT / 1e5
        PN = C[-1] * self.RT / 1e5
        Pt = cR.sum(0) * self.RT / 1e5
        K = x.shape[1]
        u = np.empty((N + 1, K))
        a = self.area
        if kind == "RP" or kind == "BD":
            p_up = self.case.p_feed if kind == "RP" else self.case.p_vent
            u0 = cv * (p_up - P0) / (a * C[0])
            uN = np.zeros(K)
            g = (u0 - uN - vads.sum(0)) / self.case.length
            u[0] = u0
            u[1:] = u0 - np.cumsum(vads + dz * g, axis=0)
            u[N] = uN
        elif kind == "FEED":
            u0 = cv * (self.case.p_feed - P0) / (a * C[0])
            u[0] = u0
            u[1:] = u0 - np.cumsum(vads, axis=0)
        elif kind == "PURGE":
            uN = cv * (PN - Pt) / (a * C[-1])
            u[N] = uN
            u[:N] = uN + np.cumsum(vads[::-1], axis=0)[::-1]
        else:
            raise ValueError(kind)
        return u, c, q, cR, C, dq

    def rhs(self, t, x, kind, cv, acc):
        self.n_rhs += 1
        vec = x.ndim == 2
        X = x if vec else x[:, None]
        N, dz, eps = self.N, self.dz, self.eps
        u, c, q, cR, C, dq = self.velocities(X, kind, cv)
        K = X.shape[1]
        cf = np.empty((2, N + 1, K))
        cf[:, 1:N] = np.where(u[1:N] >= 0.0, c[:, :N - 1], c[:, 1:])
        cf[:, 0] = np.where(u[0] >= 0.0, self.yf[:, None] * C[0], c[:, 0])
        yR = cR / cR.sum(0)
        cf[:, N] = np.where(u[N] >= 0.0, c[:, N - 1], yR * C[N - 1])
        flux = u * cf  # mol/(m^2 s), (2, N+1, K)
        out = np.empty_like(X)
        dc = (flux[:, :-1] - flux[:, 1:]) / (eps * dz) - (self.rho_s / eps) * dq
        out[self.s_c] = dc.reshape(2 * N, K)
        out[self.s_q] = dq.reshape(2 * N, K)
        a = self.area
        ex = a * flux[:, N] if kind in ("FEED", "PURGE") else np.zeros((2, K))
        if acc:
            prod = np.zeros((2, K))
        else:
            prod = np.maximum(0.0, ex.sum(0)) * yR
        out[self.s_t] = (ex - prod) / self.case.tank_volume
        f0 = a * flux[:, 0]
        cnt = np.zeros((8, K))
        if kind in ("RP", "FEED"):
            cnt[0:2] = f0  # feed in
        else:
            cnt[2:4] = -f0  # vented (extract)
        cnt[4:6] = ex  # column -> tank (net)
        cnt[6:8] = prod  # light product withdrawn
        out[self.s_cnt] = cnt
        return out if vec else out[:, 0]

    # ---------------------------------------------------------- integrate
    def _solve(self, x0, t0, t1, kind, cv, acc, events=None):
        sol = solve_ivp(self.rhs, (t0, t1), x0, method="BDF", args=(kind, cv, acc),
                        rtol=self.case.rtol, atol=self.atol, vectorized=True,
                        dense_output=True, events=events)
        if sol.status < 0:
            raise RuntimeError(f"{kind}: {sol.message}")
        return sol

    def run_step(self, x_start, step: Step):
        """Integrate one step. Returns (segments [(t0, t1, dense)], x_end, prefill_time)."""
        x0 = x_start.copy()
        x0[self.s_cnt] = 0.0
        segs = []
        t0 = 0.0
        prefill_t = None
        if step.kind == "FEED":
            p_set = self.case.tank_p_set
            if (self.tank_pressure(x0) - p_set) / p_set < -1e-9:
                def ev(t, y, *args):
                    return self.tank_pressure(y) - p_set
                ev.terminal, ev.direction = True, 1
                sol = self._solve(x0, 0.0, step.duration, step.kind, step.cv, True, events=[ev])
                segs.append((0.0, sol.t[-1], sol.sol))
                x0 = sol.y[:, -1].copy()
                t0 = sol.t[-1]
                if sol.status == 1:
                    prefill_t = t0
        if step.duration - t0 > 1e-12 * step.duration:
            sol = self._solve(x0, t0, step.duration, step.kind, step.cv, False)
            segs.append((t0, step.duration, sol.sol))
            x_end = sol.y[:, -1].copy()
        else:
            x_end = x0
        return segs, x_end, prefill_t

    @staticmethod
    def sample(segs, t):
        """Evaluate a step's piecewise dense output at times t (array)."""
        t = np.atleast_1d(t)
        out = np.empty((segs[0][2](segs[0][0]).shape[0], t.size))
        for j, tj in enumerate(t):
            for (a, b, f) in segs:
                if tj <= b + 1e-12:
                    out[:, j] = f(min(max(tj, a), b))
                    break
            else:
                out[:, j] = segs[-1][2](segs[-1][1])
        return out

    def run_cycle(self, x_start):
        """One cycle. Returns (x_end, per-step records)."""
        x = x_start.copy()
        recs = []
        for s, step in enumerate(self.case.steps):
            inv0 = self.inventory(x)
            segs, x_end, prefill_t = self.run_step(x, step)
            cnt = x_end[self.s_cnt]
            inv1 = self.inventory(x_end)
            feed_in, vent, to_tank, prod = cnt[0:2], cnt[2:4], cnt[4:6], cnt[6:8]
            # column + tank system: in = feed, out = vent + product
            imbalance = feed_in - vent - prod - (inv1 - inv0)
            recs.append(dict(step=s, kind=step.kind, segs=segs, x0=x.copy(), x_end=x_end.copy(),
                             feed_in=feed_in.copy(), vent=vent.copy(), to_tank=to_tank.copy(),
                             product=prod.copy(), imbalance=imbalance, prefill_t=prefill_t))
            x = x_end.copy()
        return x, recs

    def css_change(self, x_new, x_old):
        d = np.concatenate([(x_new[self.s_c] - x_old[self.s_c]) / self.c_ref,
                            (x_new[self.s_q] - x_old[self.s_q]) / float(self.qs.max()),
                            (x_new[self.s_t] - x_old[self.s_t]) / self.c_ref])
        return float(np.sqrt(np.mean(d * d)))

    @staticmethod
    def cycle_metrics(recs):
        feed = sum(r["feed_in"] for r in recs)
        vent = sum(r["vent"] for r in recs)
        prod = sum(r["product"] for r in recs)
        imb = np.array([r["imbalance"] for r in recs])  # (steps, 2)
        out = dict(
            feed_mol=feed.tolist(), extract_mol=vent.tolist(), raffinate_mol=prod.tolist(),
            purity_B=float(prod[1] / prod.sum()) if prod.sum() > 0 else float("nan"),
            recovery_B=float(prod[1] / feed[1]),
            purity_A_extract=float(vent[0] / vent.sum()),
            recovery_A=float(vent[0] / feed[0]),
            step_balance_rel=float(np.max(np.abs(imb) / feed[None, :])),
            css_balance_rel=float(np.max(np.abs(feed - vent - prod) / feed)),
            prefill_s=[r["prefill_t"] for r in recs if r["kind"] == "FEED"][0],
        )
        return out

    def run_to_css(self, x0=None, progress=None):
        x = self.initial_state() if x0 is None else x0.copy()
        hist = []
        recs = None
        for cyc in range(1, self.case.max_cycles + 1):
            x_new, recs = self.run_cycle(x)
            dx = self.css_change(x_new, x)
            m = self.cycle_metrics(recs)
            m.update(cycle=cyc, css_change=dx)
            hist.append(m)
            if progress:
                progress(m)
            x = x_new
            if dx < self.case.css_tol:
                break
        return x, recs, hist

    # ------------------------------------------------------- frame output
    def profiles(self, recs, n_per_step):
        """Sample a cycle's records into arrays for plotting/rendering.

        ``n_per_step``: list of sample counts per step (or an int). Returns a dict of
        arrays over samples: t (s, from cycle start), step index, y_A[N], q[2,N],
        P, u[N+1], tank pressure and y_A, cumulative counters within the step.
        """
        if np.isscalar(n_per_step):
            n_per_step = [int(n_per_step)] * len(recs)
        T, S, Y, Q, P, U, PT, YT, CNT, TL = ([] for _ in range(10))
        t_off = 0.0
        for r, ns in zip(recs, n_per_step):
            step = self.case.steps[r["step"]]
            tt = np.linspace(0.0, step.duration, ns, endpoint=False) if r is not recs[-1] \
                else np.linspace(0.0, step.duration, ns)
            X = self.sample(r["segs"], tt)
            u, c, q, cR, C, _ = self.velocities(X, step.kind, step.cv)
            T.append(t_off + tt)
            TL.append(tt)
            S.append(np.full(tt.size, r["step"]))
            Y.append((c[0] / C).T)
            Q.append(np.moveaxis(q, -1, 0))
            P.append(C.mean(0) * self.RT / 1e5)
            U.append(u.T)
            PT.append(cR.sum(0) * self.RT / 1e5)
            YT.append(cR[0] / cR.sum(0))
            CNT.append(X[self.s_cnt].T)
            t_off += step.duration
        cat = np.concatenate
        return dict(t=cat(T), t_in_step=cat(TL), step=cat(S), y_A=cat(Y), q=cat(Q), P=cat(P),
                    u=cat(U), p_tank=cat(PT), y_tank=cat(YT), counters=cat(CNT),
                    z=(np.arange(self.N) + 0.5) / self.N)
