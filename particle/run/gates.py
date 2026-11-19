"""Phase 2 gates G2a-G2e for the particle models (env viz). Writes gates_phase2.json.

Run: conda run -n viz python -I gates.py
"""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1]))
from psaviz.cases import PELLET, T, Y_FEED, equilibrium_particle, kinetic_particle  # noqa: E402
from psaviz.particle import crank_sphere  # noqa: E402

N = 100
T_END = 3e4
res = {}


def run(p, n=500):
    t = p.times(T_END, n, 1e-3)
    sol = p.solve(t)
    return t, sol.y


models = {"equilibrium": equilibrium_particle, "kinetic": kinetic_particle}
out = {k: run(f(N=N)) + (f(N=N),) for k, f in models.items()}

# G2a mass balance: held moles vs integrated surface inflow
g2a = {}
for k, (t, X, p) in out.items():
    held = p.moles(X) - p.moles(X[:, 0])[:, None]
    cum = X[-2:] if k == "equilibrium" else p.cum_moles(X)
    g2a[k] = float(np.max(np.abs(held - cum)) / np.max(np.abs(held[:, -1])))
res["G2a"] = dict(rel_err=g2a, limit=1e-4, **{"pass": max(g2a.values()) < 1e-4})

# G2b analytic: single component, linear isotherm limit vs Crank
tau = np.geomspace(1e-4, 1.5, 300)
pm = equilibrium_particle(N=N, qs=(3e5, 3e5), b=(1e-5, 1e-5), y=(1.0, 0.0), k_crystal=1e5)
K = pm.rho * pm.qs[0] * pm.b[0] * pm.RT / 1e5
D_eff = pm.eps * pm.D[0] / (pm.eps + K)
R2 = pm.g.R ** 2
sol = pm.solve(np.concatenate([[0.0], tau * R2 / D_eff]))
F_m = pm.moles(sol.y)[0, 1:] / (pm.g.Vtot * (pm.eps + K) * pm.cs[0])
pk = kinetic_particle(N=N, b=(1e-5, 1e-5), y=(1.0, 0.0))
sol = pk.solve(np.concatenate([[0.0], tau * R2 / pk.D[0]]))
F_k = pk.moles(sol.y)[0, 1:] / (pk.g.Vtot * pk.rho * pk.qs[0] * pk.th_s[0])
F_c = crank_sphere(tau)
res["G2b"] = dict(maxabs_macropore=float(np.max(np.abs(F_m - F_c))), maxabs_micropore=float(np.max(np.abs(F_k - F_c))),
                  tau_range=[1e-4, 1.5], limit=0.005)
res["G2b"]["pass"] = max(res["G2b"]["maxabs_macropore"], res["G2b"]["maxabs_micropore"]) < 0.005

# G2c final loadings vs extended Langmuir at the surface
g2c = {}
for k, (t, X, p) in out.items():
    q_end = p.mean_loading(X)[:, -1]
    g2c[k] = dict(final=q_end.tolist(), langmuir=p.q_eq.tolist(), max_rel=float(np.max(np.abs(q_end / p.q_eq - 1))))
res["G2c"] = dict(**g2c, limit=5e-3, **{"pass": all(v["max_rel"] < 5e-3 for v in g2c.values())})

# G2d grid N vs 2N
g2d = {}
for k, f in models.items():
    t, X, p = out[k]
    p2 = f(N=2 * N)
    t2, X2 = run(p2)
    d = np.abs(p.mean_loading(X) - p2.mean_loading(X2)) / p.qs[:, None]
    g2d[k] = float(d.max())
res["G2d"] = dict(maxabs_dq_over_qs=g2d, limit=5e-3, **{"pass": max(g2d.values()) < 5e-3})

# G2e mechanism
yr = Y_FEED[0] / Y_FEED[1]
t, X, p = out["equilibrium"]
qe = p.mean_loading(X)
S_eq_final = float(qe[0, -1] / qe[1, -1] / yr)
t, X, p = out["kinetic"]
qk = p.mean_loading(X)
j90 = int(np.argmax(qk[0] >= 0.9 * qk[0].max()))
S_k90 = float(qk[0, j90] / qk[1, j90] / yr)
S_k_final = float(qk[0, -1] / qk[1, -1] / yr)
res["G2e"] = dict(S_equilibrium_final=S_eq_final, t_A90_kinetic_s=float(t[j90]), S_kinetic_at_tA90=S_k90,
                  S_kinetic_final=S_k_final,
                  overshoot_B_equilibrium=float(qe[1].max() / qe[1, -1]),
                  overshoot_A_kinetic=float(qk[0].max() / qk[0, -1]),
                  limits=dict(S_eq_final=">= 3", S_kin_at_tA90=">= 3", S_kin_final="0.7-1.4"))
res["G2e"]["pass"] = S_eq_final >= 3 and S_k90 >= 3 and 0.7 <= S_k_final <= 1.4
res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
(here / "gates_phase2.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
