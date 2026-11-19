"""Phase 1 gates G1a-G1e for the column model (env viz). Writes gates_phase1.json.

Run: conda run -n viz python -I gates.py   (~5 min; the 2N grid run dominates)
"""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1]))
from psaviz.cases import column_case  # noqa: E402
from psaviz.column import ColumnModel, Step  # noqa: E402
from psaviz.isotherm import loading  # noqa: E402

N = 120
res = {}

# ---- G1a / G1b on the render case
m = ColumnModel(column_case(n_cells=N))
x, recs, hist = m.run_to_css()
res["G1a"] = dict(max_step_balance_rel_all_cycles=max(h["step_balance_rel"] for h in hist), limit=1e-4)
res["G1a"]["pass"] = res["G1a"]["max_step_balance_rel_all_cycles"] < 1e-4
res["G1b"] = dict(cycles=hist[-1]["cycle"], css_change=hist[-1]["css_change"], css_change_limit=1e-6,
                  css_balance_rel=hist[-1]["css_balance_rel"], css_balance_limit=5e-3)
res["G1b"]["pass"] = hist[-1]["css_change"] < 1e-6 and hist[-1]["css_balance_rel"] < 5e-3

# ---- G1c trace-A breakthrough into a B-saturated bed at constant pressure
P0 = 4.2
c = column_case(n_cells=N, y_feed=(1e-4, 1 - 1e-4))
mt = ColumnModel(c)
cv = c.steps[1].cv
C0 = P0 * 1e5 / mt.RT
u = cv * (c.p_feed - P0) / (mt.area * C0)
slope = c.qs[0] * c.b[0] / (1 + c.b[1] * P0) * mt.RT / 1e5  # dq_A/dc_A, m^3/kg
t_R = c.length / u * (c.bed_void + mt.rho_s * slope)
x0 = np.zeros(mt.n)
cc, qq, cR, _ = mt.split(x0)
cc[1] = C0
qq[:] = loading(np.array([0.0, P0]), c.qs, c.b)[:, None]
cR[1] = c.tank_p_set * 1e5 / mt.RT
t_end = 4.0 * t_R
segs, _, _ = mt.run_step(x0, Step("FEED", t_end, cv))
tt = np.linspace(0, t_end, 8001)
X = mt.sample(segs, tt)
cg, _, _, _ = mt.split(X)
y_out = cg[0, -1] / cg[:, -1].sum(0)
first_moment = float(np.trapezoid(1.0 - y_out / c.y_feed[0], tt))
res["G1c"] = dict(t_R_theory_s=t_R, first_moment_s=first_moment, rel_err=abs(first_moment - t_R) / t_R,
                  y_out_end_over_feed=float(y_out[-1] / c.y_feed[0]), limit=0.01)
res["G1c"]["pass"] = res["G1c"]["rel_err"] < 0.01 and res["G1c"]["y_out_end_over_feed"] > 0.999

# ---- G1d grid: N vs 2N
m2 = ColumnModel(column_case(n_cells=2 * N))
_, _, h2 = m2.run_to_css()
a, b2 = hist[-1], h2[-1]
res["G1d"] = dict(N=N, purity_B=[a["purity_B"], b2["purity_B"]], recovery_B=[a["recovery_B"], b2["recovery_B"]],
                  d_purity_pp=100 * abs(a["purity_B"] - b2["purity_B"]),
                  d_recovery_pp=100 * abs(a["recovery_B"] - b2["recovery_B"]), limits_pp=[0.2, 1.0])
res["G1d"]["pass"] = res["G1d"]["d_purity_pp"] < 0.2 and res["G1d"]["d_recovery_pp"] < 1.0

# ---- G1e from the committed cross-check
xc = json.loads((here.parent / "crosscheck" / "crosscheck.json").read_text())
res["G1e"] = dict(purity_diff_pp=xc["purity_B"]["diff_pp"], recovery_diff_pp=xc["recovery_B"]["diff_pp"],
                  yA_step_end_maxabs=max(xc["yA_step_end_maxabs"].values()), P_maxrel=xc["P_col_maxrel"],
                  **{"pass": xc["G1e_pass"]})
res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
(here / "gates_phase1.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
