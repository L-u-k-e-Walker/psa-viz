"""Gate G1e: psaviz column model vs psa-sim on the same case and grid (env viz).

Needs psasim_profiles.csv / psasim_pressure.csv / psasim_metrics.json from run_psasim.py
(committed). Writes crosscheck.json and crosscheck.png.
Run: conda run -n viz python -I compare.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1]))
from psaviz.cases import column_case  # noqa: E402
from psaviz.column import ColumnModel  # noqa: E402

cfg = json.loads((here / "psasim_config.json").read_text())
N = cfg["numerics"]["n_cells"]
ref_m = json.loads((here / "psasim_metrics.json").read_text())

# psa-sim profiles: rows (step, t_in_cycle, z)
rows = list(csv.DictReader(open(here / "psasim_profiles.csv")))
steps = np.array([int(r["step"]) - 1 for r in rows])
tic = np.array([float(r["t_in_cycle_s"]) for r in rows])
yA = np.array([float(r["y_A"]) for r in rows])
qA = np.array([float(r["q_A_mol_kg"]) for r in rows])
qB = np.array([float(r["q_B_mol_kg"]) for r in rows])
P = np.array([float(r["P_bar"]) for r in rows])
ns = len(rows) // N
yA, qA, qB, P = (a.reshape(ns, N) for a in (yA, qA, qB, P))
st, tc = steps.reshape(ns, N)[:, 0], tic.reshape(ns, N)[:, 0]
prow = list(csv.DictReader(open(here / "psasim_pressure.csv")))
p_tank_ref = np.array([float(r["raffinate_tank_bar"]) for r in prow])

m = ColumnModel(column_case(n_cells=N, css_tol=1e-7))
x, recs, hist = m.run_to_css()
met = hist[-1]
starts = np.cumsum([0.0] + [s.duration for s in m.case.steps])
my_y = np.empty_like(yA); my_qA = np.empty_like(yA); my_qB = np.empty_like(yA); my_P = np.empty(ns); my_pt = np.empty(ns)
for j in range(ns):
    s = st[j]
    X = m.sample(recs[s]["segs"], np.array([tc[j] - starts[s]]))[:, 0]
    c, q, cR, _ = m.split(X)
    my_y[j] = c[0] / c.sum(0); my_qA[j] = q[0]; my_qB[j] = q[1]
    my_P[j] = m.column_pressure(X); my_pt[j] = m.tank_pressure(X)

end_idx = [np.where(st == s)[0][-1] for s in range(4)]
d_end = [float(np.max(np.abs(my_y[i] - yA[i]))) for i in end_idx]
out = dict(
    N=N, psasim_cycles=ref_m["n_cycles"], psaviz_cycles=met["cycle"],
    purity_B=dict(psaviz=met["purity_B"], psasim=ref_m["raff_purity"],
                  diff_pp=100 * (met["purity_B"] - ref_m["raff_purity"])),
    recovery_B=dict(psaviz=met["recovery_B"], psasim=ref_m["raff_recovery"],
                    diff_pp=100 * (met["recovery_B"] - ref_m["raff_recovery"])),
    yA_step_end_maxabs=dict(zip(["RP", "FEED", "BD", "PURGE"], d_end)),
    yA_all_times_maxabs=float(np.max(np.abs(my_y - yA))),
    q_all_times_maxabs_mol_kg=float(max(np.max(np.abs(my_qA - qA)), np.max(np.abs(my_qB - qB)))),
    P_col_maxrel=float(np.max(np.abs(my_P - P.mean(1)) / P.mean(1))),
    P_tank_maxrel=float(np.max(np.abs(my_pt - p_tank_ref) / p_tank_ref)),
    prefill_s=dict(psaviz=met["prefill_s"], psasim=ref_m["prefill_s"][1]),
)
crit = dict(purity=abs(out["purity_B"]["diff_pp"]) < 0.3, recovery=abs(out["recovery_B"]["diff_pp"]) < 0.3,
            yA=max(d_end) < 0.02, P=out["P_col_maxrel"] < 0.01)
out["criteria"] = crit
out["G1e_pass"] = all(crit.values())
(here / "crosscheck.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
z = (np.arange(N) + 0.5) / N
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for s, i in enumerate(end_idx):
    ax[0].plot(z, yA[i], lw=3, alpha=0.35, color=f"C{s}")
    ax[0].plot(z, my_y[i], lw=1, color=f"C{s}", label=f"end of {m.case.steps[s].kind}")
ax[0].set(xlabel="z/L", ylabel="y_A", title="Gas-phase A: psa-sim (thick) vs psaviz (thin)")
ax[0].legend(frameon=False)
ax[1].plot(tc, P.mean(1), lw=3, alpha=0.35, color="k", label="column, psa-sim")
ax[1].plot(tc, my_P, lw=1, color="k", label="column, psaviz")
ax[1].plot(tc, p_tank_ref, lw=3, alpha=0.35, color="C3", label="tank, psa-sim")
ax[1].plot(tc, my_pt, lw=1, color="C3", label="tank, psaviz")
ax[1].set(xlabel="t in cycle (s)", ylabel="P (bar)", title="Pressures")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(here / "crosscheck.png", dpi=130)
