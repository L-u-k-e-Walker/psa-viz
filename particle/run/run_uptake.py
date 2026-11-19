"""Uptake curves for both particles (env viz). Writes uptake.csv and uptake_metrics.json.

Rendering re-solves the particles (< 1 s each), so radial profiles are not stored.
Run: conda run -n viz python -I run_uptake.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1]))
from psaviz.cases import PARTICLE_PROVENANCE, Y_FEED, equilibrium_particle, kinetic_particle  # noqa: E402

t = np.concatenate([[0.0], np.geomspace(1e-3, 3e4, 400)])
cols, meta = {"t_s": t}, {}
for k, f in (("eq", equilibrium_particle), ("kin", kinetic_particle)):
    p = f()
    X = p.solve(t).y
    q = p.mean_loading(X)
    cols[f"{k}_qA_mol_kg"], cols[f"{k}_qB_mol_kg"] = q
    meta[k] = dict(q_eq=p.q_eq.tolist(), peak_qA=float(q[0].max()), t_peak_qA_s=float(t[q[0].argmax()]),
                   peak_qB=float(q[1].max()), t_peak_qB_s=float(t[q[1].argmax()]),
                   t_half_A_s=float(np.interp(0.5 * p.q_eq[0], q[0][:q[0].argmax() + 1], t[:q[0].argmax() + 1])),
                   t_half_B_s=float(np.interp(0.5 * p.q_eq[1], q[1][:q[1].argmax() + 1], t[:q[1].argmax() + 1])))
meta["y_feed"] = list(Y_FEED)
meta["provenance"] = PARTICLE_PROVENANCE
with open(here / "uptake.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols.keys())
    w.writerows(zip(*[np.round(v, 10) for v in cols.values()]))
(here / "uptake_metrics.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
