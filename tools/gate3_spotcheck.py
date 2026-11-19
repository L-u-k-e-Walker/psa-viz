"""Gate 3 spot check: values drawn in the stills come straight from the solvers."""
import json, sys
import numpy as np
sys.path[:0] = ["column/render", "particle/render", "."]
from scene import CycleData
from psaviz.column import ColumnCase, ColumnModel
from render_particle import ParticleData
from psaviz.cases import equilibrium_particle, kinetic_particle
out = {}
cyc = CycleData()
d = np.load("column/run/css_state.npz")
x0 = d["x_start"]
x_end = cyc.recs[-1]["x_end"]
m = cyc.m
dn = np.concatenate([(x_end[m.s_c] - x0[m.s_c]) / m.c_ref, (x_end[m.s_q] - x0[m.s_q]) / m.qs.max(),
                     (x_end[m.s_t] - x0[m.s_t]) / m.c_ref])
out["column_periodicity_maxabs_normalized"] = float(np.max(np.abs(dn)))  # loop seam: end vs start of the CSS cycle
m2 = ColumnModel(ColumnCase.from_dict(json.loads(str(d["case"]))))
_, recs2 = m2.run_cycle(x0)
diffs = []
for t in (3.0, 52.0, 105.0):
    st = cyc.state(t)
    s = int(np.searchsorted(cyc.starts, t, side="right") - 1)
    X = m2.sample(recs2[s]["segs"], np.array([t - cyc.starts[s]]))[:, 0]
    c, q, _, _ = m2.split(X)
    diffs.append(max(np.max(np.abs(st["yA"] - c[0] / c.sum(0))), np.max(np.abs(st["qA"] - q[0])), np.max(np.abs(st["qB"] - q[1]))))
out["column_spotcheck_maxabs"] = float(max(diffs))
ts = [0.3, 3.0, 3000.0]
pdata = ParticleData(ts)
pd_diffs = []
for k, f in enumerate((equilibrium_particle, kinetic_particle)):
    p = f()
    X = p.solve(np.array([0.0] + ts)).y
    th = (p.split(X)[1] / p.qs[:, None, None]) if hasattr(p, "eps") else p.split(X)[0]
    for i, t in enumerate(ts):
        prof, j = pdata.at(t)
        pd_diffs.append(max(np.max(np.abs(prof[k]["thA"] - th[0, :, i + 1])), np.max(np.abs(prof[k]["thB"] - th[1, :, i + 1]))))
out["particle_spotcheck_maxabs"] = float(max(pd_diffs))
out["pass"] = out["column_spotcheck_maxabs"] < 1e-6 and out["particle_spotcheck_maxabs"] < 1e-6 and out["column_periodicity_maxabs_normalized"] < 1e-4
print(json.dumps(out, indent=1))
json.dump(out, open("docs/gate3_spotcheck.json", "w"), indent=1)
