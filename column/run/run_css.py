"""Run the render case (psaviz.cases.column_case) to CSS and save it (env viz).

Writes css_state.npz (CSS state + case), metrics.json (CSS-cycle performance),
cycle_history.csv (approach to CSS). Rendering re-integrates one cycle from
css_state.npz, so dense profiles are never stored in git.
Run: conda run -n viz python -I run_css.py [N]
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1]))
from psaviz.cases import column_case  # noqa: E402
from psaviz.column import ColumnModel  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
m = ColumnModel(column_case(n_cells=N))
t0 = time.time()
x, recs, hist = m.run_to_css()
wall = time.time() - t0
np.savez(here / "css_state.npz", x_start=recs[0]["x0"], case=json.dumps(m.case.to_dict()))
met = dict(hist[-1])
met.update(wall_s=wall, P_step_end_bar=[m.column_pressure(r["x_end"]) for r in recs],
           P_tank_step_end_bar=[m.tank_pressure(r["x_end"]) for r in recs],
           purge_mol=(-recs[3]["to_tank"]).tolist(),
           purge_to_feed=float(-recs[3]["to_tank"].sum() / sum(r["feed_in"].sum() for r in recs)))
(here / "metrics.json").write_text(json.dumps(met, indent=1, default=float))
with open(here / "cycle_history.csv", "w", newline="") as f:
    keys = ["cycle", "css_change", "purity_B", "recovery_B", "purity_A_extract", "recovery_A",
            "step_balance_rel", "css_balance_rel", "prefill_s"]
    w = csv.writer(f)
    w.writerow(keys)
    for h in hist:
        w.writerow([h[k] if h[k] is not None else "" for k in keys])
print(json.dumps(met, indent=1, default=float))
