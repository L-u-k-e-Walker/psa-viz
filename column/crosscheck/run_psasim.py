"""Run psa-sim on psasim_config.json and save the CSS cycle (env psa; reference only).

Run: conda run -n psa python -I run_psasim.py
Writes psasim_profiles.csv (last cycle, all stored times) and psasim_metrics.json.
"""
import importlib.util  # noqa: F401  (pygaps import order on this machine)
import json
import time
from pathlib import Path

from psasim.config import Config
from psasim.cycle.runner import simulate
from psasim.results import column_profiles, pressure_history

here = Path(__file__).parent
cfg = Config.from_dict(json.loads((here / "psasim_config.json").read_text()))
t0 = time.time()
res = simulate(cfg)
wall = time.time() - t0
last = res.n_cycles
prof = column_profiles(res, last)
prof.to_csv(here / "psasim_profiles.csv", index=False, float_format="%.10g")
ph = pressure_history(res, cycles=[last])
ph.to_csv(here / "psasim_pressure.csv", index=False, float_format="%.10g")
m = res.cycle_metrics[-1]
keep = {k: m[k] for k in ("cycle", "raff_purity", "raff_recovery", "extr_purity", "extr_recovery",
                         "feed_mol", "raff_prod_mol", "extr_waste_mol", "mass_balance_rel", "accumulation_rel")}
keep.update(converged=bool(res.converged), css=float(res.css[-1]), wall_s=wall, n_cycles=res.n_cycles,
            prefill_s=[s.prefill_time for s in res.cycle_steps(last)])
(here / "psasim_metrics.json").write_text(json.dumps(keep, indent=1, default=float))
print(json.dumps(keep, indent=1, default=float))
