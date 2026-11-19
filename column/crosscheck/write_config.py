"""Write the psa-sim config equivalent to psaviz.cases.column_case (env viz).

psa-sim orders light keys first, so components are [B, A] there.
Run: PYTHONPATH=../.. conda run -n viz python -I write_config.py [N]
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from psaviz.cases import P_HIGH_NOMINAL, column_case  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
c = column_case(n_cells=N)
A, B = 0, 1
order = [B, A]
st = {s.kind: s for s in c.steps}


def tank(p, vol, ic):
    r = 0.5
    return dict(inner_radius=r, outer_radius=r + 0.005, height=vol / (math.pi * r * r), pressure=p,
                temperature=c.temperature, initial_condition=ic, wall_density=7830.0, wall_cp=500.0,
                h_in=0.0, h_out=0.0, p_set=None, p_full=None, cv_full=None)


cfg = dict(
    name=f"psaviz_crosscheck_N{N}",
    description="Single-column 4-step Skarstrom cycle equivalent to psaviz.cases.column_case (cross-check only)",
    components=[dict(name=n, mw=30.0, cp=29.1, cv=20.8, dH=0.0, z=1.0, k_ldf=float(c.k_ldf[i]))
                for n, i in (("B", B), ("A", A))],
    feed=dict(y=[float(c.y_feed[i]) for i in order], temperature=c.temperature, density=1.2, viscosity=1.8e-5),
    products=dict(n_light_keys=1, y_raffinate=[1.0, 0.0], y_extract=[0.0, 1.0]),
    adsorbent=dict(name="generic", pellet_diameter=2e-3, pellet_density=c.pellet_density,
                   bulk_density=c.pellet_density * (1 - c.bed_void), solid_cp=900.0,
                   mass_transfer_resistance=1, macropore_void=0.0, micropore_void=0.0,
                   isotherm=dict(model="langmuir",
                                 params=[dict(qs=float(c.qs[i]), b=float(c.b[i]), dH=0.0) for i in order],
                                 t_ref=c.temperature, temperature_dependent=False, validity=None)),
    column=dict(n_columns=1, height=c.length, inner_radius=c.radius, outer_radius=c.radius + 0.01,
                bed_void=c.bed_void, adsorbent_mass=c.pellet_density * (1 - c.bed_void) * math.pi * c.radius**2 * c.length,
                wall_density=7830.0, wall_cp=500.0, h_in=0.0, h_out=0.0,
                p_high=P_HIGH_NOMINAL, p_low=c.p_init, temperature=c.temperature, initial_condition=[5],
                p_high_set=None, p_high_full=None, p_low_set=None, p_low_full=None),
    feed_tank=tank(c.p_feed, 1.0, 1),
    raffinate_tank=tank(c.tank_p_set, c.tank_volume, 2),
    extract_tank=tank(c.p_vent, 1.0, 2),
    machines=dict(feed_compressor_efficiency=0.75, extract_compressor_efficiency=0.75, vacuum_pump_efficiency=0.75,
                  feed_compressor_stages=1, extract_compressor_stages=1, vacuum_pump_stages=1,
                  vacuum_pressure=c.p_vent, feed_compression=False, recover_extract=False, vacuum=False),
    ambient=dict(temperature=c.temperature, pressure=1.01325, t_std=273.15, p_std=1.01325),
    cycle=dict(durations=[s.duration for s in c.steps],
               steps=[["RP-FEE-XXX", "HP-FEE-RAF", "DP-ATM-XXX", "LP-ATM-RAF"]],
               valve_feed=[[st["RP"].cv, st["FEED"].cv, st["BD"].cv, "free"]],
               valve_product=[[0, "free", 0, st["PURGE"].cv]],
               flow_dir=[[0, 0, 1, 1]], step_type=None, events=None, max_cycles=200),
    numerics=dict(n_cells=N, rtol=1e-6, atol=1e-9, css_tol=1e-14, css_metric="all", css_inventory_tol=1e-3,
                  n_time_points=51, method="BDF", topsail_compat=False),
    options=dict(energy_balance=False, pressure_drop="none"),
)
out = Path(__file__).with_name("psasim_config.json")
out.write_text(json.dumps(cfg, indent=1))
print("wrote", out)
