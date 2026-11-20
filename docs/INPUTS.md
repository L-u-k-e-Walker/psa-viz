# Inputs and where they come from

**Nothing in this project is measured.** The species are generic (A, B), and
every number is either *assumed* (an illustrative value, chosen to look like a
real sorbent class, with the reason given) or *calculated* from other inputs.
The visuals show mechanisms, not a specific material. Code: `psaviz/cases.py`.

A = heavy component = extract (more strongly adsorbed in the column and the
equilibrium particle, faster in the kinetic particle). B = light component =
raffinate (the product of the feed step).

## Column (`psaviz.cases.column_case`)

| Input | Value | Source |
|---|---|---|
| Bed length, radius | 1.0 m, 0.10 m (L/D = 5) | assumed |
| Bed void ε | 0.40 | assumed: typical random packing of spheres |
| Pellet density ρ_p | 1100 kg/m³ | assumed: zeolite-pellet-like |
| Temperature | 298.15 K, isothermal | assumed (user choice: no energy balance) |
| Isotherm | extended Langmuir, q_s = 3.0 mol/kg (A and B), b_A = 0.60 /bar, b_B = 0.06 /bar | assumed: Henry selectivity b_A/b_B = 10 gives a clear separation; equal q_s keeps the extended Langmuir model thermodynamically consistent |
| LDF k_A, k_B | 0.347 /s, 4.92 /s | **calculated**: Glueckauf from the pellet below, k_i = 15 ε_p D_p / (R_p² (ε_p + ρ_p Δq_i/Δc_i)), chord from pure B at 1 bar to feed at 4.2 bar. Same pore diffusivity for A and B, so the difference is the isotherm slope |
| Feed | y_A = 0.30 | assumed |
| Feed header | 4.3 bar, constant | assumed |
| Vent (feed end) | 1.0 bar | assumed |
| Light-product tank | 1.0 m³, set pressure 3.9 bar | assumed: large enough that the purge draws it down by < 10 % |
| Initial state | pure B at 1 bar, equilibrated; tank pure B at 3.9 bar | assumed (CSS does not depend on it) |
| Steps (duration, valve Cv) | RP 10 s, 1.5 · FEED 50 s, 7.0 · BD 10 s, 5.0 · PURGE 50 s, 0.11 mol/(s·bar) | assumed: tuned so the A front reaches ~0.6 L at the end of feed, B purity > 99.5 % and recovery ~ 34 % (realistic for a 4-step Skarstrom cycle at pressure ratio ~4) |
| Grid | 120 cells, first-order upwind (≈ axial dispersion with Pe ≈ 240) | numerical choice, grid-checked (G1d) |

Outcomes, not inputs: the column reaches **P_H = 4.208 bar** at the end of
pressurization and stays there in feed (feed flow = Cv (4.3 − P_H)); blowdown
ends at **1.009 bar**. On screen, pressure is shown as P/P_H with P_H = 4.208 bar.
CSS performance: B purity 99.858 %, B recovery 33.8 %, extract 39.3 % A,
molar purge/feed 0.33 (`column/run/metrics.json`).

Simplifications (all as in psa-sim's flow-controlled, isothermal,
no-pressure-drop mode): no momentum balance (uniform pressure along the bed),
gas in the pellet pores neglected in the column balance, linear valves, LDF
mass transfer, ideal gas.

### Render geometry
The 3D bed in the column render is schematic. Its particles are drawn far
larger than 2 mm pellets in a 0.2 m column would be, so that they are visible.
Colours come from the solver's profiles at each particle's height z/L.

## Particles (`psaviz.cases.equilibrium_particle`, `kinetic_particle`)

Both start clean. At t = 0 their surface sees the column feed (y_A = 0.30,
4.2 bar, 298.15 K), held constant (standalone uptake experiment, not driven by
the column).

| Input | Equilibrium-selective (left) | Kinetic-selective (right) | Source |
|---|---|---|---|
| Model | macropore diffusion + crystal LDF | micropore (adsorbed-phase) diffusion, Maxwell–Stefan/Langmuir flux | assumed: the usual controlling resistance for each class (zeolite pellet vs carbon molecular sieve) |
| Radius | 1.0 mm | 1.0 mm | assumed: equal so they compare directly |
| Porosity, density | ε_p 0.35, ρ_p 1100 kg/m³ | ρ_p 1100 kg/m³ | assumed |
| Isotherm | b_A 0.60, b_B 0.06 /bar, q_s 3 (same as column) | b_A = b_B = 0.25 /bar, q_s 3 | assumed: selectivity 10 vs 1 at equilibrium |
| Transport | D_p = 1.7e-6 m²/s for A and B; crystal LDF 100 /s | Đ_A/R² = 1e-2 /s, Đ_B/R² = 1e-4 /s | assumed: D_p from Bosanquet (Knudsen in ~0.1 µm pores + molecular at ~4 bar, tortuosity 3), equal for equal molar masses; Đ ratio 100 (CMS O2/N2 is ~30–100) |
| Grid | 100 shells | 100 shells | numerical, grid-checked (G2d) |

Outcomes (`particle/run/uptake_metrics.json`): equilibrium particle, B
overshoots to 1.23x its final loading at ~2 s and is displaced as A arrives
(A half-time 1.1 s, done by ~30 s, final S = 10). Kinetic particle: A
overshoots to 1.43x its final loading at ~36 s, then is slowly displaced by B
(B half-time 230 s, done by ~10^4 s, final S = 1). With the same isotherm for A and B, the
final loadings follow the gas: q_B/q_A = 0.70/0.30 = 2.3, so B ends ahead; the render says so in
a callout, and this is why a kinetic PSA keeps its feed step short. Both overshoots are
predicted by the models, not added for the visual.
