# Log

Gate evidence and decisions, newest last.

## 2026-11-19 Phase 0: setup

- User decisions recorded in docs/MISSION.md (three question rounds).
- Added matplotlib 3.11.2 and pytest 9.2.0 to env `viz` (numpy 2.5.5,
  scipy 1.18.1, OVITO 3.17.2 already there; OVITO has `TriangleMesh`).
- **Gate 0 PASS:** `import numpy, scipy, matplotlib, ovito, pytest` works in `viz`;
  repo initialised on `main`.

## 2026-11-19 Phase 1: column model

- `psaviz/column.py`: compact rewrite of psa-sim's flow-controlled isothermal
  no-pressure-drop model (`model/rhs.py` + `cycle/runner.py` logic: valve / free
  ends, uniform-rate velocity march, upwind fluxes, raffinate tank with pre-fill).
  numpy + scipy only (`solve_ivp` BDF, vectorized RHS). ~0.7 s per cycle at N = 120.
- Case design (`psaviz/cases.py`, provenance in docs/INPUTS.md). First drafts
  (P_H 3 bar, b_A/b_B 5) gave B recovery 3–19 %: blowdown losses of B plus the
  molar P/F ≈ P_L/P_H needed by the purge nearly use up the B fed. Final: b_A/b_B
  = 10, header 4.3 bar (P_H = 4.208 bar), feed Cv 7 so the front reaches ~0.6 L.
  Column LDF from Glueckauf on the equilibrium pellet (same sorbent as the left particle).
- Cross-check (`column/crosscheck/`): `write_config.py` → psa-sim run in env `psa`
  (62 cycles, 57 s) → `compare.py`. Reference CSVs committed.
- **Gate evidence** (`column/run/gates_phase1.json`, `column/crosscheck/crosscheck.json`):
  - G1a PASS: max per-step component imbalance over all 62 cycles 3.5e-13 (limit 1e-4).
  - G1b PASS: CSS change 9.7e-8 (limit 1e-6) after 62 cycles; CSS cycle balance 3.2e-6 (limit 5e-3).
  - G1c PASS: trace-A breakthrough first moment 181.87 s vs equilibrium theory 181.90 s (0.019 %).
  - G1d PASS: N = 120 vs 240: purity 99.858 vs 99.887 % (0.030 pp), recovery 33.82 vs 33.99 % (0.17 pp).
  - G1e PASS: vs psa-sim at N = 120: purity diff 4e-6 pp, recovery diff 2e-5 pp,
    y_A max abs diff 1.0e-6 over all stored times, P 4.4e-7 rel, tank pre-fill 28.9703 s in both.
- CSS render state saved in `column/run/css_state.npz` (+ metrics.json, cycle_history.csv).

## 2026-11-19 Phase 2: particle models

- `psaviz/particle.py`: `MacroporeParticle` (pore diffusion + fast crystal LDF) and
  `MicroporeParticle` (Maxwell–Stefan/Langmuir flux), spherical finite volumes, BDF with
  banded Jacobian sparsity; < 0.3 s per solve. Cases in `psaviz/cases.py`, provenance in docs/INPUTS.md.
- `particle/run/run_uptake.py` → uptake.csv, uptake_metrics.json.
- **Gate evidence** (`particle/run/gates_phase2.json`):
  - G2a PASS: held moles vs integrated surface inflow, rel err 3.6e-11 (eq) / 7.6e-10 (kin) (limit 1e-4).
  - G2b PASS: linear single-component limit vs Crank, tau 1e-4..1.5: max |dF| 0.0024 (macropore) / 0.0024 (micropore) (limit 0.005).
  - G2c PASS: final loadings = extended Langmuir to 3e-16 / 1.3e-11 rel (limit 5e-3).
  - G2d PASS: N 100 vs 200: max |dq/q_s| 3.0e-4 / 1.4e-3 (limit 5e-3).
  - G2e PASS: equilibrium final S = 10.0 (>= 3); kinetic S = 9.26 at t = 13.9 s (A at 90 % of peak) (>= 3),
    final S = 1.00 (0.7–1.4). Overshoots: B 1.23x (equilibrium), A 1.43x (kinetic).

## 2026-11-19 Phase 3: render prototypes (style checkpoint)

- Column (`column/render/`): `packing.py` (soft-sphere settling in a cylinder: 7757 spheres,
  D/d = 14, bed L/D = 2.89, core packing fraction 0.66, max overlap 0.10 d),
  `scene.py` (CSS cycle re-integrated from css_state.npz; gas dots whose density follows C(z,t)
  and species follow y_A(z,t); OVITO bed cut away at y = 0, particles coloured by q_A/q_A,feed,
  back half of a glass wall, caps, pipes, orthographic camera at 14 deg so z maps linearly to
  screen rows), `render_column.py` (matplotlib composite: profile panels aligned with the bed
  height, step list, P/P_H trace, flow arrows with A/B labels, legend, footnote).
- Particles (`particle/render/`): `scene_particle.py` (orange-slice cutaway spheres as OVITO
  meshes; left face A loading, right face B loading on one shared theta scale 0–0.42; adsorbed
  molecules on fixed random sites shown when u < theta/0.42; gas bath random walk, 30 % A),
  `render_particle.py` (titles, face tags, uptake curves on a log clock t/tau_A, legend).
- Colours: dataviz reference palette slots 1–2 (A orange, B blue), surfaces and ink from the
  palette; particle loading ramp in OKLab from the bare-sorbent grey to a darker step of the
  species hue (so A molecules stay visible over A-loaded particles). Validator not run (no node);
  the pair is documented as passing all-pairs CVD/normal-vision checks in both modes.
- OVITO lessons: `TriangleMesh.set_vertex_colors` needs RGBA; `set_normals` needs (3·faces)×3;
  after a render the source DataCollection is shared, so give the StaticSource a fresh clone per
  frame; remove pipelines from the scene before building the next theme in the same process.
- **Gate 3 PASS** (`tools/gate3_spotcheck.py` → docs/gate3_spotcheck.json): drawn column values
  vs an independent re-integration at t = 3, 52, 105 s: max |diff| 0.0; particle profiles vs a
  fresh solve at t = 0.3, 3, 3000 s: 2.2e-9 (limit 1e-6). Loop seam of the CSS cycle 2.0e-7
  (normalized). Stills: column/render/still_{dark,white}_t{003,052,062,105}.png,
  particle/render/still_{dark,white}_t{0.3,3,40,3000}.png.
