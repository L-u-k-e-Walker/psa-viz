# psa-viz: rendered PSA dynamics for a generic binary A/B

Two 20 s animations for slides:

1. **Column**: one adsorption column running a 4-step Skarstrom cycle at
   cyclic steady state. A 3D packed bed (particles coloured by how much A they
   hold, gas molecules whose density follows the pressure) beside live profiles
   aligned with bed height, the step list and the pressure trace. Loops seamlessly.
2. **Particle zoom**: an equilibrium-selective and a kinetic-selective particle
   side by side, hit by the same feed step on the same clock. Each is cut open
   to show radial loading gradients (left face A, right face B) and adsorbed
   molecules, with uptake curves underneath on a log clock.

**A = heavy component, extract** (held by the sorbent, leaves at the feed end
in blowdown and purge). **B = light component, raffinate** (the product of the
feed step). Both are generic. **Every input is an illustrative assumption or is
calculated from one; nothing is measured** (docs/INPUTS.md). The motion is
computed, not scripted: every colour, profile, count and curve comes from the
solvers in `psaviz/`.

## Outputs

| Visual | Dark | White |
|---|---|---|
| Column | `column/render/psa_column_dark.{mp4,gif}` | `column/render/psa_column_white.{mp4,gif}` |
| Particles | `particle/render/psa_particles_dark.{mp4,gif}` | `particle/render/psa_particles_white.{mp4,gif}` |

1920x1080, 20 s. MP4: 30 fps, 4–11 MB. GIF: 15 fps, full 1080p, 28–37 MB.
The column video loops seamlessly (it is one CSS cycle). Use the MP4s in
PowerPoint. Stills: `column/render/still_*.png`,
`particle/render/still_*.png`.

## What is simulated

| Piece | Model | Code |
|---|---|---|
| Column | Isothermal, no axial pressure drop, extended Langmuir, LDF. Linear valves in pressurization/blowdown, fixed flow in feed/purge, light-product tank feeding the purge. A compact rewrite of psa-sim's flow-controlled model, matching psa-sim to ~1e-6 on the same case | `psaviz/column.py` |
| Equilibrium particle | Macropore (Knudsen-type) diffusion with near-local equilibrium on the crystals; A held 10x more strongly, equal diffusivities | `psaviz/particle.py` |
| Kinetic particle | Maxwell–Stefan/Langmuir micropore diffusion; A diffuses 100x faster, same isotherm for A and B. A leads early; at equilibrium B is loaded 2.3x more because the gas is 70 % B (an on-screen callout says so) | `psaviz/particle.py` |

Column at CSS: B purity 99.86 %, B recovery 33.8 %, extract 39 % A (feed 30 %
A), pressure ratio 4.2. The particle overshoots (B in the equilibrium particle,
A in the kinetic one) come out of the models; they were not added for effect.

Not to scale: the 3D bed geometry (L/D 2.9, 14 particles across), particle
size, molecule size, and how fast the molecules drift (it rises with the real
gas velocity, but is slowed for watchability). The column video stretches the
pressure-changing steps (4 / 6 / 4 / 6 s of video for 10 / 50 / 10 / 50 s of
cycle); the on-screen clock shows true t/t_cycle.

## Pipeline

```bash
# env viz (numpy, scipy, matplotlib, OVITO 3.17, ffmpeg); see environment.yml
cd column/run && conda run -n viz python -I run_css.py        # CSS state (~40 s)
conda run -n viz python -I gates.py                           # Phase 1 gates (~5 min)
cd ../render && conda run -n viz python -I packing.py         # 3D bed packing (~2 min, seeded)
conda run -n viz python -I render_column.py --stills          # 4 stills x 2 themes
for th in dark white; do conda run -n viz python -I render_column.py --all --theme $th --workers 4; done

cd ../../particle/run && conda run -n viz python -I run_uptake.py && conda run -n viz python -I gates.py
cd ../render && conda run -n viz python -I render_particle.py --stills
for th in dark white; do conda run -n viz python -I render_particle.py --all --theme $th --workers 4; done
```

Each `--all` run takes about 8–12 min per theme (Tachyon uses every core, so
use 4 workers). Frames go to `frames/` (gitignored), and existing frames are
skipped, so a run can be resumed. Delete `frames/` after changing anything.

To change the look, edit `psaviz/style.py` (colours, ramps) or the constants
at the top of `render_column.py` / `render_particle.py`. To change the column
timing, edit `STEP_VIDEO_S`; for the particle clock, edit `LOG_T0`, `LOG_T1`,
`N_HOLD0` and `N_HOLD1`.

The psa-sim cross-check (`column/crosscheck/`) is the only part that uses
psa-sim (env `psa`). Its reference output is committed, so `compare.py` runs
without it.

## Docs

- `docs/MISSION.md`: scope, user decisions, phases and gate criteria.
- `docs/LOG.md`: gate evidence and decisions.
- `docs/INPUTS.md`: every input with its source.
