# psa-viz mission

Source of truth for this project. Two rendered visuals of PSA dynamics for a
generic binary mixture:

1. **Column**: one adsorption column running a 4-step Skarstrom cycle at
   cyclic steady state (CSS), shown as a 3D packed bed with live 2D plots.
2. **Particle zoom**: two spherical particles side by side under the same
   feed step and clock. Left: equilibrium-selective. Right: kinetic-selective.
   Each is a cutaway sphere with a radial concentration colour gradient and
   molecules.

Decisions (user, 2026-11-19):

| Topic | Decision |
|---|---|
| Physics | psa-sim's model, rewritten compactly in this project (no import of `psasim`). No momentum balance: fixed flow in the constant-pressure steps, linear valves in the pressure-changing steps |
| Thermal | isothermal |
| Cycle | full 4-step cycle at CSS: pressurization (feed), feed, blowdown (counter-current), purge (counter-current, from the raffinate tank) |
| Column sorbent | equilibrium-selective; particles are a standalone uptake experiment (not driven by the column) |
| Particle look | spherical cutaway, radial concentration colour gradient, molecules |
| Column look | 3D packed bed (OVITO/Tachyon) + 2D plots composited beside it |
| Species | **A** and **B**. A is the preferentially adsorbed species in every view = **heavy, extract** (leaves at the feed end in blowdown/purge). B = **light, raffinate** (leaves at the product end in feed). The labels must say this on screen |
| Axes | dimensionless: z/L, t/t_cycle, y_A, q/q_s, P/P_H. The run itself uses dimensional, provenance-labelled inputs (docs/INPUTS.md) |
| Output | 1920x1080, ~20 s, MP4 (H.264) + GIF, dark and white versions, for each visual |
| Repo | own git repo `psa-viz/`, remote git@github.com:L-u-k-e-Walker/psa-viz.git (the user creates it on the website); ask before pushing |

## Rules

- Self-contained: the package `psaviz/` needs only numpy and scipy;
  rendering adds matplotlib and OVITO. Env `viz`. The psa-sim cross-check
  (`column/crosscheck/`) is the only place that runs psa-sim (env `psa`). It
  writes a reference CSV that is committed, so it never needs re-running.
- Every input has a stated source (docs/INPUTS.md): all are *assumed
  (illustrative, generic)* or *calculated* from other inputs; none is
  measured. Say so on the README and in any caption.
- At most 8 parallel processes (P-cores); `OPENBLAS_NUM_THREADS=1` per
  worker. Rendered frames are gitignored.
- Self-checked gates: check each gate against the criteria below, log the
  evidence in docs/LOG.md, commit, continue. Stop and ask only if a gate fails
  in a way that needs a decision, or at the style checkpoint (Phase 3).
- Feature branch per phase, merged into main.

## Phases and gates

### Phase 0: setup
Folder tree, git, `.gitignore`, `environment.yml`, docs.
**Gate 0:** `numpy, scipy, matplotlib, ovito` import in `viz`; initial commit.

### Phase 1: column model (`psaviz/column.py`)
Finite-volume, first-order upwind, N cells; states c_A, c_B, q_A, q_B per cell
plus raffinate-tank composition. Extended Langmuir, LDF. Face velocities from
the overall balance with uniform pressure, like psa-sim's `DP0DT0` model.
Constant-pressure steps have a fixed inlet flow and a free outlet; in the
pressure-changing steps the flow is Cv (P_up - P_down).

- **G1a mass balance:** per step and component, |in - out - Δinventory| /
  (feed moles of that component per cycle) < 1e-4.
- **G1b CSS:** relative cycle-to-cycle state change < 1e-6, and the CSS
  component balance over a cycle (feed = raffinate product + extract) closes
  within 0.5 %.
- **G1c analytic:** the first moment of a trace-A breakthrough into a
  B-saturated bed matches the equilibrium-theory retention time within 1 %.
- **G1d grid:** at the render grid N and 2N, B purity differs by < 0.2 pp and
  B recovery by < 1 pp.
- **G1e psa-sim cross-check:** same single-column case and N in both codes.
  Purity and recovery agree within 0.3 pp, y_A(z) at each step end within
  0.02 (max abs), and P(t) within 1 % (max rel).

### Phase 2: particle models (`psaviz/particle.py`)
Spherical finite volume, N_r shells, step change in surface gas at t = 0 from a
clean particle.
- Equilibrium particle: macropore (Knudsen) diffusion in the pores with
  near-local equilibrium (fast crystal LDF), A more strongly adsorbed, D_A ≈ D_B.
- Kinetic particle: micropore diffusion with the Maxwell–Stefan/Langmuir
  (Habgood-type) flux N_i = -q_s Đ_i [∇θ_i + θ_i/(1-θ_t) ∇θ_t], Đ_A >> Đ_B,
  similar affinities.

- **G2a mass balance:** uptake vs integrated surface flux, relative error < 1e-4.
- **G2b analytic:** single-component linear-isotherm limit vs the Crank
  series solution, max |ΔF| < 0.005.
- **G2c equilibrium:** final loadings within 0.5 % of extended Langmuir at the
  surface conditions.
- **G2d grid:** N_r vs 2N_r uptake curves, max |Δq/q_s| < 0.005.
- **G2e mechanism:** equilibrium particle final selectivity
  S = (q_A/q_B)/(y_A/y_B) ≥ 3. Kinetic particle: S ≥ 3 at the display time
  when A reaches 90 % of its peak, and final S between 0.7 and 1.4.

### Phase 3: render prototypes (style checkpoint)
Bed packing, OVITO scene and matplotlib panels; stills (dark + white) at key
times for both visuals.
**Gate 3:** stills exist and the plotted values match the solver output
(spot-check 3 frames per visual, max abs diff < 1e-6). Then **stop**: show
the stills and ask the style questions (colours, cutaway, molecule density,
time mapping, layout).

### Phase 4: final renders
Full frame sequences; ffmpeg to MP4 (H.264, yuv420p) and GIF (palette method).
**Gate 4:** each output is 1920x1080 (GIF may be downscaled if > 50 MB; say
so), 19–21 s long with the expected frame count (ffprobe), no missing frames,
and the on-screen labels name A = heavy/extract and B = light/raffinate.

### Phase 5: docs and handoff
README (pipeline, re-render commands, provenance statement), update
`../CLAUDE.md`, final commit. Ask before pushing.
