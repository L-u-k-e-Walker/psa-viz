"""The generic A/B cases used for the renders, with the source of every input.

Nothing here is measured. Values are *assumed* (illustrative, chosen to be
typical of an equilibrium-selective zeolite pellet and a kinetic-selective
carbon molecular sieve) or *calculated* from other inputs. docs/INPUTS.md
explains the choices.
"""

from __future__ import annotations

import numpy as np

from .column import ColumnCase, Step
from .isotherm import R_GAS, loading

# ----------------------------------------------------------- shared inputs
T = 298.15
P_HIGH_NOMINAL = 4.2  # bar, column pressure during FEED at CSS (outcome of RP; used for scaling/LDF)
P_LOW = 1.0  # bar, vent
Y_FEED = (0.30, 0.70)

# equilibrium-selective pellet (column sorbent and left particle)
PELLET = dict(
    radius=1.0e-3,  # m
    porosity=0.35,  # macropore void fraction
    density=1100.0,  # kg/m^3 pellet
    pore_diffusivity=1.7e-6,  # m^2/s, same for A and B
    qs=(3.0, 3.0),  # mol/kg
    b=(0.60, 0.06),  # 1/bar  -> Henry selectivity 10
    k_crystal=100.0,  # 1/s, crystal LDF inside the pellet (fast: near local equilibrium)
)

# kinetic-selective particle (right particle; micropore diffusion)
KINETIC = dict(
    radius=1.0e-3,  # m (same size as the pellet, so the two particles compare directly)
    density=1100.0,
    qs=(3.0, 3.0),
    b=(0.25, 0.25),  # equal affinity: equilibrium selectivity 1
    D_over_R2=(1.0e-2, 1.0e-4),  # 1/s, corrected (Maxwell-Stefan) diffusivity / R^2
)


def glueckauf_ldf(p_high=P_HIGH_NOMINAL, p_low=P_LOW, y=Y_FEED, pellet=PELLET):
    """Column LDF constants from the pellet's macropore diffusion (Glueckauf).

    k_i = 15 eps_p D_p / (R_p^2 (eps_p + rho_p dq_i/dc_i)), with the chord slope
    dq_i/dc_i between the regenerated state (pure B at p_low) and the feed at p_high.
    """
    RT = R_GAS * T
    qs, b = pellet["qs"], pellet["b"]
    p_feed = np.array([y[0] * p_high, y[1] * p_high])
    p_regen = np.array([0.0, p_low])
    q_f = loading(p_feed, qs, b)
    q_r = loading(p_regen, qs, b)
    dc = (p_feed - p_regen) * 1e5 / RT
    slope = (q_f - q_r) / dc  # m^3/kg
    eps, D, R, rho = pellet["porosity"], pellet["pore_diffusivity"], pellet["radius"], pellet["density"]
    k = 15.0 * eps * D / (R**2 * (eps + rho * slope))
    return tuple(float(v) for v in k), slope


def column_case(n_cells=120, **kw) -> ColumnCase:
    k, _ = glueckauf_ldf()
    prov = {
        "length, radius": "assumed: 1 m bed, 0.1 m radius (L/D = 5)",
        "bed_void": "assumed: 0.40, typical random packing of spheres",
        "pellet_density": "assumed: 1100 kg/m^3, zeolite-pellet-like",
        "temperature": "assumed: 298.15 K, isothermal",
        "qs, b": "assumed: extended Langmuir, equal q_s = 3 mol/kg, b_A/b_B = 10",
        "k_ldf": "calculated: Glueckauf from the pellet's macropore diffusion (psaviz.cases.glueckauf_ldf)",
        "y_feed": "assumed: 30 % A",
        "p_feed, p_vent, tank": "assumed: 4.3 bar feed header, 1.0 bar vent, 1 m^3 light-product tank at 3.9 bar",
        "steps": "assumed: tuned for a textbook-looking cycle (front to ~0.7 L in feed, B purity > 99 %)",
        "pore gas in pellets": "neglected in the column balance (psa-sim macropore_void = 0)",
    }
    args = dict(
        name="generic_AB_skarstrom", length=1.0, radius=0.1, bed_void=0.40,
        pellet_density=PELLET["density"], temperature=T, qs=PELLET["qs"], b=PELLET["b"],
        k_ldf=k, y_feed=Y_FEED, p_feed=4.3, p_vent=P_LOW, tank_volume=1.0, tank_p_set=3.9,
        p_init=P_LOW,
        steps=[Step("RP", 10.0, 1.5), Step("FEED", 50.0, 7.0), Step("BD", 10.0, 5.0),
               Step("PURGE", 50.0, 0.11)],
        n_cells=n_cells, provenance=prov)
    args.update(kw)
    return ColumnCase(**args)


# ---------------------------------------------------------------- particles
PARTICLE_PROVENANCE = {
    "both: radius": "assumed: 1 mm (2 mm particles); same size so the two compare directly",
    "both: step": "assumed: clean particle, surface gas switched at t = 0 to the column feed (y_A 0.30, 4.2 bar)",
    "equilibrium: porosity, density": "assumed: eps_p 0.35, rho_p 1100 kg/m^3 (zeolite pellet)",
    "equilibrium: pore diffusivity": "assumed: D_p 1.7e-6 m^2/s for A and B (Bosanquet of Knudsen in ~0.1 um "
                                     "pores and molecular diffusion at ~4 bar, tortuosity 3); equal because the "
                                     "generic species have equal molar mass",
    "equilibrium: isotherm": "assumed: same as the column (b_A/b_B = 10)",
    "equilibrium: crystal LDF": "assumed: 100 /s, fast enough that the crystals are nearly at local equilibrium",
    "kinetic: isotherm": "assumed: b_A = b_B = 0.25 /bar, q_s 3 mol/kg (equilibrium selectivity 1)",
    "kinetic: diffusivities": "assumed: Đ_A/R^2 = 1e-2 /s, Đ_B/R^2 = 1e-4 /s (ratio 100; CMS O2/N2 is ~30-100)",
    "kinetic: flux model": "Maxwell-Stefan/Langmuir without exchange correlations (Habgood-type)",
}


def equilibrium_particle(N=100, **kw):
    from .particle import MacroporeParticle
    p = PELLET
    args = dict(radius=p["radius"], porosity=p["porosity"], density=p["density"],
                pore_diffusivity=p["pore_diffusivity"], qs=p["qs"], b=p["b"], k_crystal=p["k_crystal"],
                T=T, P=P_HIGH_NOMINAL, y=Y_FEED, N=N)
    args.update(kw)
    return MacroporeParticle(**args)


def kinetic_particle(N=100, **kw):
    from .particle import MicroporeParticle
    p = KINETIC
    args = dict(radius=p["radius"], density=p["density"], qs=p["qs"], b=p["b"], D_over_R2=p["D_over_R2"],
                T=T, P=P_HIGH_NOMINAL, y=Y_FEED, N=N)
    args.update(kw)
    return MicroporeParticle(**args)
