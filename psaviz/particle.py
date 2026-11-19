"""Single spherical particle hit by a step in surrounding gas (standalone uptake).

Two models, each typical of its sorbent class:

* ``MacroporeParticle`` (equilibrium-selective pellet, zeolite-like): gas
  diffuses through the macropores (independent Knudsen-type fluxes,
  J_i = -eps_p D_p dc_i/dr) and adsorbs on crystals that are nearly in local
  equilibrium (fast crystal LDF, k_c). A is held much more strongly than B, but
  both have the same pore diffusivity. Loading per pellet volume: rho_p q.
* ``MicroporeParticle`` (kinetic-selective, carbon-molecular-sieve-like):
  adsorbed-phase (micropore) diffusion with the Maxwell-Stefan / Langmuir
  flux without exchange correlations (Habgood-type):
      N_i = -rho_p q_s Đ_i [dtheta_i/dr + theta_i/(1 - theta_t) dtheta_t/dr]
  Đ_A >> Đ_B with similar affinities. Requires equal q_s.

Finite volumes on N equal-thickness spherical shells; the surface face uses
the half-cell distance to the fixed surface value. Initially clean; at t = 0 the
surface sees gas at pressure P with composition y. Index 0 = A, 1 = B.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from .isotherm import R_GAS, coverage, loading


class SphereGrid:
    def __init__(self, R, N):
        self.R, self.N = R, N
        self.rf = np.linspace(0.0, R, N + 1)
        self.rc = 0.5 * (self.rf[1:] + self.rf[:-1])
        self.V = 4.0 / 3.0 * np.pi * (self.rf[1:] ** 3 - self.rf[:-1] ** 3)
        self.Af = 4.0 * np.pi * self.rf**2
        self.dr_c = np.diff(self.rc)
        self.dr_s = R - self.rc[-1]
        self.Vtot = 4.0 / 3.0 * np.pi * R**3


def _sparsity(n_fields, N, extra):
    """Jacobian pattern: each field couples to all fields in the same and neighbouring shells."""
    n = n_fields * N + extra
    S = lil_matrix((n, n), dtype=np.int8)
    for a in range(n_fields):
        for bb in range(n_fields):
            for k in range(N):
                for d in (-1, 0, 1):
                    if 0 <= k + d < N:
                        S[a * N + k, bb * N + k + d] = 1
    S[n_fields * N:, :] = 1
    return S.tocsr()


class _Base:
    def times(self, t_end, n=400, t_min=1e-3):
        return np.concatenate([[0.0], np.geomspace(t_min, t_end, n - 1)])

    def solve(self, t_eval, rtol=1e-7):
        sol = solve_ivp(self.rhs, (0.0, t_eval[-1]), self.x0(), method="BDF", t_eval=t_eval,
                        rtol=rtol, atol=self.atol, jac_sparsity=self.jac_sparsity)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol


class MacroporeParticle(_Base):
    def __init__(self, radius, porosity, density, pore_diffusivity, qs, b, k_crystal,
                 T, P, y, N=100):
        self.g = SphereGrid(radius, N)
        self.N = N
        self.eps, self.rho = porosity, density
        self.D = np.broadcast_to(np.asarray(pore_diffusivity, float), (2,)).copy()
        self.qs, self.b = np.asarray(qs, float), np.asarray(b, float)
        self.kc = float(k_crystal)
        self.RT = R_GAS * T
        self.P, self.y = P, np.asarray(y, float)
        self.cs = self.y * P * 1e5 / self.RT  # surface gas, mol/m^3
        self.q_eq = loading(self.y * P, self.qs, self.b)  # final loadings
        c_ref = max(P * 1e5 / self.RT, 1e-300)
        q_ref = max(float(self.q_eq.max()), 1e-300)
        self.atol = np.concatenate([np.full(2 * N, 1e-10 * c_ref), np.full(2 * N, 1e-10 * q_ref),
                                    np.full(2, 1e-10 * q_ref * density * self.g.Vtot)])
        self.jac_sparsity = _sparsity(4, N, 2)

    def x0(self):
        return np.zeros(4 * self.N + 2)

    def split(self, x):
        N = self.N
        return x[:2 * N].reshape((2, N) + x.shape[1:]), x[2 * N:4 * N].reshape((2, N) + x.shape[1:]), x[4 * N:]

    def rhs(self, t, x):
        g, eps = self.g, self.eps
        c, q, _ = self.split(x)
        p = c * (self.RT / 1e5)
        dq = self.kc * (loading(p, self.qs, self.b) - q)
        F = np.zeros((2, self.N + 1))  # outward molar flow through each face, mol/s
        eD = eps * self.D[:, None]
        F[:, 1:-1] = -g.Af[1:-1] * eD * (c[:, 1:] - c[:, :-1]) / g.dr_c
        F[:, -1] = -g.Af[-1] * eD[:, 0] * (self.cs - c[:, -1]) / g.dr_s
        dc = (F[:, :-1] - F[:, 1:]) / (eps * g.V) - (self.rho / eps) * dq
        return np.concatenate([dc.ravel(), dq.ravel(), -F[:, -1]])

    def moles(self, x):
        """Moles of A and B held in the particle (pore gas + adsorbed)."""
        c, q, _ = self.split(x)
        return (self.eps * c + self.rho * q) @ self.g.V if x.ndim == 1 else \
            np.einsum("ink,n->ik", self.eps * c + self.rho * q, self.g.V)

    def mean_loading(self, x):
        _, q, _ = self.split(x)
        return np.einsum("in...,n->i...", q, self.g.V) / self.g.Vtot


class MicroporeParticle(_Base):
    def __init__(self, radius, density, qs, b, D_over_R2, T, P, y, N=100):
        qs = np.asarray(qs, float)
        if not np.allclose(qs, qs[0]):
            raise ValueError("Maxwell-Stefan/Langmuir flux here needs equal q_s")
        self.g = SphereGrid(radius, N)
        self.N = N
        self.rho, self.qs, self.b = density, qs, np.asarray(b, float)
        self.D = np.asarray(D_over_R2, float) * radius**2  # Đ_i, m^2/s
        self.RT = R_GAS * T
        self.P, self.y = P, np.asarray(y, float)
        self.th_s = coverage(self.y * P, self.b)
        self.q_eq = self.qs * self.th_s
        self.atol = np.concatenate([np.full(2 * N, 1e-11), np.full(2, 1e-11 * self.g.Vtot)])
        self.jac_sparsity = _sparsity(2, N, 2)

    def x0(self):
        return np.zeros(2 * self.N + 2)

    def split(self, x):
        N = self.N
        return x[:2 * N].reshape((2, N) + x.shape[1:]), x[2 * N:]

    def face_flux(self, th):
        """Outward theta-volume flow through each face (theta m^3/s), shape (2, N+1)."""
        g = self.g
        F = np.zeros((2, self.N + 1))
        thf = 0.5 * (th[:, 1:] + th[:, :-1])
        grad = (th[:, 1:] - th[:, :-1]) / g.dr_c
        F[:, 1:-1] = -g.Af[1:-1] * self.D[:, None] * (grad + thf / (1.0 - thf.sum(0)) * grad.sum(0))
        thf_s = 0.5 * (self.th_s + th[:, -1])
        grad_s = (self.th_s - th[:, -1]) / g.dr_s
        F[:, -1] = -g.Af[-1] * self.D * (grad_s + thf_s / (1.0 - thf_s.sum()) * grad_s.sum())
        return F

    def rhs(self, t, x):
        th, _ = self.split(x)
        F = self.face_flux(th)
        dth = (F[:, :-1] - F[:, 1:]) / self.g.V
        return np.concatenate([dth.ravel(), -F[:, -1]])

    def moles(self, x):
        th, _ = self.split(x)
        return self.rho * self.qs[0] * (np.einsum("in...,n->i...", th, self.g.V))

    def mean_loading(self, x):
        th, _ = self.split(x)
        return self.qs[:, None] * np.einsum("in...,n->i...", th, self.g.V) / self.g.Vtot if x.ndim > 1 \
            else self.qs * (th @ self.g.V) / self.g.Vtot

    def cum_moles(self, x):
        _, cum = self.split(x)
        return self.rho * self.qs[0] * cum


def crank_sphere(tau, n_terms=4000):
    """Fractional uptake of a sphere, constant surface concentration, linear: tau = D t / R^2."""
    tau = np.atleast_1d(np.asarray(tau, float))
    n = np.arange(1, n_terms + 1)[:, None]
    return 1.0 - 6.0 / np.pi**2 * np.sum(np.exp(-(n**2) * np.pi**2 * tau[None, :]) / n**2, axis=0)
