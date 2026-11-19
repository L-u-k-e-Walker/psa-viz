"""Extended (competitive) Langmuir isotherm for the binary A/B.

q_i* = q_s,i b_i p_i / (1 + sum_j b_j p_j), p in bar, q in mol/kg.
Index 0 = A (heavy, extract), index 1 = B (light, raffinate) everywhere in psaviz.
"""

from __future__ import annotations

import numpy as np

R_GAS = 8.314462618  # J/(mol K)
A, B = 0, 1
NAMES = ("A", "B")


def loading(p_bar, qs, b):
    """Equilibrium loadings. ``p_bar`` has shape (2, ...); returns the same shape."""
    p = np.maximum(np.asarray(p_bar, float), 0.0)
    qs = np.asarray(qs, float).reshape((2,) + (1,) * (p.ndim - 1))
    b = np.asarray(b, float).reshape((2,) + (1,) * (p.ndim - 1))
    bp = b * p
    return qs * bp / (1.0 + bp.sum(0))


def coverage(p_bar, b):
    """Fractional coverages theta_i = b_i p_i / (1 + sum b_j p_j) (equal q_s)."""
    p = np.maximum(np.asarray(p_bar, float), 0.0)
    b = np.asarray(b, float).reshape((2,) + (1,) * (p.ndim - 1))
    bp = b * p
    return bp / (1.0 + bp.sum(0))


def chord_slope(qs, b, i, p_lo_bar, p_hi_bar):
    """Chord slope dq_i/dp_i (mol/kg/bar) of component i between two (2,) partial-pressure states."""
    q_hi = loading(np.asarray(p_hi_bar, float), qs, b)[i]
    q_lo = loading(np.asarray(p_lo_bar, float), qs, b)[i]
    return (q_hi - q_lo) / (p_hi_bar[i] - p_lo_bar[i])
