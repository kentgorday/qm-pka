"""Quadrature grids for the inside and the outside of a union-of-spheres cavity.

The dispersion and exchange integrals of `nonelectrostatic` are cut off at the
PCM cavity's surface.  On an ordinary molecular grid that cut falls between
quadrature shells, and the integral converges only linearly in the shell
spacing: for the formate anion at HF/6-31+G* the exchange integral is still 9%
short on a (99, 590) grid and jumps as atoms move.  Here every radial rule ends
exactly on the surface.

Space is split among the atoms with Becke's fuzzy cells.  From each atom, along
each Lebedev direction, the ray is cut analytically into the stretches that lie
inside the cavity and those that lie outside (each sphere it crosses contributes
one interval), and each stretch gets its own rule: Gauss-Legendre on a finite
one, Gauss-Laguerre on the final stretch to infinity.  The points depend
continuously on the geometry, so the integrals do too.

Everything is in bohr.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import lebedev_rule

Array = NDArray[np.float64]

Region = Literal["inside", "outside"]

# Defaults, from the convergence table in docs/nonelectrostatic.md.
LEBEDEV_ORDER = 41  # 590 directions per atom
N_RADIAL = 40  # nodes per stretch of a ray

# Length scale (bohr) of the Gauss-Laguerre rule on the stretch to infinity.
# The rule is exact for polynomial x exp(-t / scale); density tails decay as
# exp(-2 sqrt(2 IP) t), i.e. over 0.3-0.5 bohr, and the polynomial absorbs the
# mismatch.
LAGUERRE_SCALE = 0.5

# Stretches shorter than this (bohr) are dropped, as are points whose Becke
# weight has fallen below _MIN_WEIGHT (they belong to another atom's cell).
_MIN_LENGTH = 1e-10
_MIN_WEIGHT = 1e-14


def becke_partition(points: Array, atom_coords: Array) -> Array:
    """Becke's fuzzy-cell weights, shape (n_points, n_atoms); each row sums to 1.

    The plain scheme (JCP 88, 2547 (1988)), with three iterations of the cutoff
    polynomial and no atomic-size adjustment.  Any smooth partition of unity
    gives the same integrals; this one only decides which atom's rays resolve
    which part of space.
    """
    dist = np.linalg.norm(points[:, None, :] - atom_coords[None, :, :], axis=2)
    cell = np.ones_like(dist)
    for i in range(len(atom_coords)):
        for j in range(i):
            mu = (dist[:, i] - dist[:, j]) / np.linalg.norm(atom_coords[i] - atom_coords[j])
            for _ in range(3):
                mu = 1.5 * mu - 0.5 * mu**3
            cell[:, i] *= 0.5 * (1.0 - mu)
            cell[:, j] *= 0.5 * (1.0 + mu)
    return np.asarray(cell / cell.sum(axis=1, keepdims=True))


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of intervals as a sorted list of disjoint ones."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _stretches(
    inside: list[tuple[float, float]], region: Region
) -> tuple[list[tuple[float, float]], float | None]:
    """Finite stretches of a ray in the region, and where its infinite stretch starts."""
    if region == "inside":
        return inside, None
    gaps = [(a[1], b[0]) for a, b in pairwise(inside)]
    return gaps, inside[-1][1]


def cavity_grid(
    atom_coords: Array,
    atom_radii: Array,
    region: Region,
    lebedev_order: int = LEBEDEV_ORDER,
    n_radial: int = N_RADIAL,
) -> tuple[Array, Array]:
    """Points (n, 3) and weights (n,) integrating over the inside or outside of the cavity."""
    directions, w_ang = lebedev_rule(lebedev_order)
    unit = np.ascontiguousarray(directions.T)
    x_leg, w_leg = np.polynomial.legendre.leggauss(n_radial)
    x_lag, w_lag = np.polynomial.laguerre.laggauss(n_radial)
    t_lag = LAGUERRE_SCALE * x_lag
    w_lag = LAGUERRE_SCALE * w_lag * np.exp(x_lag)

    all_coords: list[Array] = []
    all_weights: list[Array] = []
    for ia, (centre, radius) in enumerate(zip(atom_coords, atom_radii, strict=True)):
        # Where each ray enters and leaves every other sphere: |t u - d|^2 = a^2.
        offset = np.delete(atom_coords, ia, axis=0) - centre
        b = unit @ offset.T  # (n_directions, n_other)
        disc = b**2 - (np.sum(offset**2, axis=1) - np.delete(atom_radii, ia) ** 2)[None, :]
        root = np.sqrt(np.where(disc > 0.0, disc, 0.0))
        enter, leave = b - root, b + root
        crosses = (disc > 0.0) & (leave > 0.0)

        t_parts: list[Array] = []
        w_parts: list[Array] = []
        ray_of_part: list[int] = []
        for k in range(len(unit)):
            others = [
                (max(float(e), 0.0), float(v))
                for e, v in zip(enter[k, crosses[k]], leave[k, crosses[k]], strict=True)
            ]
            finite, tail = _stretches(_merge([(0.0, float(radius)), *others]), region)
            for start, end in finite:
                if end - start < _MIN_LENGTH:
                    continue
                half = 0.5 * (end - start)
                t_parts.append(half * x_leg + 0.5 * (end + start))
                w_parts.append(half * w_leg)
                ray_of_part.append(k)
            if tail is not None:
                t_parts.append(tail + t_lag)
                w_parts.append(w_lag)
                ray_of_part.append(k)
        if not t_parts:
            continue
        t = np.concatenate(t_parts)
        ray = np.repeat(ray_of_part, n_radial)
        coords = centre + t[:, None] * unit[ray]
        weights = np.concatenate(w_parts) * t**2 * w_ang[ray]
        weights *= becke_partition(coords, atom_coords)[:, ia]
        keep = np.abs(weights) > _MIN_WEIGHT
        all_coords.append(coords[keep])
        all_weights.append(weights[keep])
    return np.vstack(all_coords), np.concatenate(all_weights)
