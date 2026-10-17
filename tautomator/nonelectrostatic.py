"""Nonelectrostatic solvation terms, as an energy-only correction after a PCM SCF.

IEF-PCM is electrostatics only.  Following CMIRS (Pomogaeva & Chipman; equations
as collected in Herbert, arXiv:2203.06846, eqs. 4.18-4.24), the rest of the
solvation free energy is modelled as

    G = A * dispersion + B * exchange + short-range field term

with, for a cavity Omega, solute density rho and uniform solvent density rho_s:

  dispersion   int_{r in Omega} rho(r) I(r) / (w(r) (w(r) + sqrt(rho_s))) dr
               I(r) = int_{r' outside Omega} dr' / (|r - r'|^6 + delta^6)
               w(r) = sqrt(rho + (3C/4pi) |grad rho|^4 / rho^4)
  exchange     int_{r outside Omega} |grad rho(r)| dr

CMIRS's third term, FESR, is C |F_min|^g + D F_max^g: a power of the two
extrema of the normal field F on the cavity surface, which can describe one
hydrogen-bond acceptor site and one donor site.  It is replaced here by the
surface integral the extrema are a limit of,

  short range  sum_i A_i (c- |F_i|^n-  if F_i < 0  else  c+ F_i^n+)

so every strong-field site contributes.  F_i is the solute's own normal field,
or optionally the PCM apparent surface charge density recast as a field (see
docs/nonelectrostatic.md for why the field is the default).

The functions return the bare integrals; the linear coefficients (A, B, c-, c+)
and the exponents are applied by `Descriptors.energy`, so they can be fitted
without recomputing anything.  CMIRS's published parameters belong to an
isodensity cavity and do not transfer to the switching/Gaussian sphere cavity
used here.  Nothing is fitted yet.

The inputs (`Solute`) are backend-neutral; `pyscf_nonelectrostatic` fills one
from a converged PySCF mean field.  All quantities are in atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Damping length of the dispersion kernel (bohr); CMIRS's fixed value.
DELTA = 7.0

# The gradient-correction constant of the VV10 local frequency.  Herbert's
# review says CMIRS takes it "without modification from VV10" (0.0093); the
# functional CMIRS is named after, VV09, used 0.0089.  The difference is
# absorbed by the fitted prefactor A.
VV_C = 0.0093

# Densities below this are treated as zero when forming |grad rho| / rho.
_RHO_FLOOR = 1e-14

# Pair-kernel evaluations per block in the dispersion double sum (~6 temporaries
# of this many float64 each).
_BLOCK_PAIRS = 4_000_000

FieldSource = Literal["field", "charge"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """A discretized PCM cavity surface and the converged apparent surface charges.

    The cavity is the union of the atomic spheres ``atom_radii``.  ``areas`` are
    the quadrature weights of the surface elements; on the switching/Gaussian
    discretization both drivers use, that is Lebedev weight x R^2 x switching
    function, so elements buried near a sphere intersection fade out smoothly.

    ``charge_scale`` is the factor the model's charges carry relative to those a
    conductor would put on the same surface, for a point charge at the centre of
    a sphere: (eps - 1)/eps for IEF-PCM and C-PCM (exact for IEF-PCM there),
    (eps - 1)/(eps + 1/2) for COSMO.  `charge_field` divides it out.
    """

    coords: Array  # (n_elements, 3), bohr
    normals: Array  # (n_elements, 3), outward unit normals
    areas: Array  # (n_elements,), bohr^2
    charges: Array  # (n_elements,), apparent surface charge q_i in e
    atom_coords: Array  # (n_atoms, 3), bohr
    atom_radii: Array  # (n_atoms,), bohr
    eps: float  # dielectric constant the charges were solved with
    charge_scale: float  # the model's charges relative to a conductor's

    def __post_init__(self) -> None:
        n = len(self.areas)
        if self.coords.shape != (n, 3) or self.normals.shape != (n, 3):
            raise ValueError(
                f"coords and normals must have shape ({n}, 3), "
                f"got {self.coords.shape} and {self.normals.shape}"
            )
        if self.charges.shape != (n,):
            raise ValueError(f"charges must have shape ({n},), got {self.charges.shape}")
        if self.atom_coords.shape != (len(self.atom_radii), 3):
            raise ValueError(
                f"atom_coords must have shape ({len(self.atom_radii)}, 3), "
                f"got {self.atom_coords.shape}"
            )


@dataclass(frozen=True)
class DensityGrid:
    """The solute's electron density and its gradient on a quadrature grid.

    The grids come from `cavity_quadrature.cavity_grid`, which covers either the
    inside or the outside of the cavity and ends exactly on its surface.
    """

    coords: Array  # (n_points, 3), bohr
    weights: Array  # (n_points,)
    rho: Array  # (n_points,), electrons / bohr^3
    grad: Array  # (n_points, 3)

    def __post_init__(self) -> None:
        n = len(self.weights)
        if self.coords.shape != (n, 3) or self.grad.shape != (n, 3) or self.rho.shape != (n,):
            raise ValueError(
                f"expected {n} points; got coords {self.coords.shape}, "
                f"rho {self.rho.shape}, grad {self.grad.shape}"
            )


@dataclass(frozen=True)
class Solute:
    """Everything the three terms need from one converged PCM calculation."""

    surface: Surface
    interior: DensityGrid  # the cavity, for the dispersion integral
    exterior: DensityGrid  # the rest of space, for the exchange integral
    # The solute's own electric field (nuclei + electrons, no reaction field)
    # projected on the outward normal at each surface element.  This is the
    # quantity whose extrema are FESR's F_min and F_max.
    normal_field: Array  # (n_elements,), a.u.

    def __post_init__(self) -> None:
        if self.normal_field.shape != self.surface.areas.shape:
            raise ValueError(
                f"normal_field must have shape {self.surface.areas.shape}, "
                f"got {self.normal_field.shape}"
            )


# ---------------------------------------------------------------------------
# The integrals
# ---------------------------------------------------------------------------


def interior_kernel_integral(points: Array, surface: Surface, delta: float = DELTA) -> Array:
    """J(r) = int_{r' in Omega} dr' / (|r - r'|^6 + delta^6) for each point r.

    The volume integral is done on the cavity surface.  1/(R^6 + delta^6) is the
    divergence of g(R) R_hat with g = arctan(R^3/delta^3) / (3 delta^3 R^2), so
    by the divergence theorem

        J(r) = sum_i A_i (R_i . n_i) arctan(t_i) / (3 delta^3 R_i^3),
        R_i = s_i - r,  t_i = R_i^3 / delta^3.

    g vanishes like R at R = 0, so unlike the usual double-layer kernel this has
    no singularity as r approaches the surface and holds for r inside or outside
    the cavity alike.  Written with arctan(t)/t it needs no division by R.
    """
    out = np.empty(len(points))
    block = max(1, _BLOCK_PAIRS // len(surface.areas))
    for p0 in range(0, len(points), block):
        sep = surface.coords[None, :, :] - points[p0 : p0 + block, None, :]
        t = (np.sum(sep**2, axis=2) ** 1.5) / delta**3
        small = t < 1e-8
        atan_over_t = np.where(small, 1.0, np.arctan(t) / np.where(small, 1.0, t))
        r_dot_n = np.einsum("pix,ix->pi", sep, surface.normals)
        out[p0 : p0 + block] = (r_dot_n * atan_over_t) @ surface.areas / (3.0 * delta**6)
    return out


def exterior_kernel_integral(points: Array, surface: Surface, delta: float = DELTA) -> Array:
    """I(r) = int_{r' outside Omega} dr' / (|r - r'|^6 + delta^6).

    All of space gives 2 pi^2 / (3 delta^3); the cavity's share is subtracted.
    """
    whole_space = 2.0 * np.pi**2 / (3.0 * delta**3)
    return whole_space - interior_kernel_integral(points, surface, delta)


def dispersion_integral(
    interior: DensityGrid,
    surface: Surface,
    solvent_rho: float,
    delta: float = DELTA,
    vv_c: float = VV_C,
) -> float:
    """The CMIRS dispersion integral, without its prefactor A.

    The solute density inside the cavity interacts with uniform solvent outside
    it through the damped R^-6 kernel; w is the VV local frequency (in units
    where the plasma term is just rho) and sqrt(solvent_rho) is the same thing
    for a uniform density.  The density factor rho / (w (w + sqrt(rho_s)))
    saturates at 1 where rho >> rho_s, so the nuclear cusps do not show up in
    the integrand.  ``solvent_rho`` is the liquid's mean electron density in
    e / bohr^3 (`config.solvent_electron_density`).
    """
    use = interior.rho > _RHO_FLOOR
    rho = interior.rho[use]
    grad_sq = np.sum(interior.grad[use] ** 2, axis=1)
    w = np.sqrt(rho + 3.0 * vv_c / (4.0 * np.pi) * grad_sq**2 / rho**4)
    kernel = exterior_kernel_integral(interior.coords[use], surface, delta)
    integrand = rho * kernel / (w * (w + np.sqrt(solvent_rho)))
    return float(interior.weights[use] @ integrand)


def exchange_integral(exterior: DensityGrid) -> float:
    """int |grad rho| over the region outside the cavity, without its prefactor B."""
    return float(exterior.weights @ np.linalg.norm(exterior.grad, axis=1))


def electrons_outside(exterior: DensityGrid) -> float:
    """Number of solute electrons outside the cavity (the "escaped charge")."""
    return float(exterior.weights @ exterior.rho)


def charge_field(surface: Surface) -> Array:
    """The apparent surface charge density recast as a normal field.

    A conductor's surface charge density is sigma = -E_n / (4 pi), with E_n the
    total normal field just inside the cavity (the solute's plus the surface
    charges' own), and a dielectric model scales it by ``charge_scale``
    ((eps - 1)/eps at a true dielectric boundary).  Inverting that puts q_i / A_i
    on the same scale and sign convention as the solute's normal field, to which
    it reduces exactly for a charge at the centre of a sphere.
    """
    return -4.0 * np.pi / surface.charge_scale * surface.charges / surface.areas


def surface_power_integral(
    field: Array, areas: Array, exponent: float, sign: Literal[-1, 1]
) -> float:
    """sum_i A_i |F_i|^exponent over the elements whose field has the given sign.

    The power is on the field alone and the area enters once, so the sum
    converges to a surface integral as the discretization is refined.
    """
    mask = field * sign > 0
    return float(areas[mask] @ np.abs(field[mask]) ** exponent)


# ---------------------------------------------------------------------------
# Parameters and descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """Fitted parameters of the nonelectrostatic model (atomic units)."""

    dispersion: float  # A
    exchange: float  # B
    neg_coeff: float  # c-, on elements with negative field (hydrogen-bond acceptors)
    neg_exponent: float  # n-
    pos_coeff: float  # c+, on elements with positive field (hydrogen-bond donors)
    pos_exponent: float  # n+


@dataclass(frozen=True)
class Energies:
    """The three contributions, in hartree."""

    dispersion: float
    exchange: float
    short_range: float

    @property
    def total(self) -> float:
        return self.dispersion + self.exchange + self.short_range


@dataclass(frozen=True)
class Descriptors:
    """Parameter-free integrals for one solute; all a fit needs to keep."""

    dispersion: float  # G_disp / A
    exchange: float  # G_exch / B
    areas: Array  # (n_elements,)
    normal_field: Array  # solute's normal field at each element
    charge_field: Array  # the same from the apparent surface charges
    electrons_outside: float

    def field(self, source: FieldSource = "field") -> Array:
        return self.normal_field if source == "field" else self.charge_field

    def field_extrema(self, source: FieldSource = "field") -> tuple[float, float]:
        """(F_min, F_max): the two numbers the original FESR term is built on."""
        f = self.field(source)
        return float(f.min()), float(f.max())

    def surface_integral(
        self, exponent: float, sign: Literal[-1, 1], source: FieldSource = "field"
    ) -> float:
        return surface_power_integral(self.field(source), self.areas, exponent, sign)

    def energy(self, params: Params, source: FieldSource = "field") -> Energies:
        short_range = params.neg_coeff * self.surface_integral(
            params.neg_exponent, -1, source
        ) + params.pos_coeff * self.surface_integral(params.pos_exponent, 1, source)
        return Energies(
            dispersion=params.dispersion * self.dispersion,
            exchange=params.exchange * self.exchange,
            short_range=short_range,
        )


def descriptors(
    solute: Solute, solvent_rho: float, delta: float = DELTA, vv_c: float = VV_C
) -> Descriptors:
    """Evaluate every parameter-free integral for one solute in one solvent.

    ``solvent_rho`` (e / bohr^3) enters the dispersion integral only; take it
    from `config.solvent_electron_density`.
    """
    return Descriptors(
        dispersion=dispersion_integral(solute.interior, solute.surface, solvent_rho, delta, vv_c),
        exchange=exchange_integral(solute.exterior),
        areas=solute.surface.areas,
        normal_field=solute.normal_field,
        charge_field=charge_field(solute.surface),
        electrons_outside=electrons_outside(solute.exterior),
    )
