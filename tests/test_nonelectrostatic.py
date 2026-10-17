"""Tests for the nonelectrostatic terms against closed forms on a spherical cavity."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import lebedev_rule

from tautomator import nonelectrostatic
from tautomator.nonelectrostatic import Array, DensityGrid, Surface

RADIUS = 3.0
DELTA = 2.5  # comparable to the sphere, so the kernel varies across it


def _lebedev(order: int = 41) -> tuple[Array, Array]:
    """Unit vectors (n, 3) and weights summing to 4 pi."""
    x, w = lebedev_rule(order)
    return np.ascontiguousarray(x.T), np.asarray(w)


def _sphere(
    radius: float = RADIUS, charge: float = 0.0, eps: float = 78.39, scale: float | None = None
) -> Surface:
    """One-sphere cavity at the origin carrying the surface charge of a central point charge.

    ``scale`` is the model's charge scaling; (eps - 1)/eps, a true dielectric's, by default.
    """
    unit, w = _lebedev()
    areas = w * radius**2
    scale = (eps - 1.0) / eps if scale is None else scale
    return Surface(
        coords=radius * unit,
        normals=unit,
        areas=areas,
        charges=-scale * charge * areas / (4.0 * np.pi * radius**2),
        atom_coords=np.zeros((1, 3)),
        atom_radii=np.array([radius]),
        eps=eps,
        charge_scale=scale,
    )


def _radial_grid(r_min: float, r_max: float, n_radial: int = 120) -> tuple[Array, Array]:
    """Product grid (Gauss-Legendre in r, Lebedev in angle) over a spherical shell."""
    x, wx = np.polynomial.legendre.leggauss(n_radial)
    r = 0.5 * (r_max - r_min) * x + 0.5 * (r_max + r_min)
    wr = 0.5 * (r_max - r_min) * wx * r**2
    unit, w = _lebedev()
    coords = (r[:, None, None] * unit[None, :, :]).reshape(-1, 3)
    return coords, (wr[:, None] * w[None, :]).ravel()


def _hydrogenic(coords: Array, weights: Array) -> DensityGrid:
    """rho = exp(-2r)/pi, the 1s density, for which |grad rho| = 2 rho."""
    r = np.linalg.norm(coords, axis=1)
    rho = np.exp(-2.0 * r) / np.pi
    return DensityGrid(coords, weights, rho, -2.0 * rho[:, None] * coords / r[:, None])


class TestKernelIntegrals:
    def test_centre_of_sphere(self) -> None:
        j = nonelectrostatic.interior_kernel_integral(np.zeros((1, 3)), _sphere(), DELTA)
        expected = 4.0 * np.pi / (3.0 * DELTA**3) * np.arctan(RADIUS**3 / DELTA**3)
        assert j[0] == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("distance", [0.0, 1.7, 2.999, 3.001, 4.5, 20.0])
    def test_matches_volume_quadrature(self, distance: float) -> None:
        """Inside, at the surface and outside: no singularity as the point crosses it."""
        point = np.array([[0.0, 0.0, distance]])
        coords, weights = _radial_grid(0.0, RADIUS)
        sep = np.linalg.norm(coords - point, axis=1)
        expected = weights @ (1.0 / (sep**6 + DELTA**6))
        j = nonelectrostatic.interior_kernel_integral(point, _sphere(), DELTA)
        assert j[0] == pytest.approx(expected, rel=1e-9)

    def test_interior_and_exterior_fill_space(self) -> None:
        points = np.array([[0.3, -1.0, 0.8], [5.0, 1.0, 0.0]])
        surface = _sphere()
        total = nonelectrostatic.interior_kernel_integral(
            points, surface, DELTA
        ) + nonelectrostatic.exterior_kernel_integral(points, surface, DELTA)
        assert total == pytest.approx(2.0 * np.pi**2 / (3.0 * DELTA**3), rel=1e-12)

    def test_blocks_agree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        points = np.random.default_rng(0).normal(size=(37, 3))
        whole = nonelectrostatic.interior_kernel_integral(points, _sphere(), DELTA)
        monkeypatch.setattr(nonelectrostatic, "_BLOCK_PAIRS", 5 * len(_sphere().areas))
        assert nonelectrostatic.interior_kernel_integral(
            points, _sphere(), DELTA
        ) == pytest.approx(whole, rel=1e-13)


class TestDensityIntegrals:
    def test_exchange_is_density_gradient_outside(self) -> None:
        # int_{r > a} 2 exp(-2r)/pi dV = exp(-2a) (4a^2 + 4a + 2)
        exterior = _hydrogenic(*_radial_grid(RADIUS, 40.0, n_radial=200))
        expected = np.exp(-2.0 * RADIUS) * (4.0 * RADIUS**2 + 4.0 * RADIUS + 2.0)
        assert nonelectrostatic.exchange_integral(exterior) == pytest.approx(expected, rel=1e-10)

    def test_electrons_outside(self) -> None:
        exterior = _hydrogenic(*_radial_grid(RADIUS, 40.0, n_radial=200))
        expected = np.exp(-2.0 * RADIUS) * (2.0 * RADIUS**2 + 2.0 * RADIUS + 1.0)
        assert nonelectrostatic.electrons_outside(exterior) == pytest.approx(expected, rel=1e-10)

    def test_dispersion_against_radial_quadrature(self) -> None:
        surface = _sphere()
        solvent_rho, vv_c = 0.05, 0.0093
        # Reference: the same integral done in one dimension.  |grad rho|/rho = 2,
        # and I depends on |r| only.
        x, wx = np.polynomial.legendre.leggauss(200)
        r = 0.5 * RADIUS * (x + 1.0)
        rho = np.exp(-2.0 * r) / np.pi
        w = np.sqrt(rho + 3.0 * vv_c / (4.0 * np.pi) * 16.0)
        on_axis = np.column_stack([np.zeros_like(r), np.zeros_like(r), r])
        kernel = nonelectrostatic.exterior_kernel_integral(on_axis, surface, DELTA)
        expected = (0.5 * RADIUS * wx * 4.0 * np.pi * r**2) @ (
            rho * kernel / (w * (w + np.sqrt(solvent_rho)))
        )
        interior = _hydrogenic(*_radial_grid(0.0, RADIUS))
        value = nonelectrostatic.dispersion_integral(interior, surface, solvent_rho, DELTA, vv_c)
        assert value == pytest.approx(expected, rel=1e-9)

    def test_dispersion_density_factor_saturates(self) -> None:
        """Where rho >> rho_solvent the density factor is 1, so the integral is that of I."""
        surface = _sphere()
        coords, weights = _radial_grid(0.0, RADIUS)
        dense = DensityGrid(
            coords, weights, np.full(len(weights), 1e6), np.zeros((len(weights), 3))
        )
        expected = weights @ nonelectrostatic.exterior_kernel_integral(coords, surface, DELTA)
        assert nonelectrostatic.dispersion_integral(dense, surface, 0.05, DELTA) == pytest.approx(
            expected, rel=1e-3
        )


class TestSurfaceTerm:
    @pytest.mark.parametrize("eps", [78.39, 2.0402])
    @pytest.mark.parametrize("cosmo", [False, True], ids=["dielectric", "cosmo"])
    def test_charge_field_of_central_charge(self, eps: float, cosmo: bool) -> None:
        """For a charge at the centre of a sphere the surface charge gives back q/a^2,
        whichever way the model scales its charges."""
        scale = (eps - 1.0) / (eps + 0.5) if cosmo else (eps - 1.0) / eps
        for charge in (-1.0, 1.0):
            surface = _sphere(charge=charge, eps=eps, scale=scale)
            field = nonelectrostatic.charge_field(surface)
            assert field == pytest.approx(charge / RADIUS**2, rel=1e-12)

    def test_power_integral_of_uniform_field(self) -> None:
        surface = _sphere()
        field = np.full(len(surface.areas), -0.04)
        expected = 4.0 * np.pi * RADIUS**2 * 0.04**3.2
        assert nonelectrostatic.surface_power_integral(
            field, surface.areas, 3.2, -1
        ) == pytest.approx(expected, rel=1e-12)
        assert nonelectrostatic.surface_power_integral(field, surface.areas, 3.2, 1) == 0.0

    def test_power_integral_splits_by_sign(self) -> None:
        field = np.array([-2.0, 0.0, 3.0, 1.0])
        areas = np.array([1.0, 5.0, 0.5, 2.0])
        assert nonelectrostatic.surface_power_integral(field, areas, 2.0, -1) == pytest.approx(4.0)
        assert nonelectrostatic.surface_power_integral(field, areas, 2.0, 1) == pytest.approx(6.5)

    def test_refining_the_surface_leaves_the_integral_unchanged(self) -> None:
        """Splitting every element in two must not change the result (F^n A, not (F A)^n)."""
        field = np.array([-0.05, 0.02, 0.08])
        areas = np.array([1.0, 2.0, 0.5])
        coarse = nonelectrostatic.surface_power_integral(field, areas, 3.0, 1)
        fine = nonelectrostatic.surface_power_integral(
            np.repeat(field, 2), np.repeat(areas / 2, 2), 3.0, 1
        )
        assert fine == pytest.approx(coarse, rel=1e-14)


class TestDescriptors:
    def test_energy_assembles_the_three_terms(self) -> None:
        d = nonelectrostatic.Descriptors(
            dispersion=2.0,
            exchange=0.5,
            areas=np.array([1.0, 2.0]),
            normal_field=np.array([-0.1, 0.2]),
            charge_field=np.array([-0.2, 0.1]),
            electrons_outside=0.0,
        )
        params = nonelectrostatic.Params(
            dispersion=-0.01,
            exchange=0.04,
            neg_coeff=-100.0,
            neg_exponent=3.0,
            pos_coeff=-10.0,
            pos_exponent=2.0,
        )
        e = d.energy(params)
        assert e.dispersion == pytest.approx(-0.02)
        assert e.exchange == pytest.approx(0.02)
        assert e.short_range == pytest.approx(-100.0 * 1.0 * 0.1**3 - 10.0 * 2.0 * 0.2**2)
        assert e.total == pytest.approx(e.dispersion + e.exchange + e.short_range)
        by_charge = d.energy(params, source="charge")
        assert by_charge.short_range == pytest.approx(-100.0 * 0.2**3 - 10.0 * 2.0 * 0.1**2)
        assert d.field_extrema() == (-0.1, 0.2)
