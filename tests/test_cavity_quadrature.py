"""Tests for the inside/outside cavity quadrature."""

from __future__ import annotations

import numpy as np
import pytest

from tautomator.cavity_quadrature import Array, Region, becke_partition, cavity_grid

# Two overlapping spheres, and a five-sphere cavity with one sphere mostly buried.
PAIR = (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2]]), np.array([3.4, 2.5]))
CLUSTER = (
    np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.3], [2.4, 0.0, -1.5], [-2.4, 0.0, -1.5], [0.3, 0.2, 1.0]]
    ),
    np.array([3.85, 3.45, 3.85, 3.85, 2.0]),
)


def _inside(points: Array, centres: Array, radii: Array) -> Array:
    dist = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
    return np.asarray((dist < radii[None, :]).any(axis=1))


def _tails(points: Array, centres: Array, decay: float) -> Array:
    """Sum of exp(-decay r) on every centre: cusps inside, density-like tails outside."""
    dist = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
    return np.asarray(np.exp(-decay * dist).sum(axis=1))


class TestBeckePartition:
    def test_rows_sum_to_one_and_nuclei_own_themselves(self) -> None:
        centres, _radii = CLUSTER
        points = np.vstack([centres, np.random.default_rng(1).normal(scale=4.0, size=(50, 3))])
        cells = becke_partition(points, centres)
        assert cells.sum(axis=1) == pytest.approx(1.0, abs=1e-14)
        assert np.diag(cells[: len(centres)]) == pytest.approx(1.0, abs=1e-14)


class TestCavityGrid:
    def test_single_sphere_volume_is_exact(self) -> None:
        _coords, weights = cavity_grid(np.zeros((1, 3)), np.array([2.7]), "inside")
        assert weights.sum() == pytest.approx(4.0 / 3.0 * np.pi * 2.7**3, rel=1e-13)

    def test_volume_of_two_overlapping_spheres(self) -> None:
        centres, radii = PAIR
        d, (r1, r2) = 2.2, radii
        lens = (
            np.pi * (r1 + r2 - d) ** 2 * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2) / (12 * d)
        )
        expected = 4.0 / 3.0 * np.pi * (r1**3 + r2**3) - lens
        _coords, weights = cavity_grid(centres, radii, "inside")
        # Limited by the angular rule: the surface each ray ends on has a crease
        # where the spheres meet.
        assert weights.sum() == pytest.approx(expected, rel=3e-4)

    @pytest.mark.parametrize("region", ["inside", "outside"])
    def test_points_lie_in_their_region(self, region: Region) -> None:
        centres, radii = CLUSTER
        coords, _weights = cavity_grid(centres, radii, region)
        assert np.all(_inside(coords, centres, radii) == (region == "inside"))

    @pytest.mark.parametrize("cavity", [PAIR, CLUSTER])
    def test_inside_and_outside_fill_space(self, cavity: tuple[Array, Array]) -> None:
        """The two regions share their boundary, so its quadrature error cancels in the sum."""
        centres, radii = cavity
        decay = 0.9
        total = 0.0
        region: Region
        for region in ("inside", "outside"):
            coords, weights = cavity_grid(centres, radii, region)
            total += float(weights @ _tails(coords, centres, decay))
        assert total == pytest.approx(len(centres) * 8.0 * np.pi / decay**3, rel=1e-7)

    def test_outside_integral_converges(self) -> None:
        centres, radii = CLUSTER
        values = []
        for order, n_radial in [(41, 40), (59, 60)]:
            coords, weights = cavity_grid(centres, radii, "outside", order, n_radial)
            values.append(weights @ _tails(coords, centres, 0.9))
        assert values[0] == pytest.approx(values[1], rel=1e-3)

    def test_moves_continuously_with_the_geometry(self) -> None:
        """A small displacement changes the integral by a small amount: no shells flip."""
        centres, radii = PAIR
        steps = []
        for shift in (0.0, 1e-4, 2e-4):
            moved = centres + np.array([[0.0, 0.0, 0.0], [0.0, shift, shift]])
            coords, weights = cavity_grid(moved, radii, "outside")
            steps.append(weights @ _tails(coords, moved, 0.9))
        first, second = steps[1] - steps[0], steps[2] - steps[1]
        assert abs(first) < 1e-3
        assert second == pytest.approx(first, rel=1e-2)
