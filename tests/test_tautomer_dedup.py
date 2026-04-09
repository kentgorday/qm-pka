import numpy as np
import pytest

from qm_pka.tautomer_dedup import (
    assign_hydrogens,
    deduplicate_tautomers,
    h_assignment_fingerprint,
    validate_heavy_atom_ordering,
)
from qm_pka.types import Geometry


def _make_water() -> Geometry:
    """O-H-H with H's close to O."""
    return Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ]),
    )


def _make_methanol() -> Geometry:
    """CH3-OH: C has 3 H's, O has 1 H."""
    return Geometry(
        symbols=("C", "O", "H", "H", "H", "H"),
        coords=np.array([
            [0.0, 0.0, 0.0],       # C
            [1.43, 0.0, 0.0],      # O
            [-0.5, 0.9, 0.0],      # H on C
            [-0.5, -0.45, 0.78],   # H on C
            [-0.5, -0.45, -0.78],  # H on C
            [1.80, 0.85, 0.0],     # H on O
        ]),
    )


def _make_methoxide() -> Geometry:
    """CH3-O⁻: C has 3 H's, O has 0 H's."""
    return Geometry(
        symbols=("C", "O", "H", "H", "H"),
        coords=np.array([
            [0.0, 0.0, 0.0],       # C
            [1.43, 0.0, 0.0],      # O
            [-0.5, 0.9, 0.0],      # H on C
            [-0.5, -0.45, 0.78],   # H on C
            [-0.5, -0.45, -0.78],  # H on C
        ]),
    )


class TestAssignHydrogens:
    def test_water(self) -> None:
        geom = _make_water()
        result = assign_hydrogens(geom)
        # O is the only heavy atom, both H's are bonded to it
        assert result == (2,)

    def test_methanol(self) -> None:
        geom = _make_methanol()
        result = assign_hydrogens(geom)
        # C has 3 H's, O has 1 H
        assert result == (3, 1)

    def test_methoxide(self) -> None:
        geom = _make_methoxide()
        result = assign_hydrogens(geom)
        # C has 3 H's, O has 0 H's
        assert result == (3, 0)

    def test_no_heavy_atoms_raises(self) -> None:
        geom = Geometry(symbols=("H", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]))
        with pytest.raises(ValueError, match="no heavy atoms"):
            assign_hydrogens(geom)


class TestFingerprint:
    def test_same_tautomer_same_fp(self) -> None:
        g1 = _make_methanol()
        # Second methanol with slightly different coords but same connectivity
        g2 = Geometry(
            symbols=("C", "O", "H", "H", "H", "H"),
            coords=g1.coords + np.random.default_rng(42).normal(0, 0.01, g1.coords.shape),
        )
        assert h_assignment_fingerprint(g1) == h_assignment_fingerprint(g2)

    def test_different_tautomer_different_fp(self) -> None:
        assert h_assignment_fingerprint(_make_methanol()) != h_assignment_fingerprint(_make_methoxide())


class TestDeduplicateTautomers:
    def test_groups_same_tautomers(self) -> None:
        g1 = _make_methanol()
        g2 = Geometry(
            symbols=g1.symbols,
            coords=g1.coords + 0.001,  # tiny perturbation, same connectivity
        )
        g3 = _make_methoxide()
        groups = deduplicate_tautomers([g1, g2, g3])
        assert len(groups) == 2
        # One group has 2 geometries, the other has 1
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [1, 2]


class TestValidateHeavyAtomOrdering:
    def test_same_ordering(self) -> None:
        assert validate_heavy_atom_ordering(_make_methanol(), _make_methoxide())

    def test_different_ordering(self) -> None:
        g1 = _make_methanol()
        g2 = Geometry(
            symbols=("O", "C", "H", "H", "H", "H"),  # O and C swapped
            coords=g1.coords,
        )
        assert not validate_heavy_atom_ordering(g1, g2)
