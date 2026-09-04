import numpy as np
import pytest

from qm_pka.rdkit_utils import smiles_to_3d
from qm_pka.tautomer_dedup import (
    assign_hydrogens,
    assign_protons,
    deduplicate_tautomers,
    geometric_fingerprint,
    geometric_identity,
    heavy_frameworks_agree,
)
from qm_pka.types import Geometry


def _make_water() -> Geometry:
    """O-H-H with H's close to O."""
    return Geometry(
        symbols=("O", "H", "H"),
        coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]
        ),
    )


def _make_methanol() -> Geometry:
    """CH3-OH: C has 3 H's, O has 1 H."""
    return Geometry(
        symbols=("C", "O", "H", "H", "H", "H"),
        coords=np.array(
            [
                [0.0, 0.0, 0.0],  # C
                [1.43, 0.0, 0.0],  # O
                [-0.5, 0.9, 0.0],  # H on C
                [-0.5, -0.45, 0.78],  # H on C
                [-0.5, -0.45, -0.78],  # H on C
                [1.80, 0.85, 0.0],  # H on O
            ]
        ),
    )


def _make_methoxide() -> Geometry:
    """CH3-O⁻: C has 3 H's, O has 0 H's."""
    return Geometry(
        symbols=("C", "O", "H", "H", "H"),
        coords=np.array(
            [
                [0.0, 0.0, 0.0],  # C
                [1.43, 0.0, 0.0],  # O
                [-0.5, 0.9, 0.0],  # H on C
                [-0.5, -0.45, 0.78],  # H on C
                [-0.5, -0.45, -0.78],  # H on C
            ]
        ),
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
        assert geometric_fingerprint(g1) == geometric_fingerprint(g2)

    def test_different_tautomer_different_fp(self) -> None:
        fp_methanol = geometric_fingerprint(_make_methanol())
        fp_methoxide = geometric_fingerprint(_make_methoxide())
        assert fp_methanol != fp_methoxide


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


class TestHeavyFrameworksAgree:
    """Everything here is indexed by heavy-atom position, so this must hold."""

    def test_two_protomers_of_one_molecule_agree(self) -> None:
        assert heavy_frameworks_agree(_make_methanol(), _make_methoxide())

    def test_a_swapped_element_sequence_is_caught(self) -> None:
        g1 = _make_methanol()
        g2 = Geometry(
            symbols=("O", "C", "H", "H", "H", "H"),  # O and C swapped
            coords=g1.coords,
        )
        assert not heavy_frameworks_agree(g1, g2)

    def test_a_matching_sequence_with_a_different_framework_is_caught(self) -> None:
        """The check element sequence alone would pass: same symbols, different bonds."""
        joined = Geometry(
            symbols=("C", "C", "H", "H"),
            coords=np.array([[0.0, 0, 0], [1.5, 0, 0], [-1.0, 0, 0], [2.5, 0, 0]]),
        )
        apart = Geometry(
            symbols=("C", "C", "H", "H"),
            coords=np.array([[0.0, 0, 0], [6.0, 0, 0], [-1.0, 0, 0], [7.0, 0, 0]]),
        )
        assert [joined.symbols[i] for i in joined.heavy_atom_indices] == [
            apart.symbols[i] for i in apart.heavy_atom_indices
        ]
        assert not heavy_frameworks_agree(joined, apart)


class TestOneAssignmentPrimitive:
    """The approach-2 fingerprint and the migration check must not drift apart.

    `tautomer_id` is written at sampling and read back by
    `protomer_geometry.repair_migrated_conformers`. A second implementation of
    "which heavy atom owns this hydrogen" makes them disagree on a bridging H,
    and the repair then reads a proton migration that never happened. Sharing
    the hash does not prevent that -- only sharing the assignment does.
    """

    @staticmethod
    def _bridging() -> Geometry:
        """C...H...O with d(C-H)=1.28 and d(O-H)=1.20.

        The distances that separated the two implementations: an element-cutoff
        rule assigns to carbon (oxygen is outside its 1.15 A cutoff), while
        nearest-heavy-atom assigns to oxygen.
        """
        return Geometry(
            symbols=("C", "O", "H"),
            coords=np.array([[0.0, 0.0, 0.0], [2.48, 0.0, 0.0], [1.28, 0.0, 0.0]]),
        )

    def test_the_count_vector_comes_from_one_place(self) -> None:
        geom = self._bridging()
        assert assign_hydrogens(geom) == assign_protons(geom).counts

    def test_the_fingerprint_survives_a_round_trip_through_repair(self) -> None:
        geom = self._bridging()
        written_at_sampling = geometric_fingerprint(geom)
        read_back_by_repair = geometric_fingerprint(geom)
        assert written_at_sampling == read_back_by_repair

    def test_nearest_heavy_atom_wins_with_no_cutoff(self) -> None:
        """Pins the rule itself, so a cutoff cannot be reintroduced silently."""
        geom = self._bridging()
        heavy = geom.heavy_atom_indices
        assert assign_protons(geom).owner == (heavy[1],)  # the oxygen, at 1.20 A


def _geom(smiles: str) -> Geometry:
    return smiles_to_3d(smiles)[0]


class TestTetrahedralParity:
    """Configuration the hydrogen counts alone cannot see."""

    def test_a_nitrogen_chiral_only_once_protonated(self) -> None:
        """The case this exists for.

        A tertiary amine inverts freely and is not a stereocentre. Protonating it
        quaternises the nitrogen, which cannot invert without breaking a bond --
        so the two faces become distinct species. With a second stereocentre they
        are diastereomers, not enantiomers, and must not share a microstate.
        """
        first = _geom("CC[N@@H+](C)C[C@@H](C)O")
        second = _geom("CC[N@H+](C)C[C@@H](C)O")
        assert geometric_identity(first).counts == geometric_identity(second).counts
        assert geometric_fingerprint(first) != geometric_fingerprint(second)

    def test_a_1_4_ring_cis_trans_pair(self) -> None:
        """Neither ring carbon is a stereocentre on its own.

        Both ring branches are constitutionally identical, so no amount of
        constitutional refinement separates cis from trans -- the configuration
        lives in the pair. Recording a parity at each captures it.
        """
        cis = _geom("C[C@H](c1ccccc1)[C@H]2CC[C@@H](C(=O)O)CC2")
        trans = _geom("C[C@H](c1ccccc1)[C@H]2CC[C@H](C(=O)O)CC2")
        assert geometric_identity(cis).counts == geometric_identity(trans).counts
        assert geometric_fingerprint(cis) != geometric_fingerprint(trans)

    def test_no_parity_where_hydrogens_are_interchangeable(self) -> None:
        """CH2 and CH3 are excluded: their hydrogen ordering is not stable."""
        geom = _geom("CCCO")
        identity = geometric_identity(geom)
        heavy = geom.heavy_atom_indices
        for pos, count in enumerate(identity.counts):
            if count >= 2:
                assert identity.parities[pos] == 0, (
                    f"{geom.symbols[heavy[pos]]} with {count} hydrogens must not carry a parity"
                )

    def test_an_ordinary_carbon_stereocentre_is_recorded(self) -> None:
        """Three heavy neighbours and one hydrogen -- missed by any heavy-only rule."""
        geom = _geom("N[C@@H](C)C(=O)O")
        identity = geometric_identity(geom)
        assert any(p != 0 for p in identity.parities)

    def test_parity_is_stable_across_conformers_of_one_species(self) -> None:
        """A rotation is not a configuration change."""
        geom = _geom("N[C@@H](C)C(=O)O")
        rotated = Geometry(
            symbols=geom.symbols,
            coords=geom.coords @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        assert geometric_fingerprint(geom) == geometric_fingerprint(rotated)

    def test_a_reflection_is_a_configuration_change(self) -> None:
        geom = _geom("N[C@@H](C)C(=O)O")
        mirrored = Geometry(symbols=geom.symbols, coords=geom.coords * np.array([-1.0, 1.0, 1.0]))
        assert geometric_fingerprint(geom) != geometric_fingerprint(mirrored)
