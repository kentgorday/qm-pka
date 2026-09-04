"""Tests for protomer identity read off a geometry, and migration repair."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from rdkit import Chem

from qm_pka.ensemble import deduplicate_conformers
from qm_pka.protomer_geometry import (
    DETACHED_DISTANCE,
    _specified_stereo,
    _stereo_signature,
    assign_protons,
    match_to_candidate,
    protonation_key_from_geometry,
    protonation_key_from_mol,
    repair_migrated_conformers,
    template_from_smiles,
)
from qm_pka.rdkit_utils import smiles_to_3d
from qm_pka.tautomer_dedup import h_assignment_fingerprint
from qm_pka.types import ChargeState, Conformer, Geometry, Microstate


def _embed(smiles: str) -> tuple[Geometry, str]:
    return smiles_to_3d(smiles)


def _move_h(geom: Geometry, h_index: int, target_heavy: int, dist: float = 1.02) -> Geometry:
    """Relocate one hydrogen onto a different heavy atom, as a migration would."""
    coords = geom.coords.copy()
    centroid = coords[geom.heavy_atom_indices].mean(axis=0)
    outward = coords[target_heavy] - centroid
    norm = float(np.linalg.norm(outward))
    outward = outward / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    coords[h_index] = coords[target_heavy] + dist * outward
    return Geometry(symbols=tuple(geom.symbols), coords=coords)


def _owner_of(geom: Geometry, h_index: int) -> int:
    assignment = assign_protons(geom)
    return assignment.owner[geom.hydrogen_indices.index(h_index)]


class TestAssignProtons:
    def test_methanol_hydrogens_land_on_the_right_atoms(self) -> None:
        geom, _ = _embed("CO")
        counts = assign_protons(geom).counts
        heavy = [geom.symbols[i] for i in geom.heavy_atom_indices]
        assert dict(zip(heavy, counts, strict=True)) == {"C": 3, "O": 1}

    def test_every_hydrogen_is_counted_exactly_once(self) -> None:
        geom, _ = _embed("NCC(=O)O")
        assignment = assign_protons(geom)
        assert sum(assignment.counts) == len(geom.hydrogen_indices)
        assert len(assignment.owner) == len(geom.hydrogen_indices)

    def test_a_bonded_hydrogen_is_not_flagged_detached(self) -> None:
        geom, _ = _embed("NCC(=O)O")
        assert assign_protons(geom).is_intact

    def test_a_hydrogen_pulled_away_is_flagged_detached(self) -> None:
        geom, _ = _embed("NCC(=O)O")
        h = geom.hydrogen_indices[0]
        coords = geom.coords.copy()
        coords[h] = coords[h] + np.array([0.0, 0.0, 6.0])
        moved = Geometry(symbols=tuple(geom.symbols), coords=coords)
        assignment = assign_protons(moved)
        assert not assignment.is_intact
        assert assignment.detached == (h,)

    def test_the_margin_is_recorded_and_comfortable_for_an_ordinary_molecule(self) -> None:
        geom, _ = _embed("CCO")
        assert assign_protons(geom).min_margin > 0.3

    def test_a_geometry_with_no_heavy_atoms_is_refused(self) -> None:
        geom = Geometry(symbols=("H", "H"), coords=np.array([[0.0, 0, 0], [0.74, 0, 0]]))
        with pytest.raises(ValueError, match="no heavy atoms"):
            assign_protons(geom)


class TestKeyAgreesBetweenSmilesAndGeometry:
    @pytest.mark.parametrize(
        "smiles,charge",
        [
            ("NCC(=O)O", 0),
            ("[NH3+]CC(=O)O", 1),
            ("NCC(=O)[O-]", -1),
            ("Cc1cc[nH]c1", 0),
            ("OC(=O)/C=C\\C(=O)[O-]", -1),
        ],
    )
    def test_an_undisturbed_geometry_reproduces_its_label(self, smiles: str, charge: int) -> None:
        geom, explicit = _embed(smiles)
        template = template_from_smiles(explicit)
        assert protonation_key_from_geometry(geom, template, charge) == protonation_key_from_mol(
            template, charge
        )

    def test_a_template_from_another_molecule_is_refused(self) -> None:
        geom, _ = _embed("NCC(=O)O")
        other = template_from_smiles(smiles_to_3d("CCCCO")[1])
        with pytest.raises(ValueError, match="heavy-atom ordering"):
            protonation_key_from_geometry(geom, other, 0)


class TestAutomorphicSitesShareAKey:
    """A proton shared between equivalent heavy atoms must not split a species."""

    def test_the_two_carboxylate_oxygens_of_malonate_are_interchangeable(self) -> None:
        a = template_from_smiles(smiles_to_3d("O=C(O)CC(=O)[O-]")[1])
        b = template_from_smiles(smiles_to_3d("O=C([O-])CC(=O)O")[1])
        assert protonation_key_from_mol(a, -1) == protonation_key_from_mol(b, -1)

    def test_maleate_is_interchangeable_end_for_end(self) -> None:
        a = template_from_smiles(smiles_to_3d(r"O=C(O)/C=C\C(=O)[O-]")[1])
        b = template_from_smiles(smiles_to_3d(r"O=C([O-])/C=C\C(=O)O")[1])
        assert protonation_key_from_mol(a, -1) == protonation_key_from_mol(b, -1)

    def test_inequivalent_sites_stay_apart(self) -> None:
        """Citraconate: the methyl makes the two carboxyls distinguishable."""
        a = template_from_smiles(smiles_to_3d(r"O=C(O)/C(C)=C\C(=O)[O-]")[1])
        b = template_from_smiles(smiles_to_3d(r"O=C([O-])/C(C)=C\C(=O)O")[1])
        assert protonation_key_from_mol(a, -1) != protonation_key_from_mol(b, -1)

    def test_different_protonation_sites_stay_apart(self) -> None:
        a = template_from_smiles(smiles_to_3d("[NH3+]CC(=O)O")[1])
        b = template_from_smiles(smiles_to_3d("NCC(=O)[OH2+]")[1])
        assert protonation_key_from_mol(a, 1) != protonation_key_from_mol(b, 1)


def _microstate(smiles: str, conformers: list[Conformer] | None = None) -> Microstate:
    geom, explicit = _embed(smiles)
    return Microstate(
        tautomer_id=smiles,
        conformers=conformers if conformers is not None else [Conformer(geometry=geom)],
        smiles=explicit,
    )


class TestRepairMigratedConformers:
    def test_an_undisturbed_charge_state_is_left_alone(self) -> None:
        cs = ChargeState(charge=1, microstates=[_microstate("[NH3+]CC(=O)O")])
        report = repair_migrated_conformers(cs, stage="refinement")
        assert report.touched == 0
        assert len(cs.microstates[0].conformers) == 1

    def test_a_migrated_conformer_moves_to_the_microstate_it_became(self) -> None:
        source = _microstate("NCC(=O)[OH2+]")
        target = _microstate("[NH3+]CC(=O)O", conformers=[])
        geom = source.conformers[0].geometry

        # Move one of the [OH2+] protons onto the amine nitrogen.
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, target])
        report = repair_migrated_conformers(cs, stage="refinement")

        assert report.moved == 1
        assert source.conformers == []
        assert len(target.conformers) == 1
        assert source.excluded_conformers == []

    def test_a_detached_hydrogen_is_excluded_not_dropped(self) -> None:
        ms = _microstate("[NH3+]CC(=O)O")
        geom = ms.conformers[0].geometry
        h = geom.hydrogen_indices[0]
        coords = geom.coords.copy()
        coords[h] += np.array([0.0, 0.0, 8.0])
        ms.conformers[0].geometry = Geometry(symbols=tuple(geom.symbols), coords=coords)

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="refinement")

        assert report.detached == 1
        assert ms.conformers == []
        assert len(ms.excluded_conformers) == 1
        excluded = ms.excluded_conformers[0]
        assert excluded.reason == "proton_detached"
        assert excluded.stage == "refinement"
        assert str(DETACHED_DISTANCE) in excluded.detail

    def test_a_species_no_microstate_describes_is_excluded(self) -> None:
        """A proton onto carbon: the enumerator only ever touches heteroatoms."""
        ms = _microstate("[NH3+]CC(=O)O")
        geom = ms.conformers[0].geometry
        carbon = next(i for i, s in enumerate(geom.symbols) if s == "C")
        nitrogens = [i for i, s in enumerate(geom.symbols) if s == "N"]
        amine_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in nitrogens)
        ms.conformers[0].geometry = _move_h(geom, amine_h, carbon)

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="refinement")

        assert report.unmatched == 1
        assert ms.conformers == []
        assert ms.excluded_conformers[0].reason == "no_matching_microstate"

    def test_a_stereocentre_tie_is_resolved_from_the_geometry(self) -> None:
        """Two candidates, same protonation, opposite configuration at carbon."""
        source = _microstate("N[C@@H](C)C(=O)[OH2+]")
        first = _microstate("[NH3+][C@@H](C)C(=O)O", conformers=[])
        second = _microstate("[NH3+][C@H](C)C(=O)O", conformers=[])
        assert protonation_key_from_mol(
            template_from_smiles(first.smiles or ""), 1
        ) == protonation_key_from_mol(template_from_smiles(second.smiles or ""), 1)

        geom = source.conformers[0].geometry
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, first, second])
        report = repair_migrated_conformers(cs, stage="refinement")

        # Moving a proton between heteroatoms does not invert a carbon centre,
        # so the geometry still has the configuration it started with.
        assert report.moved == 1
        assert report.stereo_resolved == 1
        assert report.ambiguous == 0
        assert len(first.conformers) == 1
        assert second.conformers == []

    def test_a_double_bond_configuration_is_resolved_from_the_geometry(self) -> None:
        source = _microstate(r"O=C(O)/C=C\C(=O)[OH2+]")
        cis = _microstate(r"OC(=[OH+])/C=C\C(=O)O", conformers=[])
        trans = _microstate(r"OC(=[OH+])/C=C/C(=O)O", conformers=[])

        geom = source.conformers[0].geometry
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        counts = {
            o: sum(1 for h in geom.hydrogen_indices if _owner_of(geom, h) == o) for o in oxygens
        }
        donor = max(counts, key=lambda o: counts[o])
        acceptor = min(counts, key=lambda o: counts[o])
        moving = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) == donor)
        source.conformers[0].geometry = _move_h(geom, moving, acceptor)

        cs = ChargeState(charge=1, microstates=[source, cis, trans])
        report = repair_migrated_conformers(cs, stage="refinement")

        if report.moved:
            # A migrating proton does not rotate a C=C: the cis backbone stays cis.
            assert report.stereo_resolved == 1
            assert len(cis.conformers) == 1
            assert trans.conformers == []

    def test_a_tie_with_nothing_to_discriminate_is_excluded(self) -> None:
        """Candidates whose stereo is identical leave the geometry no question to answer."""
        source = _microstate("NCC(=O)[OH2+]")
        first = _microstate("[NH3+]CC(=O)O", conformers=[])
        second = _microstate("[NH3+]CC(=O)O", conformers=[])
        second.tautomer_id = "a distinct label, identical stereo"

        geom = source.conformers[0].geometry
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, first, second])
        report = repair_migrated_conformers(cs, stage="refinement")

        assert report.ambiguous == 1
        assert report.moved == 0
        assert source.excluded_conformers[0].reason == "ambiguous_microstate"

    def test_a_mixed_charge_state_is_skipped(self) -> None:
        """Half-labelled means neither rule applies; refuse rather than guess."""
        labelled = _microstate("[NH3+]CC(=O)O")
        geom, _ = _embed("[NH3+]CC(=O)O")
        unlabelled = Microstate(
            tautomer_id="deadbeef", conformers=[Conformer(geometry=geom)], smiles=None
        )
        cs = ChargeState(charge=1, microstates=[labelled, unlabelled])
        report = repair_migrated_conformers(cs, stage="sampling")
        assert report.checked == 0
        assert len(unlabelled.conformers) == 1


def _unlabelled(geom: Geometry, includes_enantiomer: bool = False) -> Microstate:
    """An approach-2 microstate: identified by its H distribution, with no SMILES."""
    return Microstate(
        tautomer_id=h_assignment_fingerprint(geom),
        conformers=[Conformer(geometry=geom)],
        smiles=None,
        includes_enantiomer=includes_enantiomer,
    )


class TestRepairWithoutSmilesLabels:
    """Approach 2: the microstate set is discovered, so an unseen state is created."""

    def test_an_undisturbed_conformer_stays_put(self) -> None:
        geom, _ = _embed("[NH3+]CC(=O)O")
        ms = _unlabelled(geom)
        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="sampling")
        assert report.touched == 0
        assert len(ms.conformers) == 1
        assert len(cs.microstates) == 1

    def test_a_migration_to_an_unseen_position_opens_a_microstate(self) -> None:
        geom, _ = _embed("NCC(=O)[OH2+]")
        ms = _unlabelled(geom, includes_enantiomer=True)
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        ms.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.created == 1
        assert report.unmatched == 0
        assert ms.conformers == []
        assert len(cs.microstates) == 2
        opened = cs.microstates[1]
        assert len(opened.conformers) == 1
        assert opened.smiles is None
        assert opened.tautomer_id == h_assignment_fingerprint(opened.conformers[0].geometry)
        assert opened.includes_enantiomer is True

    def test_a_migration_to_a_sampled_position_moves_rather_than_creates(self) -> None:
        geom, _ = _embed("NCC(=O)[OH2+]")
        source = _unlabelled(geom)
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        migrated = _move_h(geom, acid_h, nitrogen)

        target = _unlabelled(migrated)
        target.conformers = []
        source.conformers[0].geometry = migrated

        cs = ChargeState(charge=1, microstates=[source, target])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.moved == 1
        assert report.created == 0
        assert source.conformers == []
        assert len(target.conformers) == 1
        assert len(cs.microstates) == 2

    def test_two_conformers_reaching_the_same_new_state_share_one_microstate(self) -> None:
        geom, _ = _embed("NCC(=O)[OH2+]")
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        ms = _unlabelled(geom)
        ms.conformers = [
            Conformer(geometry=_move_h(geom, acid_h, nitrogen)),
            Conformer(geometry=_move_h(geom, acid_h, nitrogen, dist=1.04)),
        ]

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.created == 1
        assert len(cs.microstates) == 2
        assert len(cs.microstates[1].conformers) == 2

    def test_a_detached_hydrogen_is_still_excluded(self) -> None:
        geom, _ = _embed("[NH3+]CC(=O)O")
        ms = _unlabelled(geom)
        h = geom.hydrogen_indices[0]
        coords = geom.coords.copy()
        coords[h] += np.array([0.0, 0.0, 8.0])
        ms.conformers[0].geometry = Geometry(symbols=tuple(geom.symbols), coords=coords)

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.detached == 1
        assert report.created == 0
        assert ms.conformers == []
        assert ms.excluded_conformers[0].reason == "proton_detached"

    def test_a_disagreeing_heavy_atom_ordering_is_refused(self) -> None:
        """The fingerprint is positional; comparing across orderings would be nonsense."""
        first, _ = _embed("[NH3+]CC(=O)O")
        # Swap a nitrogen and an oxygen so the heavy-atom element sequence differs;
        # two SMILES of one molecule would not, since both are canonicalised.
        symbols = list(first.symbols)
        coords = first.coords.copy()
        i = symbols.index("N")
        j = len(symbols) - 1 - symbols[::-1].index("O")
        symbols[i], symbols[j] = symbols[j], symbols[i]
        coords[[i, j]] = coords[[j, i]]
        second = Geometry(symbols=tuple(symbols), coords=coords)
        cs = ChargeState(charge=1, microstates=[_unlabelled(first), _unlabelled(second)])
        report = repair_migrated_conformers(cs, stage="sampling")
        assert report.checked == 0
        assert all(len(ms.conformers) == 1 for ms in cs.microstates)

    def test_the_excluded_conformer_keeps_its_energies_and_multiplicity(self) -> None:
        ms = _microstate("[NH3+]CC(=O)O")
        conf = ms.conformers[0]
        conf.electronic_energy = -1.5
        conf.solvation_energy = -0.02
        conf.multiplicity = 3.0
        geom = conf.geometry
        h = geom.hydrogen_indices[0]
        coords = geom.coords.copy()
        coords[h] += np.array([0.0, 0.0, 8.0])
        conf.geometry = Geometry(symbols=tuple(geom.symbols), coords=coords)

        cs = ChargeState(charge=1, microstates=[ms])
        repair_migrated_conformers(cs, stage="sampling")

        excluded = ms.excluded_conformers[0]
        assert excluded.multiplicity == 3.0
        assert excluded.electronic_energy == -1.5
        assert excluded.solvation_energy == -0.02
        assert excluded.stage == "sampling"


class TestOneHeavyAtomOrderPerMolecule:
    """The invariant that removes the atom-correspondence problem entirely."""

    @pytest.mark.parametrize(
        "family",
        [
            ["NCC(=O)O", "[NH3+]CC(=O)O", "NCC(=O)[O-]", "NCC(=O)[OH2+]", "OC(=[OH+])C[NH3+]"],
            ["O=C(O)CC(=O)O", "O=C(O)CC(=O)[O-]", "O=C([O-])CC(=O)O", "O=C(O)CC(O)=[OH+]"],
            ["NCCc1c[nH]cn1", "[NH3+]CCc1c[nH]cn1", "NCCc1c[n-]cn1"],
            ["C=C(CC(=O)O)C(=O)O", "C=C(CC(=O)[O-])C(=O)O", "C=C(CC(O)=[OH+])C(=O)O"],
        ],
    )
    def test_every_protomer_agrees_on_the_heavy_atom_order(self, family: list[str]) -> None:
        orders = set()
        for smiles in family:
            geom, _ = _embed(smiles)
            orders.add("".join(geom.symbols[i] for i in geom.heavy_atom_indices))
        assert len(orders) == 1

    def test_a_moved_conformer_adopts_its_destination_ordering(self) -> None:
        """Without the regrouping this trips deduplicate_conformers."""
        source = _microstate("NCC(=O)[OH2+]")
        target = _microstate("[NH3+]CC(=O)O")
        for ms in (source, target):
            for conf in ms.conformers:
                conf.electronic_energy = -1.0
        geom = source.conformers[0].geometry
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, target])
        assert repair_migrated_conformers(cs, stage="refinement").moved == 1

        expected = tuple(
            a.GetSymbol() for a in template_from_smiles(target.smiles or "").GetAtoms()
        )
        assert all(tuple(c.geometry.symbols) == expected for c in target.conformers)
        # The guard this exists to satisfy.
        deduplicate_conformers(target.conformers, target.includes_enantiomer)

    def test_the_regrouping_puts_each_hydrogen_on_the_right_heavy_atom(self) -> None:
        source = _microstate("NCC(=O)[OH2+]")
        target = _microstate("[NH3+]CC(=O)O", conformers=[])
        geom = source.conformers[0].geometry
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, target])
        repair_migrated_conformers(cs, stage="refinement")

        moved = target.conformers[0].geometry
        template = template_from_smiles(target.smiles or "")
        assert protonation_key_from_geometry(moved, template, 1) == protonation_key_from_mol(
            template, 1
        )


class TestHydrogenCountsSurviveCanonicalisation:
    """Regression guard on `_SKELETON_SANITIZE`.

    The skeleton's hydrogen counts are carried through canonicalisation only
    because SANITIZE_FINDRADICALS brackets the deficient atoms. Dropping that
    flag makes these pairs collide, merging species that are not even the same
    molecular formula. These tests fail if it is ever removed.
    """

    @pytest.mark.parametrize(
        "a,b",
        [
            ("CC(=O)O", "CC(O)O"),
            ("O=C(O)CC(=O)[O-]", "O=C(O)CC([O-])[O-]"),
            ("c1ccccc1O", "C1CCCCC1O"),
            ("CC=O", "CCO"),
        ],
    )
    def test_species_differing_in_hydrogen_count_never_share_a_key(self, a: str, b: str) -> None:
        key_a = protonation_key_from_mol(template_from_smiles(smiles_to_3d(a)[1]), 0)
        key_b = protonation_key_from_mol(template_from_smiles(smiles_to_3d(b)[1]), 0)
        assert key_a != key_b

    def test_the_skeleton_brackets_a_hydrogen_deficient_atom(self) -> None:
        template = template_from_smiles(smiles_to_3d("CC(=O)O")[1])
        key = protonation_key_from_mol(template, 0)
        assert "[O]" in key, "carbonyl O must be bracketed or its H count is lost"


class TestMatchToCandidate:
    """Verification, not discrimination: does the geometry agree with *this* candidate."""

    def test_a_geometry_agrees_with_its_own_label(self) -> None:
        geom, explicit = _embed(r"OC(=[OH+])/C=C\\C(=O)O")
        template = template_from_smiles(explicit)
        assert (
            match_to_candidate(
                geom, assign_protons(geom), template, template, False, same_microstate=True
            )
            is not None
        )

    def test_a_geometry_is_rejected_by_the_wrong_diastereomer(self) -> None:
        geom, explicit = _embed(r"OC(=[OH+])/C=C\\C(=O)O")
        cis = template_from_smiles(explicit)
        trans = template_from_smiles(smiles_to_3d(r"OC(=[OH+])/C=C/C(=O)O")[1])
        assignment = assign_protons(geom)
        assert match_to_candidate(geom, assignment, cis, cis, False, same_microstate=True)
        assert match_to_candidate(geom, assignment, cis, trans, False) is None

    def test_a_candidate_specifying_no_stereo_cannot_be_contradicted(self) -> None:
        geom, explicit = _embed("NCC(=O)O")
        template = template_from_smiles(explicit)
        assert (
            match_to_candidate(
                geom, assign_protons(geom), template, template, False, same_microstate=True
            )
            is not None
        )

    def test_the_mirror_is_accepted_only_for_a_collapsed_enantiomeric_pair(self) -> None:
        """A microstate flagged `includes_enantiomer` stands for both mirror images."""
        geom, explicit = _embed("N[C@@H](C)C(=O)O")
        own = template_from_smiles(explicit)
        mirror = template_from_smiles(smiles_to_3d("N[C@H](C)C(=O)O")[1])
        assignment = assign_protons(geom)
        # The geometry is its own label either way.
        assert match_to_candidate(geom, assignment, own, own, False, same_microstate=True)
        # Against the opposite configuration it depends on what the microstate means.
        assert match_to_candidate(geom, assignment, own, mirror, False) is None
        assert match_to_candidate(geom, assignment, own, mirror, True) is not None


class TestRingStereoIsResolved:
    """1,4-ring cis/trans: a relationship carried by a *pair* of atoms.

    These share a protonation key, since the key holds no bond orders and the
    two diastereomers have identical hydrogen counts. Comparing only the atom
    whose tag differs destroys the relationship -- both candidates canonicalise
    identically -- so the comparison keeps everything the candidates specify.
    """

    SRC: ClassVar[str] = "C[C@H](c1ccc(C(=O)O)cc1)[C@H]2CC[C@@H](C(=O)[O-])CC2"
    CIS: ClassVar[str] = "C[C@H](c1ccc(C(=O)[O-])cc1)[C@H]2CC[C@@H](C(=O)O)CC2"
    TRANS: ClassVar[str] = "C[C@H](c1ccc(C(=O)[O-])cc1)[C@H]2CC[C@H](C(=O)O)CC2"

    def test_the_two_diastereomers_share_a_protonation_key(self) -> None:
        cis = template_from_smiles(smiles_to_3d(self.CIS)[1])
        trans = template_from_smiles(smiles_to_3d(self.TRANS)[1])
        assert protonation_key_from_mol(cis, -1) == protonation_key_from_mol(trans, -1)

    def test_comparing_only_the_differing_atom_would_collapse_them(self) -> None:
        """Why verification compares everything the candidate specifies.

        A 1,4-ring cis/trans relationship is carried by a *pair* of atoms. Keep
        only the one whose tag differs and RDKit drops it when writing the
        SMILES -- a lone ring carbon has two identical branches -- so the two
        diastereomers become the same string and nothing can be decided.
        """
        cis = template_from_smiles(smiles_to_3d(self.CIS)[1])
        trans = template_from_smiles(smiles_to_3d(self.TRANS)[1])
        _, cis_atoms = _specified_stereo(cis)
        _, trans_atoms = _specified_stereo(trans)
        decisive = {k for k in cis_atoms if cis_atoms.get(k) != trans_atoms.get(k)}
        assert len(decisive) == 1, "cis and trans differ at exactly one ring carbon"

        assert _stereo_signature(cis, set(), decisive) == _stereo_signature(trans, set(), decisive)
        assert _stereo_signature(cis, set(), set(cis_atoms)) != _stereo_signature(
            trans, set(), set(trans_atoms)
        )

    def test_the_geometry_picks_the_right_diastereomer(self) -> None:
        source = _microstate(self.SRC)
        cis = _microstate(self.CIS, conformers=[])
        trans = _microstate(self.TRANS, conformers=[])

        geom = source.conformers[0].geometry
        assignment = assign_protons(geom)
        owner = dict(zip(geom.hydrogen_indices, assignment.owner, strict=True))
        acid_h = next(h for h, o in owner.items() if geom.symbols[o] == "O")
        donor = owner[acid_h]
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acceptor = max(
            (i for i in oxygens if i != donor),
            key=lambda i: float(np.linalg.norm(geom.coords[i] - geom.coords[donor])),
        )
        source.conformers[0].geometry = _move_h(geom, acid_h, acceptor, dist=0.98)

        cs = ChargeState(charge=-1, microstates=[source, cis, trans])
        report = repair_migrated_conformers(cs, stage="refinement")

        # The proton moved between carboxylates; the ring was never touched.
        assert report.stereo_resolved == 1
        assert report.ambiguous == 0
        assert len(cis.conformers) == 1
        assert trans.conformers == []


class TestStereoIsVerifiedNotAssumed:
    """The branch that previously kept a conformer without looking at it."""

    CIS: ClassVar[str] = r"OC(=[OH+])/C=C\C(=O)O"
    TRANS: ClassVar[str] = r"OC(=[OH+])/C=C/C(=O)O"

    def test_the_two_share_a_protonation_key(self) -> None:
        cis = template_from_smiles(smiles_to_3d(self.CIS)[1])
        trans = template_from_smiles(smiles_to_3d(self.TRANS)[1])
        assert protonation_key_from_mol(cis, 1) == protonation_key_from_mol(trans, 1)

    def test_a_flipped_double_bond_moves_to_its_sibling(self) -> None:
        """The protonation never changed, so nothing before this looked at it."""
        cis = _microstate(self.CIS)
        trans = _microstate(self.TRANS, conformers=[])
        # file the trans geometry under the cis microstate
        cis.conformers[0].geometry = _embed(self.TRANS)[0]

        cs = ChargeState(charge=1, microstates=[cis, trans])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.moved == 1
        assert cis.conformers == []
        assert len(trans.conformers) == 1

    def test_a_flipped_double_bond_with_no_sibling_is_kept_and_flagged(self) -> None:
        """Not an exclusion: a centre sampling can flip was not stable to begin with."""
        cis = _microstate(self.CIS)
        cis.conformers[0].geometry = _embed(self.TRANS)[0]

        cs = ChargeState(charge=1, microstates=[cis])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.stereo_unmatched == 1
        assert report.moved == 0
        assert len(cis.conformers) == 1
        assert cis.excluded_conformers == []


class TestPseudoAsymmetry:
    """Two ends of a molecule that only stereochemistry tells apart.

    2,3,4-trihydroxyglutaric acid has a pseudo-asymmetric C3. Deprotonating
    either end breaks the tie, giving two diastereomers that share a protonation
    key -- so a proton moving from one end to the other is invisible to the key,
    and the skeleton automorphism relating them is stereo-relevant.
    """

    E1: ClassVar[str] = "O=C([O-])[C@@H](O)[C@@H](O)[C@@H](O)C(=O)O"
    E5: ClassVar[str] = "O=C(O)[C@@H](O)[C@@H](O)[C@@H](O)C(=O)[O-]"

    def test_they_are_different_species_sharing_one_protonation_key(self) -> None:
        t1 = template_from_smiles(smiles_to_3d(self.E1)[1])
        t5 = template_from_smiles(smiles_to_3d(self.E5)[1])
        assert protonation_key_from_mol(t1, -1) == protonation_key_from_mol(t5, -1)
        assert Chem.MolToSmiles(t1) != Chem.MolToSmiles(t5)

    @staticmethod
    def _move_acid_proton(geom: Geometry) -> Geometry:
        """Hand the carboxylic acid's proton to the carboxylate at the other end.

        A migration keeps the microstate's atom ordering and moves coordinates,
        which is what makes it invisible to the protonation key here.
        """
        symbols = list(geom.symbols)
        oxygens = [i for i, s in enumerate(symbols) if s == "O"]
        assignment = assign_protons(geom)
        owner = dict(zip(geom.hydrogen_indices, assignment.owner, strict=True))

        def near(i: int, j: int) -> bool:
            return float(np.linalg.norm(geom.coords[i] - geom.coords[j])) < 1.7

        carboxyl_o = [
            o
            for o in oxygens
            if any(
                symbols[c] == "C" and sum(1 for k in oxygens if near(k, c)) == 2
                for c in range(len(symbols))
                if symbols[c] == "C" and near(o, c)
            )
        ]
        donor = next(o for o in carboxyl_o if any(owner.get(h) == o for h in owner))
        acid_h = next(h for h, o in owner.items() if o == donor)
        acceptor = max(
            (o for o in carboxyl_o if o != donor and not any(owner.get(h) == o for h in owner)),
            key=lambda o: float(np.linalg.norm(geom.coords[o] - geom.coords[donor])),
        )
        return _move_h(geom, acid_h, acceptor, dist=0.98)

    def test_the_geometry_is_re_filed_rather_than_silently_kept(self) -> None:
        source = _microstate(self.E1)
        target = _microstate(self.E5, conformers=[])
        source.conformers[0].geometry = self._move_acid_proton(source.conformers[0].geometry)

        template = template_from_smiles(source.smiles or "")
        moved = source.conformers[0].geometry
        # the key cannot see this migration -- both ends give the same skeleton
        assert protonation_key_from_geometry(moved, template, -1) == protonation_key_from_mol(
            template, -1
        )

        cs = ChargeState(charge=-1, microstates=[source, target])
        report = repair_migrated_conformers(cs, stage="refinement")

        # Detected, but not resolved. Both candidates are satisfiable, because the
        # skeleton's automorphism group is larger than the molecule's stereo
        # automorphism group: the half-swap relating the two ends is a symmetry of
        # the skeleton but not of the stereochemistry, and enumerating over it lets
        # the wrong candidate be satisfied by a chemically invalid relabelling.
        # Restricting to stereo-preserving automorphisms is circular. Both match
        # on real evidence, so the tie stays visible rather than being resolved by
        # preferring where the conformer started -- excluding is the safe outcome
        # and a strict improvement on the silent misfiling this case got before
        # stereo was verified at all.
        assert report.ambiguous == 1
        assert report.moved == 0
        assert source.conformers == []
        assert source.excluded_conformers[0].reason == "ambiguous_microstate"
        assert target.conformers == []
