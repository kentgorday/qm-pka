import random
from typing import ClassVar

import pytest
from rdkit import Chem

from qm_pka.rdkit_utils import (
    canonical_smiles,
    deduplicate_protomers,
    enumerate_tautomers,
    get_atom_mapped_smiles,
    get_formal_charge,
    protomer_key,
    smiles_to_3d,
    validate_input_smiles,
)
from qm_pka.stereo import enumerate_and_deduplicate


class TestSmilesTo3d:
    def test_water(self) -> None:
        geom, smi = smiles_to_3d("O")
        assert geom.n_atoms == 3
        assert geom.coords.shape == (3, 3)
        assert "H" in smi

    def test_methane(self) -> None:
        geom, smi = smiles_to_3d("C")
        assert geom.n_atoms == 5
        assert smi.count("H") == 4

    def test_geometry_matches_smiles(self) -> None:
        """Geometry should have same element counts as the SMILES."""
        from collections import Counter

        from rdkit import Chem

        geom, smi = smiles_to_3d("CCO")
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smi, params)
        assert mol is not None
        assert mol.GetNumAtoms() == geom.n_atoms
        smi_counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
        geom_counts = Counter(geom.symbols)
        assert smi_counts == geom_counts


class TestEnumerateTautomers:
    def test_acetone_enol(self) -> None:
        # Acetone has a keto-enol tautomer
        tautomers = enumerate_tautomers("CC(=O)C")
        assert len(tautomers) >= 1
        # Should contain the canonical form
        assert any("O" in t for t in tautomers)

    def test_max_tautomers_limit(self) -> None:
        tautomers = enumerate_tautomers("CC(=O)C", max_tautomers=1)
        assert len(tautomers) >= 1

    def test_no_duplicates(self) -> None:
        tautomers = enumerate_tautomers("c1cc[nH]c1")
        assert len(tautomers) == len(set(tautomers))


class TestCanonicalSmiles:
    def test_reorders(self) -> None:
        assert canonical_smiles("OCC") == canonical_smiles("CCO")

    def test_charged(self) -> None:
        result = canonical_smiles("[O-]C(=O)C")
        assert "-" in result


class TestGetAtomMappedSmiles:
    def test_has_map_numbers(self) -> None:
        mapped = get_atom_mapped_smiles("CCO")
        assert ":" in mapped


class TestGetFormalCharge:
    def test_neutral(self) -> None:
        assert get_formal_charge("CCO") == 0

    def test_anion(self) -> None:
        assert get_formal_charge("[O-]C(=O)C") == -1

    def test_cation(self) -> None:
        assert get_formal_charge("[NH3+]CC") == 1


# Two Lewis structures of one 4-substituted imidazolium: the cation can be
# written on either ring nitrogen.  The enumerator emits both.
HISTAMINE_A = "[H]c1c(C([H])([H])C([H])([H])N([H])[H])n([H])c([H])[n+]1[H]"
HISTAMINE_B = "[H]c1c(C([H])([H])C([H])([H])N([H])[H])[n+]([H])c([H])n1[H]"

# E and Z of the cyanoacetate enol.  Both reach a common stereo-destroyed
# cumulene under resonance, so they are the pair most at risk of a false merge.
CYANOENOL_E = r"[H]O/C([O-])=C(/[H])C#N"
CYANOENOL_Z = "[H]O/C([O-])=C(\\[H])C#N"


class TestProtomerKeyMerges:
    """Pairs that are one species written two ways."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (HISTAMINE_A, HISTAMINE_B),
            ("Cc1c[nH+]c[nH]1", "Cc1c[nH]c[nH+]1"),  # 4-methylimidazolium
            ("Cc1c[n-]cn1", "Cc1cnc[n-]1"),  # imidazolate
            ("Cc1nnn[n-]1", "Cc1[n-]nnn1"),  # tetrazolate
            ("CC(=[NH2+])NC", "CC(N)=[NH+]C"),  # amidinium
            ("c1ccc2c(c1)[nH]c[nH+]2", "c1ccc2c(c1)[nH+]c[nH]2"),  # benzimidazolium
            # Delocalised tetronic-acid enolate: the charge sits on either oxygen.
            (
                "[H]C([H])([H])C1=C([O-])C([H])([H])OC1=O",
                "[H]C([H])([H])C1=C([O-])OC([H])([H])C1=O",
            ),
        ],
    )
    def test_resonance_forms_share_a_key(self, a: str, b: str) -> None:
        assert protomer_key(a) == protomer_key(b)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("CS(=O)C", "C[S+]([O-])C"),  # sulfoxide
            ("C[N+](C)(C)[O-]", "CN(C)(C)=O"),  # amine oxide
            ("C[N+](=O)[O-]", "CN(=O)=O"),  # nitro, hypervalent form
            ("CP(C)(C)=O", "C[P+](C)(C)[O-]"),  # phosphine oxide
        ],
    )
    def test_hypervalent_and_charge_separated_forms_agree(self, a: str, b: str) -> None:
        """Obligatory charge-separated groups need no special handling.

        Erasing charge and bond order reaches these without any enumeration,
        which is why no SMARTS is needed to recognise nitro, N-oxides and the
        rest as fixed rather than mobile.
        """
        assert protomer_key(a) == protomer_key(b)


class TestProtomerKeySeparates:
    """Pairs that are genuinely different species and must never merge."""

    @pytest.mark.parametrize(
        ("a", "b", "why"),
        [
            ("Cc1c[nH]cn1", "Cc1cnc[nH]1", "4- vs 5-methylimidazole tautomers"),
            ("CCc1c[nH]cn1", "CCc1cnc[nH]1", "N1-H vs N3-H"),
            (
                "[NH3+]c1ccc(-c2ccncc2)cc1",
                "Nc1ccc(-c2cc[nH+]cc2)cc1",
                "aniline vs pyridine nitrogen protonated",
            ),
            ("CNC(N)=[NH2+]", "CN=C(N)[NH3+]", "guanidinium tautomers"),
            ("NCC(=O)O", "[NH3+]CC(=O)[O-]", "neutral vs zwitterion"),
            ("CC(=O)O", "CC(=O)[O-]", "different net charge"),
            ("C1=CC=CCC1", "C1=CCC=CC1", "1,3- vs 1,4-cyclohexadiene"),
        ],
    )
    def test_distinct_species_get_distinct_keys(self, a: str, b: str, why: str) -> None:
        assert protomer_key(a) != protomer_key(b), why

    def test_e_and_z_do_not_merge(self) -> None:
        """Stereochemistry is preserved, not erased alongside charge and bond order."""
        assert protomer_key(CYANOENOL_E) != protomer_key(CYANOENOL_Z)

    def test_enantiomers_do_not_merge(self) -> None:
        assert protomer_key("C[C@@H](N)C(=O)O") != protomer_key("C[C@H](N)C(=O)O")


class TestProtomerKeyIsAFunctionOfTheMolecule:
    """The key must not depend on how the SMILES happened to be written.

    RDKit's canonical ranking uses formal charge, so two resonance forms of one
    molecule canonicalise to *different* atom orders.  Ranking on the charge-
    and bond-order-free skeleton is what makes the labelling stable; without it
    the key varies with atom ordering.
    """

    ALL: ClassVar[list[str]] = [
        HISTAMINE_A,
        HISTAMINE_B,
        "Cc1c[nH+]c[nH]1",
        "CCc1cccc([O-])c1",
        "C[C@@H](N)C(=O)O",
        CYANOENOL_E,
    ]

    @pytest.mark.parametrize("smi", ALL)
    def test_invariant_to_atom_renumbering(self, smi: str) -> None:
        mol = Chem.MolFromSmiles(smi)
        expected = protomer_key(smi)
        rng = random.Random(0)
        for _ in range(25):
            perm = list(range(mol.GetNumAtoms()))
            rng.shuffle(perm)
            shuffled = Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm))
            assert protomer_key(shuffled) == expected

    @pytest.mark.parametrize("smi", ALL)
    def test_invariant_to_explicit_hydrogens(self, smi: str) -> None:
        implicit = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        assert protomer_key(smi) == protomer_key(implicit)

    def test_rejects_unparseable(self) -> None:
        with pytest.raises(ValueError, match="could not parse"):
            protomer_key("not_a_smiles((")


class TestDeduplicateProtomers:
    def test_collapses_the_histamine_pair(self) -> None:
        assert len(deduplicate_protomers([HISTAMINE_A, HISTAMINE_B])) == 1

    def test_representative_does_not_depend_on_input_order(self) -> None:
        assert deduplicate_protomers([HISTAMINE_A, HISTAMINE_B]) == deduplicate_protomers(
            [HISTAMINE_B, HISTAMINE_A]
        )

    def test_keeps_distinct_species(self) -> None:
        assert len(deduplicate_protomers(["Cc1c[nH]cn1", "Cc1cnc[nH]1"])) == 2

    def test_empty(self) -> None:
        assert deduplicate_protomers([]) == []


class TestValidateInputSmiles:
    def test_rejects_radicals(self) -> None:
        with pytest.raises(ValueError, match="Open-shell"):
            validate_input_smiles("[O][O]")

    def test_rejects_multiple_fragments(self) -> None:
        with pytest.raises(ValueError, match="Multi-component"):
            validate_input_smiles("CC(=O)[O-].[Na+]")

    def test_rejects_unparseable(self) -> None:
        with pytest.raises(ValueError, match="could not parse"):
            validate_input_smiles("not_a_smiles((")

    @pytest.mark.parametrize(
        "smi", ["NCCc1c[nH]cn1", "CC(=O)[O-]", "C[N+](=O)[O-]", "C[C@@H](N)C(=O)O"]
    )
    def test_accepts_supported_inputs(self, smi: str) -> None:
        validate_input_smiles(smi)  # must not raise


class TestRepresentativeChoiceIsExplicit:
    """The survivor seeds stereoisomer enumeration and the ETKDG geometry.

    Resonance forms do not all expose the same stereochemistry, so the choice
    must be stated rather than left to where a bracket sorts in ASCII.
    """

    def test_prefers_the_form_that_exposes_stereochemistry(self) -> None:
        carbanion = "CC(=O)[CH-]C"  # no stereogenic bond
        enolate = "CC([O-])=CC"  # stereogenic C=C
        assert protomer_key(carbanion) == protomer_key(enolate)
        kept = deduplicate_protomers([carbanion, enolate])
        assert kept == [canonical_smiles(enolate)]

    def test_choice_survives_input_order(self) -> None:
        pair = ["CC(=O)[CH-]C", "CC([O-])=CC"]
        assert deduplicate_protomers(pair) == deduplicate_protomers(pair[::-1])

    def test_no_stereoisomer_is_lost_by_the_choice(self) -> None:
        """The kept member must enumerate at least as many stereoisomers as any
        member it displaced."""
        groups = [
            ["CC(=O)[CH-]C", "CC([O-])=CC"],
            ["CC(=O)C=C([O-])C", "CC([O-])=CC(=O)C"],
            [HISTAMINE_A, HISTAMINE_B],
        ]
        for group in groups:
            kept = deduplicate_protomers(group)[0]
            n_kept = len(enumerate_and_deduplicate(kept))
            for member in group:
                assert n_kept >= len(enumerate_and_deduplicate(member))


class TestProtomerKeyIgnoresAnnotations:
    def test_atom_map_numbers_do_not_split_a_species(self) -> None:
        smi = "NCCc1c[nH]cn1"
        assert protomer_key(smi) == protomer_key(get_atom_mapped_smiles(smi))
