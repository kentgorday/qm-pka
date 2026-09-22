from typing import ClassVar

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_pka.charge_enumeration import (
    _DEPROTONATION_SMARTS,
    _PROTONATION_SMARTS,
    charge_separated_variants,
    deprotonate_all_sites,
    enumerate_charge_state,
    protonate_all_sites,
)
from qm_pka.rdkit_utils import canonical_smiles, deduplicate_protomers, get_formal_charge


class TestDeprotonateAllSites:
    def test_acetic_acid(self) -> None:
        # CH3COOH has one acidic OH
        products = deprotonate_all_sites("CC(=O)O")
        assert len(products) >= 1
        for p in products:
            assert get_formal_charge(p) == -1

    def test_glycine_neutral(self) -> None:
        # Glycine (NH2CH2COOH) has acidic OH and NH2
        products = deprotonate_all_sites("NCC(=O)O")
        assert len(products) >= 1
        for p in products:
            assert get_formal_charge(p) == -1

    def test_all_products_correct_charge(self) -> None:
        products = deprotonate_all_sites("O")  # water
        for p in products:
            assert get_formal_charge(p) == -1


class TestProtonateAllSites:
    def test_amine(self) -> None:
        products = protonate_all_sites("CCN")
        assert len(products) >= 1
        for p in products:
            assert get_formal_charge(p) == 1

    def test_pyridine(self) -> None:
        products = protonate_all_sites("c1ccncc1")
        assert len(products) >= 1
        for p in products:
            assert get_formal_charge(p) == 1


class TestEnumerateChargeState:
    def test_same_charge_returns_canonical(self) -> None:
        result = enumerate_charge_state("CC(=O)O", target_charge=0)
        assert len(result) == 1

    def test_single_deprotonation(self) -> None:
        result = enumerate_charge_state("CC(=O)O", target_charge=-1)
        assert len(result) >= 1
        for smi in result:
            assert get_formal_charge(smi) == -1

    def test_double_deprotonation(self) -> None:
        # Sulfuric acid can lose two protons
        result = enumerate_charge_state("OS(=O)(=O)O", target_charge=-2)
        assert len(result) >= 1
        for smi in result:
            assert get_formal_charge(smi) == -2

    def test_protonation(self) -> None:
        result = enumerate_charge_state("CCN", target_charge=1)
        assert len(result) >= 1
        for smi in result:
            assert get_formal_charge(smi) == 1

    def test_glycine_all_charge_states(self) -> None:
        # Glycine should have species at -1, 0, +1
        for q in [-1, 0, 1]:
            result = enumerate_charge_state("NCC(=O)O", target_charge=q)
            assert len(result) >= 1, f"No species found at charge {q}"
            for smi in result:
                assert get_formal_charge(smi) == q

    def test_unreachable_target_returns_empty(self) -> None:
        # 4-chlorothiophenol has one ionizable site (-SH); asking for q=-2
        # is unreachable. Must return [], not a fallback species at the
        # wrong charge — sampling would otherwise feed a neutral SMILES
        # to DFT as a dianion.
        assert enumerate_charge_state("Sc1cccc(Cl)c1", target_charge=-2) == []
        # Same for the over-protonation direction on a mono-base.
        assert enumerate_charge_state("CCN", target_charge=2) == []
        # And from a tautomer with no ionizable sites at all (thione form
        # of chlorothiophenol — no H on S, no path to anion).
        assert enumerate_charge_state("S=C1C=C(Cl)C=CC1", target_charge=-1) == []


class TestNeutralizingRulesFire:
    """Every rule pins H count and charge on both sides.

    A product template inherits any property it does not state, so a product
    written ``[NH2:1]`` keeps the reactant's +1 and never reaches the target
    charge. That silently disabled every rule that neutralizes an ion.
    """

    @pytest.mark.parametrize(
        ("smiles", "target", "expected"),
        [
            ("C[NH3+]", 0, "CN"),
            ("c1cc[nH+]cc1", 0, "c1ccncc1"),
            ("CC(=O)[O-]", 0, "CC(=O)O"),
            ("[O-]c1ccccc1", 0, "Oc1ccccc1"),
            ("C[S-]", 0, "CS"),
        ],
    )
    def test_charged_reference_reaches_neutral(
        self, smiles: str, target: int, expected: str
    ) -> None:
        assert canonical_smiles(expected) in enumerate_charge_state(smiles, target)

    def test_aromatic_nh_deprotonates(self) -> None:
        """[n;H1;+0]>>[n;H0;-1] was dead: the product inherited H=1, giving an
        invalid [nH-]. Histamine's imidazolate was missing from every run."""
        out = enumerate_charge_state("NCCc1c[nH]cn1", -1)
        assert canonical_smiles("NCCc1c[n-]cn1") in out

    @pytest.mark.parametrize(
        ("rules", "kind"),
        [(_DEPROTONATION_SMARTS, "deprot"), (_PROTONATION_SMARTS, "prot")],
    )
    def test_no_rule_is_dead(self, rules: list[str], kind: str) -> None:
        """Every rule that matches something must be able to yield a product."""
        substrates = [
            "C[NH3+]",
            "CN",
            "C[NH-]",
            "CC(=O)O",
            "CC(=O)[O-]",
            "C[OH2+]",
            "CO",
            "CS",
            "C[S-]",
            "c1cc[nH]c1",
            "c1cc[nH+]cc1",
            "c1ccncc1",
            "CNC",
            "[NH4+]",
            "O",
            "CP",
            "CC(=O)[NH-]",
            "C[NH2+]C",
            "CO[OH2+]",
        ]
        for src in rules:
            rxn = AllChem.ReactionFromSmarts(src)
            assert rxn is not None, f"{src} does not compile"
            matched = survived = 0
            for smi in substrates:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                target = Chem.GetFormalCharge(mol) + (-1 if kind == "deprot" else 1)
                for tup in rxn.RunReactants((mol,)):
                    for prod in tup:
                        matched += 1
                        try:
                            Chem.SanitizeMol(prod)
                        except Exception:
                            continue
                        if Chem.GetFormalCharge(prod) == target:
                            survived += 1
            if matched:
                assert survived, f"{src} matches {matched} substrate(s) but never yields a product"


class TestRuleTableIsClosedUnderReversal:
    """Every deprotonation must have a protonation that undoes it, and vice versa.

    ``test_no_rule_is_dead`` catches a rule that never fires; it cannot catch a
    rule with no inverse. ``[P;H1;+0]>>[P;H0;-1]`` shipped without one, making
    ``CPC -> C[P-]C -> []`` a one-way trip.
    """

    SUBSTRATES: ClassVar[list[str]] = [
        "CPC",
        "CPCC",
        "CP",
        "P",
        "CN",
        "CNC",
        "CNCC",
        "C[NH3+]",
        "N",
        "CC(=O)N",
        "CC(=O)[NH-]",
        "CO",
        "CC(=O)O",
        "CC(=O)[O-]",
        "C[OH2+]",
        "O",
        "CS",
        "C[S-]",
        "CSC",
        "c1cc[nH]c1",
        "c1cc[nH+]cc1",
        "c1ccncc1",
        "NCCc1c[nH]cn1",
        "COP(=O)(O)O",
        "Oc1ccccc1",
        "[O-]c1ccccc1",
        "CC(=[NH2+])N",
        "C[P-]C",
    ]

    @pytest.mark.parametrize("smiles", SUBSTRATES)
    def test_deprotonation_round_trips(self, smiles: str) -> None:
        for product in deprotonate_all_sites(smiles):
            assert canonical_smiles(smiles) in protonate_all_sites(product), (
                f"{smiles} -> {product} cannot be protonated back"
            )

    @pytest.mark.parametrize("smiles", SUBSTRATES)
    def test_protonation_round_trips(self, smiles: str) -> None:
        for product in protonate_all_sites(smiles):
            assert canonical_smiles(smiles) in deprotonate_all_sites(product), (
                f"{smiles} -> {product} cannot be deprotonated back"
            )

    def test_the_phosphine_case_specifically(self) -> None:
        assert deprotonate_all_sites("CPC") == [canonical_smiles("C[P-]C")]
        assert canonical_smiles("CPC") in protonate_all_sites("C[P-]C")
        assert enumerate_charge_state("C[P-]C", 0) == [canonical_smiles("CPC")]


class TestChargeSeparatedVariants:
    """The reference charge needs protomers whose charge is internally separated.

    Neither source of microstates produces one: RDKit's tautomer enumerator
    moves protons but never separates charge, and `enumerate_charge_state` walks
    *between* charge states. So glycine's q=0 set held only the neutral form,
    although the zwitterion dominates in water.
    """

    def test_the_glycine_zwitterion_is_produced(self) -> None:
        got = charge_separated_variants("NCC(=O)O")
        assert canonical_smiles("[NH3+]CC(=O)[O-]") in got

    def test_the_input_itself_is_not_returned(self) -> None:
        """It is already a microstate; returning it would double-count."""
        for smi in ("NCC(=O)O", "CC(=O)O", "O=C(O)CC(=O)O"):
            assert canonical_smiles(smi) not in charge_separated_variants(smi)

    def test_every_product_keeps_the_input_charge(self) -> None:
        for smi in ("NCC(=O)O", "CN(C)CC(=O)O", "O=C(O)[C@@H]1CCCN1", "CCN"):
            q = get_formal_charge(smi)
            for out in charge_separated_variants(smi):
                assert get_formal_charge(out) == q, f"{smi} -> {out}"

    def test_a_molecule_with_no_basic_site_gains_only_resonance_forms(self) -> None:
        """Acetic acid has nowhere for the proton to go but its own carbonyl.

        `CC([O-])=[OH+]` is acetic acid drawn charge-separated, so
        `deduplicate_protomers` collapses it rather than scoring it twice.
        """
        got = charge_separated_variants("CC(=O)O")
        merged = deduplicate_protomers(sorted(set(got) | {canonical_smiles("CC(=O)O")}))
        assert merged == [canonical_smiles("CC(=O)O")]

    def test_the_charge_state_walk_is_left_alone(self) -> None:
        """`enumerate_charge_state` keeps its BFS contract: already there, nothing added.

        The gap is closed in `run_approach1` at the reference charge instead, so
        callers walking between charge states see no change.
        """
        assert enumerate_charge_state("CC(=O)O", target_charge=0) == [canonical_smiles("CC(=O)O")]
        assert enumerate_charge_state("NCC(=O)O", target_charge=0) == [
            canonical_smiles("NCC(=O)O")
        ]
