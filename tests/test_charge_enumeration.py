from qm_pka.charge_enumeration import (
    deprotonate_all_sites,
    enumerate_charge_state,
    protonate_all_sites,
)
from qm_pka.rdkit_utils import get_formal_charge


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
