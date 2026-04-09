from qm_pka.rdkit_utils import (
    canonical_smiles,
    enumerate_tautomers,
    get_atom_mapped_smiles,
    get_formal_charge,
    smiles_to_3d,
)


class TestSmilesTo3d:
    def test_water(self) -> None:
        geom = smiles_to_3d("O")
        assert geom.symbols == ("O", "H", "H")
        assert geom.n_atoms == 3
        assert geom.coords.shape == (3, 3)

    def test_methane(self) -> None:
        geom = smiles_to_3d("C")
        assert geom.n_atoms == 5
        assert geom.symbols[0] == "C"


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
