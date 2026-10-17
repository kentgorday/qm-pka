"""Tests for perceiving MNSol solutes from their geometries, on hand-built inputs.

The database is distributed under its own license and is not in the repository,
so nothing here reads it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from tautomator import mnsol

_HYDROXIDE = (
    "i000 H1O1 hydroxide m062x_mg3s_geom\n\n{charge} {multiplicity}\n"
    "  8    0.0000000000    0.0000000000    0.0000000000\n"
    "  1    0.0000000000    0.0000000000    0.9640000000\n"
)


def _canonical(smiles: str) -> str:
    return str(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))


def _recombined(smiles: str) -> str:
    return str(Chem.MolToSmiles(mnsol.recombine_charges(Chem.MolFromSmiles(smiles))))


def _database(tmp_path: Path, charge: int = -1, multiplicity: int = 1) -> Path:
    """A database directory holding one geometry, hydroxide's, under handle i000."""
    (tmp_path / mnsol.GEOMETRIES).mkdir()
    xyz = _HYDROXIDE.format(charge=charge, multiplicity=multiplicity)
    (tmp_path / mnsol.GEOMETRIES / "i000.xyz").write_text(xyz)
    return tmp_path


class TestReadXyz:
    def test_header_charge_and_atomic_numbers(self, tmp_path: Path) -> None:
        database = _database(tmp_path)
        charge, multiplicity, numbers, coords = mnsol.read_xyz(
            database / mnsol.GEOMETRIES / "i000.xyz"
        )
        assert (charge, multiplicity, numbers) == (-1, 1, [8, 1])
        assert coords == pytest.approx(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.964]]))


class TestPerceive:
    def test_hydroxide(self, tmp_path: Path) -> None:
        mol = mnsol.perceive(_database(tmp_path), "i000", "H1O1")
        assert mnsol.to_smiles(mol) == "[OH-]"

    def test_a_formula_that_does_not_match_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(mnsol.PerceptionError, match="formula or charge"):
            mnsol.perceive(_database(tmp_path), "i000", "H2O1")

    def test_open_shell_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(mnsol.PerceptionError, match="open shell"):
            mnsol.perceive(_database(tmp_path, charge=0, multiplicity=2), "i000", "H1O1")


class TestMolFromGeometry:
    def test_monatomic_ion_carries_its_charge(self) -> None:
        mol = mnsol.mol_from_geometry([9], np.zeros((1, 3)), -1)
        assert mnsol.to_smiles(mol) == "[F-]"

    def test_formaldehyde_gets_its_double_bond(self) -> None:
        coords = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.205], [0.0, 0.943, -0.587], [0.0, -0.943, -0.587]]
        )
        mol = mnsol.mol_from_geometry([6, 8, 1, 1], coords, 0)
        assert mnsol.to_smiles(mol) == "C=O"


class TestFormulaCounts:
    @pytest.mark.parametrize(
        "formula,expected",
        [
            ("H5C6O1BR1", {"H": 5, "C": 6, "O": 1, "Br": 1}),
            ("C1CL4", {"C": 1, "Cl": 4}),
            ("H18C8", {"H": 18, "C": 8}),
            ("F1", {"F": 1}),
        ],
    )
    def test_database_formulas(self, formula: str, expected: dict[str, int]) -> None:
        assert mnsol.formula_counts(formula) == expected


class TestRecombineCharges:
    @pytest.mark.parametrize(
        "perceived,expected",
        [
            ("COP([S-])(OC)=[S+]C", "COP(=S)(OC)SC"),  # thiophosphate ester
            ("CCN(CC)C([O-])=[S+]CCC", "CCCSC(=O)N(CC)CC"),  # thiocarbamate
            ("[S+]#C[S-]", "S=C=S"),  # a triple bond on the path
            ("COC([O-])=C1[S+]=CC=C1", "COC(=O)c1cccs1"),  # along a conjugated path
            ("CCOP([S-])(OCC)=[S+]C[S+]=P([S-])(OCC)OCC", "CCOP(=S)(OCC)SCSP(=S)(OCC)OCC"),
        ],
    )
    def test_removes_avoidable_charge_separation(self, perceived: str, expected: str) -> None:
        assert _recombined(perceived) == _canonical(expected)

    @pytest.mark.parametrize(
        "smiles",
        ["C[N+](=O)[O-]", "C[S+](C)[O-]", "C=[N+]([O-])[O-]", "CC(=O)[O-]", "C[NH3+]", "c1ccccc1"],
    )
    def test_leaves_genuine_charges_alone(self, smiles: str) -> None:
        assert _recombined(smiles) == _canonical(smiles)

    def test_restores_the_rotatable_bond(self) -> None:
        """The point of the repair: P-S-C is a single bond, and so rotatable."""
        perceived = Chem.MolFromSmiles("CCOP([S-])(OCC)=[S+]CC")
        fixed = mnsol.recombine_charges(perceived)
        before = rdMolDescriptors.CalcNumRotatableBonds(perceived, strict=True)
        assert rdMolDescriptors.CalcNumRotatableBonds(fixed, strict=True) == before + 1


class TestToSmiles:
    @pytest.mark.parametrize(
        "perceived,expected",
        [
            ("O=C1C=CC=CC1=[N+]([O-])[O-]", "O=[N+]([O-])c1ccccc1[O-]"),  # 2-nitrophenolate
            ("O=C1C=CC(=C[O-])C=C1", "O=Cc1ccc([O-])cc1"),  # 4-formylphenolate
            ("N=C1C=CC(=[N+]([O-])[O-])C=C1", "[NH-]c1ccc([N+](=O)[O-])cc1"),
        ],
    )
    def test_prefers_the_aromatic_resonance_structure(self, perceived: str, expected: str) -> None:
        mol = Chem.AddHs(Chem.MolFromSmiles(perceived))
        assert mnsol.to_smiles(mol) == _canonical(expected)

    def test_keeps_stereochemistry_when_nothing_needs_fixing(self) -> None:
        for smiles in ("CC[C@H](C)Cl", "Cl/C=C\\Cl", "C[S@+]([O-])c1ccccc1"):
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            assert mnsol.to_smiles(mol) == _canonical(smiles)

    def test_drops_stereo_defined_only_by_a_hydrogen(self) -> None:
        mol = Chem.MolFromSmiles("[H]/[O+]=C(/C)N")
        assert "[H]" not in mnsol.to_smiles(mol)


class TestDescribe:
    def test_polar_rotors_are_counted_apart_from_rotatable_bonds(self) -> None:
        diol = mnsol.describe(Chem.AddHs(Chem.MolFromSmiles("OCCO")))
        assert (diol.n_rotatable, diol.n_polar_rotors) == (1, 2)
        for planar in ("CC(N)=O", "Nc1ccccc1", "C[NH3+]"):
            assert mnsol.describe(Chem.AddHs(Chem.MolFromSmiles(planar))).n_polar_rotors == 0

    def test_sizes_and_fragments(self) -> None:
        water = mnsol.describe(Chem.AddHs(Chem.MolFromSmiles("O")))
        assert (water.n_atoms, water.n_heavy, water.n_fragments) == (3, 1, 1)
        cluster = mnsol.describe(Chem.AddHs(Chem.MolFromSmiles("C[O-].O")))
        assert cluster.n_fragments == 2
