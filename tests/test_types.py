import numpy as np
import pytest

from qm_pka.types import (
    ChargeState,
    Conformer,
    Ensemble,
    Geometry,
    Microstate,
)


class TestGeometry:
    def test_basic_construction(self) -> None:
        geom = Geometry(
            symbols=("C", "H", "H", "H", "H"),
            coords=np.zeros((5, 3)),
        )
        assert geom.n_atoms == 5
        assert geom.heavy_atom_indices == [0]
        assert geom.hydrogen_indices == [1, 2, 3, 4]

    def test_symbols_coords_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="symbols length"):
            Geometry(symbols=("C", "H"), coords=np.zeros((3, 3)))

    def test_coords_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            Geometry(symbols=("C",), coords=np.zeros((1, 4)))

    def test_symbols_is_tuple(self) -> None:
        geom = Geometry(symbols=("O", "H", "H"), coords=np.zeros((3, 3)))
        assert isinstance(geom.symbols, tuple)

    def test_symbols_normalized_to_canonical_case(self) -> None:
        # Psi4's save_xyz_file emits uppercase symbols ("CL"); RDKit's
        # PeriodicTable lookup in n_electrons/multiplicity is case-sensitive.
        geom = Geometry(symbols=("c", "CL", "BR", "h"), coords=np.zeros((4, 3)))
        assert geom.symbols == ("C", "Cl", "Br", "H")
        # n_electrons/multiplicity must not raise.
        assert geom.n_electrons(0) == 6 + 17 + 35 + 1
        assert geom.multiplicity(0) == 2  # 59 electrons -> doublet


class TestConformer:
    def test_construction(self) -> None:
        geom = Geometry(symbols=("H", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]))
        conf = Conformer(geometry=geom, electronic_energy=-1.5)
        assert conf.electronic_energy == -1.5
        assert conf.solvation_energy is None
        assert conf.rrho_correction is None
        assert conf.weight is None

    def test_free_energy_electronic_only(self) -> None:
        geom = Geometry(symbols=("H", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]))
        conf = Conformer(geometry=geom, electronic_energy=-1.5)
        assert conf.free_energy == -1.5

    def test_free_energy_all_components(self) -> None:
        geom = Geometry(symbols=("H", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]))
        conf = Conformer(
            geometry=geom,
            electronic_energy=-1.5,
            solvation_energy=-0.01,
            rrho_correction=0.02,
        )
        assert conf.free_energy == pytest.approx(-1.49)

    def test_free_energy_no_components_raises(self) -> None:
        geom = Geometry(symbols=("H", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]))
        conf = Conformer(geometry=geom)
        with pytest.raises(ValueError, match="no energy components"):
            _ = conf.free_energy


class TestMicrostate:
    def test_construction(self) -> None:
        geom = Geometry(symbols=("H",), coords=np.zeros((1, 3)))
        conf = Conformer(geometry=geom, electronic_energy=-1.0)
        ms = Microstate(tautomer_id="[H][H]", conformers=[conf], smiles="[H][H]")
        assert ms.smiles == "[H][H]"
        assert len(ms.conformers) == 1


class TestChargeState:
    def test_construction(self) -> None:
        cs = ChargeState(charge=-1, microstates=[])
        assert cs.charge == -1


class TestEnsemble:
    def test_construction(self) -> None:
        ens = Ensemble(input_smiles="CC(=O)O")
        assert ens.charge_states == {}
        assert ens.settings == {}
