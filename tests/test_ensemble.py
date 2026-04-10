from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from qm_pka.ensemble import (
    HARTREE_TO_KCAL,
    assign_weights,
    boltzmann_weights,
    charge_state_free_energy,
    ensemble_free_energy,
    ensemble_to_sdf,
    load_ensemble,
    serialize_ensemble,
)
from qm_pka.types import ChargeState, Conformer, Ensemble, Geometry, Microstate


def _make_conformer(energy: float) -> Conformer:
    geom = Geometry(symbols=("H",), coords=np.zeros((1, 3)))
    return Conformer(geometry=geom, energy=energy)


class TestBoltzmannWeights:
    def test_single_energy(self) -> None:
        w = boltzmann_weights([-1.0])
        assert w == pytest.approx([1.0])

    def test_degenerate_energies(self) -> None:
        w = boltzmann_weights([-1.0, -1.0])
        assert w == pytest.approx([0.5, 0.5])

    def test_lower_energy_higher_weight(self) -> None:
        w = boltzmann_weights([-1.1, -1.0])
        assert w[0] > w[1]

    def test_sums_to_one(self) -> None:
        w = boltzmann_weights([-1.5, -1.4, -1.3, -1.2])
        assert sum(w) == pytest.approx(1.0)


class TestEnsembleFreeEnergy:
    def test_single_conformer(self) -> None:
        g = ensemble_free_energy([-1.0])
        assert g == pytest.approx(-1.0)

    def test_degenerate_lowers_free_energy(self) -> None:
        g_single = ensemble_free_energy([-1.0])
        g_double = ensemble_free_energy([-1.0, -1.0])
        # Two degenerate states -> lower free energy (by kT*ln2)
        assert g_double < g_single

    def test_high_energy_conformer_negligible(self) -> None:
        g_one = ensemble_free_energy([-1.0])
        # Adding a conformer 100 kcal/mol higher should barely change G
        g_two = ensemble_free_energy([-1.0, -1.0 + 100.0 / HARTREE_TO_KCAL])
        assert abs(g_one - g_two) < 1e-10


class TestChargeStateFreeEnergy:
    def test_basic(self) -> None:
        cs = ChargeState(
            charge=0,
            microstates=[
                Microstate(tautomer_id="a", conformers=[_make_conformer(-1.0)]),
                Microstate(tautomer_id="b", conformers=[_make_conformer(-1.01)]),
            ],
        )
        g = charge_state_free_energy(cs)
        assert g < -1.0  # Lower than the lowest individual energy

    def test_empty_raises(self) -> None:
        cs = ChargeState(charge=0, microstates=[])
        with pytest.raises(ValueError):
            charge_state_free_energy(cs)


class TestAssignWeights:
    def test_weights_across_microstates(self) -> None:
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(tautomer_id="a", conformers=[_make_conformer(-1.0)]),
                        Microstate(tautomer_id="b", conformers=[_make_conformer(-1.0)]),
                    ],
                ),
            },
        )
        assign_weights(ens)
        # Two degenerate conformers across microstates should each get 0.5
        w0 = ens.charge_states[0].microstates[0].conformers[0].weight
        w1 = ens.charge_states[0].microstates[1].conformers[0].weight
        assert w0 == pytest.approx(0.5)
        assert w1 == pytest.approx(0.5)


class TestSerialization:
    def test_round_trip(self, tmp_path: Path) -> None:
        geom = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [-0.24, 0.93, 0.0],
                ]
            ),
        )
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(
                            tautomer_id="abc123",
                            conformers=[Conformer(geometry=geom, energy=-76.43, weight=1.0)],
                            smiles="O",
                        ),
                    ],
                ),
            },
            settings={"solvent": "water"},
        )
        json_path = serialize_ensemble(ens, tmp_path / "output")
        ens2 = load_ensemble(json_path)
        assert ens2.input_smiles == "O"
        assert 0 in ens2.charge_states
        assert len(ens2.charge_states[0].microstates) == 1
        conf = ens2.charge_states[0].microstates[0].conformers[0]
        assert conf.energy == pytest.approx(-76.43)
        np.testing.assert_allclose(conf.geometry.coords, geom.coords, atol=1e-8)


def _water_geom() -> Geometry:
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


class TestEnsembleToSdf:
    def test_with_smiles(self, tmp_path: Path) -> None:
        """Approach 1: explicit-H SMILES provides bond orders."""
        geom = _water_geom()
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(
                            tautomer_id="O",
                            conformers=[Conformer(geometry=geom, energy=-76.4, weight=1.0)],
                            smiles="O([H])[H]",
                        ),
                    ],
                ),
            },
        )
        sdf_path = ensemble_to_sdf(ens, tmp_path / "test.sdf")
        assert sdf_path.exists()
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = list(suppl)
        assert len(mols) == 1
        mol = mols[0]
        assert mol is not None
        assert mol.GetNumAtoms() == 3
        assert mol.GetNumBonds() == 2
        assert int(mol.GetProp("charge")) == 0
        assert mol.GetProp("tautomer_id") == "O"
        assert float(mol.GetDoubleProp("energy_hartree")) == pytest.approx(-76.4)

    def test_without_smiles(self, tmp_path: Path) -> None:
        """Approach 2: no SMILES, bonds from rdDetermineBonds."""
        geom = _water_geom()
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(
                            tautomer_id="fp_abc",
                            conformers=[Conformer(geometry=geom, energy=-76.4, weight=1.0)],
                            smiles=None,
                        ),
                    ],
                ),
            },
        )
        sdf_path = ensemble_to_sdf(ens, tmp_path / "test.sdf")
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = list(suppl)
        assert len(mols) == 1
        mol = mols[0]
        assert mol is not None
        assert mol.GetNumAtoms() == 3
        assert mol.GetNumBonds() == 2

    def test_multiple_charge_states(self, tmp_path: Path) -> None:
        geom = _water_geom()
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(
                            tautomer_id="O",
                            conformers=[
                                Conformer(geometry=geom, energy=-76.4, weight=0.6),
                                Conformer(geometry=geom, energy=-76.3, weight=0.4),
                            ],
                            smiles="O([H])[H]",
                        ),
                    ],
                ),
                -1: ChargeState(
                    charge=-1,
                    microstates=[
                        Microstate(
                            tautomer_id="[OH-]",
                            conformers=[
                                Conformer(
                                    geometry=Geometry(
                                        symbols=("O", "H"),
                                        coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]]),
                                    ),
                                    energy=-75.8,
                                    weight=1.0,
                                ),
                            ],
                            smiles="[O-][H]",
                        ),
                    ],
                ),
            },
        )
        sdf_path = ensemble_to_sdf(ens, tmp_path / "test.sdf")
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = list(suppl)
        assert len(mols) == 3  # 2 neutral conformers + 1 anion
