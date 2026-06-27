import logging
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from qm_pka.config import load_config


@pytest.fixture()
def tmp_toml(tmp_path: Path) -> Callable[[str], Path]:
    """Helper to write a TOML string and return its path."""

    def _write(content: str) -> Path:
        p = tmp_path / "config.toml"
        p.write_text(textwrap.dedent(content))
        return p

    return _write


class TestMinimalConfig:
    def test_molecule_only(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.molecule.smiles == "CCO"
        assert cfg.molecule.charge_range == (-1, 0)

    def test_missing_molecule_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="must have a \\[molecule\\]"):
            load_config(tmp_toml(""))

    def test_missing_smiles_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="must specify 'smiles'"):
            load_config(
                tmp_toml("""
                [molecule]
                charge_min = -1
            """)
            )


class TestChargeRange:
    def test_custom_range(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "NCC(=O)O"
            charge_min = -2
            charge_max = 2
        """)
        )
        assert cfg.molecule.charge_range == (-2, 2)

    def test_inverted_range_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"charge_min.*>.*charge_max"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                charge_min = 1
                charge_max = -1
            """)
            )


class TestDriverDefaults:
    def test_pyscf_defaults_with_solvent(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            solvent = "water"
            [scoring]
            solvent = "water"
            [compute]
            driver = "pyscf"
        """)
        )
        assert cfg.refinement.method == "wB97X-3c"
        assert cfg.refinement.basis == "vDZP"
        assert cfg.refinement.solvent_model == "SSVPE"
        assert cfg.scoring.method == "wB97M-V"
        assert cfg.scoring.basis == "def2-QZVPPD"
        assert cfg.scoring.solvent_model == "SSVPE"

    def test_pyscf_defaults_gas_phase(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [compute]
            driver = "pyscf"
        """)
        )
        assert cfg.refinement.method == "wB97X-3c"
        assert cfg.refinement.basis == "vDZP"
        assert cfg.refinement.solvent_model is None
        assert cfg.scoring.solvent_model is None

    def test_psi4_defaults(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [scoring]
            solvent = "water"
            [compute]
            driver = "psi4"
        """)
        )
        assert cfg.refinement.method == "wB97X-3c"
        assert cfg.refinement.basis == "vDZP"
        assert cfg.refinement.solvent_model is None
        assert cfg.scoring.method == "wB97M-V"
        assert cfg.scoring.solvent_model == "IEFPCM"

    def test_unknown_driver_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="Unknown driver"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [compute]
                driver = "gaussian"
            """)
            )


class TestDriverOverrides:
    def test_override_refinement_method(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "B3LYP"
            basis = "6-31G*"
            solvent_model = "SMD"
            solvent = "water"
            [compute]
            driver = "pyscf"
        """)
        )
        assert cfg.refinement.method == "B3LYP"
        assert cfg.refinement.basis == "6-31G*"

    def test_override_scoring(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [scoring]
            method = "B2PLYP"
            basis = "def2-TZVPP"
            solvent_model = "SMD"
            solvent = "water"
            [compute]
            driver = "pyscf"
        """)
        )
        assert cfg.scoring.method == "B2PLYP"
        assert cfg.scoring.basis == "def2-TZVPP"


class TestSolventValidation:
    def test_solvent_model_without_solvent_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"solvent_model.*but no solvent"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvent_model = "SMD"
            """)
            )

    def test_scoring_solvent_model_without_solvent_raises(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        with pytest.raises(ValueError, match=r"solvent_model.*but no solvent"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [scoring]
                solvent_model = "IEFPCM"
            """)
            )

    def test_psi4_refinement_solvent_warns(
        self,
        tmp_toml: Callable[[str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvent_model = "IEFPCM"
                solvent = "water"
                [compute]
                driver = "psi4"
            """)
            )
        assert "no analytical gradients" in caplog.text


class TestSamplingConfig:
    def test_defaults(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.sampling.approach == "rdkit_first"
        assert cfg.sampling.ewin == 10.0

    def test_crest_first(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [sampling]
            approach = "crest_first"
            prescreen_mode = "squick"
        """)
        )
        assert cfg.sampling.approach == "crest_first"
        assert cfg.sampling.prescreen_mode == "squick"

    def test_unknown_approach_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="Unknown sampling approach"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [sampling]
                approach = "magic"
            """)
            )


class TestRRHOMethod:
    def test_default_is_xtb(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.refinement.rrho_method == "xtb"

    def test_dft_accepted(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            rrho_method = "dft"
        """)
        )
        assert cfg.refinement.rrho_method == "dft"

    def test_invalid_rrho_method_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="Unknown rrho_method"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                rrho_method = "cheap"
            """)
            )

    def test_dft_rrho_psi4_solvent_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="rrho_method = 'dft'"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                rrho_method = "dft"
                solvent_model = "IEFPCM"
                solvent = "water"
                [compute]
                driver = "psi4"
            """)
            )


class TestExampleConfigs:
    def test_glycine_pyscf(self) -> None:
        cfg = load_config(Path("examples/glycine_pyscf.toml"))
        assert cfg.molecule.smiles == "NCC(=O)O"
        assert cfg.compute.driver == "pyscf"
        assert cfg.refinement.solvent_model == "SSVPE"

    def test_glycine_psi4(self) -> None:
        cfg = load_config(Path("examples/glycine_psi4.toml"))
        assert cfg.molecule.smiles == "NCC(=O)O"
        assert cfg.compute.driver == "psi4"
        assert cfg.refinement.solvent_model is None
