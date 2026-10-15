import logging
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from tautomator.config import _has_diffuse, load_config


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
        assert cfg.molecule.solvent == "water"

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
    def test_pyscf_minimizes_in_solvent(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [compute]
            driver = "pyscf"
        """)
        )
        assert (cfg.refinement.method, cfg.refinement.basis) == ("wB97X-3c", "vDZP")
        assert cfg.refinement.engine == "pyscf"
        # PySCF has analytical PCM gradients, so the cheapest way to a solvated
        # refinement energy is to minimize in solvent outright.
        assert cfg.refinement.solvation == "optimize"
        assert (cfg.scoring.method, cfg.scoring.basis) == ("wB97M-V", "def2-QZVPPD")
        assert cfg.scoring.solvent_model == "IEFPCM"

    def test_psi4_adds_a_single_point_instead(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [compute]
            driver = "psi4"
        """)
        )
        assert cfg.refinement.solvation == "single_point"
        assert cfg.scoring.solvent_model == "IEFPCM"

    def test_gas_phase_scoring_makes_refinement_gas_phase(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [scoring]
            solvent_model = "none"
        """)
        )
        assert cfg.scoring.solvent_model is None
        assert cfg.refinement.solvation == "none"
        assert cfg.refinement.ewin == 10.0  # wider, because gas rankings rerank

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

    def test_solvent_model_not_on_this_driver_raises(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        with pytest.raises(ValueError, match="not available on the psi4 driver"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [scoring]
                solvent_model = "SMD"
                [compute]
                driver = "psi4"
            """)
            )

    def test_pyscf_does_not_offer_ssvpe(self, tmp_toml: Callable[[str], Path]) -> None:
        # Its symmetrized system matrix costs ions several kcal/mol on PySCF's
        # surface, so the model is refused rather than left as a trap.
        with pytest.raises(ValueError, match="not available on the pyscf driver"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [scoring]
                solvent_model = "SSVPE"
                [compute]
                driver = "pyscf"
            """)
            )


class TestProtocols:
    def test_fast(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.sampling.ewin == 15.0
        assert cfg.refinement.engine == "orbmol"
        assert cfg.refinement.method == "OrbMol-v2"
        assert cfg.refinement.solvation == "single_point"
        assert (cfg.refinement.solvation_method, cfg.refinement.solvation_basis) == (
            "wB97X-3c",
            "vDZP",
        )
        assert cfg.refinement.solvation_prefilter_ewin == 12.0
        assert cfg.refinement.ewin == 6.0
        assert (cfg.scoring.method, cfg.scoring.basis) == ("wB97M-V", "def2-TZVPPD")

    def test_high_quality(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "high_quality"
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.sampling.ewin == 10.0
        assert (cfg.refinement.method, cfg.refinement.basis) == ("wB97X-3c", "vDZP")
        assert cfg.refinement.ewin == 8.0
        assert (cfg.scoring.method, cfg.scoring.basis) == ("wB97M-V", "def2-QZVPPD")

    def test_file_overrides_preset(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
            [refinement]
            ewin = 4.0
            solvation_prefilter_ewin = 14.0
        """)
        )
        assert cfg.refinement.ewin == 4.0
        assert cfg.refinement.solvation_prefilter_ewin == 14.0
        assert cfg.refinement.method == "OrbMol-v2"  # still the preset's

    def test_unknown_protocol_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="Unknown protocol"):
            load_config(
                tmp_toml("""
                protocol = "cheapest"
                [molecule]
                smiles = "CCO"
            """)
            )

    def test_provenance_records_what_each_stage_computed(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
        """)
        )
        prov = cfg.provenance()
        assert prov["protocol"] == "fast"
        assert prov["refinement"]["engine"] == "orbmol"
        # The solvation level is the DFT pair's, not OrbMol's.
        assert prov["refinement"]["solvation"] == {
            "mode": "single_point",
            "level": "wB97X-3c/vDZP",
            "model": "IEFPCM",
            "driver": "pyscf",
        }
        assert prov["scoring"]["basis"] == "def2-TZVPPD"


class TestOrbmolRefinement:
    def test_selected_by_method(self, tmp_toml: Callable[[str], Path]) -> None:
        # The method names the engine: [compute] driver stays the DFT backend,
        # because it still runs the solvation single point and all of scoring.
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "OrbMol-v2"
            [compute]
            driver = "pyscf"
        """)
        )
        assert cfg.refinement.engine == "orbmol"
        assert cfg.refinement.basis == ""
        assert cfg.compute.driver == "pyscf"
        assert cfg.refinement.solvation == "single_point"
        assert cfg.refinement.solvation_method == "wB97X-3c"

    def test_solvation_level_is_configurable(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "OrbMol-v2"
            solvation_method = "r2SCAN-3c"
            [compute]
            driver = "psi4"
        """)
        )
        assert (cfg.refinement.solvation_method, cfg.refinement.solvation_basis) == (
            "r2SCAN-3c",
            "def2-mTZVPP",
        )

    def test_cannot_score(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"\[compute\] driver = 'orbmol'"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [compute]
                driver = "orbmol"
            """)
            )

    def test_cannot_minimize_in_solvent(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="gas-phase potential"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                method = "OrbMol-v2"
                solvation = "optimize"
            """)
            )

    def test_basis_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="no meaning for an OrbMol model"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                method = "OrbMol-v2"
                basis = "def2-SVP"
            """)
            )

    def test_own_hessian_accepted(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "OrbMol-v2"
            rrho = "refinement"
        """)
        )
        assert cfg.refinement.rrho == "refinement"


class TestSolvationValidation:
    def test_solvation_method_rejected_for_dft(self, tmp_toml: Callable[[str], Path]) -> None:
        # DFT refinement takes its single point at the refinement level, reusing
        # the optimizer's gas energy; a second level would just be waste.
        with pytest.raises(ValueError, match="apply only to an OrbMol refinement"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation_method = "wB97X-3c"
            """)
            )

    def test_psi4_cannot_minimize_in_solvent(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="no analytical PCM gradients"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation = "optimize"
                [compute]
                driver = "psi4"
            """)
            )

    def test_solvated_refinement_with_gas_scoring_raises(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        with pytest.raises(ValueError, match="different physics from scoring"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation = "single_point"
                [scoring]
                solvent_model = "none"
            """)
            )

    def test_gas_refinement_with_solvated_scoring_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation = "none"
            """)
            )
        assert cfg.refinement.solvation == "none"
        assert "rerank against scoring" in caplog.text

    def test_unknown_solvation_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"Unknown \[refinement\] solvation"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation = "implicit"
            """)
            )


class TestSolvationPrefilter:
    def test_accepted_below_the_window(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.refinement.solvation_prefilter_ewin == 12.0

    def test_needs_a_solvation_single_point(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="only applies to solvation = 'single_point'"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation_prefilter_ewin = 12.0
                [compute]
                driver = "pyscf"
            """)
            )

    def test_inside_the_window_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="must exceed ewin"):
            load_config(
                tmp_toml("""
                protocol = "fast"
                [molecule]
                smiles = "CCO"
                [refinement]
                ewin = 6.0
                solvation_prefilter_ewin = 5.0
            """)
            )

    def test_narrow_margin_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                protocol = "fast"
                [molecule]
                smiles = "CCO"
                [refinement]
                ewin = 6.0
                solvation_prefilter_ewin = 7.0
            """)
            )
        assert "9.1 kcal/mol above the gas-phase minimum" in caplog.text

    def test_pointless_for_dft_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation = "single_point"
                solvation_prefilter_ewin = 14.0
                [compute]
                driver = "psi4"
            """)
            )
        assert "saves little" in caplog.text


class TestCompositeMethods:
    def test_basis_is_implied(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "r2SCAN-3c"
            [compute]
            driver = "psi4"
        """)
        )
        assert cfg.refinement.basis == "def2-mTZVPP"

    def test_matching_basis_accepted(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "wB97X-3c"
            basis = "vdzp"
        """)
        )
        assert cfg.refinement.basis == "vdzp"

    def test_other_basis_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="parameterized with vDZP"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                method = "wB97X-3c"
                basis = "def2-TZVP"
            """)
            )

    def test_pyscf_has_only_wb97x_3c(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="not available on the pyscf driver"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                method = "r2SCAN-3c"
                [compute]
                driver = "pyscf"
            """)
            )

    @pytest.mark.parametrize("header", ["", 'protocol = "high_quality"', 'protocol = "fast"'])
    def test_composite_scoring_gets_its_basis(
        self, header: str, tmp_toml: Callable[[str], Path]
    ) -> None:
        # Neither the scoring default nor a preset's basis applies to a composite.
        cfg = load_config(
            tmp_toml(f'{header}\n[molecule]\nsmiles = "CCO"\n[scoring]\nmethod = "wB97X-3c"\n')
        )
        assert (cfg.scoring.method, cfg.scoring.basis) == ("wB97X-3c", "vDZP")

    def test_preset_basis_kept_for_a_plain_functional(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
            [scoring]
            method = "B3LYP"
        """)
        )
        assert cfg.scoring.basis == "def2-TZVPPD"

    def test_plain_functional_keeps_its_basis(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "B3LYP"
            basis = "6-31G*"
        """)
        )
        assert (cfg.refinement.method, cfg.refinement.basis) == ("B3LYP", "6-31G*")


class TestDiffuseBasis:
    @pytest.mark.parametrize(
        "basis",
        [
            "def2-SVPD",
            "def2-TZVPPD",
            "DEF2-QZVPPD",
            "ma-def2-SVP",
            "aug-cc-pVDZ",
            "d-aug-cc-pVTZ",
            "heavy-aug-cc-pVDZ",
            "jun-cc-pVDZ",
            "aug-cc-pV(D+d)Z",
            "aug-pcseg-1",
            "6-31+G*",
            "6-31+G(d)",
            "6-311++G(2d,2p)",
            "augccpvdz",  # PySCF ignores case, hyphens and underscores
            "def2_svpd",
        ],
    )
    def test_names_with_diffuse_functions(self, basis: str) -> None:
        assert _has_diffuse(basis)

    @pytest.mark.parametrize(
        "basis",
        [
            "def2-SVP",
            "def2-TZVPP",
            "def2-QZVPP",
            "def2-mTZVPP",
            "cc-pVDZ",
            "cc-pV(D+d)Z",  # a tight d, not a diffuse shell
            "pcseg-1",
            "6-31G*",
            "6-31G(d)",
            "6-311G(2d,2p)",
            "vDZP",
            "STO-3G",
        ],
    )
    def test_names_without_diffuse_functions(self, basis: str) -> None:
        assert not _has_diffuse(basis)

    def test_scoring_basis_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CC(=O)O"
                [scoring]
                basis = "def2-TZVPP"
            """)
            )
        assert "[scoring] basis = 'def2-TZVPP' has no diffuse functions" in caplog.text

    def test_vdzp_scoring_does_not_warn(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CC(=O)O"
                [scoring]
                method = "wB97M-V"
                basis = "vDZP"
            """)
            )
        assert "diffuse" not in caplog.text

    def test_refinement_basis_does_not_warn(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        # Refinement only ranks within a charge state.
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CC(=O)O"
                [refinement]
                method = "B3LYP"
                basis = "6-31G*"
            """)
            )
        assert "diffuse" not in caplog.text

    def test_composite_scoring_does_not_warn(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CC(=O)O"
                [scoring]
                method = "wB97X-3c"
            """)
            )
        assert "diffuse" not in caplog.text

    def test_cation_only_run_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        # No anion charge state, but zwitterions and plain amine deprotonation
        # still need diffuse functions.
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "NCC(=O)O"
                charge_min = 0
                charge_max = 1
                [scoring]
                basis = "def2-TZVPP"
            """)
            )
        assert "[scoring] basis = 'def2-TZVPP' has no diffuse functions" in caplog.text

    @pytest.mark.parametrize("header", ["", 'protocol = "high_quality"', 'protocol = "fast"'])
    def test_defaults_are_quiet(
        self, header: str, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(tmp_toml(f'{header}\n[molecule]\nsmiles = "CC(=O)O"\n'))
        assert "diffuse" not in caplog.text


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


class TestRrho:
    def test_default_is_xtb(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.refinement.rrho == "xtb"

    def test_invalid_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"Unknown \[refinement\] rrho"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                rrho = "cheap"
            """)
            )

    def test_psi4_hessians_warn(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        # Allowed, but a finite-difference Hessian per conformer dwarfs the
        # optimization it follows.
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                rrho = "refinement"
                [compute]
                driver = "psi4"
            """)
            )
        assert cfg.refinement.rrho == "refinement"
        assert "finite-difference" in caplog.text


class TestDeprecatedSpellings:
    def test_refinement_driver_orbmol(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                driver = "orbmol"
            """)
            )
        assert cfg.refinement.method == "OrbMol-v2"
        assert cfg.refinement.engine == "orbmol"
        assert "[refinement] driver is deprecated" in caplog.text

    def test_refinement_driver_disagreeing_with_compute_raises(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        with pytest.raises(ValueError, match="no longer selects a DFT backend"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                driver = "psi4"
                [compute]
                driver = "pyscf"
            """)
            )

    def test_rrho_method_dft(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                rrho_method = "dft"
            """)
            )
        assert cfg.refinement.rrho == "refinement"
        assert "rrho_method = 'dft' is deprecated" in caplog.text

    def test_refinement_solvent_model(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvent_model = "CPCM"
                [compute]
                driver = "pyscf"
            """)
            )
        assert cfg.refinement.solvation == "optimize"
        assert cfg.scoring.solvent_model == "CPCM"
        assert "[refinement] solvent_model = 'CPCM' is deprecated" in caplog.text

    def test_per_stage_solvent_moves_to_molecule(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvent = "methanol"
                [scoring]
                solvent = "methanol"
            """)
            )
        assert cfg.molecule.solvent == "methanol"
        assert "[refinement] solvent is deprecated" in caplog.text

    def test_conflicting_solvents_raise(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="a run has one solvent"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                solvent = "water"
                [scoring]
                solvent = "methanol"
            """)
            )


class TestExampleConfigs:
    @pytest.mark.parametrize(
        "name,driver,engine,solvation",
        [
            ("glycine_pyscf", "pyscf", "pyscf", "optimize"),
            ("glycine_psi4", "psi4", "psi4", "single_point"),
            ("glycine_orbmol", "pyscf", "orbmol", "single_point"),
            ("aminobutanol_crest_pyscf", "pyscf", "pyscf", "optimize"),
            ("aminobutanol_crest_psi4", "psi4", "psi4", "single_point"),
            ("aminobutanol_rdkit_pyscf", "pyscf", "pyscf", "optimize"),
            ("aminobutanol_rdkit_psi4", "psi4", "psi4", "single_point"),
        ],
    )
    def test_loads(self, name: str, driver: str, engine: str, solvation: str) -> None:
        cfg = load_config(Path(f"examples/{name}.toml"))
        assert cfg.compute.driver == driver
        assert cfg.refinement.engine == engine
        assert cfg.refinement.solvation == solvation

    def test_examples_use_current_spellings(self, caplog: pytest.LogCaptureFixture) -> None:
        # The deprecation shims exist for configs already on disk, not for the
        # examples people copy from.
        with caplog.at_level(logging.WARNING):
            for path in sorted(Path("examples").glob("*.toml")):
                load_config(path)
        assert "deprecated" not in caplog.text
