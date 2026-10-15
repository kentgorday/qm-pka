"""Tests for the Psi4 DFT runner, focusing on the PCM cavity radii."""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qm_pka import psi4_runner, refinement, scoring
from qm_pka.config import DEFAULT_MEMORY_GB
from qm_pka.psi4_runner import (
    _MODIFIED_BONDI_RADII,
    _PCM_RADII_SCALING,
    _df_basis_block,
    _pcm_block,
    _run_psi4,
    _scratch_dir,
    psi4_executable,
)
from qm_pka.types import Geometry


def _water() -> Geometry:
    return Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]),
    )


class TestModifiedBondiRadii:
    def test_equals_pyscf_modified_bondi(self) -> None:
        # Single source of truth: the hardcoded Psi4 table must be EXACTLY
        # PySCF's modified_Bondi (same elements and values) so both backends
        # build the same cavity.  Dev/test environments have both installed; the
        # Psi4 runtime itself never imports PySCF.
        pcm = pytest.importorskip("pyscf.solvent.pcm")
        from pyscf import gto
        from pyscf.data import radii

        pyscf_table = {
            gto.elements.ELEMENTS[z]: round(pcm.modified_Bondi[z] * radii.BOHR, 4)
            for z in range(1, len(pcm.modified_Bondi))
            if pcm.modified_Bondi[z] > 0
        }
        assert pyscf_table == _MODIFIED_BONDI_RADII

    def test_hydrogen_is_modified(self) -> None:
        assert _MODIFIED_BONDI_RADII["H"] == 1.10  # not Bondi's 1.20


class TestPCMBlock:
    def test_emits_explicit_prescaled_spheres(self) -> None:
        block = _pcm_block("IEFPCM", "water", _water())
        assert "Mode = Explicit" in block
        assert "RadiiSet" not in block  # explicit spheres replace the named set
        # Radii are pre-scaled (Explicit mode applies no scaling):
        assert f"{1.52 * _PCM_RADII_SCALING:.6f}" in block  # O -> 1.824000
        assert f"{1.10 * _PCM_RADII_SCALING:.6f}" in block  # H -> 1.320000

    def test_hydrogen_radius_is_a_free_parameter(self) -> None:
        # H radius is user-settable (default 1.10); other elements are unaffected.
        custom = _pcm_block("IEFPCM", "water", _water(), pcm_hydrogen_radius=1.30)
        assert f"{1.30 * _PCM_RADII_SCALING:.6f}" in custom  # H -> 1.560000
        assert f"{1.52 * _PCM_RADII_SCALING:.6f}" in custom  # O unchanged
        default = _pcm_block("IEFPCM", "water", _water())
        assert f"{1.10 * _PCM_RADII_SCALING:.6f}" in default  # H -> 1.320000

    def test_block_not_indented(self) -> None:
        # Regression: dedent once left "pcm = {" indented -> Psi4 IndentationError.
        block = _pcm_block("IEFPCM", "water", _water())
        assert block.startswith("pcm = {")
        assert "\n  Units = Angstrom" in block

    def test_unknown_element_raises(self) -> None:
        # The table covers H-Lr (Z<=103); a superheavy beyond it raises clearly.
        geom = Geometry(symbols=("Rf",), coords=np.array([[0.0, 0.0, 0.0]]))
        with pytest.raises(ValueError, match="modified-Bondi radius"):
            _pcm_block("IEFPCM", "water", geom)

    def test_full_periodic_table_coverage(self) -> None:
        # Parity with PySCF: every element PySCF defines must be present here.
        assert len(_MODIFIED_BONDI_RADII) == 103  # H..Lr


def _thioacetate() -> Geometry:
    return Geometry(
        symbols=("C", "C", "O", "S", "H", "H", "H"),
        coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.52, 0.0, 0.0],
                [2.13, 1.07, 0.0],
                [2.36, -1.50, 0.0],
                [-0.38, 1.02, 0.0],
                [-0.38, -0.51, 0.89],
                [-0.38, -0.51, -0.89],
            ]
        ),
    )


def _gbs_shells(block: str, elem: str) -> list[list[object]]:
    """One element's shells from a Psi4 basis block, in PySCF's [l, [exp, coef], ...] form."""
    # An element's entry opens with "Cl     0"; a shell with "S    6   1.00".
    entry = re.split(rf"(?m)^{elem}\s+0\s*$", block)[1].split("****", 1)[0]
    lines = entry.strip().splitlines()
    shells: list[list[object]] = []
    while lines:
        letter, n_prim, _scale = lines.pop(0).split()
        rows = [
            [float(x.replace("D", "E")) for x in lines.pop(0).split()] for _ in range(int(n_prim))
        ]
        shells.append(["SPDFGHI".index(letter), *rows])
    return shells


class TestFittingBasisBlock:
    """The extra s primitive of config.AUX_DIFFUSE_S, as Psi4 is told about it."""

    def test_nothing_to_add_means_no_block(self) -> None:
        assert _df_basis_block("def2-QZVPPD", _water()) == ""

    @pytest.mark.parametrize("basis", ["def2-QZVPPD", "def2-SVP", "def2-mTZVPP", "vDZP"])
    def test_bases_psi4_fits_with_that_set_get_it(self, basis: str) -> None:
        assert "def2-universal-jkfit-plus-s" in _df_basis_block(basis, _thioacetate())

    @pytest.mark.parametrize("basis", ["cc-pVTZ", "aug-cc-pVTZ", "6-31+G*", "pcseg-2"])
    def test_other_bases_keep_psi4s_own_fit(self, basis: str) -> None:
        # Psi4 fits these with a JKFIT set of its own, which for an augmented
        # basis does better by small anions than ours; see config.AUXBASIS.
        assert _df_basis_block(basis, _thioacetate()) == ""

    def test_only_those_elements_get_a_set_of_their_own(self) -> None:
        block = _df_basis_block("def2-QZVPPD", _thioacetate())
        assert 'key="df_basis_scf"' in block
        assigns = [line for line in block.splitlines() if line.startswith("assign")]
        assert assigns == ["assign def2-universal-jkfit", "assign S def2-universal-jkfit-plus-s"]

    def test_carries_the_functions_pyscf_fits_with(self) -> None:
        # Psi4's shells come from basis-set-exchange and PySCF's from its own
        # library; both backends must end up with the same fit, primitive included.
        pytest.importorskip("pyscf")
        from qm_pka.config import AUX_DIFFUSE_S
        from qm_pka.pyscf_runner import _auxbasis

        elements = tuple(AUX_DIFFUSE_S)
        geom = Geometry(symbols=elements, coords=np.zeros((len(elements), 3)))
        block = _df_basis_block("def2-QZVPPD", geom)
        pyscf_aux = _auxbasis(elements)
        assert isinstance(pyscf_aux, dict)
        for elem in elements:
            assert _gbs_shells(block, elem) == pyscf_aux[elem], elem

    def test_every_kind_of_job_carries_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs: list[str] = []

        def fake_run(input_text: str, work_dir: Path, **_kwargs: object) -> str:
            inputs.append(input_text)
            (work_dir / "optimized.xyz").write_text("1\n\nCl 0.0 0.0 0.0\n")
            (work_dir / "frequencies.dat").write_text("")
            return "=== FINAL ENERGY: -1.000000000000 ===\n=== CONVERGED: 1 ===\n"

        monkeypatch.setattr(psi4_runner, "_run_psi4", fake_run)
        chloride = Geometry(symbols=("Cl",), coords=np.zeros((1, 3)))
        psi4_runner.single_point(chloride, -1, "PBE0", "def2-SVP")
        psi4_runner.optimize(chloride, -1, "PBE0", "def2-SVP")
        psi4_runner.frequencies(chloride, -1, "PBE0", "def2-SVP")
        assert len(inputs) == 3
        for text in inputs:
            assert _df_basis_block("def2-SVP", chloride) in text


@pytest.fixture
def isolated_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point tempfile at tmp_path so scratch never lands in the real TMPDIR.

    tempfile.gettempdir() caches its answer on first call, so setting the
    TMPDIR env var after interpreter start has no effect; the module-level
    tempfile.tempdir must be patched instead.  Without this, tests that keep
    scratch on purpose leak a directory into the real temp dir on every run.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


class TestScratchCleanup:
    """Psi4 scratch must not leak.

    Orphaned scratch is what filled the disk during the first training-set
    run: on the out-of-core DFJK path a single job writes several GB of
    three-index integrals, and nothing ever deleted the directories.
    """

    @staticmethod
    def _stub_psi4(tmp_path: Path, body: str) -> None:
        exe = tmp_path / "psi4"
        exe.write_text(f"#!/bin/sh\n{body}\n")
        exe.chmod(0o755)

    def test_removed_on_success(self, isolated_tmpdir: Path) -> None:
        with _scratch_dir() as tmp:
            captured = Path(tmp)
            assert captured.exists()
        assert not captured.exists()

    def test_removed_on_exception(self, isolated_tmpdir: Path) -> None:
        captured = None
        with pytest.raises(RuntimeError, match="boom"), _scratch_dir() as tmp:
            captured = Path(tmp)
            raise RuntimeError("boom")
        assert captured is not None and not captured.exists()

    def test_kept_when_env_set(
        self, isolated_tmpdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QM_PKA_KEEP_SCRATCH", "1")
        with _scratch_dir() as tmp:
            captured = Path(tmp)
        # retained on purpose -- isolated_tmpdir keeps it inside pytest's
        # tmp_path so it is still cleaned up with the rest of the test run
        assert captured.exists()
        assert captured.parent == isolated_tmpdir


class TestMemoryFlag:
    """memory_gb is passed as psi4's --memory CLI flag, not baked into input."""

    @staticmethod
    def _recording_psi4(tmp_path: Path) -> None:
        exe = tmp_path / "psi4"
        # record argv, then write a minimal parseable output to $3 (the -o path)
        exe.write_text(
            "#!/bin/sh\n"
            f'echo "$@" > {tmp_path}/argv.txt\n'
            f'cp "$1" {tmp_path}/seen_input.dat\n'
            'echo "=== FINAL ENERGY: -1.000000000000 ===" > "$3"\n'
        )
        exe.chmod(0o755)

    def test_memory_flag_present(
        self, isolated_tmpdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: tmp_path / "psi4")
        self._recording_psi4(tmp_path)
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp), memory_gb=12.0)
        argv = (tmp_path / "argv.txt").read_text()
        assert "--memory 12.0GB" in argv
        # the input file itself stays clean
        assert "memory" not in (tmp_path / "seen_input.dat").read_text()

    def test_default_memory_always_passed(
        self, isolated_tmpdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Psi4 must never silently fall back to its own 500 MiB default.
        monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: tmp_path / "psi4")
        self._recording_psi4(tmp_path)
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp))
        assert f"--memory {DEFAULT_MEMORY_GB}GB" in (tmp_path / "argv.txt").read_text()

    def test_scratch_pinned_to_work_dir(
        self, isolated_tmpdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # -s keeps psi4's binary scratch inside the dir we delete
        monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: tmp_path / "psi4")
        self._recording_psi4(tmp_path)
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp))
            assert f"-s {tmp}" in (tmp_path / "argv.txt").read_text()


# ---------------------------------------------------------------------------
# Which psi4 gets launched
# ---------------------------------------------------------------------------
class TestPsi4Executable:
    """Psi4 has a pixi environment of its own, beside the one the tests run in."""

    @staticmethod
    def _psi4_in(env: Path, body: str = "") -> Path:
        exe = env / "bin" / "psi4"
        exe.parent.mkdir(parents=True)
        exe.write_text(f"#!/bin/sh\n{body}\n")
        exe.chmod(0o755)
        return exe

    @pytest.fixture()
    def envs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A directory of environments, with the interpreter placed in envs/default."""
        envs = tmp_path / "envs"
        monkeypatch.setattr(sys, "prefix", str(envs / "default"))
        monkeypatch.setenv("PATH", str(tmp_path / "elsewhere" / "bin"))
        return envs

    def test_prefers_its_own_environment_to_path(self, envs: Path, tmp_path: Path) -> None:
        # the locked Psi4 must not be shadowed by whatever else is on PATH
        sibling = self._psi4_in(envs / "psi4")
        self._psi4_in(tmp_path / "elsewhere")
        assert psi4_executable() == sibling

    def test_falls_back_to_path(self, envs: Path, tmp_path: Path) -> None:
        on_path = self._psi4_in(tmp_path / "elsewhere")
        assert psi4_executable() == on_path

    def test_missing_psi4_says_how_to_install_it(self, envs: Path) -> None:
        with pytest.raises(FileNotFoundError, match="pixi install -e psi4"):
            psi4_executable()

    def test_missing_psi4_stops_the_stage(self, envs: Path) -> None:
        # refine() and score() wrap each conformer in a handler that records the
        # failure and moves on; a Psi4 that is not installed must not get that far
        with pytest.raises(FileNotFoundError):
            refinement._get_driver("psi4")
        with pytest.raises(FileNotFoundError):
            scoring._get_driver("psi4")

    def test_psi4_runs_with_its_own_bin_first_on_path(
        self, envs: Path, tmp_path: Path, isolated_tmpdir: Path
    ) -> None:
        # Psi4 finds helpers such as mctc-gcp on PATH; they are installed beside it
        exe = self._psi4_in(
            envs / "psi4",
            f'echo "$PATH" > {tmp_path}/path.txt\n'
            'echo "=== FINAL ENERGY: -1.000000000000 ===" > "$3"',
        )
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp))
        seen = (tmp_path / "path.txt").read_text().strip().split(":")
        assert seen[0] == str(exe.parent)
        assert str(tmp_path / "elsewhere" / "bin") in seen[1:]


# ---------------------------------------------------------------------------
# PCM cavity validity
# ---------------------------------------------------------------------------
def test_pcm_block_emits_the_requested_tessera_area() -> None:
    """The area must reach the input deck; a silent default would make the
    cavity search a no-op that still reports success."""
    geom = Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.9266, 0.0]]),
    )
    default = psi4_runner._pcm_block("IEFPCM", "water", geom)
    custom = psi4_runner._pcm_block("IEFPCM", "water", geom, area=0.05)
    assert f"Area = {psi4_runner.DEFAULT_TESSERA_AREA}" in default
    assert "Area = 0.05" in custom


def test_run_psi4_rejects_a_suppressed_pcm_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PCMSolver's cavity checks are patched to warnings in this build, so a
    job can exit 0 with a meaningless energy.  Exit status alone must not be
    treated as success."""

    class _Result:
        returncode = 0
        stdout = ""
        stderr = (
            "Absolute value of the relative difference between the Gauss' theorem "
            "(-15.7959) and computed (-16.0515) values of the total nuclear ASC "
            "higher than threshold (0.0161811).\n"
            "Normally an error, this has been commuted to a warning via patch.\n"
        )

    monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: tmp_path / "psi4")
    monkeypatch.setattr("qm_pka.psi4_runner.subprocess.run", lambda *a, **k: _Result())
    (tmp_path / "output.dat").write_text("=== FINAL ENERGY: -1.0 ===")
    with pytest.raises(RuntimeError, match="invalid cavity"):
        psi4_runner._run_psi4("", tmp_path)


def test_choose_tessera_area_prefers_the_default_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy geometry costs one probe; an unhealthy one searches the ladder
    rather than refining blindly, because validity is not monotonic in area."""
    geom = Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.9266, 0.0]]),
    )
    psi4_runner._cavity_cache.clear()
    monkeypatch.setattr(psi4_runner, "_cavity_is_valid", lambda *a, **k: True)
    assert psi4_runner.choose_tessera_area(geom, 0, "IEFPCM", "water") == (
        psi4_runner.DEFAULT_TESSERA_AREA
    )

    tried: list[float] = []

    def _only_third_works(*args: object, **kwargs: object) -> bool:
        tried.append(args[5] if len(args) > 5 else kwargs["area"])  # type: ignore[arg-type]
        return len(tried) == 3

    psi4_runner._cavity_cache.clear()
    monkeypatch.setattr(psi4_runner, "_cavity_is_valid", _only_third_works)
    chosen = psi4_runner.choose_tessera_area(geom, 0, "IEFPCM", "water")
    assert chosen == tried[2]
    assert tried[0] == psi4_runner.DEFAULT_TESSERA_AREA


def test_choose_tessera_area_raises_when_no_area_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Better a dropped conformer than a converged, meaningless energy."""
    geom = Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.9266, 0.0]]),
    )
    psi4_runner._cavity_cache.clear()
    monkeypatch.setattr(psi4_runner, "_cavity_is_valid", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="No usable PCM cavity"):
        psi4_runner.choose_tessera_area(geom, 0, "IEFPCM", "water")
