"""Tests for the Psi4 DFT runner, focusing on the PCM cavity radii."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qm_pka import psi4_runner
from qm_pka.config import DEFAULT_MEMORY_GB
from qm_pka.psi4_runner import (
    _MODIFIED_BONDI_RADII,
    _PCM_RADII_SCALING,
    _pcm_block,
    _run_psi4,
    _scratch_dir,
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
        monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
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
        monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
        self._recording_psi4(tmp_path)
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp))
        assert f"--memory {DEFAULT_MEMORY_GB}GB" in (tmp_path / "argv.txt").read_text()

    def test_scratch_pinned_to_work_dir(
        self, isolated_tmpdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # -s keeps psi4's binary scratch inside the dir we delete
        monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
        self._recording_psi4(tmp_path)
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp))
            assert f"-s {tmp}" in (tmp_path / "argv.txt").read_text()


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
