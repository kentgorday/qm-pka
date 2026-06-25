"""Tests for the Psi4 DFT runner, focusing on the PCM cavity radii."""

from __future__ import annotations

import numpy as np
import pytest

from qm_pka.psi4_runner import (
    _MODIFIED_BONDI_RADII,
    _PCM_RADII_SCALING,
    _pcm_block,
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
