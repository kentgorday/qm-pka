"""Tests for PySCF DFT runner, focusing on D4 composite method registration."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.dft import libxc

# Import triggers D4 composite registration and the PCM ECP-cavity patch
from qm_pka.pyscf_runner import (
    _D4_COMPOSITES,
    _HESSIAN_FALLBACK,
    _build_mf,
    _resolve_basis,
    _resolve_method,
)
from qm_pka.types import Geometry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def water_mol() -> gto.Mole:
    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0 0 0.96; H 0 0.96 0"
    mol.basis = "def2-svp"
    mol.verbose = 0
    mol.build()
    return mol


# ---------------------------------------------------------------------------
# _resolve_method tests
# ---------------------------------------------------------------------------
class TestResolveMethod:
    @pytest.mark.parametrize(
        "method,expected_d4_param",
        [
            ("wb97x-d4", "wb97x"),
            ("wb97x-d4rev", "wb97x-rev"),
            ("wb97x-3c", "wb97x-3c"),
        ],
    )
    def test_d4_composites_detected(self, method: str, expected_d4_param: str) -> None:
        xc_string, d4_param = _resolve_method(method)
        assert d4_param == expected_d4_param
        assert "-d4" in xc_string  # internal name has -d4 suffix

    @pytest.mark.parametrize("method", ["wb97x-v", "wb97m-v", "wb97x-d3bj", "b3lyp"])
    def test_non_composites_pass_through(self, method: str) -> None:
        xc_string, d4_param = _resolve_method(method)
        assert xc_string == method
        assert d4_param is None


# ---------------------------------------------------------------------------
# _resolve_basis tests
# ---------------------------------------------------------------------------
class TestResolveBasis:
    def test_vdzp_loaded_from_bse(self) -> None:
        basis, _ecp = _resolve_basis("vDZP", ["H", "O"])
        assert isinstance(basis, dict)
        assert "H" in basis
        assert "O" in basis

    def test_vdzp_ecp_loaded_for_core_elements(self) -> None:
        # vDZP is ECP-designed: oxygen carries a core ECP, hydrogen does not.
        _basis, ecp = _resolve_basis("vDZP", ["H", "O"])
        assert ecp is not None
        assert "O" in ecp
        assert "H" not in ecp

    def test_vdzp_case_insensitive(self) -> None:
        basis, _ecp = _resolve_basis("vdzp", ["C", "H"])
        assert isinstance(basis, dict)

    def test_normal_basis_passes_through(self) -> None:
        basis, ecp = _resolve_basis("def2-svp", ["H", "O"])
        assert basis == "def2-svp"
        assert ecp is None

    def test_def2_no_ecp_for_light_elements(self) -> None:
        # def2 is all-electron through Z=36; no ECP for an organic element set.
        _basis, ecp = _resolve_basis("def2-QZVPPD", ["C", "H", "O", "Br"])
        assert ecp is None

    def test_def2_ecp_for_heavy_elements(self) -> None:
        # Z >= 37 (here iodine) needs the def2-ECP, applied by basis name.
        basis, ecp = _resolve_basis("def2-QZVPPD", ["I", "H"])
        assert ecp == {"I": "def2-QZVPPD"}
        assert basis == "def2-QZVPPD"


# ---------------------------------------------------------------------------
# PCM cavity correction for ECP atoms (monkeypatch of pyscf.solvent.pcm)
# ---------------------------------------------------------------------------
class TestPCMECPCavity:
    """Importing pyscf_runner patches ``pcm.gen_surface`` so the solvent cavity
    switching radii use the true nuclear charge rather than the ECP-reduced
    ``mol.atom_charges()``."""

    def _cavity_npts(self, basis: str) -> int:
        geom = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]),
        )
        _mol, mf = _build_mf(geom, 0, "pbe", basis, "SSVPE", "water", 1)
        mf.with_solvent.build()
        return int(mf.with_solvent.surface["grid_coords"].shape[0])

    def test_patch_installed(self) -> None:
        import pyscf.solvent.pcm as pcm

        assert getattr(pcm.gen_surface, "_ecp_cavity_patched", False)

    def test_ecp_basis_matches_all_electron_cavity(self) -> None:
        # Same geometry: an ECP basis (vDZP) must build the SAME PCM cavity as an
        # all-electron basis (def2-SVP), since the switching radii use true
        # nuclear charge in both.  Without the patch the vDZP (ECP) cavity differs
        # (oxygen's switching radius collapses to carbon's), changing the count.
        assert self._cavity_npts("vDZP") == self._cavity_npts("def2-svp")

    # Upstream-drift tripwire: the monkeypatch wraps an internal PySCF function
    # and relies on it reading mol.atom_charges() for the switching radii.  Pin
    # the source so any change to that function fails here until someone
    # re-reviews _patch_pcm_ecp_cavity, then re-blesses the values below.
    _PINNED_PYSCF = "2.11.0"
    _PINNED_HASH = "c3664f90b7f761deb85cef6bd5b339965bbd911c461f59786c453606c343e57c"

    def test_upstream_gen_surface_unchanged(self) -> None:
        import hashlib
        import inspect

        import pyscf
        import pyscf.solvent.pcm as pcm

        original = getattr(pcm.gen_surface, "_ecp_cavity_original", None)
        assert original is not None, "ECP cavity patch not applied"
        try:
            src = inspect.getsource(original)
        except (OSError, TypeError):
            pytest.skip("pyscf.solvent.pcm.gen_surface source unavailable")

        # Actionable invariant: the exact behaviour the patch corrects.
        assert "mol.atom_charges()" in src, (
            "pyscf gen_surface no longer reads mol.atom_charges() for the cavity "
            "radii — the ECP cavity patch may be obsolete or broken. Re-review "
            "qm_pka.pyscf_runner._patch_pcm_ecp_cavity."
        )

        actual = hashlib.sha256(src.encode()).hexdigest()
        assert actual == self._PINNED_HASH, (
            "pyscf.solvent.pcm.gen_surface source changed "
            f"(pinned pyscf {self._PINNED_PYSCF}, running {pyscf.__version__}). "
            "Re-review qm_pka.pyscf_runner._patch_pcm_ecp_cavity against the new "
            f"implementation, then update _PINNED_PYSCF/_PINNED_HASH to {actual!r}."
        )


# ---------------------------------------------------------------------------
# D4 composite registration: correct XC functional
# ---------------------------------------------------------------------------
class TestD4CompositeXC:
    """All D4 composites must use wB97X-V (libxc 466), not wB97X (464)."""

    LIBXC_WB97X_V = 466
    LIBXC_WB97X = 464

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_uses_wb97x_v_functional(self, method: str) -> None:
        """D4 composites must use the reparameterized wB97X-V (466)."""
        xc_string, _ = _resolve_method(method)
        _, fac_list = libxc.parse_xc(xc_string)
        func_id = int(fac_list[0][0])
        assert func_id == self.LIBXC_WB97X_V, (
            f"{method} uses libxc {func_id}, expected {self.LIBXC_WB97X_V} (wB97X-V)"
        )


# ---------------------------------------------------------------------------
# D4 composite registration: no VV10
# ---------------------------------------------------------------------------
class TestD4CompositeNoVV10:
    """D4 composites must NOT include VV10 nonlocal correlation."""

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_no_vv10(self, method: str, water_mol: gto.Mole) -> None:
        xc_string, _ = _resolve_method(method)
        mf = dft.RKS(water_mol)
        mf.xc = xc_string
        assert not mf.do_nlc(), f"{method} has VV10 enabled"


# ---------------------------------------------------------------------------
# D4 composite registration: dispersion is applied
# ---------------------------------------------------------------------------
class TestD4CompositeDispersion:
    """D4 composites must have non-zero D4 dispersion energy."""

    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    @pytest.mark.slow
    def test_d4_dispersion_applied(self, method: str, water_mol: gto.Mole) -> None:
        xc_string, _ = _resolve_method(method)
        mf = dft.RKS(water_mol)
        mf.xc = xc_string
        mf.grids.atom_grid = (50, 194)  # small grid for speed
        mf.verbose = 0
        mf.kernel()
        d4_energy = mf.scf_summary.get("dispersion", 0.0)
        assert abs(d4_energy) > 1e-8, f"{method} has no D4 dispersion"


# ---------------------------------------------------------------------------
# D4 parameters are distinct
# ---------------------------------------------------------------------------
class TestD4ParametersDistinct:
    """Each D4 composite must use different parameters."""

    @pytest.mark.slow
    def test_d4_energies_differ(self, water_mol: gto.Mole) -> None:
        d4_energies: dict[str, float] = {}
        for method in _D4_COMPOSITES:
            xc_string, _ = _resolve_method(method)
            mf = dft.RKS(water_mol)
            mf.xc = xc_string
            mf.grids.atom_grid = (50, 194)
            mf.verbose = 0
            mf.kernel()
            d4_energies[method] = float(mf.scf_summary.get("dispersion", 0.0))

        # All pairwise comparisons
        methods = list(d4_energies)
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1 :]:
                assert not np.isclose(d4_energies[m1], d4_energies[m2], atol=1e-8), (
                    f"{m1} and {m2} have same D4 energy: {d4_energies[m1]}"
                )


# ---------------------------------------------------------------------------
# D4 dispersion in gradients
# ---------------------------------------------------------------------------
class TestD4Gradients:
    """D4 dispersion must be included in analytical gradients."""

    @pytest.mark.slow
    @pytest.mark.parametrize("method", list(_D4_COMPOSITES))
    def test_gradient_includes_d4(self, method: str, water_mol: gto.Mole) -> None:
        """Compare gradient with and without D4 to verify it contributes."""
        xc_string, _ = _resolve_method(method)

        # With D4
        mf_d4 = dft.RKS(water_mol)
        mf_d4.xc = xc_string
        mf_d4.grids.atom_grid = (50, 194)
        mf_d4.verbose = 0
        mf_d4.kernel()
        grad_d4 = mf_d4.nuc_grad_method()
        grad_d4.verbose = 0
        g_with = grad_d4.kernel()

        # Without D4 (plain wb97x-v, no VV10, no dispersion)
        mf_bare = dft.RKS(water_mol)
        mf_bare.xc = "wb97x-v"
        mf_bare.nlc = 0
        mf_bare.grids.atom_grid = (50, 194)
        mf_bare.verbose = 0
        mf_bare.kernel()
        grad_bare = mf_bare.nuc_grad_method()
        grad_bare.verbose = 0
        g_without = grad_bare.kernel()

        # Gradients should differ (D4 contributes)
        assert not np.allclose(g_with, g_without, atol=1e-10), (
            f"{method} gradient unchanged by D4 dispersion"
        )


# ---------------------------------------------------------------------------
# Hessian fallback
# ---------------------------------------------------------------------------
class TestHessianFallback:
    @pytest.mark.parametrize("method", list(_HESSIAN_FALLBACK))
    def test_d4_methods_fall_up_to_vv10(self, method: str) -> None:
        assert _HESSIAN_FALLBACK[method] == "wb97x-v"

    def test_non_d4_methods_not_in_fallback(self) -> None:
        for method in ["wb97m-v", "wb97x-d3bj", "b3lyp"]:
            assert method not in _HESSIAN_FALLBACK
