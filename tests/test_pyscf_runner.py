"""Tests for PySCF DFT runner, focusing on D4 composite method registration."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.dft import libxc

# Import triggers D4 composite registration
from qm_pka.pyscf_runner import (
    _D4_COMPOSITES,
    _HESSIAN_FALLBACK,
    _resolve_basis,
    _resolve_method,
)


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
        result = _resolve_basis("vDZP", ["H", "O"])
        assert isinstance(result, dict)
        assert "H" in result
        assert "O" in result

    def test_vdzp_case_insensitive(self) -> None:
        result = _resolve_basis("vdzp", ["C", "H"])
        assert isinstance(result, dict)

    def test_normal_basis_passes_through(self) -> None:
        result = _resolve_basis("def2-svp", ["H", "O"])
        assert result == "def2-svp"


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
