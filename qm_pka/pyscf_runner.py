"""DFT calculations using PySCF as a Python library.

Provides geometry optimization, single-point energy, and frequency
calculations. Uses an (99, 590) integration grid for all DFT calculations
and a (50, 194) NLC grid for VV10-containing functionals.

Composite methods based on wB97X-V (wB97X-D4, wB97X-D4rev, wB97X-3c) are
registered with PySCF's dispersion dispatch so that the correct D4 parameters
from dftd4 are used. All three share the reparameterized wB97X-V exchange-
correlation functional (libxc 466), which is distinct from the original wB97X
(libxc 464).
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qm_pka.types import Geometry

log = logging.getLogger(__name__)

# Angstrom <-> Bohr conversion
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

# Functionals that include VV10 nonlocal correlation
_VV10_XC = {"wb97m-v", "wb97x-v", "b97m-v"}

# ---------------------------------------------------------------------------
# wB97X-V + D4 composite methods
# ---------------------------------------------------------------------------
# PySCF natively maps "wb97x-d4" to libxc 464 (original wB97X, 2008) + D4.
# This is WRONG for wB97X-D4 as defined by dftd4, which uses the
# reparameterized wB97X-V functional (libxc 466, 2013) with D4 dispersion
# instead of VV10.  We register these composite methods correctly.
#
# Mapping: user method name -> (internal xc string, dftd4 param name)
# All use wB97X-V (466) as the XC functional with VV10 disabled.
_D4_COMPOSITES: dict[str, tuple[str, str]] = {
    "wb97x-d4": ("wb97x-v", "wb97x"),
    "wb97x-d4rev": ("wb97x-v", "wb97x-rev"),
    "wb97x-3c": ("wb97x-v", "wb97x-3c"),
}

# Methods where the Hessian should "fall up" to wB97X-V (with VV10) for
# analytical second derivatives, avoiding expensive finite-difference D4
# Hessians.  The VV10 Hessian is analytical in PySCF.
_HESSIAN_FALLBACK: dict[str, str] = {
    "wb97x-d4": "wb97x-v",
    "wb97x-d4rev": "wb97x-v",
    "wb97x-3c": "wb97x-v",
}


def _register_d4_composites() -> None:
    """Register wB97X-V+D4 composite methods with PySCF's dispersion dispatch.

    PySCF's libxc.parse_xc uses pyscf.scf.dispersion.parse_dft to extract
    the base XC code from method strings containing '-D3' or '-D4'.  We add
    entries to the internal _white_list and XC_MAP so that:
      - The XC functional evaluates as wB97X-V (libxc 466)
      - VV10 nonlocal correlation is disabled (nlc=False)
      - D4 dispersion is requested
      - The correct dftd4 parameter name is used for the D4 calculation

    Each composite is registered under an internal name with a '-d4' suffix
    (e.g. 'wb97x-3c-d4') so that libxc.parse_xc recognizes the '-D4' token
    and delegates to parse_dft.
    """
    from pyscf.scf import dispersion

    for _user_name, (_xc, d4_param) in _D4_COMPOSITES.items():
        internal = f"{d4_param}-d4"
        dispersion._white_list[internal] = (_xc, False, "d4")
        dispersion.XC_MAP[internal] = d4_param
    dispersion.parse_dft.cache_clear()
    dispersion.parse_disp.cache_clear()


_register_d4_composites()


def _resolve_method(method: str) -> tuple[str, str | None]:
    """Resolve a user-facing method name to a PySCF xc string.

    For D4 composite methods (wB97X-D4, wB97X-D4rev, wB97X-3c), returns
    the internal name registered with PySCF's dispersion dispatch.

    Returns (xc_string, d4_param_name_or_None).
    """
    key = method.lower().replace("_", "-")
    if key in _D4_COMPOSITES:
        _xc, d4_param = _D4_COMPOSITES[key]
        # Internal name: e.g. "wb97x-3c" -> "wb97x-3c-d4"
        return f"{d4_param}-d4", d4_param
    return method, None


def _resolve_basis(basis: str, elements: list[str]) -> str | dict[str, Any]:
    """Resolve a basis set name, loading from basis-set-exchange if needed.

    PySCF doesn't include vDZP natively; we load it from BSE.
    """
    if basis.lower() == "vdzp":
        import basis_set_exchange as bse
        from pyscf.gto.basis import parse_nwchem

        nwchem_str = bse.get_basis("Grimme vDZP", fmt="nwchem")
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".nw", delete=False) as f:
            f.write(nwchem_str)
            tmppath = f.name

        import os

        try:
            unique_elements = sorted(set(elements))
            basis_dict: dict[str, Any] = {}
            for elem in unique_elements:
                basis_dict[elem] = parse_nwchem.load(tmppath, elem)
            return basis_dict
        finally:
            os.unlink(tmppath)
    return basis


def _build_mf(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
) -> tuple[Any, Any]:
    """Build a PySCF Mole and mean-field object with the requested settings.

    Returns (mol, mf) where mf is ready for .kernel().
    """
    from pyscf import dft, gto

    xc_string, _d4_param = _resolve_method(method)
    resolved_basis = _resolve_basis(basis, list(geom.symbols))

    mol = gto.Mole()
    mol.atom = [
        (sym, tuple(float(x) for x in coord))
        for sym, coord in zip(geom.symbols, geom.coords, strict=True)
    ]
    mol.basis = resolved_basis
    mol.charge = charge
    mol.spin = geom.n_electrons(charge) % 2  # 0 for singlet, 1 for doublet
    mol.verbose = 3
    mol.build()

    # Use RKS for closed-shell, UKS for open-shell
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)

    # Set the XC functional.  For most methods PySCF parses the string
    # natively (e.g. "wB97X-D3BJ" dispatches D3BJ via pyscf-dispersion,
    # "wB97M-V" enables VV10 via libxc).  For D4 composites the string
    # was rewritten by _resolve_method to the registered internal name.
    mf.xc = xc_string

    # Integration grid: (99, 590) for all DFT calculations
    mf.grids.atom_grid = (99, 590)
    mf.grids.prune = None

    # VV10 NLC grid for functionals with active nonlocal correlation.
    # D4 composites disable VV10 (nlc=False in the registration), so
    # they won't reach this branch.
    method_lower = method.lower().replace("_", "-")
    if method_lower in _VV10_XC:
        mf.nlcgrids.atom_grid = (50, 194)
        mf.nlcgrids.prune = None

    # Implicit solvent
    if solvent_model is not None and solvent is not None:
        from pyscf import solvent as pyscf_solvent

        solvent_model_upper = solvent_model.upper()
        if solvent_model_upper == "SMD":
            mf = pyscf_solvent.SMD(mf)
            mf.with_solvent.solvent = solvent
        elif solvent_model_upper in ("DDCOSMO", "COSMO", "CPCM"):
            mf = pyscf_solvent.ddCOSMO(mf)
            mf.with_solvent.eps = _solvent_dielectric(solvent)
        else:
            raise ValueError(f"Unknown PySCF solvent model: {solvent_model!r}")

    return mol, mf


def single_point(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
) -> float:
    """Run a single-point DFT energy calculation.

    Returns the total energy in Hartree.
    """
    _mol, mf = _build_mf(geom, charge, method, basis, solvent_model, solvent)
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for method={method}, basis={basis}")
    return float(energy)


def optimize(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
) -> tuple[Geometry, float]:
    """Run DFT geometry optimization.

    Returns (optimized_geometry, final_energy_hartree).
    """
    from pyscf.geomopt.geometric_solver import optimize as geom_optimize

    _mol, mf = _build_mf(geom, charge, method, basis, solvent_model, solvent)

    # geometric writes temporary files; use a temp dir
    with tempfile.TemporaryDirectory(prefix="pyscf_opt_") as tmpdir:
        mol_opt = geom_optimize(mf, maxsteps=200, tmpdir=tmpdir)

    # Extract optimized coordinates (PySCF stores in Bohr)
    coords_bohr: NDArray[np.float64] = np.asarray(mol_opt.atom_coords(), dtype=np.float64)
    coords_ang = coords_bohr * BOHR_TO_ANG
    symbols = tuple(mol_opt.elements)

    opt_geom = Geometry(symbols=symbols, coords=coords_ang)

    # Final energy from the converged SCF on the optimized geometry
    energy = float(mf.e_tot)

    return opt_geom, energy


def frequencies(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
) -> list[float]:
    """Compute harmonic vibrational frequencies.

    For D4-based methods (wB97X-D4, wB97X-D4rev, wB97X-3c), the Hessian
    "falls up" to wB97X-V (with analytical VV10) to avoid expensive
    finite-difference D4 Hessians.  The vibrational frequencies are
    insensitive to the choice of dispersion correction.

    Returns frequencies in cm⁻¹ (including imaginary frequencies as negative values).
    """
    method_lower = method.lower().replace("_", "-")
    hess_method = _HESSIAN_FALLBACK.get(method_lower, method)
    if hess_method != method:
        log.info(
            f"Hessian: falling up from {method} to {hess_method} for analytical second derivatives"
        )

    mol, mf = _build_mf(geom, charge, hess_method, basis, solvent_model, solvent)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for method={hess_method}, basis={basis}")

    hessian = mf.Hessian().kernel()

    # Reshape from (natm, natm, 3, 3) to (3*natm, 3*natm)
    n_atoms = mol.natm
    hess_2d = hessian.transpose(0, 2, 1, 3).reshape(3 * n_atoms, 3 * n_atoms)

    return _harmonic_analysis(mol, hess_2d)


def _harmonic_analysis(mol: Any, hess: NDArray[np.float64]) -> list[float]:
    """Diagonalize mass-weighted Hessian and return frequencies in cm⁻¹.

    Follows standard normal mode analysis:
    1. Mass-weight the Hessian
    2. Diagonalize
    3. Convert eigenvalues to frequencies
    """
    # Atomic masses in amu
    masses = np.array([mol.atom_mass_list()[i] for i in range(mol.natm)])

    # Build mass-weighting matrix: 1/sqrt(m_i) for each coordinate
    mass_weights = np.repeat(1.0 / np.sqrt(masses), 3)
    mass_weighted_hess = np.outer(mass_weights, mass_weights) * hess

    # Diagonalize
    eigenvalues = np.linalg.eigvalsh(mass_weighted_hess)

    # Convert eigenvalues to frequencies in cm⁻¹
    # eigenvalue is in Hartree/(Bohr² * amu)
    # frequency = sqrt(eigenvalue) / (2*pi*c) converted to cm⁻¹
    hartree_to_joule = 4.3597447222071e-18
    bohr_to_meter = 5.29177210903e-11
    amu_to_kg = 1.66053906660e-27
    speed_of_light = 2.99792458e10  # cm/s

    # Convert eigenvalue units: Hartree/(Bohr² * amu) -> J/(m² * kg) = 1/s²
    conv = hartree_to_joule / (bohr_to_meter**2 * amu_to_kg)

    freqs: list[float] = []
    for ev in sorted(eigenvalues):
        ev_si = ev * conv
        if ev_si < 0:
            freq = -np.sqrt(-ev_si) / (2.0 * np.pi * speed_of_light)
        else:
            freq = np.sqrt(ev_si) / (2.0 * np.pi * speed_of_light)
        freqs.append(float(freq))

    return freqs


# Common solvent dielectric constants
_DIELECTRIC: dict[str, float] = {
    "water": 78.39,
    "methanol": 32.7,
    "ethanol": 24.55,
    "dmso": 46.7,
    "acetonitrile": 37.5,
    "thf": 7.58,
    "toluene": 2.38,
    "chloroform": 4.81,
    "dichloromethane": 8.93,
    "hexane": 1.88,
    "acetone": 20.7,
}


def _solvent_dielectric(solvent: str) -> float:
    """Look up dielectric constant for a solvent name."""
    key = solvent.lower()
    if key not in _DIELECTRIC:
        raise ValueError(
            f"Unknown solvent {solvent!r} for dielectric lookup. "
            f"Known solvents: {', '.join(sorted(_DIELECTRIC))}"
        )
    return _DIELECTRIC[key]
