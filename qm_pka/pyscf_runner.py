"""DFT calculations using PySCF as a Python library.

Provides geometry optimization, single-point energy, and frequency
calculations. Uses an (99, 590) integration grid for all DFT calculations
and a (50, 194) NLC grid for VV10-containing functionals.
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

    mol = gto.Mole()
    mol.atom = [
        (sym, tuple(float(x) for x in coord))
        for sym, coord in zip(geom.symbols, geom.coords, strict=True)
    ]
    mol.basis = basis
    mol.charge = charge
    mol.spin = geom.n_electrons(charge) % 2  # 0 for singlet, 1 for doublet
    mol.verbose = 3
    mol.build()

    # Use RKS for closed-shell, UKS for open-shell
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)

    # PySCF parses the method string natively: "wB97X-D3BJ" sets the base XC
    # functional and auto-dispatches D3BJ via pyscf-dispersion. VV10 functionals
    # like "wB97M-V" are handled entirely within libxc.
    mf.xc = method

    # Integration grid: (99, 590) for all DFT calculations
    mf.grids.atom_grid = (99, 590)
    mf.grids.prune = None

    # VV10 NLC grid for functionals with nonlocal correlation
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

    Returns frequencies in cm⁻¹ (including imaginary frequencies as negative values).
    """
    mol, mf = _build_mf(geom, charge, method, basis, solvent_model, solvent)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PySCF SCF did not converge for method={method}, basis={basis}")

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
