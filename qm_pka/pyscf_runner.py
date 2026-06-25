"""DFT calculations using PySCF as a Python library.

Provides geometry optimization, single-point energy, and frequency
calculations. Uses an (99, 590) integration grid for all DFT calculations
and a (50, 194) NLC grid for VV10-containing functionals, with NWChem
region-based grid pruning.

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


def _patch_pcm_ecp_cavity() -> None:
    """Correct PySCF's PCM cavity radii lookup for ECP atoms.

    ``pyscf.solvent.pcm.gen_surface`` indexes the inter-atomic switching-function
    vdW radii by ``mol.atom_charges()``, which an ECP *reduces* (e.g. oxygen
    8 -> 6).  ECP atoms therefore get the wrong element's radius (oxygen gets
    carbon's 1.70 A instead of 1.52).  The per-sphere radius already uses the
    true nuclear charge and is correct; only the switching radii are affected.

    Because vDZP places an ECP on every non-H atom, this corrupts the solvent
    cavity of every heavy atom in every PySCF implicit-solvent calculation
    (shifting solvation energies by ~0.26 kcal/mol on water).  We wrap
    ``gen_surface`` to expose true nuclear charges for the duration of the call.
    Every caller (PCM and SMD) reaches the function via ``pcm.gen_surface``, so a
    single reassignment covers them all.  gpu4pyscf has its own, already-correct
    ``gen_surface`` and is untouched.  This is the interim fix for an upstream
    PySCF bug (its line should use ``gto.charge`` like the per-sphere loop does).
    """
    import numpy
    import pyscf.solvent.pcm as pcm
    from pyscf import gto

    if getattr(pcm.gen_surface, "_ecp_cavity_patched", False):
        return  # idempotent: never double-wrap

    _orig_gen_surface = pcm.gen_surface

    def gen_surface(mol: Any, *args: Any, **kwargs: Any) -> Any:
        true_z = numpy.array([gto.charge(mol.atom_symbol(ia)) for ia in range(mol.natm)])
        had_attr = "atom_charges" in mol.__dict__
        saved = mol.__dict__.get("atom_charges")
        mol.atom_charges = lambda: true_z  # shadow the ECP-reduced method
        try:
            return _orig_gen_surface(mol, *args, **kwargs)
        finally:
            if had_attr:
                mol.atom_charges = saved
            else:
                del mol.atom_charges

    gen_surface._ecp_cavity_patched = True  # type: ignore[attr-defined]
    gen_surface._ecp_cavity_original = _orig_gen_surface  # type: ignore[attr-defined]
    pcm.gen_surface = gen_surface


_patch_pcm_ecp_cavity()


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


def _resolve_basis(
    basis: str, elements: list[str]
) -> tuple[str | dict[str, Any], dict[str, Any] | None]:
    """Resolve a basis set name, loading from basis-set-exchange if needed.

    PySCF doesn't include vDZP natively; we load it from BSE.  vDZP is an
    ECP-designed basis (heavier atoms have no core functions), so we must also
    load its per-element effective core potentials and apply them via
    ``mol.ecp`` — without them PySCF runs an all-electron SCF on a core-
    incapable basis and the energy is meaningless.

    Returns ``(basis, ecp)`` where ``ecp`` is ``None`` for built-in bases or a
    ``{element: ecp_data}`` dict for vDZP (elements with no core, e.g. H, are
    omitted).
    """
    if basis.lower() == "vdzp":
        import os
        import tempfile

        import basis_set_exchange as bse
        from pyscf.gto.basis import parse_nwchem

        nwchem_str = bse.get_basis("Grimme vDZP", fmt="nwchem")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".nw", delete=False) as f:
            f.write(nwchem_str)
            tmppath = f.name

        try:
            unique_elements = sorted(set(elements))
            basis_dict: dict[str, Any] = {}
            ecp_dict: dict[str, Any] = {}
            for elem in unique_elements:
                basis_dict[elem] = parse_nwchem.load(tmppath, elem)
                # vDZP replaces the core of heavier atoms with an ECP; light
                # atoms (H, He) have none, for which load_ecp returns empty.
                try:
                    ecp_data = parse_nwchem.load_ecp(tmppath, elem)
                except Exception:
                    ecp_data = None
                if ecp_data:
                    ecp_dict[elem] = ecp_data
            return basis_dict, (ecp_dict or None)
        finally:
            os.unlink(tmppath)

    # def2-family bases use effective core potentials for Z >= 37 (Rb onward).
    # PySCF does NOT auto-apply them from mol.basis, so an all-electron SCF on
    # a core-incapable basis would result (the same failure mode as vDZP).
    # Request the matching def2-ECP by name, but only for the heavy elements
    # that actually define one, so light-element calculations are untouched.
    if basis.lower().startswith("def2"):
        from pyscf import gto

        ecp_dict = {elem: basis for elem in sorted(set(elements)) if gto.charge(elem) >= 37}
        if ecp_dict:
            return basis, ecp_dict

    return basis, None


def _build_mf(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
) -> tuple[Any, Any]:
    """Build a PySCF Mole and mean-field object with the requested settings.

    Returns (mol, mf) where mf is ready for .kernel().
    """
    from pyscf import dft, gto, lib

    lib.num_threads(threads)

    xc_string, _d4_param = _resolve_method(method)
    resolved_basis, resolved_ecp = _resolve_basis(basis, list(geom.symbols))

    mol = gto.Mole()
    mol.atom = [
        (sym, tuple(float(x) for x in coord))
        for sym, coord in zip(geom.symbols, geom.coords, strict=True)
    ]
    mol.basis = resolved_basis
    if resolved_ecp is not None:
        mol.ecp = resolved_ecp
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

    # Integration grid: (99, 590) for all DFT calculations, with NWChem
    # region-based pruning (PySCF's default scheme; the closest analog to
    # Psi4's ROBUST used in the Psi4 backend).
    mf.grids.atom_grid = (99, 590)
    mf.grids.prune = dft.gen_grid.nwchem_prune

    # VV10 NLC grid for functionals with active nonlocal correlation.
    # D4 composites disable VV10 (nlc=False in the registration), so
    # they won't reach this branch.
    method_lower = method.lower().replace("_", "-")
    if method_lower in _VV10_XC:
        mf.nlcgrids.atom_grid = (50, 194)
        mf.nlcgrids.prune = dft.gen_grid.nwchem_prune

    # Implicit solvent.  Prefer the analytical PCM (with proper gradients)
    # over the experimental ddPCM/ddCOSMO domain-decomposed variants.
    if solvent_model is not None and solvent is not None:
        from pyscf import solvent as pyscf_solvent

        solvent_model_upper = solvent_model.upper().replace("-", "")
        # Map user-facing names to the analytical PCM "method" string.
        pcm_method_map = {
            "PCM": "IEF-PCM",
            "IEFPCM": "IEF-PCM",
            "CPCM": "C-PCM",
            "COSMO": "COSMO",
            "SSVPE": "SS(V)PE",
        }
        if solvent_model_upper == "SMD":
            mf = pyscf_solvent.SMD(mf)
            mf.with_solvent.solvent = solvent
        elif solvent_model_upper in pcm_method_map:
            mf = pyscf_solvent.PCM(mf)
            mf.with_solvent.method = pcm_method_map[solvent_model_upper]
            mf.with_solvent.eps = _solvent_dielectric(solvent)
            # Cavity radii: PySCF's default is modified_Bondi (H=1.10); override
            # the hydrogen radius with the user value.  radii_table holds the
            # final (already vdw_scale-scaled) radii, so replicate that formula.
            # With pcm_hydrogen_radius=1.10 this reproduces the default exactly.
            import pyscf.solvent.pcm as _pcm
            from pyscf.data import radii as _radii

            ws = mf.with_solvent
            rad = _pcm.modified_Bondi.copy()
            rad[1] = pcm_hydrogen_radius / _radii.BOHR
            ws.radii_table = ws.vdw_scale * rad + ws.r_probe / _radii.BOHR
        elif solvent_model_upper == "DDPCM":
            mf = pyscf_solvent.ddPCM(mf)
            mf.with_solvent.eps = _solvent_dielectric(solvent)
        elif solvent_model_upper == "DDCOSMO":
            mf = pyscf_solvent.ddCOSMO(mf)
            mf.with_solvent.eps = _solvent_dielectric(solvent)
        else:
            raise ValueError(f"Unknown PySCF solvent model: {solvent_model!r}")

    # SCF robustness: allow more iterations for difficult cases (especially
    # in implicit solvent).  Default is 50; bump to 100.
    mf.max_cycle = 100

    return mol, mf


def _run_scf_robust(mf: Any) -> float:
    """Run SCF; on non-convergence, retry with second-order SCF (.newton())."""
    energy = mf.kernel()
    if mf.converged:
        return float(energy)

    log.warning("SCF did not converge; retrying with second-order SCF (Newton)")
    dm = mf.make_rdm1()
    mf_newton = mf.newton()
    mf_newton.max_cycle = 100
    energy = mf_newton.kernel(dm0=dm)
    if not mf_newton.converged:
        raise RuntimeError("PySCF SCF failed to converge even with Newton solver")
    # Copy converged density back so any subsequent operations see it
    mf.converged = True
    mf.e_tot = mf_newton.e_tot
    mf.mo_coeff = mf_newton.mo_coeff
    mf.mo_occ = mf_newton.mo_occ
    mf.mo_energy = mf_newton.mo_energy
    return float(energy)


def single_point(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
) -> float:
    """Run a single-point DFT energy calculation.

    Returns the total energy in Hartree.
    """
    _mol, mf = _build_mf(
        geom, charge, method, basis, solvent_model, solvent, pcm_hydrogen_radius, threads
    )
    return _run_scf_robust(mf)


def optimize(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
) -> tuple[Geometry, float, bool]:
    """Run DFT geometry optimization.

    Strategy: first try the default TRIC internal coordinates (100 steps).
    If that doesn't converge, retry with Cartesian coordinates for another
    100 steps starting from the last internal-coords geometry.  This is
    analogous to Psi4 optking's dynamic_level mechanism.

    Returns (optimized_geometry, final_energy_hartree, converged).
    The geometry and energy are always the latest ones, regardless of
    convergence status.
    """
    from pyscf.geomopt.geometric_solver import kernel as geom_kernel

    def _run(start_geom: Geometry, coordsys: str, maxsteps: int) -> tuple[Geometry, float, bool]:
        _mol, mf_local = _build_mf(
            start_geom,
            charge,
            method,
            basis,
            solvent_model,
            solvent,
            pcm_hydrogen_radius,
            threads,
        )
        _run_scf_robust(mf_local)
        with tempfile.TemporaryDirectory(prefix="pyscf_opt_") as tmpdir:
            conv, mol_opt = geom_kernel(
                mf_local, maxsteps=maxsteps, tmpdir=tmpdir, coordsys=coordsys
            )
        coords_ang = np.asarray(mol_opt.atom_coords(), dtype=np.float64) * BOHR_TO_ANG
        out_geom = Geometry(symbols=tuple(mol_opt.elements), coords=coords_ang)
        return out_geom, float(mf_local.e_tot), bool(conv)

    # First attempt: default TRIC (internal coordinates)
    opt_geom, energy, converged = _run(geom, coordsys="tric", maxsteps=100)
    if converged:
        return opt_geom, energy, True

    # Fallback: continue from the last geometry in Cartesian coordinates
    log.info(
        "Internal-coord opt did not converge in 100 steps; "
        "retrying from last geometry with Cartesian coords"
    )
    opt_geom, energy, converged = _run(opt_geom, coordsys="cart", maxsteps=100)
    return opt_geom, energy, converged


def frequencies(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
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

    mol, mf = _build_mf(
        geom, charge, hess_method, basis, solvent_model, solvent, pcm_hydrogen_radius, threads
    )
    _run_scf_robust(mf)

    hessian = mf.Hessian().kernel()

    # Use PySCF's built-in harmonic analysis, which projects out
    # translational and rotational modes before diagonalization.
    from pyscf.hessian.thermo import harmonic_analysis

    results = harmonic_analysis(mol, hessian, imaginary_freq=False)
    freqs_cm: Any = results["freq_wavenumber"]
    return [float(f) for f in freqs_cm]


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


# Schoenflies point group -> rotational symmetry number sigma (order of proper
# rotation subgroup).  PySCF labels linear molecules "Coov"/"Dooh" and single
# atoms "SO3".
_SIGMA_ROT_TABLE: dict[str, int] = {
    "C1": 1,
    "Cs": 1,
    "Ci": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
    "C6": 6,
    "C7": 7,
    "C8": 8,
    "C2v": 2,
    "C3v": 3,
    "C4v": 4,
    "C5v": 5,
    "C6v": 6,
    "C7v": 7,
    "C8v": 8,
    "C2h": 2,
    "C3h": 3,
    "C4h": 4,
    "C5h": 5,
    "C6h": 6,
    "D2": 4,
    "D3": 6,
    "D4": 8,
    "D5": 10,
    "D6": 12,
    "D7": 14,
    "D8": 16,
    "D2h": 4,
    "D3h": 6,
    "D4h": 8,
    "D5h": 10,
    "D6h": 12,
    "D7h": 14,
    "D8h": 16,
    "D2d": 4,
    "D3d": 6,
    "D4d": 8,
    "D5d": 10,
    "D6d": 12,
    "S4": 2,
    "S6": 3,
    "S8": 4,
    "T": 12,
    "Td": 12,
    "Th": 12,
    "O": 24,
    "Oh": 24,
    "I": 60,
    "Ih": 60,
    "Coov": 1,  # linear heteronuclear (e.g. CO, HCN)
    "Dooh": 2,  # linear homonuclear / symmetric linear (e.g. N2, CO2)
    "SO3": 1,  # single atom
}


def rotational_symmetry_number(geom: Geometry) -> int:
    """Detect the rotational symmetry number sigma_rot from the 3D geometry.

    Uses `pyscf.symm.detect_symm` to get the Schoenflies label, then looks
    up sigma for the full point group.
    """
    from pyscf import symm

    atoms = [
        (sym, tuple(float(x) for x in coord))
        for sym, coord in zip(geom.symbols, geom.coords, strict=True)
    ]
    label, _origin, _axes = symm.detect_symm(atoms)
    sigma = _SIGMA_ROT_TABLE.get(label)
    if sigma is None:
        log.warning(f"Unknown PySCF point group label {label!r}; defaulting sigma_rot=1")
        return 1
    return sigma


def _solvent_dielectric(solvent: str) -> float:
    """Look up dielectric constant for a solvent name."""
    key = solvent.lower()
    if key not in _DIELECTRIC:
        raise ValueError(
            f"Unknown solvent {solvent!r} for dielectric lookup. "
            f"Known solvents: {', '.join(sorted(_DIELECTRIC))}"
        )
    return _DIELECTRIC[key]
