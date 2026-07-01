"""DFT calculations using Psi4 as a subprocess.

Runs Psi4 via its command-line interface, writing input files and parsing
output. Uses an (99, 590) integration grid for all DFT calculations
and a (50, 194) NLC grid for VV10-containing functionals, with Psi4's
recommended ROBUST grid pruning.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from qm_pka.types import Geometry
from qm_pka.xyz_io import read_xyz

log = logging.getLogger(__name__)

# Functionals that include VV10 nonlocal correlation
_VV10_FUNCTIONALS = {"wb97m-v", "wb97m_v", "wb97x-v", "wb97x_v", "b97m-v", "b97m_v"}

# PCM cavity radii (Angstrom), copied IN FULL from pyscf.solvent.pcm.modified_Bondi
# (PySCF's "vdw from ASE" composite set: Bondi 1964 + Mantina 2009 + others, with
# hydrogen forced to 1.10) so both backends build the same cavity for any element.
# Hardcoded rather than imported so the Psi4 backend keeps no runtime dependency on
# PySCF; a test cross-checks every value against pyscf.solvent.pcm.modified_Bondi.
_MODIFIED_BONDI_RADII: dict[str, float] = {
    "H": 1.1,
    "He": 1.4,
    "Li": 1.82,
    "Be": 1.53,
    "B": 1.92,
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Ne": 1.54,
    "Na": 2.27,
    "Mg": 1.73,
    "Al": 1.84,
    "Si": 2.1,
    "P": 1.8,
    "S": 1.8,
    "Cl": 1.75,
    "Ar": 1.88,
    "K": 2.75,
    "Ca": 2.31,
    "Sc": 2.0,
    "Ti": 2.0,
    "V": 2.0,
    "Cr": 2.0,
    "Mn": 2.0,
    "Fe": 2.0,
    "Co": 2.0,
    "Ni": 1.63,
    "Cu": 1.4,
    "Zn": 1.39,
    "Ga": 1.87,
    "Ge": 2.11,
    "As": 1.85,
    "Se": 1.9,
    "Br": 1.85,
    "Kr": 2.02,
    "Rb": 3.03,
    "Sr": 2.49,
    "Y": 2.0,
    "Zr": 2.0,
    "Nb": 2.0,
    "Mo": 2.0,
    "Tc": 2.0,
    "Ru": 2.0,
    "Rh": 2.0,
    "Pd": 1.63,
    "Ag": 1.72,
    "Cd": 1.58,
    "In": 1.93,
    "Sn": 2.17,
    "Sb": 2.06,
    "Te": 2.06,
    "I": 1.98,
    "Xe": 2.16,
    "Cs": 3.43,
    "Ba": 2.49,
    "La": 2.0,
    "Ce": 2.0,
    "Pr": 2.0,
    "Nd": 2.0,
    "Pm": 2.0,
    "Sm": 2.0,
    "Eu": 2.0,
    "Gd": 2.0,
    "Tb": 2.0,
    "Dy": 2.0,
    "Ho": 2.0,
    "Er": 2.0,
    "Tm": 2.0,
    "Yb": 2.0,
    "Lu": 2.0,
    "Hf": 2.0,
    "Ta": 2.0,
    "W": 2.0,
    "Re": 2.0,
    "Os": 2.0,
    "Ir": 2.0,
    "Pt": 1.75,
    "Au": 1.66,
    "Hg": 1.55,
    "Tl": 1.96,
    "Pb": 2.02,
    "Bi": 2.07,
    "Po": 1.97,
    "At": 2.02,
    "Rn": 2.2,
    "Fr": 3.48,
    "Ra": 2.83,
    "Ac": 2.0,
    "Th": 2.0,
    "Pa": 2.0,
    "U": 1.86,
    "Np": 2.0,
    "Pu": 2.0,
    "Am": 2.0,
    "Cm": 2.0,
    "Bk": 2.0,
    "Cf": 2.0,
    "Es": 2.0,
    "Fm": 2.0,
    "Md": 2.0,
    "No": 2.0,
    "Lr": 2.0,
}
# PCMSolver applies no scaling in Mode=Explicit, so the emitted radii are
# pre-scaled by the same factor PySCF uses (vdw_scale) and Psi4's Bondi+Scaling
# default alpha.
_PCM_RADII_SCALING = 1.2


def _molecule_block(geom: Geometry, charge: int) -> str:
    """Build a Psi4 molecule block from a Geometry and charge."""
    mult = geom.multiplicity(charge)
    lines = [f"  {charge} {mult}"]
    for sym, (x, y, z) in zip(geom.symbols, geom.coords, strict=True):
        lines.append(f"  {sym}  {x: .10f}  {y: .10f}  {z: .10f}")
    lines.append("  units angstrom")
    lines.append("  no_reorient")
    lines.append("  no_com")
    return "\n".join(lines)


def _options_block(
    method: str,
    basis: str,
    solvent_model: str | None,
) -> str:
    """Build the Psi4 set block with DFT grid and other options."""
    method_lower = method.lower().replace("_", "-")

    opts = [
        f"  basis {basis}",
        "  dft_radial_points 99",
        "  dft_spherical_points 590",
        # ROBUST: Psi4's recommended region-based pruning (Bragg-Slater
        # radius zones).  Single keyword; Psi4 has no separate VV10 control.
        "  dft_pruning_scheme robust",
    ]

    # VV10 NLC grid
    if method_lower in _VV10_FUNCTIONALS:
        opts.extend(
            [
                "  dft_vv10_radial_points 50",
                "  dft_vv10_spherical_points 194",
            ]
        )

    # SCF convergence
    opts.append("  e_convergence 1e-8")
    opts.append("  d_convergence 1e-8")

    # PCM solvent
    if solvent_model is not None:
        opts.append("  pcm true")
        opts.append("  pcm_scf_type total")

    return "\n".join(opts)


def _bse_basis_block(basis: str, geom: Geometry) -> str:
    """If `basis` should be loaded from basis-set-exchange, return a
    Psi4 input snippet that registers it via `basis_helper`. Otherwise "".

    Currently only vDZP is loaded from BSE: Psi4's bundled vdzp.gbs is
    missing fluorine (psi4/psi4#3205); the BSE copy carries the workaround
    (a vanishingly-small d-type ECP projector on F).
    """
    if basis.lower() != "vdzp":
        return ""
    import basis_set_exchange as bse

    elements = sorted(set(geom.symbols))
    psi4_str = bse.get_basis("Grimme vDZP", fmt="psi4", elements=elements)
    return f'\nbasis_helper("""\n{psi4_str}\n""", name="vDZP", set_option=True)\n'


def _pcm_block(
    solvent_model: str,
    solvent: str,
    geom: Geometry,
    pcm_hydrogen_radius: float = 1.1,
) -> str:
    """Build the Psi4 PCM section with modified-Bondi cavity radii.

    Psi4's PCMSolver only honours per-atom radii via ``Mode = Explicit`` (the
    block-level ``Mode = Atoms`` override is silently dropped in the embedded
    host integration), so we emit the full sphere list ourselves: one sphere per
    atom at its coordinates with radius ``scaling * modified_Bondi``, except
    hydrogen uses ``pcm_hydrogen_radius`` (default 1.10).  Explicit mode applies
    no scaling, so the radii are pre-scaled.

    The spheres are pinned to ``geom``'s coordinates, which is correct for a
    fixed-geometry ``single_point`` (the input frame is preserved by
    ``no_reorient``/``no_com``).  Psi4 has no analytical PCM gradients, so
    optimisation/frequencies in implicit solvent are unsupported regardless.
    """
    spheres = []
    for sym, (x, y, z) in zip(geom.symbols, geom.coords, strict=True):
        if sym == "H":
            base = pcm_hydrogen_radius
        else:
            try:
                base = _MODIFIED_BONDI_RADII[sym]
            except KeyError:
                raise ValueError(
                    f"No modified-Bondi radius for element {sym!r}; add it to "
                    "_MODIFIED_BONDI_RADII (must match pyscf modified_Bondi)."
                ) from None
        radius = base * _PCM_RADII_SCALING
        spheres.append(f"      {x:.10f}, {y:.10f}, {z:.10f}, {radius:.6f}")
    lines = [
        "pcm = {",
        "  Units = Angstrom",
        "  Medium {",
        f"    SolverType = {solvent_model}",
        f"    Solvent = {_psi4_solvent_name(solvent)}",
        "  }",
        "  Cavity {",
        "    Type = GePol",
        "    Area = 0.1",
        "    Mode = Explicit",
        "    Spheres = [",
        ",\n".join(spheres),
        "    ]",
        "  }",
        "}",
    ]
    return "\n".join(lines)


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
    mol_block = _molecule_block(geom, charge)
    opts_block = _options_block(method, basis, solvent_model)

    input_text = f"""\
molecule mol {{
{mol_block}
}}

set {{
{opts_block}
}}
"""
    if solvent_model is not None and solvent is not None:
        input_text += "\n" + _pcm_block(solvent_model, solvent, geom, pcm_hydrogen_radius) + "\n"

    input_text += _bse_basis_block(basis, geom)

    input_text += f"""
E = energy('{method}')
psi4.print_out(f'\\n=== FINAL ENERGY: {{E:.12f}} ===\\n')
"""

    output, _work_dir = _run_psi4(input_text, threads=threads)
    return _parse_final_energy(output)


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

    optking's dynamic_level=1 allows the optimizer to adaptively change
    parameters (coordinate system, trust radius, etc.) when progress stalls.

    Returns (optimized_geometry, final_energy_hartree, converged). The
    geometry and energy are always the latest ones, even if optking did not
    fully converge.
    """
    mol_block = _molecule_block(geom, charge)
    opts_block = _options_block(method, basis, solvent_model)

    input_text = f"""\
molecule mol {{
{mol_block}
}}

set {{
{opts_block}
  geom_maxiter 200
  dynamic_level 1
  g_convergence gau
}}
"""
    if solvent_model is not None and solvent is not None:
        input_text += "\n" + _pcm_block(solvent_model, solvent, geom, pcm_hydrogen_radius) + "\n"

    input_text += _bse_basis_block(basis, geom)

    # Wrap optimize() so that a non-convergence exception still yields a
    # geometry and energy on disk (from the last optimizer step).  Two failure
    # modes are treated the same way: psi4's OptimizationConvergenceError
    # (geom_maxiter reached) and optking's "Maximum dynamic_level reached"
    # OptError (adaptive recovery exhausted).  Both leave a usable last-step
    # geometry in the active molecule; unlike OptimizationConvergenceError the
    # OptError carries no wfn, so we recover the energy with a single-point on
    # that geometry.  Any other failure (e.g. SCF non-convergence) propagates.
    input_text += f"""
from optking.exceptions import OptError
converged = True
try:
    E, wfn = optimize('{method}', return_wfn=True)
except psi4.OptimizationConvergenceError as exc:
    converged = False
    wfn = exc.wfn
    E = wfn.energy()
    psi4.print_out('\\n=== OPT NOT CONVERGED: using last geometry ===\\n')
except OptError as exc:
    if 'dynamic_level' not in str(exc):
        raise
    converged = False
    E, wfn = energy('{method}', return_wfn=True)
    psi4.print_out('\\n=== OPT NOT CONVERGED (dynamic_level): using last geometry ===\\n')
psi4.print_out(f'\\n=== FINAL ENERGY: {{E:.12f}} ===\\n')
psi4.print_out(f'\\n=== CONVERGED: {{1 if converged else 0}} ===\\n')
wfn.molecule().save_xyz_file('optimized.xyz', True)
"""

    output, work_dir = _run_psi4(input_text, threads=threads)
    energy = _parse_final_energy(output)
    converged = _parse_converged_flag(output)
    opt_geom = read_xyz(work_dir / "optimized.xyz")

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

    Returns frequencies in cm⁻¹ (imaginary frequencies as negative values).
    """
    mol_block = _molecule_block(geom, charge)
    opts_block = _options_block(method, basis, solvent_model)

    input_text = f"""\
molecule mol {{
{mol_block}
}}

set {{
{opts_block}
}}
"""
    if solvent_model is not None and solvent is not None:
        input_text += "\n" + _pcm_block(solvent_model, solvent, geom, pcm_hydrogen_radius) + "\n"

    input_text += _bse_basis_block(basis, geom)

    input_text += f"""
E, wfn = frequency('{method}', return_wfn=True)
freqs = wfn.frequencies().to_array()
with open('frequencies.dat', 'w') as f:
    for freq in freqs:
        f.write(f'{{freq:.6f}}\\n')
psi4.print_out(f'\\n=== FINAL ENERGY: {{E:.12f}} ===\\n')
"""

    _output, work_dir = _run_psi4(input_text, threads=threads)

    freq_list: list[float] = []
    for line in (work_dir / "frequencies.dat").read_text().splitlines():
        stripped = line.strip()
        if stripped:
            freq_list.append(float(stripped))

    return freq_list


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_psi4(
    input_text: str,
    timeout: int = 86400,
    threads: int = 1,
) -> tuple[str, Path]:
    """Write a Psi4 input file, run psi4, return (output_text, work_dir).

    The work_dir is a temporary directory that persists so callers can
    read output files (optimized.xyz, frequencies.dat, etc.).
    """
    work_dir = Path(tempfile.mkdtemp(prefix="psi4_"))

    input_path = work_dir / "input.dat"
    output_path = work_dir / "output.dat"
    input_path.write_text(input_text)

    cmd = [
        "psi4",
        str(input_path),
        "-o",
        str(output_path),
        "-n",
        str(threads),
    ]

    log.info(f"Running Psi4 in {work_dir}")
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Psi4 writes to the output file, not stdout
    output_text = ""
    if output_path.exists():
        output_text = output_path.read_text()

    if result.returncode != 0:
        raise RuntimeError(
            f"Psi4 failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr[-2000:]}\n"
            f"output (last 2000 chars): {output_text[-2000:]}"
        )

    return output_text, work_dir


def _parse_final_energy(output: str) -> float:
    """Parse the energy from our sentinel line in Psi4 output."""
    match = re.search(r"=== FINAL ENERGY:\s+([-\d.]+)\s+===", output)
    if match is None:
        raise RuntimeError("Could not parse final energy from Psi4 output")
    return float(match.group(1))


def _parse_converged_flag(output: str) -> bool:
    """Parse the converged flag from our sentinel line in Psi4 output.

    If the flag is missing (older input templates), assume converged=True.
    """
    match = re.search(r"=== CONVERGED:\s+([01])\s+===", output)
    if match is None:
        return True
    return match.group(1) == "1"


# Psi4 expects specific solvent names
_PSI4_SOLVENT_NAMES: dict[str, str] = {
    "water": "Water",
    "methanol": "Methanol",
    "ethanol": "Ethanol",
    "dmso": "DMSO",
    "acetonitrile": "Acetonitrile",
    "thf": "THF",
    "toluene": "Toluene",
    "chloroform": "Chloroform",
    "dichloromethane": "DiChloroMethane",
    "hexane": "Hexane",
    "acetone": "Acetone",
}


def rotational_symmetry_number(geom: Geometry) -> int:
    """Detect the rotational symmetry number sigma_rot from the 3D geometry.

    Uses Psi4's built-in point-group detector via
    `Molecule.rotational_symmetry_number()`, which returns sigma for the full
    point group (not the abelian subgroup Psi4 uses for electronic structure).
    """
    import psi4  # type: ignore[import-untyped]

    xyz_lines = [
        f"{sym}  {x:.10f}  {y:.10f}  {z:.10f}"
        for sym, (x, y, z) in zip(geom.symbols, geom.coords, strict=True)
    ]
    mol = psi4.geometry("\n".join(xyz_lines) + "\nunits angstrom\nno_reorient\nno_com")
    mol.update_geometry()
    return int(mol.rotational_symmetry_number())


def _psi4_solvent_name(solvent: str) -> str:
    """Map a common solvent name to Psi4's expected format."""
    key = solvent.lower()
    if key not in _PSI4_SOLVENT_NAMES:
        return solvent
    return _PSI4_SOLVENT_NAMES[key]
