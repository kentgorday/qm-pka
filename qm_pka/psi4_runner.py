"""DFT calculations using Psi4 as a subprocess.

Runs Psi4 via its command-line interface, writing input files and parsing
output. Uses an (99, 590) integration grid for all DFT calculations
and a (50, 194) NLC grid for VV10-containing functionals.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent

from qm_pka.types import Geometry
from qm_pka.xyz_io import read_xyz

log = logging.getLogger(__name__)

# Functionals that include VV10 nonlocal correlation
_VV10_FUNCTIONALS = {"wb97m-v", "wb97m_v", "wb97x-v", "wb97x_v", "b97m-v", "b97m_v"}


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


def _pcm_block(solvent_model: str, solvent: str) -> str:
    """Build the Psi4 PCM section."""
    return dedent(f"""\
        pcm = {{
          Units = Angstrom
          Medium {{
            SolverType = {solvent_model}
            Solvent = {_psi4_solvent_name(solvent)}
          }}
          Cavity {{
            RadiiSet = Bondi
            Type = GePol
            Scaling = True
            Area = 0.3
          }}
        }}""")


def single_point(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
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
        input_text += "\n" + _pcm_block(solvent_model, solvent) + "\n"

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
    threads: int = 1,
) -> tuple[Geometry, float]:
    """Run DFT geometry optimization.

    Returns (optimized_geometry, final_energy_hartree).
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
}}
"""
    if solvent_model is not None and solvent is not None:
        input_text += "\n" + _pcm_block(solvent_model, solvent) + "\n"

    input_text += f"""
E, wfn = optimize('{method}', return_wfn=True)
psi4.print_out(f'\\n=== FINAL ENERGY: {{E:.12f}} ===\\n')
wfn.molecule().save_xyz_file('optimized.xyz', True)
"""

    output, work_dir = _run_psi4(input_text, threads=threads)
    energy = _parse_final_energy(output)
    opt_geom = read_xyz(work_dir / "optimized.xyz")

    return opt_geom, energy


def frequencies(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
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
        input_text += "\n" + _pcm_block(solvent_model, solvent) + "\n"

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
