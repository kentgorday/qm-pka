"""Wrapper for CREST: conformer search, tautomerization, protonation/deprotonation.

CREST 2.12 evaluates GFN2-xTB by driving the external ``xtb`` binary as a
subprocess. We pin 2.12 because the CREST 3.x in-process rewrite collapses
conformer searches to a single structure on macOS (see pixi.toml).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from qm_pka.types import Conformer, Geometry
from qm_pka.xtb_runner import single_point
from qm_pka.xyz_io import read_multi_xyz, write_xyz

log = logging.getLogger(__name__)


def conformer_search(
    geom: Geometry,
    charge: int = 0,
    solvent: str | None = None,
    ewin: float = 6.0,
    mode: str = "default",
    threads: int | None = None,
    work_dir: Path | None = None,
) -> list[Conformer]:
    """Run CREST iMTD-GC conformer search.

    Args:
        geom: Input geometry.
        charge: Molecular charge.
        solvent: Solvent name for ALPB (e.g. "water"). None for gas phase.
        ewin: Energy window in kcal/mol.
        mode: "default", "quick", "squick", or "mquick".
        threads: Number of CPU threads. None for CREST default.
        work_dir: Working directory. If None, uses a temp directory.

    Returns:
        List of Conformer objects from the ensemble, sorted by energy.
    """
    cmd, work_dir, cleanup = _build_crest_cmd(geom, charge, solvent, threads, work_dir)
    cmd.extend(["--ewin", str(ewin)])

    if mode == "quick":
        cmd.append("--quick")
    elif mode == "squick":
        cmd.append("--squick")
    elif mode == "mquick":
        cmd.append("--mquick")
    elif mode != "default":
        raise ValueError(f"Unknown mode: {mode}")

    try:
        _run_crest(cmd, work_dir)
        output_file = work_dir / "crest_conformers.xyz"
        if not output_file.exists():
            raise FileNotFoundError(f"CREST did not produce crest_conformers.xyz in {work_dir}")
        conformers = read_multi_xyz(output_file)

        # When solvent was used, the XYZ energies include ALPB solvation.
        # Run gas-phase single points to decompose into electronic + solvation.
        if solvent is not None:
            for conf in conformers:
                total = conf.electronic_energy
                assert total is not None
                gas_phase = single_point(conf.geometry, charge=charge, solvent=None)
                conf.electronic_energy = gas_phase
                conf.solvation_energy = total - gas_phase
            log.info(f"Decomposed solvation for {len(conformers)} conformer(s) via gas-phase SP")

        return conformers
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def tautomerize(
    geom: Geometry,
    charge: int = 0,
    solvent: str | None = None,
    threads: int | None = None,
    work_dir: Path | None = None,
) -> list[Geometry]:
    """Run CREST tautomer screening.

    Returns list of unique tautomer geometries (as Geometry objects).
    """
    cmd, work_dir, cleanup = _build_crest_cmd(geom, charge, solvent, threads, work_dir)
    cmd.append("--tautomerize")

    try:
        _run_crest(cmd, work_dir)
        output_file = work_dir / "tautomers.xyz"
        if not output_file.exists():
            # CREST may produce no tautomers for simple molecules
            return []
        conformers = read_multi_xyz(output_file)
        return [c.geometry for c in conformers]
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def deprotonate(
    geom: Geometry,
    charge: int = 0,
    solvent: str | None = None,
    threads: int | None = None,
    work_dir: Path | None = None,
) -> list[Geometry]:
    """Run CREST deprotonation site screening.

    Generates structures with one proton removed (charge - 1).
    Returns list of deprotonated geometries.
    """
    cmd, work_dir, cleanup = _build_crest_cmd(geom, charge, solvent, threads, work_dir)
    cmd.append("--deprotonate")

    try:
        _run_crest(cmd, work_dir)
        output_file = work_dir / "deprotonated.xyz"
        if not output_file.exists():
            return []
        conformers = read_multi_xyz(output_file)
        expected_atoms = geom.n_atoms - 1
        validated: list[Geometry] = []
        for c in conformers:
            if c.geometry.n_atoms == expected_atoms:
                validated.append(c.geometry)
            else:
                log.warning(
                    f"CREST deprotonate returned structure with {c.geometry.n_atoms} atoms "
                    f"(expected {expected_atoms}), discarding"
                )
        return validated
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def protonate(
    geom: Geometry,
    charge: int = 0,
    solvent: str | None = None,
    threads: int | None = None,
    work_dir: Path | None = None,
) -> list[Geometry]:
    """Run CREST protonation site screening.

    Generates structures with one proton added (charge + 1).
    Returns list of protonated geometries.
    """
    cmd, work_dir, cleanup = _build_crest_cmd(geom, charge, solvent, threads, work_dir)
    cmd.append("--protonate")

    try:
        _run_crest(cmd, work_dir)
        output_file = work_dir / "protonated.xyz"
        if not output_file.exists():
            return []
        conformers = read_multi_xyz(output_file)
        expected_atoms = geom.n_atoms + 1
        validated: list[Geometry] = []
        for c in conformers:
            if c.geometry.n_atoms == expected_atoms:
                validated.append(c.geometry)
            else:
                log.warning(
                    f"CREST protonate returned structure with {c.geometry.n_atoms} atoms "
                    f"(expected {expected_atoms}), discarding"
                )
        return validated
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def _build_crest_cmd(
    geom: Geometry,
    charge: int,
    solvent: str | None,
    threads: int | None,
    work_dir: Path | None,
) -> tuple[list[str], Path, bool]:
    """Build the common CREST command prefix and prepare the working directory.

    Returns (cmd, work_dir, should_cleanup).
    """
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="crest_"))
        cleanup = True
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    input_xyz = work_dir / "input.xyz"
    write_xyz(geom, input_xyz)

    cmd = [
        "crest",
        str(input_xyz),
        "--gfn2",
        "--chrg",
        str(charge),
    ]
    if solvent is not None:
        cmd.extend(["--alpb", solvent])
    if threads is not None:
        cmd.extend(["-T", str(threads)])

    return cmd, work_dir, cleanup


def _run_crest(cmd: list[str], work_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run a CREST command and check for errors."""
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=86400,  # 24h max for large conformer searches
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"CREST failed (exit {result.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout (last 2000 chars): {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    return result
