"""Wrapper for xtb: geometry optimization and single-point energy."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from qm_pka.types import Geometry
from qm_pka.xyz_io import read_xyz, write_xyz


def optimize(
    geom: Geometry,
    charge: int = 0,
    gfn: int = 2,
    solvent: str | None = None,
    opt_level: str = "tight",
    work_dir: Path | None = None,
) -> Geometry:
    """Run xtb geometry optimization.

    Returns the optimized Geometry.
    """
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="xtb_opt_"))
        cleanup = True

    try:
        input_xyz = work_dir / "input.xyz"
        write_xyz(geom, input_xyz)

        cmd = [
            "xtb",
            str(input_xyz),
            f"--gfn", str(gfn),
            "--opt", opt_level,
            "--chrg", str(charge),
        ]
        if solvent is not None:
            cmd.extend(["--alpb", solvent])

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xtb optimization failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        opt_xyz = work_dir / "xtbopt.xyz"
        if not opt_xyz.exists():
            raise FileNotFoundError(
                f"xtb did not produce xtbopt.xyz in {work_dir}"
            )
        return read_xyz(opt_xyz)

    finally:
        if cleanup:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def single_point(
    geom: Geometry,
    charge: int = 0,
    gfn: int = 2,
    solvent: str | None = None,
    work_dir: Path | None = None,
) -> float:
    """Run xtb single-point calculation.

    Returns the total energy in Hartree.
    """
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="xtb_sp_"))
        cleanup = True

    try:
        input_xyz = work_dir / "input.xyz"
        write_xyz(geom, input_xyz)

        cmd = [
            "xtb",
            str(input_xyz),
            f"--gfn", str(gfn),
            "--chrg", str(charge),
        ]
        if solvent is not None:
            cmd.extend(["--alpb", solvent])

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xtb single-point failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        return _parse_energy(result.stdout)

    finally:
        if cleanup:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def _parse_energy(stdout: str) -> float:
    """Parse total energy from xtb stdout."""
    match = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh", stdout)
    if match is None:
        raise RuntimeError("Could not parse energy from xtb output")
    return float(match.group(1))
