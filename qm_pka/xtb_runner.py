"""Wrapper for xtb/crest: geometry optimization and single-point energy.

Note: standalone xtb 6.7.1 from conda-forge has a Fortran format string bug
that crashes --opt (github.com/grimme-lab/xtb/issues/1332). We work around
this by using CREST's --optlev for optimization, which uses its own internal
xTB implementation. Single-point calculations use standalone xtb (unaffected).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from qm_pka.types import Geometry
from qm_pka.xyz_io import read_multi_xyz, write_xyz


def optimize(
    geom: Geometry,
    charge: int = 0,
    gfn: int = 2,
    solvent: str | None = None,
    opt_level: str = "tight",
    work_dir: Path | None = None,
) -> Geometry:
    """Run geometry optimization via CREST's --mdopt or --opt flag.

    Uses CREST instead of standalone xtb to work around the conda-forge
    xtb 6.7.1 optimizer bug. Returns the optimized Geometry.
    """
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="xtb_opt_"))
        cleanup = True

    try:
        input_xyz = work_dir / "input.xyz"
        write_xyz(geom, input_xyz)

        cmd = [
            "crest",
            str(input_xyz),
            "--gfn2" if gfn == 2 else f"--gfn{gfn}",
            "--chrg",
            str(charge),
            "--optlev",
            opt_level,
            "--mdopt",
            str(input_xyz),
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
                f"crest optimization failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        opt_xyz = work_dir / "crest_ensemble.xyz"
        if not opt_xyz.exists():
            raise FileNotFoundError(f"crest did not produce crest_ensemble.xyz in {work_dir}")
        # crest_ensemble.xyz is a multi-xyz; first structure is the optimized one
        conformers = read_multi_xyz(opt_xyz)
        if not conformers:
            raise RuntimeError("crest_ensemble.xyz is empty")
        return conformers[0].geometry

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
            "--gfn",
            str(gfn),
            "--chrg",
            str(charge),
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
                f"xtb single-point failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
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
