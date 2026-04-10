"""Wrapper around CREST for xTB geometry optimization and single-point energy.

All CREST commands use -newversion to ensure CREST uses its internal tblite
backend rather than the standalone xtb binary. This avoids a Fortran format
string bug in xtb 6.7.1 build 2 (github.com/grimme-lab/xtb/issues/1332)
which is forced by gcp-correction's mctc-lib <0.4 pin.
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
    """Run geometry optimization via CREST --mdopt.

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
            "crest",
            str(input_xyz),
            "--gfn2" if gfn == 2 else f"--gfn{gfn}",
            "--chrg",
            str(charge),
            "-newversion",  # use tblite backend, not standalone xtb
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
    """Run single-point energy calculation via CREST --sp.

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
            "crest",
            str(input_xyz),
            "--sp",
            "--gfn2" if gfn == 2 else f"--gfn{gfn}",
            "--chrg",
            str(charge),
            "-newversion",  # use tblite backend, not standalone xtb
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
                f"crest single-point failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        return _parse_energy(result.stdout)

    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def _parse_energy(stdout: str) -> float:
    """Parse total energy from CREST stdout."""
    match = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh", stdout)
    if match is None:
        raise RuntimeError("Could not parse energy from CREST output")
    return float(match.group(1))
