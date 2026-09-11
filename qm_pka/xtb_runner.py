"""Wrappers around CREST/xtb for xTB geometry optimization, single-point
energy, and quasi-RRHO vibrational free-energy corrections.

CREST 2.12 drives the external ``xtb`` binary as a subprocess (the CREST 3.x
in-process rewrite produces degenerate single-conformer ensembles on macOS, so
we pin 2.12). That pin covers the conformer search only. Everything here --
optimization, single points and Hessians -- calls ``xtb`` directly.

Optimization used to route through CREST ``--mdopt`` to dodge a Fortran
format-string crash in conda-forge xtb 6.7.1 *build 2*
(grimme-lab/xtb#1332). Build 4 fixed it and pixi.toml already floors us above
that, so the workaround outlived its cause -- and it was not free. CREST accepts
an optimization level, reports success, and returns the input geometry
bit-identical whenever its own optimizer considers the structure converged, so
a CREST-optimized conformer could not be tightened at all: ``xtb --opt`` calls
it converged in one cycle at crude, loose, normal *and* tight, while ``vtight``
takes tens of cycles and moves it half an angstrom. Structures left at ``tight``
carry imaginary modes of a few hundred wavenumbers, which a numerical Hessian
then reports as real.
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
) -> tuple[Geometry, bool]:
    """Optimize a geometry with ``xtb --opt``.

    Returns ``(geometry, converged)``. A geometry that did not converge is
    returned anyway, carrying xtb's last step: it is still a better starting
    point than what went in, and discarding it would lose a conformer for a
    numerical outcome rather than a chemical one. Callers decide what to do
    with the flag; nothing here drops anything.

    ``opt_level`` matters more than it looks. CREST's conformer search leaves
    structures that xtb calls converged at every level up to ``tight``, so
    re-minimizing those needs ``vtight`` to move at all. Fresh embeddings are
    far enough from a minimum that ``tight`` does real work.
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
            "--opt",
            opt_level,
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

        opt_xyz = work_dir / "xtbopt.xyz"
        if not opt_xyz.exists():
            raise RuntimeError(
                f"xtb optimization produced no geometry (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        # xtb writes xtbopt.xyz even when it runs out of cycles, and says so in
        # the log rather than in the exit code.
        converged = "GEOMETRY OPTIMIZATION CONVERGED" in result.stdout
        conformers = read_multi_xyz(opt_xyz)
        if not conformers:
            raise RuntimeError("xtbopt.xyz is empty")
        return conformers[0].geometry, converged

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
    """Run a single-point energy calculation via the standalone xtb binary.

    CREST 2.x has no single-point run mode (``--sp`` is silently ignored and a
    full conformer search runs instead), so this calls ``xtb`` directly, which
    is also the engine CREST drives internally. Returns the energy in Hartree.
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
            "--sp",
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


def frequencies(
    geom: Geometry,
    charge: int = 0,
    gfn: int = 2,
    solvent: str | None = None,
    biased: bool = False,
    threads: int | None = None,
    work_dir: Path | None = None,
) -> list[float]:
    """Compute harmonic vibrational frequencies via the standalone xtb binary.

    Args:
        biased: If True, use ``--bhess`` (Spicher-Grimme single-point Hessian),
            appropriate for geometries that are *not* stationary points on the
            xTB surface (e.g. DFT-optimized geometries during refinement). If
            False, use the plain numerical Hessian ``--hess`` for xTB minima
            (e.g. CREST-optimized geometries during sampling).
        solvent: ALPB implicit-solvent name (e.g. "water"); xTB RRHO is always
            computed in implicit solvent in this workflow.

    Returns frequencies in cm**-1 (the 3N-6 vibrational modes, with translation
    and rotation already projected out by xtb; imaginary modes as negatives).
    """
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="xtb_hess_"))
        cleanup = True

    try:
        input_xyz = work_dir / "input.xyz"
        write_xyz(geom, input_xyz)

        cmd = [
            "xtb",
            str(input_xyz),
            "--bhess" if biased else "--hess",
            "--gfn",
            str(gfn),
            "--chrg",
            str(charge),
        ]
        if solvent is not None:
            cmd.extend(["--alpb", solvent])
        if threads is not None:
            cmd.extend(["--parallel", str(threads)])

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xtb Hessian failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        g98 = work_dir / "g98.out"
        if not g98.exists():
            raise FileNotFoundError(f"xtb did not produce g98.out in {work_dir}")
        return _parse_g98_frequencies(g98.read_text())

    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def _parse_g98_frequencies(text: str) -> list[float]:
    """Parse vibrational frequencies (cm⁻¹) from xtb's Gaussian-98 output.

    g98.out lists only the real vibrational modes (translation/rotation are
    already projected out), three per ``Frequencies --`` line.
    """
    freqs: list[float] = []
    prefix = "Frequencies --"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            freqs.extend(float(tok) for tok in stripped[len(prefix) :].split())
    if not freqs:
        raise RuntimeError("Could not parse any frequencies from xtb g98.out")
    return freqs


def _parse_energy(stdout: str) -> float:
    """Parse total energy from xtb stdout."""
    match = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh", stdout)
    if match is None:
        raise RuntimeError("Could not parse energy from xtb output")
    return float(match.group(1))
