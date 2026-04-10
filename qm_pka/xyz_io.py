"""Read and write XYZ files, including CREST multi-structure ensembles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qm_pka.types import Conformer, Geometry


def read_xyz(path: Path) -> Geometry:
    """Read a single-structure XYZ file."""
    text = path.read_text()
    return _parse_single_xyz(text.splitlines())


def read_multi_xyz(path: Path) -> list[Conformer]:
    """Read a CREST multi-structure XYZ file.

    Each frame has an energy in the comment line (Hartree).
    Returns Conformer objects with parsed energies.
    """
    lines = path.read_text().splitlines()
    conformers: list[Conformer] = []
    i = 0
    while i < len(lines):
        # Skip blank lines between frames
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        frame_lines = lines[i : i + n_atoms + 2]
        geom = _parse_single_xyz(frame_lines)
        energy = _parse_comment_energy(frame_lines[1])
        conformers.append(Conformer(geometry=geom, energy=energy))
        i += n_atoms + 2
    return conformers


def write_xyz(geom: Geometry, path: Path, comment: str = "") -> None:
    """Write a single-structure XYZ file."""
    path.write_text(_format_xyz(geom, comment))


def write_multi_xyz(conformers: list[Conformer], path: Path) -> None:
    """Write a multi-structure XYZ file with energies in comment lines."""
    blocks: list[str] = []
    for conf in conformers:
        blocks.append(_format_xyz(conf.geometry, f"{conf.energy:.10f}"))
    path.write_text("\n".join(blocks))


def _parse_single_xyz(lines: list[str]) -> Geometry:
    """Parse a single XYZ frame from lines."""
    n_atoms = int(lines[0].strip())
    # line 1 is the comment line (may contain energy)
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return Geometry(
        symbols=tuple(symbols),
        coords=np.array(coords, dtype=np.float64),
    )


def _parse_comment_energy(comment: str) -> float:
    """Extract energy in Hartree from a CREST-style comment line.

    CREST writes the energy as the first token in the comment line,
    e.g. '     -15.12345678' or '-15.12345678   1.0000000000'.
    """
    return float(comment.split()[0])


def _format_xyz(geom: Geometry, comment: str = "") -> str:
    """Format a Geometry as an XYZ string."""
    lines: list[str] = [str(geom.n_atoms), comment]
    for sym, (x, y, z) in zip(geom.symbols, geom.coords, strict=True):
        lines.append(f"{sym:>2s} {x:>16.10f} {y:>16.10f} {z:>16.10f}")
    return "\n".join(lines) + "\n"
