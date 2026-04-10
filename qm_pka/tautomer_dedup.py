"""Tautomer deduplication by H-count-per-heavy-atom fingerprinting.

For approach 2 (CREST-first), tautomers come as unlabeled XYZ structures.
We identify unique tautomers by counting how many hydrogens are bonded to
each heavy atom. Two structures with the same H-assignment are the same
tautomer (possibly different conformers).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from qm_pka.types import Geometry

# Covalent bond cutoffs for X-H bonds (Angstrom).
# Based on covalent radius sum * 1.2.
_H_BOND_CUTOFFS: dict[str, float] = {
    "C": 1.30,
    "N": 1.20,
    "O": 1.15,
    "S": 1.60,
    "P": 1.60,
    "F": 1.10,
    "Cl": 1.50,
    "Br": 1.60,
    "I": 1.75,
    "B": 1.35,
    "Se": 1.65,
}

# Fallback cutoff for elements not in the table
_DEFAULT_CUTOFF: float = 1.70


def assign_hydrogens(geom: Geometry) -> tuple[int, ...]:
    """Count hydrogens bonded to each heavy atom.

    Returns a tuple indexed by heavy-atom position (in the order they appear
    in geom.symbols) giving the number of H atoms covalently bonded to each.

    A hydrogen is assigned to the nearest heavy atom within the element-specific
    covalent bond cutoff. If no heavy atom is within range, raises an error.
    """
    heavy_indices = geom.heavy_atom_indices
    h_indices = geom.hydrogen_indices

    if not heavy_indices:
        raise ValueError("Geometry has no heavy atoms")

    h_counts = [0] * len(heavy_indices)

    for h_idx in h_indices:
        h_pos = geom.coords[h_idx]
        best_heavy_pos: int | None = None
        best_dist = float("inf")

        for list_pos, heavy_idx in enumerate(heavy_indices):
            heavy_sym = geom.symbols[heavy_idx]
            cutoff = _H_BOND_CUTOFFS.get(heavy_sym, _DEFAULT_CUTOFF)
            dist = float(np.linalg.norm(geom.coords[heavy_idx] - h_pos))
            if dist < cutoff and dist < best_dist:
                best_dist = dist
                best_heavy_pos = list_pos

        if best_heavy_pos is None:
            raise ValueError(
                f"Hydrogen at index {h_idx} has no heavy atom within covalent bond distance"
            )
        h_counts[best_heavy_pos] += 1

    return tuple(h_counts)


def h_assignment_fingerprint(geom: Geometry) -> str:
    """Return a hex digest of the H-assignment tuple.

    Used as tautomer_id in approach 2 where we don't have SMILES labels.
    """
    h_tuple = assign_hydrogens(geom)
    return hashlib.sha256(repr(h_tuple).encode()).hexdigest()[:16]


def deduplicate_tautomers(
    geometries: list[Geometry],
) -> dict[str, list[Geometry]]:
    """Group geometries by their H-assignment fingerprint.

    Returns {fingerprint: [geometries_with_that_assignment]}.
    Within each group, geometries are sorted by the order they appeared
    in the input list (preserving energy-ranked order from CREST).
    """
    groups: dict[str, list[Geometry]] = defaultdict(list)
    for geom in geometries:
        fp = h_assignment_fingerprint(geom)
        groups[fp].append(geom)
    return dict(groups)


def validate_heavy_atom_ordering(reference: Geometry, candidate: Geometry) -> bool:
    """Check that heavy atoms appear in the same element order.

    CREST should preserve heavy-atom ordering across tautomerization
    (only H's are moved to the end). This validates that assumption.
    """
    ref_heavy = [reference.symbols[i] for i in reference.heavy_atom_indices]
    cand_heavy = [candidate.symbols[i] for i in candidate.heavy_atom_indices]
    return ref_heavy == cand_heavy
