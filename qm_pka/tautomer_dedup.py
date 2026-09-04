"""Which heavy atom owns each hydrogen, and the identity built from it.

For approach 2 (CREST-first), tautomers come as unlabeled XYZ structures.
We identify unique tautomers by counting how many hydrogens are bonded to
each heavy atom. Two structures with the same H-assignment are the same
tautomer (possibly different conformers).

`assign_protons` is the *sole* geometry-to-hydrogen-assignment primitive.
Everything downstream -- the approach-2 fingerprint written at sampling and
the migration check that reads it back in `qm_pka.protomer_geometry` -- goes
through it. Two implementations of "which atom owns this H" is not a
theoretical hazard: an earlier element-cutoff version and this one disagree on
a bridging hydrogen (C-H 1.28 A, O-H 1.20 A gives (1,0) against (0,1)), and a
disagreement there reads as a proton migration that never happened. Sharing
the *hash* would not have helped; the inputs have to be computed once.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from qm_pka.types import Geometry

# A hydrogen further than this from every heavy atom has left the molecule.
# The longest genuine X-H bond in the reference batch is S-H at 1.355 A; the
# dissociated hydrogens found there sit at 4.9-6.0 A.  The gap is wide enough
# that the exact value does not matter.
DETACHED_DISTANCE = 2.0


@dataclass(frozen=True)
class ProtonAssignment:
    """Which heavy atom owns each hydrogen in a geometry."""

    owner: tuple[int, ...]  # heavy-atom index for each H, in geometry H order
    counts: tuple[int, ...]  # H count per heavy atom, indexed as Geometry.heavy_atom_indices
    min_margin: float  # smallest (d2 - d1) over all H; diagnostic only, never branched on
    detached: tuple[int, ...]  # H indices with no heavy atom within DETACHED_DISTANCE

    @property
    def is_intact(self) -> bool:
        return not self.detached


def assign_protons(geom: Geometry) -> ProtonAssignment:
    """Assign every hydrogen to its nearest heavy atom.

    Detached hydrogens are still counted against their nearest heavy atom, so
    ``counts`` always sums to the hydrogen count and a key is always
    computable; ``detached`` is the separate signal that the geometry is not
    the species it claims to be.
    """
    heavy = geom.heavy_atom_indices
    if not heavy:
        raise ValueError("Geometry has no heavy atoms")

    coords = geom.coords
    owner: list[int] = []
    detached: list[int] = []
    counts = [0] * len(heavy)
    min_margin = float("inf")

    for h in geom.hydrogen_indices:
        ranked = sorted(
            (float(np.linalg.norm(coords[j] - coords[h])), pos) for pos, j in enumerate(heavy)
        )
        best_dist, best_pos = ranked[0]
        owner.append(heavy[best_pos])
        counts[best_pos] += 1
        if len(ranked) > 1:
            min_margin = min(min_margin, ranked[1][0] - best_dist)
        if best_dist > DETACHED_DISTANCE:
            detached.append(h)

    return ProtonAssignment(tuple(owner), tuple(counts), min_margin, tuple(detached))


def assign_hydrogens(geom: Geometry) -> tuple[int, ...]:
    """Count hydrogens bonded to each heavy atom, in heavy-atom order.

    A thin view over :func:`assign_protons`, kept because the count vector is
    what the fingerprint needs. It must stay a view rather than a second
    implementation -- see the module docstring.
    """
    return assign_protons(geom).counts


def fingerprint_counts(h_counts: tuple[int, ...]) -> str:
    """Return a hex digest of an H-count-per-heavy-atom tuple.

    Sole definition of the approach-2 microstate id, so that a fingerprint
    recomputed later -- by `protomer_geometry.repair_migrated_conformers`, on a
    geometry that has since been minimised -- is comparable to the one stored as
    ``tautomer_id`` at sampling.

    This is the extension point for giving approach 2 a stereo model: tetrahedral
    parity and, less straightforwardly, double-bond configuration would join the
    hydrogen counts here. Both are TODO -- see "Known limitations" in
    docs/protomer-identity.md for what each would take and why the second is
    unresolved.
    """
    return hashlib.sha256(repr(h_counts).encode()).hexdigest()[:16]


def h_assignment_fingerprint(geom: Geometry) -> str:
    """Return a hex digest of the H-assignment tuple.

    Used as tautomer_id in approach 2 where we don't have SMILES labels.
    """
    return fingerprint_counts(assign_hydrogens(geom))


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
