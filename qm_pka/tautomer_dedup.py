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


# Covalent radii (Angstrom), for heavy-heavy connectivity only. Never used to
# infer a bond *order* -- that is the thing this path deliberately does not do.
# Heavy bonds separate far more cleanly than X-H contacts: a C-C bond is ~1.5 A
# against a next-nearest approach of 2.4 A or more, so the cutoff has room that
# the hydrogen assignment never had.
_COVALENT_RADIUS: dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Se": 1.20,
    "Br": 1.20,
    "I": 1.39,
}
_DEFAULT_RADIUS = 0.80
_BOND_SCALE = 1.25


@dataclass(frozen=True)
class GeometricIdentity:
    """What a geometry says about which species it is, with no SMILES involved.

    Indexed throughout by *heavy-atom position* -- the order heavy atoms appear
    in the geometry -- which every structure of one charge state shares. See
    `heavy_frameworks_agree`.
    """

    counts: tuple[int, ...]  # hydrogens owned by each heavy atom
    parities: tuple[int, ...]  # tetrahedral parity per heavy atom; 0 where undefined
    framework: tuple[tuple[int, ...], ...]  # heavy neighbours of each heavy atom
    assignment: ProtonAssignment


def heavy_framework(geom: Geometry) -> tuple[tuple[int, ...], ...]:
    """Heavy-atom connectivity from distances, in heavy-atom position order."""
    heavy = geom.heavy_atom_indices
    position = {idx: pos for pos, idx in enumerate(heavy)}
    neighbours: list[list[int]] = [[] for _ in heavy]
    for a in range(len(heavy)):
        for b in range(a + 1, len(heavy)):
            i, j = heavy[a], heavy[b]
            limit = _BOND_SCALE * (
                _COVALENT_RADIUS.get(geom.symbols[i], _DEFAULT_RADIUS)
                + _COVALENT_RADIUS.get(geom.symbols[j], _DEFAULT_RADIUS)
            )
            if float(np.linalg.norm(geom.coords[i] - geom.coords[j])) < limit:
                neighbours[position[i]].append(position[j])
                neighbours[position[j]].append(position[i])
    return tuple(tuple(sorted(n)) for n in neighbours)


def geometric_identity(geom: Geometry) -> GeometricIdentity:
    """Everything the CREST-first identity needs, derived once from coordinates.

    One function so that the fingerprint written at sampling and the one
    recomputed by the migration check cannot be built from differently-derived
    inputs. Sharing only the hash would not achieve that; the inputs have to be
    computed in one place.

    Tetrahedral parity is recorded at heavy atoms with four connections and **at
    most one hydrogen**. That gate is doing two jobs. It admits the ordinary
    carbon stereocentre, which has three heavy neighbours and one hydrogen and
    would be missed entirely by any rule counting heavy neighbours alone; and it
    admits a 1,4-ring carbon, whose two ring branches are constitutionally
    identical so that no purely constitutional refinement can ever separate cis
    from trans -- the configuration lives in the *pair*, and recording a parity
    at both captures it. It excludes atoms with two or more hydrogens, where the
    hydrogens are interchangeable and their ordering is not stable across CREST
    outputs, which is the only case that could flip for no chemical reason.

    Recording a parity at an atom that is not a stereocentre is harmless: it is a
    constant for that species, so it never splits anything.

    Neighbours are ordered ``(is_hydrogen, position)``, never by raw index. A
    lone hydrogen can be written before or after its heavy siblings depending on
    where CREST put it, and ordering it last makes the parity depend only on the
    heavy-atom order -- the invariant this path already requires.
    """
    assignment = assign_protons(geom)
    heavy = geom.heavy_atom_indices
    framework = heavy_framework(geom)

    owned: dict[int, list[int]] = defaultdict(list)
    for h_idx, owner in zip(geom.hydrogen_indices, assignment.owner, strict=True):
        owned[owner].append(h_idx)

    parities: list[int] = []
    for pos, centre in enumerate(heavy):
        hydrogens = owned.get(centre, [])
        if len(framework[pos]) + len(hydrogens) != 4 or len(hydrogens) > 1:
            parities.append(0)
            continue
        ordered = [heavy[n] for n in framework[pos]] + hydrogens
        v = [geom.coords[i] - geom.coords[centre] for i in ordered]
        volume = float(np.dot(np.cross(v[1] - v[0], v[2] - v[0]), v[3] - v[0]))
        parities.append(int(np.sign(volume)))

    return GeometricIdentity(
        counts=assignment.counts,
        parities=tuple(parities),
        framework=framework,
        assignment=assignment,
    )


def geometric_fingerprint(geom: Geometry) -> str:
    """The approach-2 microstate id: hydrogen distribution plus configuration.

    Double-bond configuration is *not* included -- it would need a rigidity test,
    and rigidity is not recoverable from coordinates alone. See "Known
    limitations" in docs/protomer-identity.md.
    """
    identity = geometric_identity(geom)
    payload = repr((identity.counts, identity.parities))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def heavy_frameworks_agree(reference: Geometry, candidate: Geometry) -> bool:
    """Do two geometries share a heavy-atom order, as the fingerprint assumes?

    Everything in this path is indexed by heavy-atom position, so comparing
    fingerprints across structures that order their heavy atoms differently is
    meaningless. Element sequence is the weaker check and was all this path had:
    two different orderings can share one sequence. Comparing the connectivity
    tests the correspondence itself.
    """
    ref_heavy = [reference.symbols[i] for i in reference.heavy_atom_indices]
    cand_heavy = [candidate.symbols[i] for i in candidate.heavy_atom_indices]
    if ref_heavy != cand_heavy:
        return False
    return heavy_framework(reference) == heavy_framework(candidate)


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
        fp = geometric_fingerprint(geom)
        groups[fp].append(geom)
    return dict(groups)
