"""Stereoisomer enumeration and enantiomer deduplication.

Enumerates all stereoisomers (tetrahedral + E/Z) of a SMILES and deduplicates
enantiomeric pairs by choosing a canonical representative.

Enantiomer canonicalization: two stereoisomers are enantiomers iff inverting all
tetrahedral centers (CW<->CCW) while leaving E/Z bonds unchanged produces the
other (after canonicalization). We pick the lexicographically smaller canonical
SMILES as the representative, which is deterministic regardless of which
enantiomer is encountered first.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)


def enumerate_stereoisomers(smiles: str) -> list[str]:
    """Enumerate all stereoisomers (tetrahedral + E/Z) of a SMILES.

    Returns a list of unique canonical SMILES. If the molecule has no
    stereocenters, returns a single-element list with the canonical SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=True)
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    result: list[str] = []
    seen: set[str] = set()
    for iso in isomers:
        can: str | None = Chem.MolToSmiles(iso)
        if can is not None and can not in seen:
            seen.add(can)
            result.append(can)
    if not result:
        can = Chem.MolToSmiles(mol)
        if can is not None:
            result.append(can)
    return result


def mirror_smiles(smiles: str) -> str:
    """Return the mirror image of a SMILES by inverting all tetrahedral centers.

    E/Z double bond stereo is left unchanged (a mirror reflection preserves
    cis/trans relationships). Only tetrahedral chirality (CW<->CCW) is flipped.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    for atom in mol.GetAtoms():
        chiral = atom.GetChiralTag()
        if chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    result: str | None = Chem.MolToSmiles(mol)
    if result is None:
        raise RuntimeError(f"Failed to generate mirror SMILES for: {smiles}")
    return result


def canonical_enantiomer(smiles: str) -> str:
    """Return a canonical representative from the enantiomeric pair.

    Given a SMILES, computes its mirror image (all tetrahedral centers
    inverted, E/Z unchanged), canonicalizes both, and returns the
    lexicographically smaller one. This is deterministic: the same
    representative is chosen regardless of which enantiomer is input.
    """
    canon = _canonical(smiles)
    mirror = mirror_smiles(canon)
    canon_mirror = _canonical(mirror)
    return min(canon, canon_mirror)


def deduplicate_enantiomers(smiles_list: list[str]) -> list[tuple[str, bool]]:
    """Remove enantiomeric duplicates from a list of SMILES.

    For each enantiomeric pair, keeps the canonical representative
    (lexicographically smaller canonical SMILES). Preserves input order
    for the first occurrence of each unique canonical enantiomer.

    Returns list of (smiles, includes_enantiomer) tuples.
    includes_enantiomer is True when the representative was part of
    a collapsed pair (i.e., it has a distinct mirror image), False for
    achiral or meso compounds.
    """
    seen: dict[str, bool] = {}
    for smi in smiles_list:
        canon = _canonical(smi)
        canon_mirror = _canonical(mirror_smiles(canon))
        canon_enant = min(canon, canon_mirror)
        if canon_enant not in seen:
            has_enantiomer = canon != canon_mirror
            seen[canon_enant] = has_enantiomer
    return list(seen.items())


def enumerate_and_deduplicate(smiles: str) -> list[tuple[str, bool]]:
    """Enumerate all stereoisomers and deduplicate enantiomers.

    Convenience function that combines enumerate_stereoisomers and
    deduplicate_enantiomers. Returns list of (smiles, includes_enantiomer)
    tuples with one representative per enantiomeric pair.
    """
    stereoisomers = enumerate_stereoisomers(smiles)
    return deduplicate_enantiomers(stereoisomers)


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    result: str | None = Chem.MolToSmiles(mol)
    if result is None:
        raise RuntimeError(f"Failed to canonicalize: {smiles}")
    return result
