"""RDKit interface: SMILES<->3D, tautomer enumeration, protomer identity."""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler
from rdkit.Chem.MolStandardize import rdMolStandardize

from qm_pka.types import Geometry

log = logging.getLogger(__name__)


def frame_atom_order(mol: Chem.Mol) -> list[int]:
    """Heavy-atom indices of ``mol`` in canonical order of its bare frame.

    The *frame* is the heavy-atom graph with hydrogen counts, formal charges,
    bond orders, aromaticity and stereo all erased -- so it is byte-identical
    for every protomer and every tautomer of one molecule, and canonically
    ranking it labels the heavy atoms in a way none of them disagree about.

    `_skeleton` cannot serve here: it pins hydrogen counts, which is exactly
    what distinguishes two protomers, so its ranking differs between them.
    """
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetAtomMapNum(0)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    frame = rw.GetMol()
    Chem.SanitizeMol(frame, _SKELETON_SANITIZE)
    ranks = list(Chem.CanonicalRankAtoms(frame))
    return sorted(range(mol.GetNumAtoms()), key=lambda i: ranks[i])


def smiles_to_3d(smiles: str) -> tuple[Geometry, str]:
    """Generate a 3D geometry from a SMILES string via ETKDG embedding.

    Returns (geometry, explicit_h_smiles) where the geometry's atom
    ordering matches the SMILES atom ordering. Coordinates are reordered
    using _smilesAtomOutputOrder so that geometry index i corresponds
    to SMILES atom i.

    Heavy atoms are put in `frame_atom_order` before anything else, which makes
    every protomer of one molecule agree on their order. Without it each
    microstate inherits the canonical order of its *own* SMILES -- and those
    diverge, because RDKit's canonical ranking depends on formal charge and
    hydrogen count. A conformer whose proton migrates then cannot be re-filed
    under the microstate it became without solving an atom correspondence
    problem first; see `qm_pka.protomer_geometry`. Hydrogens still interleave
    differently between protomers, since SMILES writes each one attached to its
    heavy atom, but that reordering is a regrouping rather than a mapping.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.RenumberAtoms(mol, frame_atom_order(mol))
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status != 0:
        raise RuntimeError(f"ETKDG embedding failed for: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    all_coords = np.array(conf.GetPositions(), dtype=np.float64)

    explicit_h_smiles: str | None = Chem.MolToSmiles(mol, canonical=False)
    if explicit_h_smiles is None:
        raise RuntimeError(f"Failed to generate explicit-H SMILES for: {smiles}")

    # _smilesAtomOutputOrder[smi_idx] = mol_idx: reorder coords to match SMILES
    import json

    order: list[int] = json.loads(mol.GetProp("_smilesAtomOutputOrder"))
    symbols = tuple(mol.GetAtomWithIdx(order[i]).GetSymbol() for i in range(len(order)))
    coords = all_coords[order]

    return Geometry(symbols=symbols, coords=coords), explicit_h_smiles


def enumerate_tautomers(
    smiles: str,
    max_tautomers: int = 1000,
    max_transforms: int = 1000,
) -> list[str]:
    """Enumerate tautomers using RDKit's TautomerEnumerator.

    Returns a list of unique canonical SMILES including the input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    enumerator.SetMaxTransforms(max_transforms)
    tautomers = enumerator.Enumerate(mol)
    seen: set[str] = set()
    result: list[str] = []
    for t in tautomers:
        can = canonical_smiles_from_mol(t)
        if can not in seen:
            seen.add(can)
            result.append(can)
    return result


def canonical_smiles(smiles: str) -> str:
    """Return the RDKit canonical SMILES for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return canonical_smiles_from_mol(mol)


def canonical_smiles_from_mol(mol: Chem.Mol) -> str:
    """Return the canonical SMILES for an RDKit Mol object."""
    result: str | None = Chem.MolToSmiles(mol)
    if result is None:
        raise RuntimeError("Failed to generate SMILES from mol")
    return result


def get_atom_mapped_smiles(smiles: str) -> str:
    """Return SMILES with atom map numbers for tracking through transformations."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    result: str | None = Chem.MolToSmiles(mol)
    if result is None:
        raise RuntimeError("Failed to generate mapped SMILES")
    return result


def get_formal_charge(smiles: str) -> int:
    """Return the net formal charge of a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return int(Chem.GetFormalCharge(mol))


# --- protomer identity ------------------------------------------------------
#
# Resonance moves formal charge and bond order, and nothing else: it cannot
# break a heavy-atom bond or relocate a hydrogen. Our own enumeration is
# similarly restricted -- it only adds and removes H at heteroatoms. So across
# every microstate of one molecule the heavy-atom framework is fixed, and
#
#     (framework, per-atom H count, net charge, stereochemistry)
#
# identifies a species uniquely, whichever Lewis structure happens to have been
# written for it.
#
# Canonical SMILES cannot serve as that identity, because two resonance forms
# are genuinely different graphs in RDKit's model and canonicalise apart. A
# 4-substituted imidazolium is the standard case: the cation can be written on
# either ring nitrogen, and the enumerator emits both, double-counting the
# species and its whole conformer ensemble.
#
# protomer_key erases what resonance can change, keeps what it cannot, and
# canonicalises the result.

# Valence checking is skipped when sanitizing the skeleton: a framework whose
# bonds have all been reduced to single is deliberately not a valid Lewis
# structure. Ring perception and canonical ranking, which are what we need from
# it, do not care.
# Skip valence checking: a skeleton with every bond reduced to single and every
# charge zeroed is not a valid molecule, and does not need to be.
#
# SANITIZE_FINDRADICALS must stay enabled, counter-intuitive as that is on a
# graph whose valences are deliberately wrong. The deficiency we create is read
# as unpaired electrons, and the resulting radical count is what forces RDKit to
# bracket an atom -- `[O]` rather than `O` -- which is the only thing preserving
# its hydrogen count through canonicalisation. Without it acetic acid and
# ethane-1,1-diol both canonicalise to `CC(O)O`, and the key silently merges
# species with different molecular formulas. See tests/test_rdkit_utils.py.
_SKELETON_SANITIZE = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES

_UNSPECIFIED_STEREO = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)


def _skeleton(mol: Chem.Mol) -> Chem.Mol:
    """Return the heavy-atom framework with everything resonance can move erased.

    Formal charges, bond orders, aromaticity and stereo are stripped; per-atom
    hydrogen counts are pinned as explicit, because they are what distinguishes
    a genuine tautomer from a resonance form.
    """
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        # Atom maps are an annotation, not chemistry; left in place they would
        # be written into the canonical SMILES and split a species from itself.
        atom.SetAtomMapNum(0)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    skeleton = rw.GetMol()
    Chem.SanitizeMol(skeleton, _SKELETON_SANITIZE)
    return skeleton


def protomer_key(smiles: str) -> str:
    """Return a resonance-invariant identity for a protonation microstate.

    Two SMILES receive the same key exactly when they describe the same species
    written as different Lewis structures. Genuine tautomers, differing
    protonation sites, and stereoisomers all receive different keys, because
    each of those changes either a per-atom hydrogen count or a stereo
    descriptor, and neither is erased.

    Stereo descriptors are CIP labels rather than RDKit's internal E/Z tags,
    keyed by the atom's rank in the *skeleton*: the skeleton is identical across
    resonance forms, so this labelling is stable where a raw atom index is not.

    Unsupported inputs (see ``validate_input_smiles``) are keyed conservatively:
    anything this function cannot label with confidence falls back to the plain
    canonical SMILES, which merges nothing.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except RuntimeError as exc:  # pragma: no cover - not reached on any known input
        log.warning(
            f"CIP labelling failed for {smiles} ({exc}); keying it by plain "
            f"canonical SMILES, so it will not merge with any other microstate"
        )
        return canonical_smiles_from_mol(mol)

    skeleton = _skeleton(mol)
    ranks = list(Chem.CanonicalRankAtoms(skeleton))

    stereo: list[tuple[str, object, str]] = []
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode"):
            stereo.append(("A", ranks[atom.GetIdx()], atom.GetProp("_CIPCode")))
    for bond in mol.GetBonds():
        if bond.GetStereo() in _UNSPECIFIED_STEREO:
            continue
        if not bond.HasProp("_CIPCode"):  # pragma: no cover - no known input reaches this
            # A specified stereo bond that CIP could not name. Merging two of
            # these would silently collapse E with Z, so refuse to merge at all.
            log.warning(
                f"unlabelled stereo bond in {smiles}; keying it by plain "
                f"canonical SMILES, so it will not merge with any other microstate"
            )
            return canonical_smiles_from_mol(mol)
        ends = tuple(sorted((ranks[bond.GetBeginAtomIdx()], ranks[bond.GetEndAtomIdx()])))
        stereo.append(("B", ends, bond.GetProp("_CIPCode")))

    stereo.sort(key=repr)
    charge = Chem.GetFormalCharge(mol)
    return f"{Chem.MolToSmiles(skeleton)}|q{charge}|{stereo}"


def _representative_rank(smiles: str) -> tuple[int, int, str]:
    """Sort key picking the best Lewis structure to stand for a species.

    Fewest formally charged atoms first -- the textbook test for the dominant
    resonance contributor, and the one that keeps an absurd but valid structure
    like ``CC([O-])=[OH+]`` from standing in for acetic acid. Then the most
    perceivable stereo elements, since the survivor seeds stereoisomer
    enumeration and a delocalised anion drawn with its double bond in one
    position can expose a stereogenic bond the other position hides. Then the
    canonical SMILES, for determinism.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # pragma: no cover - callers pass parsed SMILES
        return (0, 0, smiles)
    charged = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge())
    return (charged, -len(Chem.FindPotentialStereo(mol)), smiles)


def deduplicate_protomers(smiles_list: list[str]) -> list[str]:
    """Collapse SMILES that describe one species written as different Lewis structures.

    Keeps one representative per :func:`protomer_key`. The survivor is not an
    arbitrary member: it seeds both stereoisomer enumeration and the ETKDG
    geometry, so it is chosen by :func:`_representative_rank` rather than by
    lexicographic order, which would decide both on where a bracket happens to
    sort in ASCII. Returned in sorted order.
    """
    groups: dict[str, list[str]] = {}
    for smi in smiles_list:
        groups.setdefault(protomer_key(smi), []).append(canonical_smiles(smi))
    return sorted(min(group, key=_representative_rank) for group in groups.values())


def validate_input_smiles(smiles: str) -> None:
    """Raise ``ValueError`` if the input is outside what the pipeline models.

    Two classes are rejected outright rather than approximated, because nothing
    downstream would treat them correctly and a plausible-looking number is
    worse than a refusal:

    Open-shell species (radicals). ``Geometry.multiplicity`` assumes the lowest
    multiplicity for the electron count, so a triplet would be evaluated as a
    singlet and quietly return an energy for the wrong state. There are no plans
    to support them.

    Multi-component inputs (salts, solvates, mixtures). Conformer search is
    meaningless when fragments translate freely, and the thermodynamic cycle
    assumes a single solute. Desalt first and submit the component of interest.
    There are no plans to support them.

    Two further limits are *not* checked here, because RDKit discards them
    before this function ever sees them, and both may be supported later:
    enhanced stereochemistry (AND/OR stereo groups) is not modelled, and
    atropisomerism is neither perceived from SMILES nor preserved by the plain
    canonical SMILES used as microstate labels -- it survives only in CXSMILES.
    Inputs relying on either are silently treated as unspecified.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    if radicals:
        raise ValueError(
            f"Open-shell input is not supported: {smiles} carries {radicals} "
            f"radical electron(s). Every stage assumes the lowest spin "
            f"multiplicity for the electron count, so an open-shell species "
            f"would be evaluated as a closed-shell one."
        )

    n_frags = len(Chem.GetMolFrags(mol))
    if n_frags > 1:
        raise ValueError(
            f"Multi-component input is not supported: {smiles} has {n_frags} "
            f"disconnected fragments. Desalt the structure and submit the "
            f"single component whose pKa is wanted."
        )
