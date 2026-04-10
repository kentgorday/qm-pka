"""RDKit interface: SMILES<->3D, tautomer enumeration, atom-mapped SMILES."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from qm_pka.types import Geometry


def smiles_to_3d(smiles: str) -> tuple[Geometry, str]:
    """Generate a 3D geometry from a SMILES string via ETKDG embedding.

    Returns (geometry, explicit_h_smiles) where the geometry's atom
    ordering matches the SMILES atom ordering. Coordinates are reordered
    using _smilesAtomOutputOrder so that geometry index i corresponds
    to SMILES atom i.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status != 0:
        raise RuntimeError(f"ETKDG embedding failed for: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    all_coords = np.array(conf.GetPositions(), dtype=np.float64)

    explicit_h_smiles: str | None = Chem.MolToSmiles(mol)
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
