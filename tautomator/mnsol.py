"""Reading the Minnesota Solvation Database (MNSol v2012) and perceiving its solutes.

MNSol is the reference set for fitting the nonelectrostatic correction
(`nonelectrostatic`): experimental solvation free energies of 790 solutes in
water and 91 other solvents, with one gas-phase M06-2X/MG3S geometry per solute.  It ships
no connectivity, so everything here turns a geometry into an RDKit molecule and
a SMILES the pipeline can start from: bonds from ``rdDetermineBonds`` with the
charge from the file, a repair for the charge-separated Lewis structures it
writes for some sulfur compounds, a check against the database's molecular
formula, and a choice of resonance structure for the SMILES.

The database is distributed under its own license, so neither it nor the set
`scripts/select_mnsol_set.py` writes from it is tracked here, as for the IUPAC
pKa data.  Unpacked, it is a directory holding ``MNSol_alldata.txt`` (one row per
solute-solvent free energy, tab-separated, Latin-1) and ``all_solutes/`` (one
XYZ file per solute, named by its file handle, with charge and multiplicity on
line 3 and atomic numbers in the first column).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolDescriptors

Array = NDArray[np.float64]

TABLE = "MNSol_alldata.txt"
GEOMETRIES = "all_solutes"

# O-H and S-H on any heavy atom; N-H on an amine nitrogen that is neither an
# amide/sulfonamide-type nor attached to an aromatic ring (those are planar).
_POLAR_ROTORS = (
    Chem.MolFromSmarts("[OX2H1,SX2H1]"),
    Chem.MolFromSmarts("[NX3;H1,H2;!$(N-[#6,#7,#15,#16]=[#7,#8,#16]);!$(N-a);!$(N=*)]"),
)

_MULTIPLE = (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
_LOWERED = {Chem.BondType.TRIPLE: Chem.BondType.DOUBLE, Chem.BondType.DOUBLE: Chem.BondType.SINGLE}


class PerceptionError(ValueError):
    """A solute whose structure could not be established from its geometry."""


@dataclass(frozen=True)
class Structure:
    """A solute's structure, size and flexibility, as perceived from its geometry.

    Rotatable bonds are RDKit's strict count.  It leaves out the polar hydrogen
    rotors (O-H, S-H and aliphatic N-H), which move a hydrogen-bond donor and so
    the surface field; those are counted separately.
    """

    smiles: str
    n_atoms: int
    n_heavy: int
    n_fragments: int
    n_rotatable: int
    n_polar_rotors: int


def read_xyz(path: Path) -> tuple[int, int, list[int], Array]:
    """(charge, multiplicity, atomic numbers, coordinates in Angstrom) of an MNSol geometry."""
    lines = path.read_text().splitlines()
    charge, multiplicity = (int(x) for x in lines[2].split())
    rows = [line.split() for line in lines[3:] if line.strip()]
    return (
        charge,
        multiplicity,
        [int(r[0]) for r in rows],
        np.array([[float(x) for x in r[1:4]] for r in rows]),
    )


def mol_from_geometry(numbers: list[int], coords: Array, charge: int) -> Chem.Mol:
    """An RDKit molecule with bond orders perceived from the coordinates."""
    rw = Chem.RWMol()
    for z in numbers:
        rw.AddAtom(Chem.Atom(z))
    conf = Chem.Conformer(len(numbers))
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, tuple(float(x) for x in xyz))
    rw.AddConformer(conf)
    mol = rw.GetMol()
    if len(numbers) == 1:  # a monatomic ion has no bonds to carry its charge
        mol.GetAtomWithIdx(0).SetFormalCharge(charge)
    else:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    Chem.SanitizeMol(mol)
    return mol


def _alternating_path(
    mol: Chem.Mol, start: int, want_multiple: bool, seen: list[int]
) -> list[int]:
    """Atoms from ``start`` to an anion along bonds alternating multiple, single, ..., single."""
    for bond in mol.GetAtomWithIdx(start).GetBonds():
        other = bond.GetOtherAtomIdx(start)
        is_multiple = bond.GetBondType() in _MULTIPLE
        is_single = bond.GetBondType() == Chem.BondType.SINGLE
        if other in seen or not (is_multiple if want_multiple else is_single):
            continue
        if not want_multiple and mol.GetAtomWithIdx(other).GetFormalCharge() == -1:
            return [*seen, other]
        path = _alternating_path(mol, other, not want_multiple, [*seen, other])
        if path:
            return path
    return []


def recombine_charges(mol: Chem.Mol) -> Chem.Mol:
    """Remove charge separation that a neutral resonance structure avoids.

    A cation and an anion joined by a path of alternating multiple and single
    bonds (X+=A-B=C-Y-) are one resonance structure of X-A=B-C=Y; each multiple
    bond on the path is lowered by one, each single bond raised to double, and
    both charges removed.  Nitro groups and sulfoxides have no such path from
    the cation and are left as they are.  ``rdDetermineBonds`` writes the
    separated form for thiophosphates and thiocarbamates, where it hides a
    rotatable P-S or C-S single bond.
    """
    rw = Chem.RWMol(mol)
    Chem.Kekulize(rw, clearAromaticFlags=True)
    changed = True
    while changed:
        changed = False
        for atom in rw.GetAtoms():
            if atom.GetFormalCharge() != 1:
                continue
            path = _alternating_path(rw, atom.GetIdx(), True, [atom.GetIdx()])
            if not path:
                continue
            for a, b in pairwise(path):
                bond = rw.GetBondBetweenAtoms(a, b)
                bond.SetBondType(_LOWERED.get(bond.GetBondType(), Chem.BondType.DOUBLE))
                bond.SetStereo(Chem.BondStereo.STEREONONE)
            rw.GetAtomWithIdx(path[0]).SetFormalCharge(0)
            rw.GetAtomWithIdx(path[-1]).SetFormalCharge(0)
            changed = True
            break
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def formula_counts(formula: str) -> dict[str, int]:
    """Element counts of a database formula such as H5C6O1BR1."""
    return {el.capitalize(): int(n) for el, n in re.findall(r"([A-Z]+?)(\d+)", formula)}


def perceive(database: Path, handle: str, formula: str) -> Chem.Mol:
    """The solute as an RDKit molecule with hydrogens, checked against the database.

    Raises `PerceptionError` for an open-shell solute, for one whose bonds
    cannot be perceived, and for one whose perceived formula or charge differs
    from the database's.
    """
    charge, multiplicity, numbers, coords = read_xyz(database / GEOMETRIES / f"{handle}.xyz")
    if multiplicity != 1:
        raise PerceptionError("open shell")
    try:
        mol = recombine_charges(mol_from_geometry(numbers, coords, charge))
    except Exception as exc:
        raise PerceptionError(f"bond perception: {exc}".splitlines()[0][:80]) from exc
    perceived: dict[str, int] = {}
    for atom in mol.GetAtoms():
        perceived[atom.GetSymbol()] = perceived.get(atom.GetSymbol(), 0) + 1
    if perceived != formula_counts(formula) or Chem.GetFormalCharge(mol) != charge:
        raise PerceptionError("perceived structure does not match the database formula or charge")
    return mol


def _charged_atoms(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge())


def _aromatic_atoms(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())


def to_smiles(mol: Chem.Mol) -> str:
    """Canonical SMILES of the solute, written as its most conventional resonance structure.

    Bond perception is free to return any valid Lewis structure, and for
    delocalised ions it sometimes picks an unusual one: 2-nitrophenolate comes
    back as a quinoid nitronate, O=C1C=CC=CC1=[N+]([O-])[O-].  That is the same
    species as [O-]c1ccccc1[N+](=O)[O-], but it reads as a different one and
    hides the rotatable C-N bond.  Among the resonance structures the one with
    the fewest charged atoms, then the most aromatic atoms, is preferred; the
    perceived structure is kept unless another is strictly better on those, so
    its stereochemistry survives wherever nothing needed fixing.
    """
    # Double-bond stereo defined only by a hydrogen (the O-H of a protonated
    # carbonyl, the N-H of an imidate) is dropped with the hydrogen: it would put
    # an explicit [H] in the SMILES and pin what is really a conformer choice.
    params = Chem.RemoveHsParameters()
    params.removeDefiningBondStereo = True
    heavy = Chem.RemoveHs(mol, params)
    best, best_rank = heavy, (_charged_atoms(heavy), -_aromatic_atoms(heavy))
    for form in Chem.ResonanceMolSupplier(heavy):
        if form is None:
            continue
        form = Chem.Mol(form)
        Chem.SanitizeMol(form)
        rank = (_charged_atoms(form), -_aromatic_atoms(form))
        if rank < best_rank:
            best, best_rank = form, rank
    return str(Chem.MolToSmiles(best))


def describe(mol: Chem.Mol) -> Structure:
    """SMILES, size and flexibility of a perceived solute (a molecule with hydrogens)."""
    smiles = to_smiles(mol)
    heavy = Chem.MolFromSmiles(smiles)
    return Structure(
        smiles=smiles,
        n_atoms=mol.GetNumAtoms(),
        n_heavy=mol.GetNumHeavyAtoms(),
        n_fragments=len(Chem.GetMolFrags(heavy)),
        n_rotatable=rdMolDescriptors.CalcNumRotatableBonds(heavy, strict=True),
        n_polar_rotors=sum(len(heavy.GetSubstructMatches(q)) for q in _POLAR_ROTORS),
    )
