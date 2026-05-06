"""Broad SMARTS-based protonation/deprotonation enumeration with BFS.

Design principle: keep SMARTS simple and general (match any heteroatom with
the right H-count/charge), not substructure-specific. Generate more variants
rather than fewer — downstream QM energetics filter unrealistic states.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from qm_pka.rdkit_utils import canonical_smiles, get_formal_charge

# Deprotonation reactions: remove one H from a heteroatom, decrease formal charge.
# Each pattern is intentionally broad.
_DEPROTONATION_SMARTS: list[str] = [
    # Neutral heteroatoms with H -> anionic
    "[NH:1]>>[N-:1]",
    "[NH2:1]>>[NH-:1]",
    "[NH3:1]>>[NH2-:1]",
    "[OH:1]>>[O-:1]",
    "[OH2:1]>>[OH-:1]",
    "[SH:1]>>[S-:1]",
    "[PH:1]>>[P-:1]",
    # Cationic heteroatoms with H -> neutral
    "[NH4+:1]>>[NH3:1]",
    "[NH3+:1]>>[NH2:1]",
    "[NH2+:1]>>[NH:1]",
    "[NH+:1]>>[N:1]",
    "[OH2+:1]>>[OH:1]",
    "[OH+:1]>>[O:1]",
    "[SH2+:1]>>[SH:1]",
    "[SH+:1]>>[S:1]",
    # Aromatic N with H
    "[nH:1]>>[n-:1]",
    "[nH+:1]>>[n:1]",
]

# Protonation reactions: add one H to a heteroatom, increase formal charge.
_PROTONATION_SMARTS: list[str] = [
    # Neutral heteroatoms -> cationic
    "[NH2:1]>>[NH3+:1]",
    "[NH:1]>>[NH2+:1]",
    "[N;H0;+0;X3:1]>>[NH+:1]",
    "[OH:1]>>[OH2+:1]",
    "[O;H0;+0:1]>>[OH+:1]",
    "[SH:1]>>[SH2+:1]",
    "[S;H0;+0:1]>>[SH+:1]",
    # Anionic heteroatoms -> neutral
    "[N-:1]>>[NH:1]",
    "[NH-:1]>>[NH2:1]",
    "[NH2-:1]>>[NH3:1]",
    "[O-:1]>>[OH:1]",
    "[OH-:1]>>[OH2:1]",
    "[S-:1]>>[SH:1]",
    # Aromatic N
    "[n;H0;+0:1]>>[nH+:1]",
    "[n-:1]>>[nH:1]",
]


def _compile_reactions(smarts_list: list[str]) -> list[AllChem.ChemicalReaction]:
    reactions: list[AllChem.ChemicalReaction] = []
    for s in smarts_list:
        rxn = AllChem.ReactionFromSmarts(s)
        if rxn is None:
            raise RuntimeError(f"Failed to compile reaction SMARTS: {s}")
        reactions.append(rxn)
    return reactions


_DEPROT_RXNS: list[AllChem.ChemicalReaction] | None = None
_PROT_RXNS: list[AllChem.ChemicalReaction] | None = None


def _get_deprot_rxns() -> list[AllChem.ChemicalReaction]:
    global _DEPROT_RXNS
    if _DEPROT_RXNS is None:
        _DEPROT_RXNS = _compile_reactions(_DEPROTONATION_SMARTS)
    return _DEPROT_RXNS


def _get_prot_rxns() -> list[AllChem.ChemicalReaction]:
    global _PROT_RXNS
    if _PROT_RXNS is None:
        _PROT_RXNS = _compile_reactions(_PROTONATION_SMARTS)
    return _PROT_RXNS


def deprotonate_all_sites(smiles: str) -> list[str]:
    """Remove one proton from every possible heteroatom site.

    Returns a deduplicated list of canonical SMILES for all single-deprotonation
    products. Each product has formal charge one unit lower than the input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    target_charge = get_formal_charge(smiles) - 1
    return _apply_reactions(mol, _get_deprot_rxns(), target_charge)


def protonate_all_sites(smiles: str) -> list[str]:
    """Add one proton to every possible heteroatom site.

    Returns a deduplicated list of canonical SMILES for all single-protonation
    products. Each product has formal charge one unit higher than the input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    target_charge = get_formal_charge(smiles) + 1
    return _apply_reactions(mol, _get_prot_rxns(), target_charge)


def _apply_reactions(
    mol: Chem.Mol,
    reactions: list[AllChem.ChemicalReaction],
    target_charge: int,
) -> list[str]:
    """Apply all reactions to mol, return deduplicated products at target charge."""
    seen: set[str] = set()
    results: list[str] = []
    for rxn in reactions:
        products = rxn.RunReactants((mol,))
        for product_tuple in products:
            for product in product_tuple:
                try:
                    Chem.SanitizeMol(product)
                    charge = Chem.GetFormalCharge(product)
                    if charge != target_charge:
                        continue
                    can = Chem.MolToSmiles(product)
                    if can is not None and can not in seen:
                        seen.add(can)
                        results.append(can)
                except Exception:
                    # Skip products that fail sanitization
                    continue
    return results


def enumerate_charge_state(smiles: str, target_charge: int) -> list[str]:
    """BFS to enumerate all unique species at the target charge.

    Starting from the input SMILES, iteratively applies single protonation
    or deprotonation steps until the target charge is reached. Returns all
    unique canonical SMILES found at the target charge, or an empty list
    if the target charge is unreachable from this input (e.g. asking for
    q=-2 on a molecule with only one ionizable site).
    """
    current_charge = get_formal_charge(smiles)
    if current_charge == target_charge:
        return [canonical_smiles(smiles)]

    step_fn = deprotonate_all_sites if target_charge < current_charge else protonate_all_sites

    n_steps = abs(target_charge - current_charge)
    current_level: set[str] = {canonical_smiles(smiles)}

    for _ in range(n_steps):
        next_level: set[str] = set()
        for smi in current_level:
            products = step_fn(smi)
            next_level.update(products)
        if not next_level:
            # No further (de)protonation sites — target charge unreachable.
            # Return [] rather than the partial-walk SMILES, which would be
            # at the wrong charge and silently feed bogus species to DFT.
            return []
        current_level = next_level

    return sorted(current_level)
