from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from rdkit.Chem import GetPeriodicTable

_PT = GetPeriodicTable()


@dataclass
class Geometry:
    """Molecular geometry: atom symbols and Cartesian coordinates."""

    symbols: tuple[str, ...]
    coords: NDArray[np.float64]  # shape (n_atoms, 3), Angstrom

    def __post_init__(self) -> None:
        if len(self.symbols) != self.coords.shape[0]:
            raise ValueError(
                f"symbols length ({len(self.symbols)}) != coords rows ({self.coords.shape[0]})"
            )
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (n, 3), got {self.coords.shape}")

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def heavy_atom_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.symbols) if s != "H"]

    @property
    def hydrogen_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.symbols) if s == "H"]

    def n_electrons(self, charge: int) -> int:
        """Total electron count for the given molecular charge."""
        return int(sum(_PT.GetAtomicNumber(s) for s in self.symbols)) - charge

    def multiplicity(self, charge: int) -> int:
        """Spin multiplicity (2S+1). Assumes lowest multiplicity (singlet or doublet)."""
        return 1 + self.n_electrons(charge) % 2


@dataclass
class Conformer:
    """A molecular conformer with its energy."""

    geometry: Geometry
    energy: float  # Hartree (xTB/CREST at sampling; replaced by DFT later)
    weight: float | None = None  # Boltzmann weight within its microstate


@dataclass
class Microstate:
    """A tautomeric/protonation microstate with its conformer ensemble."""

    tautomer_id: str  # canonical SMILES (approach 1) or fingerprint hash (approach 2)
    conformers: list[Conformer]
    smiles: str | None = None  # explicit-H canonical SMILES (approach 1), None in approach 2
    includes_enantiomer: bool = False  # True if this represents a collapsed enantiomeric pair


@dataclass
class ChargeState:
    """All microstates at a given molecular charge."""

    charge: int
    microstates: list[Microstate]


@dataclass
class Ensemble:
    """Complete ensemble for a molecule across all charge states."""

    input_smiles: str
    charge_states: dict[int, ChargeState] = field(default_factory=dict)
    settings: dict[str, object] = field(default_factory=dict)
