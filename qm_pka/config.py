"""Configuration for the three-stage pKa workflow.

Reads a single TOML file that configures:
  [molecule]   — input SMILES and charge range
  [sampling]   — CREST-based conformer/tautomer/protonation sampling
  [refinement] — geometry optimization with cheap DFT + quasi-RRHO correction
  [scoring]    — single-point energy with expensive DFT
  [compute]    — driver (psi4/pyscf), threads, output directory
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Driver-specific defaults
# ---------------------------------------------------------------------------
_REFINEMENT_DEFAULTS: dict[str, dict[str, str | None]] = {
    "psi4": {
        "method": "wB97X-3c",
        "basis": "vDZP",
        "solvent_model": None,
    },
    "pyscf": {
        "method": "wB97X-3c",
        "basis": "vDZP",
        "solvent_model": "SSVPE",
    },
}

_SCORING_DEFAULTS: dict[str, dict[str, str | None]] = {
    "psi4": {
        "method": "wB97M-V",
        "basis": "def2-QZVPPD",
        "solvent_model": "IEFPCM",
    },
    "pyscf": {
        "method": "wB97M-V",
        "basis": "def2-QZVPPD",
        "solvent_model": "SSVPE",
    },
}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MoleculeConfig:
    smiles: str
    charge_min: int = -1
    charge_max: int = 0

    @property
    def charge_range(self) -> tuple[int, int]:
        return (self.charge_min, self.charge_max)


@dataclass
class SamplingConfig:
    approach: str = "rdkit_first"  # "rdkit_first" or "crest_first"
    ewin: float = 10.0  # energy window (kcal/mol)
    # Approach 1 (rdkit_first) options
    max_tautomers: int = 1000
    max_transforms: int = 1000
    crest_mode: str = "default"  # conformer search mode
    # Approach 2 (crest_first) options
    prescreen_mode: str = "quick"
    prescreen_ewin: float = 6.0
    full_mode: str = "default"


@dataclass
class RefinementConfig:
    method: str = ""
    basis: str = ""
    solvent_model: str | None = None
    solvent: str | None = None  # e.g. "water" — required if solvent_model is set
    ewin: float = 10.0  # energy window (kcal/mol) for filtering after refinement
    pcm_hydrogen_radius: float = 1.1  # PCM cavity radius (Angstrom) for hydrogen
    # quasi-RRHO recompute on the DFT geometry: "xtb" (GFN2 single-point/biased
    # Hessian via xtb --bhess, always implicit solvent) or "dft" (Hessian at the
    # refinement DFT level, matching the refinement solvent choice).
    rrho_method: str = "xtb"


@dataclass
class ScoringConfig:
    method: str = ""
    basis: str = ""
    solvent_model: str | None = None
    solvent: str | None = None
    ewin: float = 10.0
    pcm_hydrogen_radius: float = 1.1  # PCM cavity radius (Angstrom) for hydrogen


@dataclass
class ComputeConfig:
    driver: str = "pyscf"  # "psi4" or "pyscf"
    threads: int | None = None
    output_dir: str = "output"


@dataclass
class PkaConfig:
    molecule: MoleculeConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)


# ---------------------------------------------------------------------------
# TOML loading + validation
# ---------------------------------------------------------------------------
def load_config(path: Path) -> PkaConfig:
    """Load and validate a pKa workflow config from a TOML file."""
    raw = tomllib.loads(path.read_text())

    # [molecule] — required
    mol_raw = raw.get("molecule")
    if mol_raw is None:
        raise ValueError("Config must have a [molecule] section with 'smiles'")
    if "smiles" not in mol_raw:
        raise ValueError("[molecule] must specify 'smiles'")
    molecule = MoleculeConfig(
        smiles=mol_raw["smiles"],
        charge_min=mol_raw.get("charge_min", -1),
        charge_max=mol_raw.get("charge_max", 0),
    )
    if molecule.charge_min > molecule.charge_max:
        raise ValueError(
            f"charge_min ({molecule.charge_min}) > charge_max ({molecule.charge_max})"
        )

    # [compute] — optional, needed before refinement/scoring for driver defaults
    compute_raw = raw.get("compute", {})
    compute = ComputeConfig(
        driver=compute_raw.get("driver", "pyscf"),
        threads=compute_raw.get("threads"),
        output_dir=compute_raw.get("output_dir", "output"),
    )
    if compute.driver not in ("psi4", "pyscf"):
        raise ValueError(f"Unknown driver: {compute.driver!r} (must be 'psi4' or 'pyscf')")

    # [sampling] — optional
    samp_raw = raw.get("sampling", {})
    sampling = SamplingConfig(
        approach=samp_raw.get("approach", "rdkit_first"),
        ewin=samp_raw.get("ewin", 10.0),
        max_tautomers=samp_raw.get("max_tautomers", 1000),
        max_transforms=samp_raw.get("max_transforms", 1000),
        crest_mode=samp_raw.get("crest_mode", "default"),
        prescreen_mode=samp_raw.get("prescreen_mode", "quick"),
        prescreen_ewin=samp_raw.get("prescreen_ewin", 6.0),
        full_mode=samp_raw.get("full_mode", "default"),
    )
    if sampling.approach not in ("rdkit_first", "crest_first"):
        raise ValueError(
            f"Unknown sampling approach: {sampling.approach!r} "
            f"(must be 'rdkit_first' or 'crest_first')"
        )

    # [refinement] — optional, driver-specific defaults applied
    # Default solvent_model only applies when the user specifies a solvent;
    # otherwise it stays None (gas phase).
    ref_raw = raw.get("refinement", {})
    ref_defaults = _REFINEMENT_DEFAULTS[compute.driver]
    ref_solvent = ref_raw.get("solvent")
    ref_solvent_model: str | None = ref_raw.get(
        "solvent_model",
        ref_defaults.get("solvent_model") if ref_solvent is not None else None,
    )
    refinement = RefinementConfig(
        method=ref_raw.get("method", ref_defaults["method"]),
        basis=ref_raw.get("basis", ref_defaults["basis"]),
        solvent_model=ref_solvent_model,
        solvent=ref_solvent,
        ewin=ref_raw.get("ewin", 10.0),
        pcm_hydrogen_radius=ref_raw.get("pcm_hydrogen_radius", 1.1),
        rrho_method=ref_raw.get("rrho_method", "xtb"),
    )
    if refinement.rrho_method not in ("xtb", "dft"):
        raise ValueError(
            f"Unknown rrho_method: {refinement.rrho_method!r} (must be 'xtb' or 'dft')"
        )

    # [scoring] — optional, driver-specific defaults applied
    score_raw = raw.get("scoring", {})
    score_defaults = _SCORING_DEFAULTS[compute.driver]
    score_solvent = score_raw.get("solvent")
    score_solvent_model: str | None = score_raw.get(
        "solvent_model",
        score_defaults.get("solvent_model") if score_solvent is not None else None,
    )
    scoring = ScoringConfig(
        method=score_raw.get("method", score_defaults["method"]),
        basis=score_raw.get("basis", score_defaults["basis"]),
        solvent_model=score_solvent_model,
        solvent=score_solvent,
        ewin=score_raw.get("ewin", 10.0),
        pcm_hydrogen_radius=score_raw.get("pcm_hydrogen_radius", 1.1),
    )

    # Validation: solvent_model requires solvent
    if refinement.solvent_model is not None and refinement.solvent is None:
        raise ValueError(
            f"[refinement] specifies solvent_model={refinement.solvent_model!r} "
            f"but no solvent (e.g. solvent = 'water')"
        )
    if scoring.solvent_model is not None and scoring.solvent is None:
        raise ValueError(
            f"[scoring] specifies solvent_model={scoring.solvent_model!r} "
            f"but no solvent (e.g. solvent = 'water')"
        )

    # Warning: psi4 + solvent in refinement is very slow
    if compute.driver == "psi4" and refinement.solvent_model is not None:
        log.warning(
            "Psi4 has no analytical gradients for PCM solvent models. "
            "Geometry optimization with implicit solvent will be extremely slow. "
            "Consider using pyscf as the driver, or removing solvent_model from [refinement]."
        )

    # Error: DFT-level RRHO in implicit solvent requires PCM second derivatives,
    # which Psi4 lacks entirely (no analytical PCM gradients). Use rrho_method =
    # "xtb" (ALPB Hessian) or drop the refinement solvent.
    if (
        refinement.rrho_method == "dft"
        and compute.driver == "psi4"
        and refinement.solvent_model is not None
    ):
        raise ValueError(
            "[refinement] rrho_method = 'dft' with an implicit solvent is not "
            "supported on the psi4 driver: Psi4 has no analytical PCM Hessians. "
            "Use rrho_method = 'xtb', remove solvent_model from [refinement], or "
            "switch to the pyscf driver."
        )

    return PkaConfig(
        molecule=molecule,
        sampling=sampling,
        refinement=refinement,
        scoring=scoring,
        compute=compute,
    )
