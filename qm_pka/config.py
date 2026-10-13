"""Configuration for the three-stage pKa workflow.

Reads a single TOML file that configures:
  [molecule]   — input SMILES, charge range, solvent
  [sampling]   — CREST-based conformer/tautomer/protonation sampling
  [refinement] — geometry optimization + solvation + quasi-RRHO
  [scoring]    — single-point energy with expensive DFT
  [compute]    — DFT driver (psi4/pyscf), threads, output directory

The config says which physics goes into each energy term; the runners decide how
to get it (analytic vs finite-difference Hessians, fallback functionals for
second derivatives, grids, cavity construction) and record what they did.

Refinement produces three terms, each with its own setting:
  electronic — ``method``/``basis``.  An OrbMol model name here (see
      qm_pka.orbmol_runner.MODELS) makes OrbMol the optimizer; anything else is
      DFT on the [compute] driver.
  solvation  — ``solvation``: "optimize" (minimize in implicit solvent),
      "single_point" (gas minimization, then one solvated single point) or
      "none" (gas phase; the filter then ranks on gas energies, which reranks
      badly against solvated scoring).
  RRHO       — ``rrho``: "xtb" (GFN2 biased Hessian in ALPB) or "refinement"
      (a Hessian from whatever optimized the geometry).

Scoring always recomputes electronic + solvation and carries RRHO over.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Per-QM-job memory ceiling, shared by both backends so a given config means
# the same thing on either one.  Their native defaults disagree by 8x (Psi4
# 500 MiB, PySCF 4 GB), which is an invisible backend difference in a project
# that cross-checks the two.  4 GB matches PySCF's own default, so only Psi4
# moves; measured wall time on def2-QZVPPD/wB97M-V is flat from 4 GB up, so
# there is nothing to buy by going higher.
DEFAULT_MEMORY_GB = 4.0

log = logging.getLogger(__name__)

Solvation = Literal["none", "optimize", "single_point"]
Rrho = Literal["xtb", "refinement"]

DRIVERS = ("psi4", "pyscf")

# "3c" composite methods are a functional *and* the basis they were
# parameterized with; pairing one with another basis is a different method.
COMPOSITE_BASIS: dict[str, str] = {
    "wb97x-3c": "vDZP",
    "r2scan-3c": "def2-mTZVPP",
    "b97-3c": "def2-mTZVP",
    "pbeh-3c": "def2-mSVP",
    "hf-3c": "MINIX",
}
# PySCF implements only wB97X-3c of those (pyscf.scf.dispersion: "Only wb97x-3c
# is supported for now"); Psi4 has all five.
PYSCF_COMPOSITES = {"wb97x-3c"}

# Implicit-solvent models, by the driver that can run them.  One model is used
# for every solvation energy in a run, so the refinement filter ranks on the
# same physics scoring does.  Both drivers default to IEF-PCM, which agrees
# across them to ~0.15 kcal/mol.  PySCF's SS(V)PE is not offered: it symmetrizes
# the system matrix, and on PySCF's surface that moved the solvation energy of
# acetate and methylammonium by 2.7 and 4.2 kcal/mol and did not converge with
# the surface grid (the ill behaviour Lange & Herbert describe, Chem. Phys.
# Lett. 509, 77).
_SOLVENT_MODELS: dict[str, set[str]] = {
    "pyscf": {"IEFPCM", "PCM", "CPCM", "COSMO", "SMD", "DDPCM", "DDCOSMO"},
    "psi4": {"IEFPCM", "CPCM"},
}

# The solvation single point for an OrbMol refinement: OrbMol has no implicit
# solvent, so the term comes from a cheap DFT pair (PCM and gas) on the
# [compute] driver.  wB97X-3c/vDZP is the default and the benchmarked choice.
DEFAULT_ORBMOL_SOLVATION_METHOD = "wB97X-3c"


@dataclass
class MoleculeConfig:
    smiles: str
    charge_min: int = -1
    charge_max: int = 0
    solvent: str = "water"  # used by sampling (ALPB), refinement and scoring

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
    method: str = ""  # DFT functional, or an OrbMol model (see MODELS)
    basis: str = ""  # empty for OrbMol; fixed by the method for 3c composites
    solvation: Solvation = "single_point"
    # Level for the solvation single point when OrbMol optimizes (OrbMol has no
    # implicit solvent).  Refused for DFT refinement, where the single point is
    # at the refinement level so the optimizer's gas-phase energy can be reused.
    solvation_method: str = ""
    solvation_basis: str = ""
    # Optional cost control: skip the solvation single point for conformers
    # already this far above the charge state's minimum in gas phase.  Off by
    # default; skipped conformers are recorded as excluded_conformers with
    # reason "gas_prefilter".
    solvation_prefilter_ewin: float | None = None
    rrho: Rrho = "xtb"
    ewin: float = 8.0  # energy window (kcal/mol) for filtering after refinement
    pcm_hydrogen_radius: float = 1.1  # PCM cavity radius (Angstrom) for hydrogen
    # Derived: "orbmol", or the [compute] driver for DFT refinement.
    engine: str = "pyscf"


@dataclass
class ScoringConfig:
    method: str = ""
    basis: str = ""
    solvent_model: str | None = None  # the run's implicit-solvent model
    ewin: float = 10.0
    pcm_hydrogen_radius: float = 1.1  # PCM cavity radius (Angstrom) for hydrogen


@dataclass
class ComputeConfig:
    driver: str = "pyscf"  # "psi4" or "pyscf": scoring, and any DFT in refinement
    threads: int | None = None
    memory_gb: float = DEFAULT_MEMORY_GB
    output_dir: str = "output"


@dataclass
class PkaConfig:
    molecule: MoleculeConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    protocol: str | None = None

    def provenance(self) -> dict[str, Any]:
        """What each stage computed, for the output file and for resume checks."""
        ref, sc = self.refinement, self.scoring
        solvation: dict[str, Any] = {"mode": ref.solvation}
        if ref.solvation != "none":
            solvation |= {
                "level": (
                    f"{ref.solvation_method}/{ref.solvation_basis}"
                    if ref.engine == "orbmol"
                    else f"{ref.method}/{ref.basis}"
                ),
                "model": sc.solvent_model,
                "driver": self.compute.driver,
            }
        return {
            "protocol": self.protocol,
            "solvent": self.molecule.solvent,
            "refinement": {
                "engine": ref.engine,
                "method": ref.method,
                "basis": ref.basis,
                "solvation": solvation,
                "rrho": ref.rrho,
                "ewin": ref.ewin,
                "solvation_prefilter_ewin": ref.solvation_prefilter_ewin,
            },
            "scoring": {
                "driver": self.compute.driver,
                "method": sc.method,
                "basis": sc.basis,
                "solvent_model": sc.solvent_model,
                "ewin": sc.ewin,
            },
        }


# ---------------------------------------------------------------------------
# Recommended protocols
# ---------------------------------------------------------------------------
# Defaults only: any key written in the file wins.  "fast" is ~5x cheaper than
# "high_quality" and, measured against it on training-set molecules, agrees to
# ~0.05 pKa at the scoring level and <=0.01 pKa on the refinement window.
PROTOCOLS: dict[str, dict[str, dict[str, Any]]] = {
    "high_quality": {
        "sampling": {"ewin": 10.0},
        "refinement": {"method": "wB97X-3c", "ewin": 8.0, "rrho": "xtb"},
        "scoring": {"method": "wB97M-V", "basis": "def2-QZVPPD", "ewin": 10.0},
    },
    "fast": {
        # A wider sampling window is affordable because OrbMol refinement costs
        # seconds per conformer, and xTB misranks against DFT in one direction.
        "sampling": {"ewin": 15.0},
        "refinement": {
            "method": "OrbMol-v2",
            "solvation": "single_point",
            "solvation_prefilter_ewin": 12.0,
            "ewin": 6.0,
            "rrho": "xtb",
        },
        "scoring": {"method": "wB97M-V", "basis": "def2-TZVPPD", "ewin": 10.0},
    },
}

_DFT_REFINEMENT_DEFAULT_METHOD = "wB97X-3c"
_SCORING_DEFAULTS = {"method": "wB97M-V", "basis": "def2-QZVPPD"}
_DEFAULT_SOLVENT_MODEL = {"pyscf": "IEFPCM", "psi4": "IEFPCM"}


# ---------------------------------------------------------------------------
# TOML loading + validation
# ---------------------------------------------------------------------------
def load_config(path: Path) -> PkaConfig:
    """Load and validate a pKa workflow config from a TOML file."""
    raw = tomllib.loads(path.read_text())
    protocol = raw.get("protocol")
    if protocol is not None and protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol!r} (must be one of {', '.join(PROTOCOLS)})")
    preset = PROTOCOLS.get(protocol or "", {})

    def section(name: str) -> dict[str, Any]:
        merged = dict(preset.get(name, {}))
        merged.update(raw.get(name, {}))
        return merged

    mol_raw = raw.get("molecule")
    if mol_raw is None:
        raise ValueError("Config must have a [molecule] section with 'smiles'")
    if "smiles" not in mol_raw:
        raise ValueError("[molecule] must specify 'smiles'")
    mol_raw = dict(mol_raw)

    ref_raw, score_raw, compute_raw = section("refinement"), section("scoring"), section("compute")
    _migrate(ref_raw, score_raw, compute_raw, mol_raw)

    molecule = MoleculeConfig(
        smiles=mol_raw["smiles"],
        charge_min=mol_raw.get("charge_min", -1),
        charge_max=mol_raw.get("charge_max", 0),
        solvent=mol_raw.get("solvent", "water"),
    )
    if molecule.charge_min > molecule.charge_max:
        raise ValueError(
            f"charge_min ({molecule.charge_min}) > charge_max ({molecule.charge_max})"
        )

    compute = ComputeConfig(
        driver=compute_raw.get("driver", "pyscf"),
        threads=compute_raw.get("threads"),
        memory_gb=compute_raw.get("memory_gb", DEFAULT_MEMORY_GB),
        output_dir=compute_raw.get("output_dir", "output"),
    )
    if compute.memory_gb <= 0:
        raise ValueError(f"[compute] memory_gb must be positive, got {compute.memory_gb}")
    if compute.driver == "orbmol":
        raise ValueError(
            "[compute] driver = 'orbmol' is not supported: the driver runs scoring, "
            "which needs implicit solvent, and OrbMol-v2 is gas-phase only. To refine "
            "with OrbMol set [refinement] method = 'OrbMol-v2' and keep [compute] "
            "driver = 'psi4' or 'pyscf'."
        )
    if compute.driver not in DRIVERS:
        raise ValueError(f"Unknown driver: {compute.driver!r} (must be 'psi4' or 'pyscf')")

    samp_raw = section("sampling")
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

    scoring = _build_scoring(score_raw, compute.driver)
    refinement = _build_refinement(ref_raw, compute.driver, scoring)
    _validate(refinement, scoring, compute)

    return PkaConfig(
        molecule=molecule,
        sampling=sampling,
        refinement=refinement,
        scoring=scoring,
        compute=compute,
        protocol=protocol,
    )


def _build_scoring(raw: dict[str, Any], driver: str) -> ScoringConfig:
    model = raw.get("solvent_model", _DEFAULT_SOLVENT_MODEL[driver])
    if isinstance(model, str) and model.lower() == "none":
        model = None
    return ScoringConfig(
        method=raw.get("method", _SCORING_DEFAULTS["method"]),
        basis=raw.get("basis", _SCORING_DEFAULTS["basis"]),
        solvent_model=model,
        ewin=raw.get("ewin", 10.0),
        pcm_hydrogen_radius=raw.get("pcm_hydrogen_radius", 1.1),
    )


def _build_refinement(
    raw: dict[str, Any], driver: str, scoring: ScoringConfig
) -> RefinementConfig:
    from qm_pka.orbmol_runner import MODELS

    method = raw.get("method", _DFT_REFINEMENT_DEFAULT_METHOD)
    engine = "orbmol" if method.lower() in MODELS else driver
    basis = raw.get("basis", "" if engine == "orbmol" else _default_basis(method))
    solvation: Solvation = raw.get("solvation", _default_solvation(engine, scoring))
    solvation_method = raw.get("solvation_method", "")
    if engine == "orbmol" and solvation != "none" and not solvation_method:
        solvation_method = DEFAULT_ORBMOL_SOLVATION_METHOD
    return RefinementConfig(
        method=method,
        basis=basis,
        solvation=solvation,
        solvation_method=solvation_method,
        solvation_basis=raw.get("solvation_basis", _default_basis(solvation_method)),
        solvation_prefilter_ewin=raw.get("solvation_prefilter_ewin"),
        rrho=raw.get("rrho", "xtb"),
        ewin=raw.get("ewin", 8.0 if solvation != "none" else 10.0),
        pcm_hydrogen_radius=raw.get("pcm_hydrogen_radius", 1.1),
        engine=engine,
    )


def _default_basis(method: str) -> str:
    return COMPOSITE_BASIS.get(method.lower(), "")


def _default_solvation(engine: str, scoring: ScoringConfig) -> Solvation:
    if scoring.solvent_model is None:
        return "none"  # gas-phase scoring: the filter must rank on gas energies too
    # PySCF has analytical PCM gradients, so it can minimize in solvent; Psi4
    # cannot, and OrbMol has no implicit solvent at all.
    return "optimize" if engine == "pyscf" else "single_point"


def _validate(ref: RefinementConfig, scoring: ScoringConfig, compute: ComputeConfig) -> None:
    if ref.solvation not in ("none", "optimize", "single_point"):
        raise ValueError(
            f"Unknown [refinement] solvation: {ref.solvation!r} "
            "(must be 'none', 'optimize' or 'single_point')"
        )
    if ref.rrho not in ("xtb", "refinement"):
        raise ValueError(
            f"Unknown [refinement] rrho: {ref.rrho!r} (must be 'xtb' or 'refinement')"
        )

    # Level of theory: the 3c composites carry their basis with them.
    _check_composite("[refinement] method", ref.method, ref.basis, compute.driver)
    _check_composite(
        "[refinement] solvation_method", ref.solvation_method, ref.solvation_basis, compute.driver
    )
    _check_composite("[scoring] method", scoring.method, scoring.basis, compute.driver)

    if ref.engine == "orbmol":
        if ref.basis:
            raise ValueError(
                f"[refinement] basis = {ref.basis!r} has no meaning for an OrbMol model "
                "(the model is the level of theory); remove it."
            )
        if ref.solvation == "optimize":
            raise ValueError(
                "[refinement] solvation = 'optimize' is not supported with an OrbMol "
                "model: OrbMol-v2 is a gas-phase potential. Use 'single_point', which "
                f"adds a {DEFAULT_ORBMOL_SOLVATION_METHOD} solvation energy on the "
                "OrbMol geometry."
            )
    else:
        if ref.solvation_method or ref.solvation_basis:
            raise ValueError(
                "[refinement] solvation_method/solvation_basis apply only to an OrbMol "
                "refinement; DFT refinement takes its solvation single point at the "
                "refinement level, reusing the optimizer's gas-phase energy."
            )
        if ref.engine == "psi4" and ref.solvation == "optimize":
            raise ValueError(
                "[refinement] solvation = 'optimize' is not supported on the psi4 "
                "driver: Psi4 has no analytical PCM gradients, so minimization in "
                "implicit solvent falls back to finite-difference gradients. Use "
                "'single_point', or switch [compute] driver to pyscf."
            )

    if ref.solvation != "none" and scoring.solvent_model is None:
        raise ValueError(
            "[refinement] computes a solvation energy but [scoring] is gas-phase "
            "(solvent_model = 'none'); the refinement filter would rank on different "
            "physics from scoring. Set [refinement] solvation = 'none', or give "
            "[scoring] a solvent model."
        )
    if ref.solvation == "none" and scoring.solvent_model is not None:
        log.warning(
            "[refinement] solvation = 'none' with solvated scoring: conformers are "
            "filtered on gas-phase energies, which rerank against scoring (a 4 kcal/mol "
            "window cost 3.3 pKa in testing). Keep ewin >= 10, or use "
            "solvation = 'single_point'."
        )

    if scoring.solvent_model is not None:
        allowed = _SOLVENT_MODELS[compute.driver]
        if scoring.solvent_model.upper().replace("-", "") not in {m.upper() for m in allowed}:
            raise ValueError(
                f"[scoring] solvent_model = {scoring.solvent_model!r} is not available "
                f"on the {compute.driver} driver (has: {', '.join(sorted(allowed))})."
            )

    _validate_prefilter(ref)

    if ref.rrho == "refinement" and ref.engine == "psi4":
        log.warning(
            "[refinement] rrho = 'refinement' on the psi4 driver: Psi4 has no analytical "
            "Hessian for these functionals, so every conformer costs a finite-difference "
            "Hessian (~6N gradient calculations) -- usually far more than the geometry "
            "optimization itself. rrho = 'xtb' is the default for good reason."
        )


def _validate_prefilter(ref: RefinementConfig) -> None:
    prefilter = ref.solvation_prefilter_ewin
    if prefilter is None:
        return
    if ref.solvation != "single_point":
        raise ValueError(
            "[refinement] solvation_prefilter_ewin only applies to "
            "solvation = 'single_point' (there is no solvation single point to skip "
            f"with solvation = {ref.solvation!r})."
        )
    if prefilter <= ref.ewin:
        raise ValueError(
            f"[refinement] solvation_prefilter_ewin ({prefilter}) must exceed ewin "
            f"({ref.ewin}): it would drop conformers inside the window it feeds."
        )
    if prefilter < ref.ewin + 4.0:
        log.warning(
            f"[refinement] solvation_prefilter_ewin ({prefilter}) is within 4 kcal/mol "
            f"of ewin ({ref.ewin}). Gas-phase and solvated rankings differ: conformers "
            "inside a 6 kcal/mol solvated window sat up to 9.1 kcal/mol above the "
            "gas-phase minimum in testing."
        )
    if ref.engine != "orbmol":
        log.warning(
            "[refinement] solvation_prefilter_ewin with DFT refinement saves little: "
            "the solvation single point is ~10% of one geometry optimization at the "
            "same level. It matters for OrbMol, where optimization takes seconds."
        )


def _check_composite(label: str, method: str, basis: str, driver: str) -> None:
    """A 3c composite is a functional *and* its basis; refuse other pairings."""
    if not method:
        return
    key = method.lower()
    required = COMPOSITE_BASIS.get(key)
    if required is None:
        return
    if basis and basis.lower() != required.lower():
        raise ValueError(
            f"{label} = {method!r} is parameterized with {required}, not {basis!r}; "
            f"{method} with another basis is a different method. Drop the basis to get "
            "the right one."
        )
    if driver == "pyscf" and key not in PYSCF_COMPOSITES:
        raise ValueError(
            f"{label} = {method!r} is not available on the pyscf driver (it implements "
            f"only {', '.join(sorted(PYSCF_COMPOSITES))} of the 3c composites). Use "
            "[compute] driver = 'psi4', or wB97X-3c."
        )


def _migrate(
    ref_raw: dict[str, Any],
    score_raw: dict[str, Any],
    compute_raw: dict[str, Any],
    mol_raw: dict[str, Any],
) -> None:
    """Accept the pre-protocol spellings, with a warning."""
    if (driver := ref_raw.pop("driver", None)) is not None:
        if driver == "orbmol":
            ref_raw.setdefault("method", "OrbMol-v2")
        elif driver != compute_raw.get("driver", "pyscf"):
            raise ValueError(
                f"[refinement] driver = {driver!r} no longer selects a DFT backend: "
                "[compute] driver runs every DFT job. Set [compute] driver instead."
            )
        log.warning(
            "[refinement] driver is deprecated: an OrbMol model in [refinement] method "
            "selects OrbMol, and [compute] driver runs the DFT."
        )
    if (rrho_method := ref_raw.pop("rrho_method", None)) is not None:
        ref_raw.setdefault("rrho", "refinement" if rrho_method == "dft" else rrho_method)
        log.warning(
            f"[refinement] rrho_method = {rrho_method!r} is deprecated: use "
            f"rrho = {ref_raw['rrho']!r} ('dft' is now 'refinement', a Hessian from "
            "whatever optimized the geometry)."
        )
    if (model := ref_raw.pop("solvent_model", None)) is not None:
        ref_raw.setdefault("solvation", "optimize")
        score_raw.setdefault("solvent_model", model)
        log.warning(
            f"[refinement] solvent_model = {model!r} is deprecated: [refinement] "
            "solvation says how solvation enters refinement, and [scoring] "
            "solvent_model is the model for the whole run."
        )
    for name, sec in (("refinement", ref_raw), ("scoring", score_raw)):
        if (solvent := sec.pop("solvent", None)) is not None:
            if mol_raw.setdefault("solvent", solvent) != solvent:
                raise ValueError(
                    f"[{name}] solvent = {solvent!r} conflicts with "
                    f"[molecule] solvent = {mol_raw['solvent']!r}; a run has one solvent."
                )
            log.warning(f"[{name}] solvent is deprecated: use [molecule] solvent.")
