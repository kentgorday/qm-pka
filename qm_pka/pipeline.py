"""Three-stage pKa prediction pipeline: sampling → refinement → scoring."""

from __future__ import annotations

import logging
from pathlib import Path

from qm_pka.config import PkaConfig
from qm_pka.ensemble import assign_weights, serialize_ensemble
from qm_pka.refinement import refine
from qm_pka.sampling import run_approach1, run_approach2
from qm_pka.scoring import score
from qm_pka.types import Ensemble

log = logging.getLogger(__name__)


def run_pipeline(config: PkaConfig) -> Ensemble:
    """Run the full three-stage pKa prediction pipeline.

    Stage 1 (Sampling): CREST-based conformer/tautomer/protonation enumeration,
        with an xTB-level RRHO correction folded into the energy filter.
    Stage 2 (Refinement): DFT geometry optimization + RRHO recompute on the DFT
        geometry (xtb --bhess or refinement-level DFT, per rrho_method).
    Stage 3 (Scoring): DFT single-point energy (RRHO carried over unchanged).

    Returns Ensemble with Boltzmann weights assigned.
    """
    output_dir = Path(config.compute.output_dir)

    # Aqueous solvent used for xTB (ALPB) throughout: conformer sampling and the
    # xtb-level RRHO at both sampling and refinement.
    solvent = "water"

    # Stage 1: Sampling
    log.info("=== Stage 1: Sampling ===")
    if config.sampling.approach == "rdkit_first":
        ensemble = run_approach1(
            smiles=config.molecule.smiles,
            charge_range=config.molecule.charge_range,
            solvent=solvent,
            crest_mode=config.sampling.crest_mode,
            ewin=config.sampling.ewin,
            threads=config.compute.threads,
            max_tautomers=config.sampling.max_tautomers,
            max_transforms=config.sampling.max_transforms,
        )
    else:
        ensemble = run_approach2(
            smiles=config.molecule.smiles,
            charge_range=config.molecule.charge_range,
            solvent=solvent,
            prescreen_mode=config.sampling.prescreen_mode,
            full_mode=config.sampling.full_mode,
            prescreen_ewin=config.sampling.prescreen_ewin,
            ewin=config.sampling.ewin,
            threads=config.compute.threads,
        )
    serialize_ensemble(ensemble, output_dir / "sampling")

    # Stage 2: Refinement
    log.info("=== Stage 2: Refinement ===")
    ref = config.refinement
    threads = config.compute.threads or 1
    refine(
        ensemble,
        driver_name=config.compute.driver,
        method=ref.method,
        basis=ref.basis,
        solvent_model=ref.solvent_model,
        solvent=ref.solvent,
        ewin=ref.ewin,
        pcm_hydrogen_radius=ref.pcm_hydrogen_radius,
        rrho_method=ref.rrho_method,
        xtb_rrho_solvent=solvent,
        threads=threads,
    )
    serialize_ensemble(ensemble, output_dir / "refinement")

    # Stage 3: Scoring
    log.info("=== Stage 3: Scoring ===")
    sc = config.scoring
    score(
        ensemble,
        driver_name=config.compute.driver,
        method=sc.method,
        basis=sc.basis,
        solvent_model=sc.solvent_model,
        solvent=sc.solvent,
        ewin=sc.ewin,
        pcm_hydrogen_radius=sc.pcm_hydrogen_radius,
        threads=threads,
    )

    # Final: assign Boltzmann weights
    assign_weights(ensemble)
    serialize_ensemble(ensemble, output_dir)

    return ensemble
