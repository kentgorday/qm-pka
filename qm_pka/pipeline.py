"""Three-stage pKa prediction pipeline: sampling → refinement → scoring."""

from __future__ import annotations

import logging
from pathlib import Path

from qm_pka.config import PkaConfig
from qm_pka.ensemble import (
    SCHEMA_VERSION,
    assign_weights,
    load_ensemble,
    schema_version,
    serialize_ensemble,
)
from qm_pka.refinement import refine
from qm_pka.sampling import run_approach1, run_approach2
from qm_pka.scoring import score
from qm_pka.types import Ensemble

log = logging.getLogger(__name__)

# Stages that checkpoint to <output_dir>/<name>/ensemble.json, latest first.
# Scoring has no entry: it writes the final <output_dir>/ensemble.json, and a
# molecule with that file is simply done.
_RESUMABLE_STAGES = ("refinement", "sampling")


def _find_resume_point(output_dir: Path, smiles: str) -> tuple[Ensemble, str] | None:
    """Return (ensemble, stage_name) for the latest completed stage, or None.

    The schema version and ``input_smiles`` are validated.  A stage's
    ensemble.json records the
    *sampling* settings (they are written once and carried forward unchanged),
    so nothing on disk says which method/basis produced refined geometries —
    resuming after editing [refinement] would silently mix levels of theory.
    The SMILES check still catches the dangerous case: output directories are
    numbered by CSV row, so editing the training set shifts them and would
    otherwise resume one molecule's run against another's data.
    """
    for stage in _RESUMABLE_STAGES:
        path = output_dir / stage / "ensemble.json"
        if not path.exists():
            continue
        found_version = schema_version(path)
        if found_version < SCHEMA_VERSION:
            log.warning(
                f"Ignoring {path}: schema v{found_version}, this build writes "
                f"v{SCHEMA_VERSION}. Resuming would reload every conformer at the "
                f"default multiplicity of 1.0 and silently reproduce the old, "
                f"unweighted answer. Rerun the stage instead."
            )
            return None
        ensemble = load_ensemble(path)
        if ensemble.input_smiles != smiles:
            log.warning(
                f"Ignoring {path}: it holds {ensemble.input_smiles!r}, "
                f"but this config asks for {smiles!r}"
            )
            return None
        return ensemble, stage
    return None


def run_pipeline(config: PkaConfig, resume: bool = False) -> Ensemble:
    """Run the full three-stage pKa prediction pipeline.

    Stage 1 (Sampling): CREST-based conformer/tautomer/protonation enumeration,
        with an xTB-level RRHO correction folded into the energy filter.
    Stage 2 (Refinement): DFT geometry optimization + RRHO recompute on the DFT
        geometry (xtb --bhess or refinement-level DFT, per rrho_method).
    Stage 3 (Scoring): DFT single-point energy (RRHO carried over unchanged).

    With resume=True, any stage whose ensemble.json is already on disk is
    reloaded instead of recomputed, and the run picks up at the next stage.
    Stages are days long here, so a run killed during scoring would otherwise
    redo sampling and refinement from scratch.  Resumption happens at stage
    granularity: a stage interrupted partway through restarts from its
    beginning, since neither refine() nor score() checkpoints per conformer.

    Returns Ensemble with Boltzmann weights assigned.
    """
    output_dir = Path(config.compute.output_dir)

    # Aqueous solvent used for xTB (ALPB) throughout: conformer sampling and the
    # xtb-level RRHO at both sampling and refinement.
    solvent = "water"

    resumed_from: str | None = None
    ensemble: Ensemble | None = None
    if resume:
        found = _find_resume_point(output_dir, config.molecule.smiles)
        if found is not None:
            ensemble, resumed_from = found
            n_conf = sum(
                len(ms.conformers)
                for cs in ensemble.charge_states.values()
                for ms in cs.microstates
            )
            log.info(
                f"=== Resuming after {resumed_from}: reusing {n_conf} conformer(s) "
                f"from {output_dir / resumed_from / 'ensemble.json'} ==="
            )

    # Stage 1: Sampling
    if ensemble is not None:
        log.info("=== Stage 1: Sampling (skipped, reusing checkpoint) ===")
    else:
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
    threads = config.compute.threads or 1
    memory_gb = config.compute.memory_gb
    if resumed_from == "refinement":
        log.info("=== Stage 2: Refinement (skipped, reusing checkpoint) ===")
    else:
        log.info("=== Stage 2: Refinement ===")
        ref = config.refinement
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
            memory_gb=memory_gb,
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
        memory_gb=memory_gb,
    )

    # Final: assign Boltzmann weights
    assign_weights(ensemble)
    serialize_ensemble(ensemble, output_dir)

    return ensemble
