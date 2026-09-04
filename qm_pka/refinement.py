"""Refinement stage: DFT geometry optimization for sampled conformers.

Replaces xTB-level energies with DFT energies and optionally computes
quasi-RRHO vibrational free energy corrections.
"""

from __future__ import annotations

import logging
from types import ModuleType

from qm_pka import xtb_runner
from qm_pka.config import DEFAULT_MEMORY_GB
from qm_pka.ensemble import deduplicate_charge_state, filter_charge_state_by_energy
from qm_pka.protomer_geometry import repair_migrated_conformers
from qm_pka.thermo import quasi_rrho_free_energy
from qm_pka.types import Ensemble, ExcludedConformer, ExclusionReason

log = logging.getLogger(__name__)


def _get_driver(name: str) -> ModuleType:
    """Return the DFT driver module for the given name."""
    if name == "psi4":
        from qm_pka import psi4_runner

        return psi4_runner
    if name == "pyscf":
        from qm_pka import pyscf_runner

        return pyscf_runner
    raise ValueError(f"Unknown driver: {name!r}. Must be 'psi4' or 'pyscf'.")


def refine(
    ensemble: Ensemble,
    driver_name: str,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    ewin: float = 10.0,
    pcm_hydrogen_radius: float = 1.1,
    rrho_method: str = "xtb",
    xtb_rrho_solvent: str | None = None,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> Ensemble:
    """Refine all conformers via DFT geometry optimization.

    For each conformer in each charge state:
      1. Run DFT geometry optimization (with solvent if configured).
      2. If solvent is used, run a gas-phase single-point on the optimized
         geometry to decompose into electronic and solvation components.
      3. Recompute the quasi-RRHO vibrational free-energy correction on the
         DFT geometry. ``rrho_method="xtb"`` uses a GFN2 single-point (biased)
         Hessian via ``xtb --bhess`` in implicit solvent (``xtb_rrho_solvent``);
         ``rrho_method="dft"`` computes the Hessian at the refinement DFT level,
         matching the refinement solvent. This replaces the cheap xTB RRHO that
         sampling computed on the xTB geometry.

    Runs in two passes per charge state: every conformer is optimized first,
    then symmetry-duplicate structures are collapsed, and only the survivors
    get a Hessian. Optimization frequently maps distinct sampled conformers
    onto one DFT minimum, so deduplicating first avoids paying for the same
    frequencies twice -- and avoids two independent Hessians disagreeing about
    a mirror-image pair whose vibrational spectra must be identical.

    Conformers whose optimizer ran but did not fully converge are kept
    (with refinement_converged=False) since the last-step geometry is
    usually good enough for conformer screening.  Conformers that raise
    an exception (e.g. SCF non-convergence) are dropped with a warning.

    After processing, conformers within each charge state are filtered by
    the energy window.

    Modifies the ensemble in-place and returns it.
    """
    driver = _get_driver(driver_name)

    for cs in ensemble.charge_states.values():
        log.info(f"Refining charge state q={cs.charge}...")
        for ms in cs.microstates:
            surviving = []
            for conf in ms.conformers:
                # Which half of the try raised.  The gas-phase point runs only
                # to decompose a solvated energy, so a failure there means the
                # optimization succeeded and the decomposition did not -- a
                # different thing to chase, and worth separating in the record.
                fail_reason: ExclusionReason = "optimization_failed"
                try:
                    opt_geom, opt_energy, converged = driver.optimize(
                        conf.geometry,
                        cs.charge,
                        method,
                        basis,
                        solvent_model,
                        solvent,
                        pcm_hydrogen_radius=pcm_hydrogen_radius,
                        threads=threads,
                        memory_gb=memory_gb,
                    )
                    conf.geometry = opt_geom
                    conf.refinement_converged = converged
                    if not converged:
                        log.warning(
                            f"  Geometry optimization did not fully converge "
                            f"for conformer in microstate {ms.tautomer_id[:8]}; "
                            f"keeping last-step geometry"
                        )

                    if solvent_model is not None:
                        # opt_energy includes solvation — decompose
                        fail_reason = "gas_phase_sp_failed"
                        gas_energy = driver.single_point(
                            opt_geom,
                            cs.charge,
                            method,
                            basis,
                            threads=threads,
                            memory_gb=memory_gb,
                        )
                        conf.electronic_energy = gas_energy
                        conf.solvation_energy = opt_energy - gas_energy
                    else:
                        conf.electronic_energy = opt_energy
                        conf.solvation_energy = None

                    surviving.append(conf)
                except Exception as e:
                    # Geometry non-convergence does NOT arrive here: the driver
                    # catches it and returns refinement_converged=False, keeping
                    # the last-step geometry.  That conformer stays in the
                    # ensemble, and rightly so -- an unminimized geometry sits
                    # *above* its true minimum, so it is under-weighted, never
                    # dominant.  What reaches this handler is SCF non-convergence
                    # and genuine faults, where there is no energy at all.
                    log.warning(
                        f"  Refinement failed for conformer in microstate "
                        f"{ms.tautomer_id[:8]}: {e}"
                    )
                    ms.excluded_conformers.append(
                        ExcludedConformer(
                            geometry=conf.geometry,
                            stage="refinement",
                            reason=fail_reason,
                            detail=f"{type(e).__name__}: {e}",
                            multiplicity=conf.multiplicity,
                            refinement_converged=conf.refinement_converged,
                        )
                    )
            ms.conformers = surviving

        # Re-file conformers whose proton moved during the DFT optimization
        # before deduplicating, so a migrated conformer is compared against the
        # microstate it now belongs to rather than the one it started in.
        report = repair_migrated_conformers(cs, stage="refinement")
        if report.touched:
            log.info(
                f"  q={cs.charge}: {report.moved} conformer(s) re-filed after a proton moved"
                f"{f', {report.detached} with a detached H' if report.detached else ''}"
                f"{f', {report.unmatched} matching no microstate' if report.unmatched else ''}"
                f"{f', {report.ambiguous} ambiguous' if report.ambiguous else ''}"
            )

        # Deduplicate before the Hessians, not after. Optimization routinely
        # relaxes distinct sampled conformers onto the same DFT minimum, and
        # onto mirror-image pairs; each of those would otherwise cost a second
        # Hessian and then contribute twice to the partition function.
        # The free energy used to pick each group's representative still holds
        # sampling's xTB-geometry RRHO alongside the new DFT terms, which is
        # fine: it only breaks ties between near-identical structures.
        before, after = deduplicate_charge_state(cs)
        if after < before:
            log.info(f"  q={cs.charge}: {before} conformer(s) -> {after} distinct state(s)")

        for ms in cs.microstates:
            surviving = []
            for conf in ms.conformers:
                try:
                    if rrho_method == "xtb":
                        freqs = xtb_runner.frequencies(
                            conf.geometry,
                            cs.charge,
                            solvent=xtb_rrho_solvent,
                            biased=True,
                            threads=threads,
                        )
                    else:  # "dft"
                        freqs = driver.frequencies(
                            conf.geometry,
                            cs.charge,
                            method,
                            basis,
                            solvent_model,
                            solvent,
                            pcm_hydrogen_radius=pcm_hydrogen_radius,
                            threads=threads,
                            memory_gb=memory_gb,
                        )
                    conf.rrho_correction = quasi_rrho_free_energy(freqs)
                    surviving.append(conf)
                except Exception as e:
                    # Previously the conformer was KEPT with rrho_correction
                    # unset.  free_energy sums non-None components, so it became
                    # electronic + solvation while every sibling included RRHO --
                    # and RRHO runs 11-114 kcal/mol (median 76) in this batch, so
                    # the conformer landed far below the ensemble's real ~6
                    # kcal/mol spread, took essentially all the Boltzmann weight,
                    # and the energy window evicted everything else.  A missing
                    # component must never be silently comparable to a complete
                    # one.
                    log.warning(
                        f"  RRHO failed for conformer in microstate {ms.tautomer_id[:8]}: {e}"
                    )
                    ms.excluded_conformers.append(
                        ExcludedConformer(
                            geometry=conf.geometry,
                            stage="refinement",
                            reason="rrho_failed",
                            detail=f"{type(e).__name__}: {e}",
                            multiplicity=conf.multiplicity,
                            electronic_energy=conf.electronic_energy,
                            solvation_energy=conf.solvation_energy,
                            refinement_converged=conf.refinement_converged,
                        )
                    )
            ms.conformers = surviving

        filter_charge_state_by_energy(cs, ewin)
        n_conf = sum(len(ms.conformers) for ms in cs.microstates)
        log.info(
            f"  q={cs.charge}: {len(cs.microstates)} microstate(s), "
            f"{n_conf} conformer(s) after filtering"
        )

    return ensemble
