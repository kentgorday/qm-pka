"""Refinement stage: DFT geometry optimization for sampled conformers.

Replaces xTB-level energies with DFT energies and optionally computes
quasi-RRHO vibrational free energy corrections.
"""

from __future__ import annotations

import logging
from types import ModuleType

from qm_pka import xtb_runner
from qm_pka.ensemble import filter_charge_state_by_energy
from qm_pka.thermo import quasi_rrho_free_energy
from qm_pka.types import Ensemble

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
                        gas_energy = driver.single_point(
                            opt_geom, cs.charge, method, basis, threads=threads
                        )
                        conf.electronic_energy = gas_energy
                        conf.solvation_energy = opt_energy - gas_energy
                    else:
                        conf.electronic_energy = opt_energy
                        conf.solvation_energy = None

                    if rrho_method == "xtb":
                        freqs = xtb_runner.frequencies(
                            opt_geom,
                            cs.charge,
                            solvent=xtb_rrho_solvent,
                            biased=True,
                            threads=threads,
                        )
                    else:  # "dft"
                        freqs = driver.frequencies(
                            opt_geom,
                            cs.charge,
                            method,
                            basis,
                            solvent_model,
                            solvent,
                            pcm_hydrogen_radius=pcm_hydrogen_radius,
                            threads=threads,
                        )
                    conf.rrho_correction = quasi_rrho_free_energy(freqs)

                    surviving.append(conf)
                except Exception as e:
                    log.warning(
                        f"  Refinement failed for conformer in microstate "
                        f"{ms.tautomer_id[:8]}: {e}"
                    )
            ms.conformers = surviving

        filter_charge_state_by_energy(cs, ewin)
        n_conf = sum(len(ms.conformers) for ms in cs.microstates)
        log.info(
            f"  q={cs.charge}: {len(cs.microstates)} microstate(s), "
            f"{n_conf} conformer(s) after filtering"
        )

    return ensemble
