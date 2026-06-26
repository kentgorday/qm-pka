"""Scoring stage: high-level DFT single-point energy evaluation.

Replaces refinement-level energies with higher-level DFT energies. The
quasi-RRHO vibrational free-energy correction is carried over unchanged from
refinement (scoring never recomputes frequencies).
"""

from __future__ import annotations

import logging
from types import ModuleType

from qm_pka.ensemble import filter_charge_state_by_energy
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


def score(
    ensemble: Ensemble,
    driver_name: str,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    ewin: float = 10.0,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
) -> Ensemble:
    """Score all conformers via high-level DFT single-point energy.

    For each conformer in each charge state:
      1. Run DFT single-point (with solvent if configured).
      2. If solvent is used, run a gas-phase single-point to decompose
         into electronic and solvation components.

    Any quasi-RRHO correction set during refinement is left untouched and
    contributes to the conformer free energy used for filtering and weighting.

    Conformers that fail are dropped with a warning. After processing,
    conformers within each charge state are filtered by the energy window.

    Modifies the ensemble in-place and returns it.
    """
    driver = _get_driver(driver_name)

    for cs in ensemble.charge_states.values():
        log.info(f"Scoring charge state q={cs.charge}...")
        for ms in cs.microstates:
            surviving = []
            for conf in ms.conformers:
                try:
                    if solvent_model is not None:
                        total = driver.single_point(
                            conf.geometry,
                            cs.charge,
                            method,
                            basis,
                            solvent_model,
                            solvent,
                            pcm_hydrogen_radius=pcm_hydrogen_radius,
                            threads=threads,
                        )
                        gas = driver.single_point(
                            conf.geometry, cs.charge, method, basis, threads=threads
                        )
                        conf.electronic_energy = gas
                        conf.solvation_energy = total - gas
                    else:
                        conf.electronic_energy = driver.single_point(
                            conf.geometry, cs.charge, method, basis, threads=threads
                        )
                        conf.solvation_energy = None

                    surviving.append(conf)
                except Exception as e:
                    log.warning(
                        f"  Scoring failed for conformer in microstate {ms.tautomer_id[:8]}: {e}"
                    )
            ms.conformers = surviving

        filter_charge_state_by_energy(cs, ewin)
        n_conf = sum(len(ms.conformers) for ms in cs.microstates)
        log.info(
            f"  q={cs.charge}: {len(cs.microstates)} microstate(s), "
            f"{n_conf} conformer(s) after filtering"
        )

        # Detect sigma_rot per microstate using its lowest-energy conformer.
        for ms in cs.microstates:
            if not ms.conformers:
                continue
            lowest = min(ms.conformers, key=lambda c: c.free_energy)
            try:
                ms.symmetry_number = driver.rotational_symmetry_number(lowest.geometry)
            except Exception as e:
                log.warning(
                    f"  sigma_rot detection failed for microstate "
                    f"{ms.tautomer_id[:8]}: {e}; using sigma=1"
                )
                ms.symmetry_number = 1

    return ensemble
