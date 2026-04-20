"""Scoring stage: high-level DFT single-point energy evaluation.

Replaces refinement-level energies with higher-level DFT energies and
optionally computes quasi-RRHO vibrational free energy corrections.
"""

from __future__ import annotations

import logging
from types import ModuleType

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


def score(
    ensemble: Ensemble,
    driver_name: str,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    ewin: float = 10.0,
    compute_rrho: bool = False,
) -> Ensemble:
    """Score all conformers via high-level DFT single-point energy.

    For each conformer in each charge state:
      1. Run DFT single-point (with solvent if configured).
      2. If solvent is used, run a gas-phase single-point to decompose
         into electronic and solvation components.
      3. If compute_rrho is True, compute vibrational frequencies and set
         the quasi-RRHO free energy correction.

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
                        )
                        gas = driver.single_point(conf.geometry, cs.charge, method, basis)
                        conf.electronic_energy = gas
                        conf.solvation_energy = total - gas
                    else:
                        conf.electronic_energy = driver.single_point(
                            conf.geometry, cs.charge, method, basis
                        )
                        conf.solvation_energy = None

                    if compute_rrho:
                        freqs = driver.frequencies(
                            conf.geometry,
                            cs.charge,
                            method,
                            basis,
                            solvent_model,
                            solvent,
                        )
                        conf.rrho_correction = quasi_rrho_free_energy(freqs)

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

    return ensemble
