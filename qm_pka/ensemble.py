"""Boltzmann weighting, partition functions, and ensemble serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qm_pka.types import ChargeState, Conformer, Ensemble, Geometry, Microstate

# Constants
HARTREE_TO_KCAL = 627.5094740631
KB_HARTREE = 3.1668115634556e-6  # Boltzmann constant in Hartree/K


def boltzmann_weights(
    energies: list[float],
    temperature: float = 298.15,
) -> list[float]:
    """Compute normalized Boltzmann weights from energies in Hartree."""
    e = np.array(energies)
    e_rel = e - e.min()
    log_weights = -e_rel / (KB_HARTREE * temperature)
    # Shift for numerical stability
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = weights.sum()
    result: list[float] = (weights / total).tolist()
    return result


def ensemble_free_energy(
    energies: list[float],
    temperature: float = 298.15,
) -> float:
    """Compute ensemble free energy: G = -kT * ln(Z) + E_min.

    Energies in Hartree, returns Hartree. The partition function Z is
    computed relative to the minimum energy for numerical stability.
    """
    e = np.array(energies)
    e_min = e.min()
    e_rel = e - e_min
    z = np.sum(np.exp(-e_rel / (KB_HARTREE * temperature)))
    return float(e_min - KB_HARTREE * temperature * np.log(z))


def charge_state_free_energy(
    charge_state: ChargeState,
    temperature: float = 298.15,
) -> float:
    """Compute the free energy of a charge state.

    Boltzmann-averages over all microstates and all conformers within
    the charge state, treating them as a single flat ensemble.
    """
    all_energies: list[float] = []
    for ms in charge_state.microstates:
        for conf in ms.conformers:
            all_energies.append(conf.energy)
    if not all_energies:
        raise ValueError(f"Charge state {charge_state.charge} has no conformers")
    return ensemble_free_energy(all_energies, temperature)


def assign_weights(ensemble: Ensemble, temperature: float = 298.15) -> None:
    """Assign Boltzmann weights to all conformers within each charge state.

    Weights are normalized across all conformers in a charge state
    (spanning all microstates), since macroscopic pKa depends on the
    full partition function of each charge state.
    """
    for cs in ensemble.charge_states.values():
        all_conformers: list[Conformer] = []
        all_energies: list[float] = []
        for ms in cs.microstates:
            for conf in ms.conformers:
                all_conformers.append(conf)
                all_energies.append(conf.energy)
        if not all_energies:
            continue
        weights = boltzmann_weights(all_energies, temperature)
        for conf, w in zip(all_conformers, weights, strict=True):
            conf.weight = w


def serialize_ensemble(ensemble: Ensemble, output_dir: Path) -> Path:
    """Write ensemble to a single JSON file with inline coordinates.

    Returns the path to the JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, object] = {
        "input_smiles": ensemble.input_smiles,
        "settings": ensemble.settings,
        "charge_states": {},
    }

    cs_data: dict[str, object] = {}
    for charge, cs in sorted(ensemble.charge_states.items()):
        ms_list: list[dict[str, object]] = []
        for ms in cs.microstates:
            conf_list: list[dict[str, object]] = []
            for conf in ms.conformers:
                conf_list.append(
                    {
                        "symbols": list(conf.geometry.symbols),
                        "coords": conf.geometry.coords.tolist(),
                        "energy": conf.energy,
                        "weight": conf.weight,
                    }
                )
            ms_list.append(
                {
                    "tautomer_id": ms.tautomer_id,
                    "smiles": ms.smiles,
                    "n_conformers": len(ms.conformers),
                    "conformers": conf_list,
                }
            )
        cs_data[str(charge)] = {
            "charge": charge,
            "n_microstates": len(cs.microstates),
            "microstates": ms_list,
        }

    data["charge_states"] = cs_data

    json_path = output_dir / "ensemble.json"
    json_path.write_text(json.dumps(data, indent=2))
    return json_path


def load_ensemble(path: Path) -> Ensemble:
    """Load ensemble from JSON with inline coordinates."""
    raw = json.loads(path.read_text())
    ensemble = Ensemble(
        input_smiles=raw["input_smiles"],
        settings=raw.get("settings", {}),
    )
    for charge_str, cs_data in raw.get("charge_states", {}).items():
        charge = int(charge_str)
        microstates: list[Microstate] = []
        for ms_data in cs_data.get("microstates", []):
            conformers: list[Conformer] = []
            for conf_data in ms_data.get("conformers", []):
                geom = Geometry(
                    symbols=tuple(conf_data["symbols"]),
                    coords=np.array(conf_data["coords"]),
                )
                conformers.append(
                    Conformer(
                        geometry=geom,
                        energy=conf_data["energy"],
                        weight=conf_data.get("weight"),
                    )
                )
            microstates.append(
                Microstate(
                    tautomer_id=ms_data["tautomer_id"],
                    conformers=conformers,
                    smiles=ms_data.get("smiles"),
                )
            )
        ensemble.charge_states[charge] = ChargeState(charge=charge, microstates=microstates)
    return ensemble
