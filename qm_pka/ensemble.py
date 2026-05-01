"""Boltzmann weighting, partition functions, and ensemble serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

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


def _microstate_multiplicity(ms: Microstate) -> float:
    """Effective multiplicity for partition-function weighting.

    Combines enantiomer degeneracy (x2 if `includes_enantiomer`) with the
    rotational symmetry-number correction (1/sigma_rot). Higher sigma means
    fewer distinguishable rotational orientations -> smaller q_rot -> smaller
    weight.
    """
    enant = 2 if ms.includes_enantiomer else 1
    return enant / ms.symmetry_number


def charge_state_free_energy(
    charge_state: ChargeState,
    temperature: float = 298.15,
) -> float:
    """Compute the free energy of a charge state.

    Boltzmann-averages over all microstates and all conformers within
    the charge state.  Each microstate's Boltzmann weight is multiplied
    by an effective multiplicity combining enantiomer degeneracy and the
    rotational symmetry number (see `_microstate_multiplicity`).
    """
    if not any(ms.conformers for ms in charge_state.microstates):
        raise ValueError(f"Charge state {charge_state.charge} has no conformers")

    kbt = KB_HARTREE * temperature
    e_min = min(conf.free_energy for ms in charge_state.microstates for conf in ms.conformers)
    z = 0.0
    for ms in charge_state.microstates:
        multiplicity = _microstate_multiplicity(ms)
        for conf in ms.conformers:
            z += multiplicity * np.exp(-(conf.free_energy - e_min) / kbt)
    return float(e_min - kbt * np.log(z))


def assign_weights(ensemble: Ensemble, temperature: float = 298.15) -> None:
    """Assign Boltzmann weights to all conformers within each charge state.

    Weights are normalized across all conformers in a charge state
    (spanning all microstates), since macroscopic pKa depends on the
    full partition function of each charge state. Each microstate gets
    an effective multiplicity combining enantiomer degeneracy and sigma_rot.
    """
    kbt = KB_HARTREE * temperature
    for cs in ensemble.charge_states.values():
        entries: list[tuple[Conformer, float]] = []
        for ms in cs.microstates:
            multiplicity = _microstate_multiplicity(ms)
            for conf in ms.conformers:
                entries.append((conf, multiplicity))
        if not entries:
            continue
        e_min = min(conf.free_energy for conf, _ in entries)
        raw_weights = [mult * np.exp(-(conf.free_energy - e_min) / kbt) for conf, mult in entries]
        total = sum(raw_weights)
        for (conf, _), w in zip(entries, raw_weights, strict=True):
            conf.weight = float(w / total)


def filter_charge_state_by_energy(cs: ChargeState, ewin_kcal: float) -> None:
    """Remove conformers outside the energy window within a charge state.

    The window is relative to the lowest free_energy across all conformers
    in all microstates of this charge state. Microstates with no surviving
    conformers are pruned.
    """
    all_conformers = [c for ms in cs.microstates for c in ms.conformers]
    if not all_conformers:
        return
    e_min = min(c.free_energy for c in all_conformers)
    ewin_hartree = ewin_kcal / HARTREE_TO_KCAL
    for ms in cs.microstates:
        ms.conformers = [c for c in ms.conformers if (c.free_energy - e_min) <= ewin_hartree]
    cs.microstates = [ms for ms in cs.microstates if ms.conformers]


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
                        "electronic_energy": conf.electronic_energy,
                        "solvation_energy": conf.solvation_energy,
                        "rrho_correction": conf.rrho_correction,
                        "free_energy": conf.free_energy,
                        "weight": conf.weight,
                        "refinement_converged": conf.refinement_converged,
                    }
                )
            ms_list.append(
                {
                    "tautomer_id": ms.tautomer_id,
                    "smiles": ms.smiles,
                    "includes_enantiomer": ms.includes_enantiomer,
                    "symmetry_number": ms.symmetry_number,
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
                        electronic_energy=conf_data.get("electronic_energy"),
                        solvation_energy=conf_data.get("solvation_energy"),
                        rrho_correction=conf_data.get("rrho_correction"),
                        weight=conf_data.get("weight"),
                        refinement_converged=conf_data.get("refinement_converged"),
                    )
                )
            microstates.append(
                Microstate(
                    tautomer_id=ms_data["tautomer_id"],
                    conformers=conformers,
                    smiles=ms_data.get("smiles"),
                    includes_enantiomer=ms_data.get("includes_enantiomer", False),
                    symmetry_number=ms_data.get("symmetry_number", 1),
                )
            )
        ensemble.charge_states[charge] = ChargeState(charge=charge, microstates=microstates)
    return ensemble


def _mol_from_smiles_and_coords(smiles: str, geom: Geometry, charge: int) -> Chem.Mol:
    """Build an RDKit mol from explicit-H SMILES, setting coordinates from geometry.

    Assumes the geometry atom ordering matches the SMILES atom ordering
    (i.e. geometry was reordered at creation time via _smilesAtomOutputOrder).
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    if mol.GetNumAtoms() != geom.n_atoms:
        raise ValueError(
            f"SMILES atom count ({mol.GetNumAtoms()}) != geometry atom count ({geom.n_atoms})"
        )

    conf = Chem.Conformer(geom.n_atoms)
    for i in range(geom.n_atoms):
        conf.SetAtomPosition(
            i,
            (float(geom.coords[i, 0]), float(geom.coords[i, 1]), float(geom.coords[i, 2])),
        )
    mol.AddConformer(conf, assignId=True)
    return mol


def _mol_from_coords(geom: Geometry, charge: int) -> Chem.Mol:
    """Build an RDKit mol from coordinates only, using rdDetermineBonds."""
    mol = Chem.RWMol()
    for sym in geom.symbols:
        mol.AddAtom(Chem.Atom(sym))
    conf = Chem.Conformer(geom.n_atoms)
    for i in range(geom.n_atoms):
        conf.SetAtomPosition(
            i,
            (float(geom.coords[i, 0]), float(geom.coords[i, 1]), float(geom.coords[i, 2])),
        )
    mol.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return Chem.Mol(mol)


def ensemble_to_sdf(ensemble: Ensemble, output_path: Path) -> Path:
    """Write all conformers in an ensemble to an SDF file.

    If a microstate has an explicit-H SMILES (approach 1), bond orders
    come from the SMILES. Otherwise (approach 2), bonds are determined
    from coordinates via rdDetermineBonds.

    Each conformer is written as a separate record with properties:
    charge, tautomer_id, smiles, energy_hartree, boltzmann_weight.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))

    for charge, cs in sorted(ensemble.charge_states.items()):
        sorted_ms = sorted(cs.microstates, key=lambda m: min(c.free_energy for c in m.conformers))
        for ms in sorted_ms:
            for conf in sorted(ms.conformers, key=lambda c: c.free_energy):
                if ms.smiles is not None:
                    mol = _mol_from_smiles_and_coords(ms.smiles, conf.geometry, charge)
                else:
                    mol = _mol_from_coords(conf.geometry, charge)

                mol.SetIntProp("charge", charge)
                mol.SetProp("tautomer_id", ms.tautomer_id)
                if ms.smiles is not None:
                    mol.SetProp("smiles", ms.smiles)
                mol.SetDoubleProp("free_energy_hartree", conf.free_energy)
                if conf.electronic_energy is not None:
                    mol.SetDoubleProp("electronic_energy_hartree", conf.electronic_energy)
                if conf.solvation_energy is not None:
                    mol.SetDoubleProp("solvation_energy_hartree", conf.solvation_energy)
                if conf.rrho_correction is not None:
                    mol.SetDoubleProp("rrho_correction_hartree", conf.rrho_correction)
                if conf.weight is not None:
                    mol.SetDoubleProp("boltzmann_weight", conf.weight)

                writer.write(mol)

    writer.close()
    return output_path
