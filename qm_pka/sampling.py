"""Sampling stage: tautomer/charge/conformer enumeration via CREST + xTB.

Two approaches are provided:
  - run_approach1 (RDKit-first): enumerates tautomers/protonation sites in
    SMILES space, then runs CREST conformer searches on each.
  - run_approach2 (CREST-first): uses CREST's physics-based tautomerize/
    deprotonate/protonate on multiple conformers, deduplicates by
    H-assignment fingerprint, then runs full conformer searches.

Both return an Ensemble with xTB-level energies (no Boltzmann weights
assigned — that is the caller's responsibility).
"""

from __future__ import annotations

import logging

from qm_pka.charge_enumeration import enumerate_charge_state
from qm_pka.crest_runner import (
    conformer_search,
    deprotonate,
    protonate,
    tautomerize,
)
from qm_pka.ensemble import HARTREE_TO_KCAL
from qm_pka.rdkit_utils import (
    canonical_smiles,
    enumerate_tautomers,
    get_formal_charge,
    smiles_to_3d,
)
from qm_pka.stereo import enumerate_and_deduplicate
from qm_pka.tautomer_dedup import deduplicate_tautomers
from qm_pka.types import ChargeState, Conformer, Ensemble, Geometry, Microstate
from qm_pka.xtb_runner import optimize, single_point

log = logging.getLogger(__name__)


def _filter_by_energy_window(conformers: list[Conformer], ewin_kcal: float) -> list[Conformer]:
    """Keep conformers within ewin kcal/mol of the lowest energy."""
    if not conformers:
        return []
    e_min = min(c.free_energy for c in conformers)
    ewin_hartree = ewin_kcal / HARTREE_TO_KCAL
    return [c for c in conformers if (c.free_energy - e_min) <= ewin_hartree]


# ---------------------------------------------------------------------------
# Approach 1: RDKit-first
# ---------------------------------------------------------------------------


def run_approach1(
    smiles: str,
    charge_range: tuple[int, int],
    solvent: str = "water",
    crest_mode: str = "default",
    ewin: float = 6.0,
    threads: int | None = None,
    max_tautomers: int = 1000,
    max_transforms: int = 1000,
) -> Ensemble:
    """Sampling via approach 1 (RDKit-first).

    Enumerates tautomers and protonation/deprotonation sites in SMILES space,
    then runs CREST conformer searches on each labeled microstate.

    Returns an Ensemble with xTB-level energies (no weights assigned).
    """
    ref_charge = get_formal_charge(smiles)
    ref_smiles = canonical_smiles(smiles)

    log.info(f"Input: {ref_smiles} (charge {ref_charge})")
    log.info(f"Charge range: {charge_range[0]} to {charge_range[1]}")

    ensemble = Ensemble(
        input_smiles=ref_smiles,
        settings={
            "approach": "rdkit_first",
            "solvent": solvent,
            "crest_mode": crest_mode,
            "ewin_kcal": ewin,
            "charge_range": list(charge_range),
            "max_tautomers": max_tautomers,
            "max_transforms": max_transforms,
        },
    )

    # Step 1: Enumerate tautomers at reference charge
    log.info(f"Enumerating tautomers at reference charge {ref_charge}...")
    ref_tautomers = enumerate_tautomers(
        ref_smiles, max_tautomers=max_tautomers, max_transforms=max_transforms
    )
    log.info(f"  Found {len(ref_tautomers)} tautomer(s) at charge {ref_charge}")

    # Step 2: For each target charge, enumerate species and conformer-search each
    for q in range(charge_range[0], charge_range[1] + 1):
        log.info(f"Processing charge state q={q}...")

        species_at_q: set[str]
        if q == ref_charge:
            species_at_q = set(ref_tautomers)
        else:
            # BFS from all reference tautomers to target charge
            species_at_q = set()
            for tau in ref_tautomers:
                species_at_q.update(enumerate_charge_state(tau, q))
            log.info(f"  {len(species_at_q)} species at charge {q} (before tautomers)")

            # Enumerate tautomers of each species at this charge
            expanded: set[str] = set()
            for smi in species_at_q:
                tau_list = enumerate_tautomers(
                    smi, max_tautomers=max_tautomers, max_transforms=max_transforms
                )
                expanded.update(tau_list)
            species_at_q = expanded

        log.info(f"  {len(species_at_q)} unique tautomer(s) at charge {q}")

        # Enumerate stereoisomers for each tautomer, deduplicate enantiomers
        stereo_species: dict[str, bool] = {}
        for smi in species_at_q:
            for stereo_smi, has_enant in enumerate_and_deduplicate(smi):
                if stereo_smi not in stereo_species:
                    stereo_species[stereo_smi] = has_enant
        log.info(
            f"  {len(stereo_species)} unique microstate(s) at charge {q} "
            f"(after stereoisomer enumeration + enantiomer dedup)"
        )

        microstates: list[Microstate] = []
        for smi, has_enant in sorted(stereo_species.items()):
            log.info(f"  Conformer search for {smi} (enantiomer: {has_enant}, ewin={ewin})...")
            try:
                geom_3d, explicit_h_smi = smiles_to_3d(smi)
                geom_opt = optimize(geom_3d, charge=q, solvent=solvent)
                try:
                    conformers = conformer_search(
                        geom_opt,
                        charge=q,
                        solvent=solvent,
                        ewin=ewin,
                        mode=crest_mode,
                        threads=threads,
                    )
                    log.info(f"    Found {len(conformers)} conformer(s)")
                except RuntimeError:
                    log.warning(
                        f"    Conformer search failed for {smi}, "
                        f"falling back to single optimized geometry"
                    )
                    total = single_point(geom_opt, charge=q, solvent=solvent)
                    gas_phase = single_point(geom_opt, charge=q, solvent=None)
                    conformers = [
                        Conformer(
                            geometry=geom_opt,
                            electronic_energy=gas_phase,
                            solvation_energy=total - gas_phase if solvent is not None else None,
                        )
                    ]
                microstates.append(
                    Microstate(
                        tautomer_id=smi,
                        conformers=conformers,
                        smiles=explicit_h_smi,
                        includes_enantiomer=has_enant,
                    )
                )
            except Exception as e:
                log.warning(f"    Failed for {smi}: {e}")
                continue

        ensemble.charge_states[q] = ChargeState(charge=q, microstates=microstates)

    return ensemble


# ---------------------------------------------------------------------------
# Approach 2: CREST-first
# ---------------------------------------------------------------------------


def _step_charge(
    geometries: list[Geometry],
    current_charge: int,
    target_charge: int,
    solvent: str | None,
    threads: int | None,
) -> list[Geometry]:
    """Apply one protonation or deprotonation step to a list of geometries.

    CREST may crash for chemically unreasonable charge states. Failures on
    individual geometries are logged and skipped.
    """
    results: list[Geometry] = []
    for geom in geometries:
        try:
            if target_charge < current_charge:
                results.extend(
                    deprotonate(geom, charge=current_charge, solvent=solvent, threads=threads)
                )
            elif target_charge > current_charge:
                results.extend(
                    protonate(geom, charge=current_charge, solvent=solvent, threads=threads)
                )
        except RuntimeError:
            log.warning(
                f"  CREST (de)protonation failed for one structure "
                f"(charge {current_charge} -> {target_charge}), skipping"
            )
    return results


def _run_crest_pipeline_for_stereoisomer(
    geom_3d: Geometry,
    ref_charge: int,
    charge_range: tuple[int, int],
    solvent: str,
    prescreen_mode: str,
    full_mode: str,
    prescreen_ewin: float,
    ewin: float,
    threads: int | None,
    includes_enantiomer: bool = False,
) -> dict[int, list[Microstate]]:
    """Run the CREST tautomer/charge pipeline for a single starting geometry.

    Returns a dict mapping charge -> list of Microstates found.
    """
    # Optimize starting geometry
    geom_opt = optimize(geom_3d, charge=ref_charge, solvent=solvent)

    # Quick conformer pre-screen
    log.info(f"  Quick conformer pre-screen (mode={prescreen_mode})...")
    prescreen_conformers = conformer_search(
        geom_opt,
        charge=ref_charge,
        solvent=solvent,
        ewin=prescreen_ewin,
        mode=prescreen_mode,
        threads=threads,
    )
    representatives = _filter_by_energy_window(prescreen_conformers, prescreen_ewin)
    log.info(
        f"    {len(representatives)} representative conformer(s) within {prescreen_ewin} kcal/mol"
    )
    rep_geoms = [c.geometry for c in representatives]

    # Build charge states iteratively outward from reference
    computed_geoms: dict[int, list[Geometry]] = {ref_charge: rep_geoms}
    charge_order = [ref_charge]
    for q in range(ref_charge - 1, charge_range[0] - 1, -1):
        charge_order.append(q)
    for q in range(ref_charge + 1, charge_range[1] + 1):
        charge_order.append(q)

    result: dict[int, list[Microstate]] = {}

    for q in charge_order:
        log.info(f"  Processing charge state q={q}...")

        if q == ref_charge:
            source_geoms = rep_geoms
        else:
            adjacent = q + 1 if q < ref_charge else q - 1
            if adjacent not in computed_geoms:
                log.warning(f"    No source geometries at charge {adjacent}, skipping q={q}")
                result[q] = []
                continue
            source_geoms = _step_charge(computed_geoms[adjacent], adjacent, q, solvent, threads)
            log.info(f"    {len(source_geoms)} structure(s) from (de)protonation")

        if not source_geoms:
            log.warning(f"    No structures generated for charge {q}")
            result[q] = []
            continue

        # Tautomerize each structure
        all_tautomers: list[Geometry] = list(source_geoms)
        for geom in source_geoms:
            try:
                tau_geoms = tautomerize(geom, charge=q, solvent=solvent, threads=threads)
                all_tautomers.extend(tau_geoms)
            except RuntimeError:
                log.warning(f"    CREST tautomerization failed for one structure at charge {q}")
        log.info(f"    {len(all_tautomers)} total structure(s) after tautomerization")

        # Deduplicate by H-assignment fingerprint
        groups = deduplicate_tautomers(all_tautomers)
        log.info(f"    {len(groups)} unique tautomer(s) after deduplication")

        computed_geoms[q] = [geoms[0] for geoms in groups.values()]

        # Full conformer search on each unique tautomer
        microstates: list[Microstate] = []
        for fp, geoms in groups.items():
            representative = geoms[0]
            log.info(f"    Conformer search for tautomer {fp[:8]} (ewin={ewin} kcal/mol)...")
            try:
                try:
                    conformers = conformer_search(
                        representative,
                        charge=q,
                        solvent=solvent,
                        ewin=ewin,
                        mode=full_mode,
                        threads=threads,
                    )
                    log.info(f"      Found {len(conformers)} conformer(s)")
                except RuntimeError:
                    log.warning(
                        f"      Conformer search failed for tautomer {fp[:8]}, "
                        f"falling back to single optimized geometry"
                    )
                    geom_opt = optimize(representative, charge=q, solvent=solvent)
                    total = single_point(geom_opt, charge=q, solvent=solvent)
                    gas_phase = single_point(geom_opt, charge=q, solvent=None)
                    conformers = [
                        Conformer(
                            geometry=geom_opt,
                            electronic_energy=gas_phase,
                            solvation_energy=total - gas_phase if solvent is not None else None,
                        )
                    ]
                microstates.append(
                    Microstate(
                        tautomer_id=fp,
                        conformers=conformers,
                        includes_enantiomer=includes_enantiomer,
                    )
                )
            except Exception as e:
                log.warning(f"      Failed for tautomer {fp[:8]}: {e}")
                continue

        result[q] = microstates

    return result


def run_approach2(
    smiles: str,
    charge_range: tuple[int, int],
    solvent: str = "water",
    prescreen_mode: str = "quick",
    full_mode: str = "default",
    prescreen_ewin: float = 6.0,
    ewin: float = 6.0,
    threads: int | None = None,
) -> Ensemble:
    """Sampling via approach 2 (CREST-first).

    Uses CREST's physics-based --tautomerize/--deprotonate/--protonate on
    multiple conformers, deduplicates by H-count-per-heavy-atom fingerprint,
    then runs full conformer searches.

    Returns an Ensemble with xTB-level energies (no weights assigned).
    """
    log.info(f"Input: {smiles}")
    log.info(f"Charge range: {charge_range[0]} to {charge_range[1]}")

    # Assume reference charge is 0 (neutral input)
    ref_charge = 0

    # Step 1: Enumerate stereoisomers and deduplicate enantiomers
    stereo_smiles = enumerate_and_deduplicate(smiles)
    log.info(
        f"Enumerated {len(stereo_smiles)} unique stereoisomer(s) (after enantiomer deduplication)"
    )

    ensemble = Ensemble(
        input_smiles=smiles,
        settings={
            "approach": "crest_first",
            "solvent": solvent,
            "prescreen_mode": prescreen_mode,
            "full_mode": full_mode,
            "prescreen_ewin_kcal": prescreen_ewin,
            "ewin_kcal": ewin,
            "charge_range": list(charge_range),
            "n_stereoisomers": len(stereo_smiles),
        },
    )

    # Step 2: Run the full CREST pipeline for each stereoisomer, merge results
    all_microstates: dict[int, list[Microstate]] = {}

    for si, (stereo_smi, has_enant) in enumerate(stereo_smiles):
        log.info(
            f"Stereoisomer {si + 1}/{len(stereo_smiles)}: {stereo_smi} (enantiomer: {has_enant})"
        )
        log.info("  Generating 3D coordinates and optimizing...")
        geom_3d, _ = smiles_to_3d(stereo_smi)

        ms_by_charge = _run_crest_pipeline_for_stereoisomer(
            geom_3d,
            ref_charge=ref_charge,
            charge_range=charge_range,
            solvent=solvent,
            prescreen_mode=prescreen_mode,
            full_mode=full_mode,
            prescreen_ewin=prescreen_ewin,
            ewin=ewin,
            threads=threads,
            includes_enantiomer=has_enant,
        )

        for q, microstates in ms_by_charge.items():
            all_microstates.setdefault(q, []).extend(microstates)

    # Step 3: Deduplicate microstates across stereoisomers by tautomer_id
    for q, microstates in all_microstates.items():
        seen: dict[str, Microstate] = {}
        for ms in microstates:
            if ms.tautomer_id not in seen:
                seen[ms.tautomer_id] = ms
            else:
                # Keep the one with the lower best energy
                existing_best = min(c.free_energy for c in seen[ms.tautomer_id].conformers)
                new_best = min(c.free_energy for c in ms.conformers)
                if new_best < existing_best:
                    seen[ms.tautomer_id] = ms
        ensemble.charge_states[q] = ChargeState(charge=q, microstates=list(seen.values()))

    return ensemble
