"""Approach 2: CREST-first tautomer/charge enumeration with conformer sampling.

Uses CREST's physics-based --tautomerize/--deprotonate/--protonate on multiple
conformers, deduplicates by H-count-per-heavy-atom fingerprint, then runs
full conformer searches. Produces unlabeled microstates sufficient for
macroscopic pKa calculation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qm_pka.crest_runner import (
    conformer_search,
    deprotonate,
    protonate,
    tautomerize,
)
from qm_pka.ensemble import HARTREE_TO_KCAL, assign_weights, serialize_ensemble
from qm_pka.rdkit_utils import smiles_to_3d
from qm_pka.stereo import enumerate_and_deduplicate
from qm_pka.tautomer_dedup import (
    deduplicate_tautomers,
)
from qm_pka.types import ChargeState, Conformer, Ensemble, Geometry, Microstate
from qm_pka.xtb_runner import optimize, single_point

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _filter_by_energy_window(conformers: list[Conformer], ewin_kcal: float) -> list[Conformer]:
    """Keep conformers within ewin kcal/mol of the lowest energy."""
    if not conformers:
        return []
    e_min = min(c.free_energy for c in conformers)
    ewin_hartree = ewin_kcal / HARTREE_TO_KCAL
    return [c for c in conformers if (c.free_energy - e_min) <= ewin_hartree]


def _step_charge(
    geometries: list[Geometry],
    current_charge: int,
    target_charge: int,
    solvent: str | None,
    threads: int | None,
) -> list[Geometry]:
    """Apply one protonation or deprotonation step to a list of geometries.

    CREST may crash for chemically unreasonable charge states (e.g., all
    candidate structures fragment). Failures on individual geometries are
    logged and skipped so the pipeline can continue.
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
    includes_enantiomer is propagated to all microstates created.
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
            log.info(f"    Conformer search for tautomer {fp[:8]}...")
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
    output_dir: Path,
    solvent: str = "water",
    prescreen_mode: str = "quick",
    full_mode: str = "default",
    prescreen_ewin: float = 6.0,
    ewin: float = 6.0,
    threads: int | None = None,
) -> Ensemble:
    """Run the full approach 2 pipeline.

    Stereoisomers are enumerated from the input SMILES before any CREST calls.
    Each unique stereoisomer (after enantiomer deduplication) is run through the
    full CREST pipeline independently. Results are merged by pooling microstates
    at each charge state and deduplicating by tautomer fingerprint.

    Note: stereoisomers created by CREST during protonation/deprotonation/
    tautomerization are not explicitly enumerated — only the starting structures
    are diversified. Approach 1 (RDKit-first) handles per-tautomer stereoisomer
    enumeration and should be used when thorough stereoisomer coverage is needed.

    Future improvement: investigate whether CREST conformational sampling reliably
    samples nitrogen inversions. If not, both nitrogen invertomers may need to be
    provided as explicit starting points.
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
    # (different stereoisomers may produce the same tautomer fingerprint)
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

    # Step 4: Assign weights and serialize
    assign_weights(ensemble)
    json_path = serialize_ensemble(ensemble, output_dir)
    log.info(f"Ensemble written to {json_path}")
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approach 2: CREST-first pKa microstate enumeration"
    )
    parser.add_argument("--smiles", required=True, help="Input SMILES string")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--charge-min", type=int, default=-1, help="Minimum charge state")
    parser.add_argument("--charge-max", type=int, default=0, help="Maximum charge state")
    parser.add_argument("--solvent", default="water", help="Solvent for ALPB")
    mode_choices = ["default", "quick", "squick", "mquick"]
    parser.add_argument("--prescreen-mode", default="quick", choices=mode_choices)
    parser.add_argument("--full-mode", default="default", choices=mode_choices)
    parser.add_argument(
        "--prescreen-ewin", type=float, default=6.0, help="Pre-screen energy window (kcal/mol)"
    )
    parser.add_argument(
        "--ewin", type=float, default=6.0, help="Full search energy window (kcal/mol)"
    )
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for CREST")

    args = parser.parse_args()
    run_approach2(
        smiles=args.smiles,
        charge_range=(args.charge_min, args.charge_max),
        output_dir=args.output_dir,
        solvent=args.solvent,
        prescreen_mode=args.prescreen_mode,
        full_mode=args.full_mode,
        prescreen_ewin=args.prescreen_ewin,
        ewin=args.ewin,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
