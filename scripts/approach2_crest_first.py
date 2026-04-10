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
from qm_pka.tautomer_dedup import (
    deduplicate_tautomers,
)
from qm_pka.types import ChargeState, Conformer, Ensemble, Geometry, Microstate
from qm_pka.xtb_runner import optimize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _filter_by_energy_window(conformers: list[Conformer], ewin_kcal: float) -> list[Conformer]:
    """Keep conformers within ewin kcal/mol of the lowest energy."""
    if not conformers:
        return []
    e_min = min(c.energy for c in conformers)
    ewin_hartree = ewin_kcal / HARTREE_TO_KCAL
    return [c for c in conformers if (c.energy - e_min) <= ewin_hartree]


def _step_charge(
    geometries: list[Geometry],
    current_charge: int,
    target_charge: int,
    solvent: str | None,
    threads: int | None,
) -> list[Geometry]:
    """Apply one protonation or deprotonation step to a list of geometries."""
    results: list[Geometry] = []
    if target_charge < current_charge:
        for geom in geometries:
            results.extend(
                deprotonate(geom, charge=current_charge, solvent=solvent, threads=threads)
            )
    elif target_charge > current_charge:
        for geom in geometries:
            results.extend(
                protonate(geom, charge=current_charge, solvent=solvent, threads=threads)
            )
    return results


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
    """Run the full approach 2 pipeline."""
    log.info(f"Input: {smiles}")
    log.info(f"Charge range: {charge_range[0]} to {charge_range[1]}")

    # Assume reference charge is 0 (neutral input)
    ref_charge = 0

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
        },
    )

    # Step 1: Generate initial 3D and optimize
    log.info("Generating 3D coordinates and optimizing...")
    geom_3d, _ = smiles_to_3d(smiles)
    geom_opt = optimize(geom_3d, charge=ref_charge, solvent=solvent)

    # Step 2: Quick conformer pre-screen
    log.info(f"Quick conformer pre-screen (mode={prescreen_mode})...")
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
        f"  {len(representatives)} representative conformer(s) within {prescreen_ewin} kcal/mol"
    )
    rep_geoms = [c.geometry for c in representatives]

    # Step 3: Build charge states iteratively outward from reference
    # Store computed microstates per charge for chaining
    computed_geoms: dict[int, list[Geometry]] = {ref_charge: rep_geoms}

    # Process reference charge first
    charge_order = [ref_charge]
    # Then step outward: ref-1, ref-2, ..., charge_min; ref+1, ref+2, ..., charge_max
    for q in range(ref_charge - 1, charge_range[0] - 1, -1):
        charge_order.append(q)
    for q in range(ref_charge + 1, charge_range[1] + 1):
        charge_order.append(q)

    for q in charge_order:
        log.info(f"Processing charge state q={q}...")

        if q == ref_charge:
            # Tautomerize the representative conformers
            source_geoms = rep_geoms
        else:
            # Get source geometries from the adjacent charge state
            adjacent = q + 1 if q < ref_charge else q - 1
            if adjacent not in computed_geoms:
                log.warning(f"  No source geometries at charge {adjacent}, skipping q={q}")
                continue
            source_geoms = _step_charge(computed_geoms[adjacent], adjacent, q, solvent, threads)
            log.info(f"  {len(source_geoms)} structure(s) from (de)protonation")

        if not source_geoms:
            log.warning(f"  No structures generated for charge {q}")
            ensemble.charge_states[q] = ChargeState(charge=q, microstates=[])
            continue

        # Tautomerize each structure
        all_tautomers: list[Geometry] = list(source_geoms)  # include originals
        for geom in source_geoms:
            tau_geoms = tautomerize(geom, charge=q, solvent=solvent, threads=threads)
            all_tautomers.extend(tau_geoms)
        log.info(f"  {len(all_tautomers)} total structure(s) after tautomerization")

        # Deduplicate by H-assignment fingerprint
        groups = deduplicate_tautomers(all_tautomers)
        log.info(f"  {len(groups)} unique tautomer(s) after deduplication")

        # Store representative geometries for chaining to next charge state
        computed_geoms[q] = [geoms[0] for geoms in groups.values()]

        # Full conformer search on each unique tautomer
        microstates: list[Microstate] = []
        for fp, geoms in groups.items():
            representative = geoms[0]  # lowest energy from CREST output
            log.info(f"  Conformer search for tautomer {fp[:8]}...")
            try:
                conformers = conformer_search(
                    representative,
                    charge=q,
                    solvent=solvent,
                    ewin=ewin,
                    mode=full_mode,
                    threads=threads,
                )
                log.info(f"    Found {len(conformers)} conformer(s)")
                microstates.append(
                    Microstate(
                        tautomer_id=fp,
                        conformers=conformers,
                    )
                )
            except Exception as e:
                log.warning(f"    Failed for tautomer {fp[:8]}: {e}")
                continue

        ensemble.charge_states[q] = ChargeState(charge=q, microstates=microstates)

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
