"""Approach 1: RDKit-first tautomer/charge enumeration with CREST conformer sampling.

Enumerates tautomers and protonation/deprotonation sites in SMILES space,
then runs CREST conformer searches on each labeled microstate.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qm_pka.charge_enumeration import enumerate_charge_state
from qm_pka.crest_runner import conformer_search
from qm_pka.ensemble import assign_weights, serialize_ensemble
from qm_pka.rdkit_utils import (
    canonical_smiles,
    enumerate_tautomers,
    get_formal_charge,
    smiles_to_3d,
)
from qm_pka.types import ChargeState, Ensemble, Microstate
from qm_pka.xtb_runner import optimize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def run_approach1(
    smiles: str,
    charge_range: tuple[int, int],
    output_dir: Path,
    solvent: str = "water",
    crest_mode: str = "default",
    ewin: float = 6.0,
    threads: int | None = None,
    max_tautomers: int = 1000,
    max_transforms: int = 1000,
) -> Ensemble:
    """Run the full approach 1 pipeline."""
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

        log.info(f"  {len(species_at_q)} unique microstate(s) at charge {q}")

        microstates: list[Microstate] = []
        for smi in sorted(species_at_q):
            log.info(f"  Conformer search for {smi}...")
            try:
                geom_3d, explicit_h_smi = smiles_to_3d(smi)
                geom_opt = optimize(geom_3d, charge=q, solvent=solvent)
                conformers = conformer_search(
                    geom_opt,
                    charge=q,
                    solvent=solvent,
                    ewin=ewin,
                    mode=crest_mode,
                    threads=threads,
                )
                log.info(f"    Found {len(conformers)} conformer(s)")
                microstates.append(
                    Microstate(
                        tautomer_id=smi,
                        conformers=conformers,
                        smiles=explicit_h_smi,
                    )
                )
            except Exception as e:
                log.warning(f"    Failed for {smi}: {e}")
                continue

        ensemble.charge_states[q] = ChargeState(charge=q, microstates=microstates)

    # Step 3: Assign Boltzmann weights and serialize
    assign_weights(ensemble)
    json_path = serialize_ensemble(ensemble, output_dir)
    log.info(f"Ensemble written to {json_path}")
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approach 1: RDKit-first pKa microstate enumeration"
    )
    parser.add_argument("--smiles", required=True, help="Input SMILES string")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--charge-min", type=int, default=-1, help="Minimum charge state")
    parser.add_argument("--charge-max", type=int, default=0, help="Maximum charge state")
    parser.add_argument("--solvent", default="water", help="Solvent for ALPB")
    crest_choices = ["default", "quick", "squick", "mquick"]
    parser.add_argument("--crest-mode", default="default", choices=crest_choices)
    parser.add_argument("--ewin", type=float, default=6.0, help="Energy window (kcal/mol)")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for CREST")
    parser.add_argument("--max-tautomers", type=int, default=1000, help="RDKit maxTautomers")
    parser.add_argument("--max-transforms", type=int, default=1000, help="RDKit maxTransforms")

    args = parser.parse_args()
    run_approach1(
        smiles=args.smiles,
        charge_range=(args.charge_min, args.charge_max),
        output_dir=args.output_dir,
        solvent=args.solvent,
        crest_mode=args.crest_mode,
        ewin=args.ewin,
        threads=args.threads,
        max_tautomers=args.max_tautomers,
        max_transforms=args.max_transforms,
    )


if __name__ == "__main__":
    main()
