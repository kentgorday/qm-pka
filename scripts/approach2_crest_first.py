"""CLI for Approach 2: CREST-first pKa microstate enumeration."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qm_pka.ensemble import assign_weights, serialize_ensemble
from qm_pka.sampling import run_approach2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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
    ensemble = run_approach2(
        smiles=args.smiles,
        charge_range=(args.charge_min, args.charge_max),
        solvent=args.solvent,
        prescreen_mode=args.prescreen_mode,
        full_mode=args.full_mode,
        prescreen_ewin=args.prescreen_ewin,
        ewin=args.ewin,
        threads=args.threads,
    )
    assign_weights(ensemble)
    json_path = serialize_ensemble(ensemble, args.output_dir)
    log.info(f"Ensemble written to {json_path}")


if __name__ == "__main__":
    main()
