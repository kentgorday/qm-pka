#!/usr/bin/env python
"""Run the pKa pipeline over every SMILES in a training-set CSV.

Builds a per-molecule TOML config, runs the three-stage pipeline, and writes
each molecule's outputs (sampling/, refinement/, ensemble.json) into its own
subdirectory under --output-root.

Per-molecule failures are caught and logged so a single bad compound does
not abort the whole batch.

Usage:
    pixi run python scripts/run_training_set.py \
        --training-set training_set.csv \
        --output-root training_set_output \
        --driver pyscf --threads 8

Resume after interruption:
    pixi run python scripts/run_training_set.py --resume ...
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path

from qm_pka.config import load_config
from qm_pka.pipeline import run_pipeline


def make_toml(smi: str, output_dir: Path, driver: str, threads: int) -> str:
    """Build a TOML config string for one molecule.

    Charge range is hardcoded to [-2, +2]. Empty charge states (e.g. when
    the molecule cannot reach -2) flow through harmlessly.
    """
    safe_smi = smi.replace('"', '\\"')
    return f'''\
[molecule]
smiles = "{safe_smi}"
charge_min = -2
charge_max = 2

[sampling]
approach = "rdkit_first"
ewin = 10.0

[refinement]
solvent = "water"
ewin = 10.0

[scoring]
solvent = "water"
ewin = 10.0
rrho_level = "refinement"

[compute]
driver = "{driver}"
threads = {threads}
output_dir = "{output_dir}"
'''


def safe_name(smi: str, fallback: str) -> str:
    """Sanitize a SMILES string into a filesystem-safe directory name."""
    name = "".join(c if c.isalnum() else "_" for c in smi)[:50]
    return name or fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pKa pipeline over a training-set CSV")
    parser.add_argument(
        "--training-set",
        type=Path,
        default=Path("training_set.csv"),
        help="CSV with at least a SMILES column (and optionally unique_ID)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training_set_output"),
    )
    parser.add_argument("--driver", default="pyscf", choices=["pyscf", "psi4"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N molecules (useful for smoke testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip molecules whose final ensemble.json already exists",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    if not args.training_set.exists():
        print(
            f"Error: {args.training_set} not found. Run scripts/select_training_set.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with args.training_set.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("SMILES"):
                rows.append(row)

    if args.limit:
        rows = rows[: args.limit]

    log.info(f"Running pipeline on {len(rows)} molecules")

    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows, start=1):
        smi = row["SMILES"]
        uid = row.get("unique_ID") or f"mol_{i}"
        compound_dir = args.output_root / f"{i:03d}_{safe_name(uid, f'mol_{i}')}"
        ensemble_json = compound_dir / "ensemble.json"

        if args.resume and ensemble_json.exists():
            log.info(f"[{i}/{len(rows)}] {uid} ({smi}): already done, skipping")
            n_skip += 1
            continue

        log.info(f"[{i}/{len(rows)}] {uid} ({smi}): starting")
        try:
            compound_dir.mkdir(parents=True, exist_ok=True)
            toml_text = make_toml(smi, compound_dir, args.driver, args.threads)
            toml_path = compound_dir / "config.toml"
            toml_path.write_text(toml_text)

            config = load_config(toml_path)
            run_pipeline(config)
            n_ok += 1
            log.info(f"[{i}/{len(rows)}] {uid}: done")
        except Exception as e:
            n_fail += 1
            log.error(f"[{i}/{len(rows)}] {uid}: FAILED: {e}")
            (compound_dir / "ERROR.txt").write_text(traceback.format_exc())

    print(f"\nSummary: {n_ok} ok, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
