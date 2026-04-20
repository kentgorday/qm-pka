"""Run the three-stage pKa prediction pipeline from a TOML config."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qm_pka.config import load_config
from qm_pka.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-stage pKa prediction pipeline")
    parser.add_argument("config", type=Path, help="Path to TOML config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = load_config(args.config)
    ensemble = run_pipeline(config)

    for charge, cs in sorted(ensemble.charge_states.items()):
        n_conf = sum(len(ms.conformers) for ms in cs.microstates)
        print(f"Charge {charge}: {len(cs.microstates)} microstate(s), {n_conf} conformer(s)")


if __name__ == "__main__":
    main()
