"""Compare predicted and experimental pKas for a training-set run.

Transitions come from `qm_pka.training_set.parse_pka_column`, which reads them
off the experimental label. Pairing values with transitions positionally
mis-assigns the two entries that mix `pKa` and `pKaH` labels.

Also reports, per molecule, the multiplicative factor on the deprotonated
state's solvation energy that would reproduce the experiment -- the quantity a
charge-specific PCM scaling would have to be constant in.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from qm_pka.ensemble import load_ensemble
from qm_pka.training_set import parse_pka_column

HARTREE_TO_KCAL = 627.509
RT_KCAL = 0.5924
LN10_RT = 1.364
# Gas-phase proton free energy plus its solvation, kcal/mol.  Literature values
# span roughly 2 kcal/mol; this is a fit parameter, and it shifts every pKa
# equally without touching the spread or the correlation.
PROTON_TERM = -270.29


def _components(charge_state: object) -> tuple[np.ndarray, np.ndarray]:
    gas, solv = [], []
    for ms in charge_state.microstates:  # type: ignore[attr-defined]
        for c in ms.conformers:
            gas.append(c.electronic_energy + (c.rrho_correction or 0.0))
            solv.append(c.solvation_energy or 0.0)
    return np.array(gas), np.array(solv)


def _free_energy(gas: np.ndarray, solv: np.ndarray, alpha: float = 1.0) -> float:
    """Boltzmann-summed free energy, with the solvation term scaled by alpha."""
    g = gas + alpha * solv
    lowest = g.min()
    return float(
        lowest
        - (RT_KCAL / HARTREE_TO_KCAL)
        * np.log(np.exp(-(g - lowest) * HARTREE_TO_KCAL / RT_KCAL).sum())
    )


def _solve_alpha(gas_lo: np.ndarray, solv_lo: np.ndarray, base: float, target: float) -> float:
    """Solvation scale factor on the deprotonated state reproducing ``target``.

    Bisection rather than scipy, to keep this script free of a typed-stub
    dependency for fifteen lines of root finding. Returns NaN when the target is
    unreachable in [0.5, 4.0], which happens when the error is not a solvation
    effect at all.
    """
    low, high = 0.5, 4.0
    f_low = _predict(gas_lo, solv_lo, base, low) - target
    f_high = _predict(gas_lo, solv_lo, base, high) - target
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if (f_low > 0.0) == (f_high > 0.0):
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (low + high)
        f_mid = _predict(gas_lo, solv_lo, base, mid) - target
        if (f_mid > 0.0) == (f_low > 0.0):
            low, f_low = mid, f_mid
        else:
            high = mid
    return 0.5 * (low + high)


def _predict(gas_lo: np.ndarray, solv_lo: np.ndarray, base: float, alpha: float = 1.0) -> float:
    """pKa implied by the two charge states, with the anion's solvation scaled."""
    delta = (_free_energy(gas_lo, solv_lo, alpha) - base) * HARTREE_TO_KCAL
    return (delta + PROTON_TERM) / LN10_RT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--training-set", type=Path, default=Path("training_set.csv"))
    parser.add_argument(
        "--substitute",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="use PATH's ensemble.json for molecule ID (e.g. a corrected rerun)",
    )
    args = parser.parse_args()

    meta = {r["unique_ID"]: r for r in csv.DictReader(args.training_set.open())}
    paths = {
        d.name.split("_", 1)[1]: d / "ensemble.json"
        for d in sorted(args.output_root.glob("*/"))
        if (d / "ensemble.json").exists()
    }
    for item in args.substitute:
        name, _, path = item.partition("=")
        paths[name] = Path(path)

    rows = []
    for name, path in paths.items():
        if name not in meta:
            continue
        ensemble = load_ensemble(path)
        for value in parse_pka_column(meta[name]["pkas"]):
            hi, lo = value.higher_charge, value.lower_charge
            if hi not in ensemble.charge_states or lo not in ensemble.charge_states:
                continue
            gas_hi, solv_hi = _components(ensemble.charge_states[hi])
            gas_lo, solv_lo = _components(ensemble.charge_states[lo])
            if not len(gas_hi) or not len(gas_lo):
                continue
            base = _free_energy(gas_hi, solv_hi)

            alpha = _solve_alpha(gas_lo, solv_lo, base, value.value)
            rows.append(
                (name, meta[name]["SMILES"], value, _predict(gas_lo, solv_lo, base), alpha)
            )

    print(
        f"{'molecule':<16}{'SMILES':<24}{'label':<7}{'q':>7}{'pred':>7}{'exp':>7}{'err':>7}{'alpha':>7}"
    )
    for name, smiles, value, pred, alpha in sorted(
        rows, key=lambda r: (r[2].label, -abs(r[3] - r[2].value))
    ):
        arrow = f"{value.higher_charge:+d}/{value.lower_charge:+d}"
        print(
            f"{name:<16}{smiles:<24}{value.label:<7}{arrow:>7}"
            f"{pred:>7.2f}{value.value:>7.2f}{pred - value.value:>+7.2f}{alpha:>7.3f}"
        )

    for label in sorted({r[2].label for r in rows}):
        sub = [r for r in rows if r[2].label == label]
        if len(sub) < 2:
            continue
        predicted = np.array([r[3] for r in sub])
        observed = np.array([r[2].value for r in sub])
        scales = np.array([r[4] for r in sub])
        err = predicted - observed
        line = f"\n{label}: n={len(sub)}  mean err {np.mean(err):+.2f}  sd {np.std(err):.2f}"
        if len(sub) > 2:
            fit = np.polyfit(predicted, observed, 1)
            slope, intercept = float(fit[0]), float(fit[1])
            resid = observed - (slope * predicted + intercept)
            line += (
                f"  r={np.corrcoef(predicted, observed)[0, 1]:.3f}"
                f"  | rescale slope {slope:.2f} residual {resid.std():.2f}"
                f"  | alpha {np.nanmean(scales):.3f}+-{np.nanstd(scales):.3f}"
            )
        print(line)


if __name__ == "__main__":
    main()
