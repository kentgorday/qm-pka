# qm-pka

Three-stage QM workflow for predicting macroscopic aqueous pKa values of small
organic molecules from a SMILES string. Built on RDKit + CREST/xTB +
PySCF/Psi4. Python 3.13, managed with pixi.

The pipeline produces a fully-decomposed conformer **ensemble** (geometries +
energy components + Boltzmann weights) for every charge state in a requested
range. Going from that ensemble to a numeric macroscopic pKa is **not yet
implemented** — see [Status](#status).

## The pipeline

`qm_pka.pipeline.run_pipeline(config)` orchestrates three stages. Between each
stage, conformers are filtered by an energy window *within* (never across)
each charge state.

| Stage | What it does | Tool | Sets |
|---|---|---|---|
| 1. Sampling | Enumerate tautomers, protonation states, stereoisomers, conformers | CREST + GFN2-xTB (ALPB water) | `electronic_energy`, `solvation_energy` |
| 2. Refinement | DFT geometry optimization | PySCF or Psi4 | overwrites the above with DFT values; optionally `rrho_correction` |
| 3. Scoring | High-level DFT single point | PySCF or Psi4 | overwrites electronic/solvation; optionally `rrho_correction` |

After scoring, `assign_weights` populates per-conformer Boltzmann weights and
the ensemble is serialized to `ensemble.json`.

RRHO (quasi-RRHO vibrational free energy) is computed once, at whichever stage
`scoring.rrho_level` selects (default `"refinement"`). It is not redundantly
recomputed.

## Data model

Defined in `qm_pka/types.py`. This is the most reused mental model in the
codebase; if you only read one file, read this one.

```
Ensemble
└── charge_states: dict[int, ChargeState]      # one per molecular charge
    └── microstates: list[Microstate]          # one per tautomer/protomer
        └── conformers: list[Conformer]        # one per 3D structure
```

A `Conformer.free_energy` is the sum of three optional components:

```
free_energy = electronic_energy + solvation_energy + rrho_correction
              (gas-phase DFT)    (implicit solvent) (quasi-RRHO G_vib)
```

Components are populated incrementally across stages. When a stage runs DFT
with implicit solvent, it does a second gas-phase single point on the same
geometry to keep the decomposition clean.

A `Microstate` carries two corrections that affect partition-function
weighting:

- `includes_enantiomer: bool` — true when the microstate represents a
  collapsed enantiomeric pair (multiplies its statistical weight by 2).
- `symmetry_number: int` — sigma_rot from point-group detection on the
  lowest-energy conformer (divides its weight).

Boltzmann weights are normalized **across all conformers of all microstates
within a charge state**, since the macroscopic pKa depends on the full
partition function of each charge state.

## Two sampling approaches

Configured via `sampling.approach`:

- **`"rdkit_first"` (default).** Walk SMILES space first: BFS over SMARTS
  protonation/deprotonation rules → RDKit `TautomerEnumerator` → stereoisomer
  enumeration with enantiomer dedup → CREST conformer search per labeled
  microstate. Microstates carry an explicit-H SMILES, so SDF export gets real
  bond orders.
- **`"crest_first"`.** Walk physical structures first: a quick conformer
  prescreen → CREST `--tautomerize`/`--protonate`/`--deprotonate` walking
  outward from neutral charge → dedup tautomers by H-count-per-heavy-atom
  fingerprint → full conformer search. Microstates have no SMILES (SDF export
  uses `rdDetermineBonds`).

Approach 1 is the default; approach 2 is useful when you want CREST's
physics-based site detection rather than SMARTS rules.

## Two QM backends

Configured via `compute.driver`. Both implement the same three operations
(`single_point`, `optimize`, `frequencies`) with the same signature, so
`refinement.py` and `scoring.py` are backend-agnostic.

- **`"pyscf"` (default).** In-process Python library. Supports analytical PCM
  gradients (SS(V)PE, IEF-PCM, C-PCM) and SMD. Default refinement
  `wB97X-3c/vDZP/SS(V)PE`; default scoring `wB97M-V/def2-QZVPPD/SS(V)PE`.
- **`"psi4"`.** Subprocess driver, writes input decks and parses output.
  **No analytical PCM gradients** — `config.load_config` warns if you
  request `psi4` + solvent in `[refinement]`. Use gas-phase refinement with
  Psi4, or switch to PySCF.

DFT integration grids are hardcoded: (99, 590) for the SCF grid, (50, 194)
for the VV10 NLC grid. These aren't user-tunable on purpose.

## Quickstart

```bash
pixi install
pixi run python scripts/run_pipeline.py examples/glycine_pyscf.toml
```

Outputs land in `compute.output_dir`:

```
output_dir/
├── sampling/ensemble.json     # post-stage-1 snapshot
├── refinement/ensemble.json   # post-stage-2 snapshot
└── ensemble.json              # final, with Boltzmann weights
```

Example configs in `examples/` cover both backends and both sampling
approaches; each TOML is annotated.

For batch runs over a CSV of SMILES, see `scripts/run_training_set.py`
(supports `--resume`).

## Repo layout

Source under `qm_pka/`. One-liner per module:

| Module | Role |
|---|---|
| `types.py` | Core dataclasses: `Geometry`, `Conformer`, `Microstate`, `ChargeState`, `Ensemble` |
| `config.py` | TOML loader; driver-specific defaults; cross-field validation |
| `pipeline.py` | Stage orchestration |
| `sampling.py` | Approach 1 + Approach 2 |
| `refinement.py`, `scoring.py` | Backend-agnostic stage runners |
| `pyscf_runner.py`, `psi4_runner.py` | Backend implementations |
| `crest_runner.py`, `xtb_runner.py` | CREST/xTB subprocess wrappers |
| `charge_enumeration.py` | SMARTS-based BFS to enumerate charge states |
| `rdkit_utils.py` | SMILES ↔ 3D, tautomer enumeration, canonicalization |
| `stereo.py` | Stereoisomer enumeration + enantiomer dedup via mirror-SMILES |
| `tautomer_dedup.py` | H-count-per-heavy-atom fingerprint (used by approach 2) |
| `thermo.py` | Quasi-RRHO free energy (Grimme 2012) |
| `ensemble.py` | Boltzmann weighting, energy-window filtering, JSON/SDF I/O |
| `xyz_io.py` | XYZ read/write, including CREST multi-frame ensembles |

Scripts (`scripts/`):

- `run_pipeline.py` — single molecule.
- `run_training_set.py` — batch over a CSV with per-molecule subdirectories
  and resume support.
- `select_training_set.py` — diversity-based curation from the IUPAC
  Dissociation-Constants dataset.
- `approach1_rdkit_first.py`, `approach2_crest_first.py` — sampling-only
  entry points (no DFT).

## Development

```bash
pixi run check          # ruff lint + format-check + mypy --strict
pixi run lint
pixi run format
pixi run typecheck
pytest                  # excludes external-tool tests by default
pytest -m slow          # tests that shell out to xtb/crest
```

Pre-commit hook runs ruff. Mypy is strict.

## Non-obvious design decisions

The set of things that look weird and aren't. Read before "fixing":

- **`xtb` is floored at `>=6.7.1` in `pixi.toml`.** `gcp-correction` 2.3.2
  (build 2) requires `mctc-lib >= 0.5.1`; without the floor the solver falls
  back to the ancient `xtb` 6.4.1 (the last release with no `mctc-lib`
  dependency) instead of a build rebuilt against `mctc-lib` 0.5.x. xtb 6.7.1
  build >=4 also fixes a Fortran format-string bug
  ([xtb#1332](https://github.com/grimme-lab/xtb/issues/1332)).
- **wB97X-D4 / wB97X-D4rev / wB97X-3c are re-registered with PySCF**
  (`pyscf_runner.py::_register_d4_composites`). PySCF's native `wb97x-d4`
  string maps to libxc 464 (the original 2008 wB97X) plus D4. The actual
  D4-parameterized methods use libxc 466 (the reparameterized
  wB97X-V exchange-correlation, with VV10 disabled). The registration patches
  PySCF's dispersion dispatch to do the right thing. Don't bypass it.
- **D4 composite Hessians fall up to wB97X-V** (`_HESSIAN_FALLBACK`). PySCF
  has no analytical D4 Hessian, but VV10 *is* analytical. Vibrational
  frequencies are insensitive to the dispersion correction, so this is a free
  speedup.
- **Psi4 + PCM avoidance.** Psi4 has no analytical PCM gradients;
  optimization with implicit solvent is impractically slow. The config
  loader warns. Either run Psi4 gas-phase or use PySCF.
- **Geometry optimizer fallback (PySCF).** TRIC internal coordinates first
  (100 steps); on non-convergence, restart in Cartesian coordinates from the
  last geometry (another 100 steps). Analogous to Psi4 optking's
  `dynamic_level=1`.
- **Non-converged optimizations are kept**, with `refinement_converged=False`.
  The last-step geometry is usually good enough for conformer screening.
  Conformers that *raise* (e.g. SCF failure) are dropped with a warning.
- **Solvation is decomposed via a second gas-phase SP** at every stage that
  uses implicit solvent. This keeps `electronic_energy` and
  `solvation_energy` cleanly separated even though most QM codes return only
  the total.
- **Sampling solvent is hardcoded to ALPB water** and not user-exposed.

## Status

What's implemented:

- Three-stage pipeline end-to-end, both sampling approaches, both QM backends.
- Energy decomposition, quasi-RRHO with frequency scaling, sigma_rot
  detection, enantiomer multiplicity.
- Batch runner with per-molecule failure isolation and resume.
- Training-set selection from IUPAC dissociation constants.

What's not yet here:

- **No pKa-from-ensemble step.** The pipeline produces ensembles and
  Boltzmann-weighted free energies; the LFER fit / macroscopic pKa formula
  is not committed.
- Training-set batch outputs are not committed beyond a few sample runs in
  `smoke/` and `train_set/`.
