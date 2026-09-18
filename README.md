# qm-pka

Three-stage QM workflow for predicting macroscopic aqueous pKa values of small
organic molecules from a SMILES string. Built on RDKit + CREST/xTB +
PySCF/Psi4 (plus the OrbMol-v2 machine-learned potential as an optional
refinement driver). Python 3.13, managed with pixi.

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
| 1. Sampling | Enumerate tautomers, protonation states, stereoisomers, conformers | CREST + GFN2-xTB (ALPB water) | `electronic_energy`, `solvation_energy`, `rrho_correction` (xTB `--hess`), `multiplicity` |
| 2. Refinement | Geometry optimization + RRHO recompute on the refined geometry | PySCF, Psi4 or OrbMol-v2 (+ xtb) | overwrites electronic/solvation with refinement-level values; recomputes `rrho_correction` and `multiplicity` |
| 3. Scoring | High-level DFT single point | PySCF or Psi4 | overwrites electronic/solvation; leaves `rrho_correction` and `multiplicity` |

After scoring, `assign_weights` populates per-conformer Boltzmann weights and
the ensemble is serialized to `ensemble.json`.

RRHO (quasi-RRHO vibrational free energy) is always included. It is first
computed cheaply at sampling — an xTB Hessian (`xtb --hess`, ALPB water) on the
xTB geometry, folded into the sampling energy window so filtering ranks by free
energy. It is then recomputed once on the refined DFT geometry, controlled by
`refinement.rrho_method`:

- `"xtb"` (default): GFN2 single-point biased Hessian (`xtb --bhess`, ALPB
  water) — the Spicher–Grimme SPH scheme. Cheap and robust on the non-stationary
  DFT geometry; recommended.
- `"dft"`: Hessian from the refinement driver at the refinement level, matching
  the refinement solvent — for `"orbmol"`, the OrbMol-v2 Hessian (despite the
  name). More expensive and only as good as the backend's Hessians. Unsupported
  on Psi4 with implicit solvent (no analytical PCM Hessians).

Scoring never recomputes RRHO; the refinement value carries through.

Sampling and refinement each deduplicate conformers before spending Hessians on
them: a structure that is another one relabelled, or its mirror image, is
collapsed and the survivor carries the weight of both. Refinement is where this
matters most — DFT optimization relaxes a large fraction of distinct sampled
conformers onto the same minimum. Scoring does not deduplicate, since it does
not move geometries. See [docs/symmetry.md](docs/symmetry.md).

Microstates in the RDKit-first path are identified by `protomer_key`, not by
canonical SMILES. Two SMILES that describe one species with a delocalised charge
written on different atoms — a 4-substituted imidazolium, an enolate — are
otherwise computed twice and double-counted in the partition function. Radicals
and multi-component inputs are rejected at entry rather than approximated.

The same identity is read back off the geometry after sampling and after
refinement, because protons migrate between heavy atoms during minimisation —
often enough that about a quarter of conformers in the first training batch were
filed under a protomer they no longer matched. Those are re-filed under the
microstate they became, before deduplication, so they collapse against it rather
than buying a Hessian twice; a conformer that fragmented or became a species no
microstate describes is recorded in `excluded_conformers`. Every protomer of a
molecule is embedded in one heavy-atom order (`frame_atom_order`) so that
re-filing is a regrouping rather than an atom-correspondence problem. See
[docs/protomer-identity.md](docs/protomer-identity.md).

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

Two corrections affect partition-function weighting, at different levels:

- `Microstate.includes_enantiomer: bool` — true when the microstate represents
  a collapsed *configurational* enantiomer pair (R vs S), multiplying its
  weight by 2.
- `Conformer.multiplicity: float` — how many physical states this conformer
  stands for, divided by its own rotational symmetry number. Set by
  `ensemble.deduplicate_charge_state()` after every deduplication pass.

The two are disjoint by construction and never both fire on the same
reflection. See [docs/symmetry.md](docs/symmetry.md) for the full treatment —
what gets deduplicated, why the count of relabelled copies is not a weight, and
how to check a multiplicity by hand.

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

## QM backends

Configured via `compute.driver`, which runs both QM stages unless
`refinement.driver` names a different one for refinement. All drivers implement
the same three operations (`single_point`, `optimize`, `frequencies`) with the
same signature, so `refinement.py` and `scoring.py` are backend-agnostic.

- **`"pyscf"` (default).** In-process Python library. Supports analytical PCM
  gradients (SS(V)PE, IEF-PCM, C-PCM) and SMD. Default refinement
  `wB97X-3c/vDZP/SS(V)PE`; default scoring `wB97M-V/def2-QZVPPD/SS(V)PE`.
- **`"psi4"`.** Subprocess driver, writes input decks and parses output.
  **No analytical PCM gradients** — `config.load_config` warns if you
  request `psi4` + solvent in `[refinement]`. Use gas-phase refinement with
  Psi4, or switch to PySCF.
- **`"orbmol"` (refinement only).** The OrbMol-v2 machine-learned potential
  (orb-models; Apache-2.0 code and weights), trained on gas-phase
  wB97M-V/def2-TZVPD data (OMol25 + OPoly26), with total charge and spin as
  inputs. Set `refinement.driver = "orbmol"`; `method` defaults to
  `"OrbMol-v2"` and there is no basis. It has **no implicit solvent**, so it is
  gas-phase only and cannot score — `config.load_config` rejects it as
  `compute.driver` and rejects a refinement `solvent_model`. Measured against
  wB97M-V (807 training-set structures; 29 geometry pairs): conformer/tautomer
  energies within a charge state MAE 0.11 kcal/mol (wB97X-3c: 0.31), gas-phase
  deprotonation energies MAE 0.14 kcal/mol vs def2-TZVPD, and its minima are
  lower in wB97M-V energy than wB97X-3c's in 29/29 cases. Dianions are weaker.
  CPU only, about 1 GB of RAM, seconds per optimization. See
  `examples/glycine_orbmol.toml`.

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

Example configs in `examples/` cover both DFT backends, OrbMol-v2 refinement,
and both sampling approaches; each TOML is annotated.

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
| `pyscf_runner.py`, `psi4_runner.py`, `orbmol_runner.py` | Backend implementations (OrbMol: refinement only) |
| `crest_runner.py`, `xtb_runner.py` | CREST/xTB subprocess wrappers |
| `charge_enumeration.py` | SMARTS-based BFS to enumerate charge states |
| `rdkit_utils.py` | SMILES ↔ 3D, tautomer enumeration, protomer identity, input validation |
| `protomer_geometry.py` | Protomer identity from coordinates; re-files conformers whose proton moved |
| `stereo.py` | Stereoisomer enumeration + enantiomer dedup via mirror-SMILES |
| `tautomer_dedup.py` | H-count-per-heavy-atom fingerprint (used by approach 2) |
| `thermo.py` | Quasi-RRHO free energy (Grimme 2012) |
| `conformer_symmetry.py` | Symmetry-aware conformer dedup and multiplicities, from coordinates alone |
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
- **`crest` is pinned to `2.12.*`, not 3.x.** The CREST 3.x rewrite moved
  optimization/MD/MTD in-process, and on macOS its conformer search collapses
  every ensemble to a single structure — reproduced as n-pentane giving 1
  conformer on 3.0.2 vs 6 on 2.12, on both osx-arm64 and osx-64-under-Rosetta
  (so it's the 3.x source, not arch codegen), independent of threads/`--ewin`/
  TOML-vs-CLI; the 3.x optimizer itself is fine, the iMTD-GC pipeline is not.
  See [crest#407](https://github.com/crest-lab/crest/issues/407),
  [#411](https://github.com/crest-lab/crest/issues/411),
  [#412](https://github.com/crest-lab/crest/issues/412). 2.12 drives the
  external `xtb` binary and works; pinning it moves only `crest` (xtb and all
  DFT deps stay put). Consequence: `single_point()` and `frequencies()` call
  `xtb` directly, since CREST 2.x has no `--sp` run mode.
- **wB97X-D4 / wB97X-D4rev / wB97X-3c are built by `pyscf_runner.py::_set_method`,
  never passed to PySCF by name.** All three are the reparameterized wB97X-V
  exchange-correlation (libxc 466) with VV10 off and D4 on; PySCF's native
  `wb97x-d4` means the original 2008 wB97X (libxc 464) plus D4. How the dftd4
  parameter name reaches PySCF depends on its dispersion dispatch, which PySCF
  2.13.1 rewrote (PR #3209), so `_d4_colon_syntax` probes for it: on older PySCF,
  `_register_d4_composites` registers internal XC names in PySCF's dispatch
  tables; from 2.13.1 nothing is registered and the name goes in
  `mf.disp = "d4:<name>"` (registering there would be wrong — XC_MAP is keyed
  on the parsed functional, and 2.14 has a native `wb97x-d4` of its own). Both
  routes were verified (2.11, 2.13.1, 2.14); the tests build composites through
  `_set_method`, so they exercise whichever route the installed PySCF takes.
- **The D4 parameters of those composites are pinned** (`_D4_PARAMS`, installed
  by `_pin_d4_parameters`). PySCF evaluates D4 with the libdftd4 bundled inside
  pyscf-dispersion — not the conda `dftd4` Psi4 uses — and its parameter names
  move between releases: in pyscf-dispersion 1.3.0's copy, `wb97x` is the fit
  for the 2008 functional, which is what wB97X-D4 silently got before the pin.
  The pin sets the numbers (with the ATM three-body term) through the dftd4 C
  API; pyscf-dispersion's own `set_param` passes the wrong arguments. Don't
  bypass either.
- **OrbMol-v2 is loaded on CPU, in float64, from a checksummed file.**
  orb-models 0.7.0 cannot target MPS (it calls `.cuda()` for any non-CPU
  device, and its Warp neighbour list has no MPS backend), and at ~20 atoms CPU
  is faster anyway. float64 because geomeTRIC and the finite-difference Hessian
  want a smooth surface; orb-models sets torch's global default dtype, so it is
  one precision per process. The weights come from Hugging Face rather than
  orb-models' default S3 URL (same file, ~1000x faster here) and their sha256
  is checked on every load; `QM_PKA_ORBMOL_WEIGHTS` points at a local copy.
- **orb-models' dependencies are bent in `pixi.toml`.** It pins
  `dm-tree==0.1.8`, which has no py3.13 wheel and whose conda-forge builds
  conflict with current pytorch, so the pin is overridden (orb-models only uses
  `tree.map_structure`/`flatten`). torch comes from conda-forge and is held at
  2.8/2.9 because newer builds need pybind11-abi 11 while PySCF's xcfun needs
  abi 4; on linux-64 it is the MKL build, which moves libblas to MKL.
- **D4 composite Hessians fall up to wB97X-V** (`_HESSIAN_FALLBACK`). PySCF
  has no analytical D4 Hessian, but VV10 *is* analytical. Vibrational
  frequencies are insensitive to the dispersion correction, so this is a free
  speedup.
- **Psi4 + PCM avoidance.** Psi4 has no analytical PCM gradients;
  optimization with implicit solvent is impractically slow. The config
  loader warns. Either run Psi4 gas-phase or use PySCF.
- **Every Psi4 PCM job pre-screens its cavity.** PCMSolver's two validity
  checks — Gauss' theorem on the nuclear surface charge, and positive-
  definiteness of the single-layer operator — are fatal upstream but patched to
  warnings in this build, so Psi4 exits 0 and returns a meaningless energy
  (this corrupted 324 of 941 conformers in the first training batch, one by
  22,000 kcal/mol). So `_run_psi4` reads stderr on *success* too, and a cheap
  HF/STO-3G probe checks the cavity before each real job: the checks depend
  only on geometry, radii and area, never on the wavefunction, so a minimal
  basis gives the same verdict for ~1% of the cost. On failure it *searches*
  `_TESSERA_AREA_LADDER` rather than refining, because validity is not
  monotonic in area — observed invalid bands at 0.10–0.12 and 0.25–0.40, and
  one molecule needed a **coarser** area than the 0.1 default. The ladder stops
  at **0.15** on accuracy, not stability: a coarse cavity can pass both of
  PCMSolver's checks and still integrate the surface badly, which would swap a
  detectable failure for a silent one. Measured against the 0.1 default over 37
  conformers (charges −2…+2, 4–19 atoms), the worst deviation is 0.10 kcal/mol
  at area 0.15 but 0.24 at 0.18 and 0.40 at 0.30 — and it is overwhelmingly a
  *cationic* effect, an order of magnitude worse at q=+1/+2 than for neutrals
  and anions, since protonated species carry the most hydrogens and hydrogen
  spheres get the fewest tesserae at a fixed area. Conformers with no valid
  cavity at or below 0.15 are dropped rather than served from a coarser grid.
  Not applied to
  PySCF: it discretizes with Lebedev grids plus a switching function rather
  than GePol tesserae, and showed no failure on any of the 941, nor under 20x
  grid degradation.
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
