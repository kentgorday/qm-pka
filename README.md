# tautomator

Three-stage QM workflow for predicting macroscopic aqueous pKa values of small
organic molecules from a SMILES string. Built on RDKit + CREST/xTB +
PySCF/Psi4 (plus the OrbMol-v2 machine-learned potential as an optional
refinement driver). Python 3.13, managed with pixi.

The pipeline produces a fully-decomposed conformer **ensemble** (geometries +
energy components + Boltzmann weights) for every charge state in a requested
range. Going from that ensemble to a numeric macroscopic pKa is **not yet
implemented** — see [Status](#status).

## The pipeline

`tautomator.pipeline.run_pipeline(config)` orchestrates three stages. Between each
stage, conformers are filtered by an energy window *within* (never across)
each charge state.

| Stage | What it does | Tool | Sets |
|---|---|---|---|
| 1. Sampling | Enumerate tautomers, protonation states, stereoisomers, conformers | CREST + GFN2-xTB (ALPB water) | `electronic_energy`, `solvation_energy`, `rrho_correction` (xTB `--hess`), `multiplicity` |
| 2. Refinement | Geometry optimization + solvation + RRHO recompute on the refined geometry | PySCF, Psi4 or OrbMol-v2 (+ xtb) | overwrites electronic/solvation with refinement-level values; recomputes `rrho_correction` and `multiplicity` |
| 3. Scoring | High-level DFT single point | PySCF or Psi4 | overwrites electronic/solvation; leaves `rrho_correction` and `multiplicity` |

After scoring, `assign_weights` populates per-conformer Boltzmann weights and
the ensemble is serialized to `ensemble.json`.

RRHO (quasi-RRHO vibrational free energy) is always included. It is first
computed cheaply at sampling — an xTB Hessian (`xtb --hess`, ALPB water) on the
xTB geometry, folded into the sampling energy window so filtering ranks by free
energy. It is then recomputed once on the refined geometry, controlled by
`refinement.rrho`:

- `"xtb"` (default): GFN2 single-point biased Hessian (`xtb --bhess`, ALPB
  water) — the Spicher–Grimme SPH scheme. Cheap and robust on the non-stationary
  refined geometry; recommended.
- `"refinement"`: a Hessian from whatever optimized the geometry, at the
  refinement level — OrbMol-v2's own Hessian for an OrbMol refinement, the DFT
  backend's otherwise. More expensive and only as good as that Hessian.

Neither choice dominates, which is why the cheap default stands. Measured over
38 structures (v3 molecules 1–11, lowest conformer per charge state, at OrbMol
minima), the two error sources are the same size: moving the xTB Hessian from
gas to ALPB shifts ΔG_RRHO by 0.62 kcal/mol sd (0.44 pKa on measured
transitions), while swapping xTB for OrbMol in gas phase shifts it by 0.71
kcal/mol sd. Against a wB97X-3c gas Hessian on 10 small transitions, OrbMol is
the better *gas* Hessian by about 4x (sd 0.19 vs 0.81) — but a gas Hessian is
the wrong ensemble, and xTB is the only one of the two that can carry the
solvent. `rrho = "refinement"` on the Psi4 driver is allowed and warns loudly:
Psi4 has no analytical Hessian for these functionals, so each conformer costs
~6N gradient calculations, usually more than the optimization it follows.

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

Defined in `tautomator/types.py`. This is the most reused mental model in the
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

`Microstate.excluded_conformers` holds conformers that left the ensemble with
their geometry and a `reason`, deliberately typed as a `Literal` so a typo is a
type error rather than a category that silently never matches. It is **not** a
complete record of every conformer that was dropped, and the asymmetry is on
purpose: it holds the ones that failed (`optimization_failed`,
`rrho_failed`, `solvation_failed`, `scoring_failed`, …), the ones whose
*identity* turned out wrong (`fragmented`, `proton_detached`, …), and the ones
the solvation pre-filter skipped (`gas_prefilter`) — but the energy-window
filters themselves drop conformers without a record. A conformer outside the
window has a complete, correct free energy and was ranked against its siblings
on it; there is nothing to diagnose, and at a 15 kcal/mol sampling window the
records would outnumber the ensemble. A conformer skipped by the pre-filter is
different in kind: it was dropped on a *cost* decision, using an energy missing
the solvation term, so the record is what lets a finished run be checked for
near misses rather than taken on faith.

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

`compute.driver` is the DFT backend for the whole run: all of scoring, and any
DFT in refinement. It is `"psi4"` or `"pyscf"` — never `"orbmol"`, which cannot
score. All three drivers implement the same three operations (`single_point`,
`optimize`, `frequencies`) with the same signature, so `refinement.py` and
`scoring.py` are backend-agnostic.

- **`"pyscf"` (default).** In-process Python library. Supports analytical PCM
  gradients (IEF-PCM, the default; C-PCM) and SMD, so it is the only backend that
  can *minimize* in implicit solvent. Of the 3c composites it implements only
  wB97X-3c.
- **`"psi4"`.** Subprocess driver, writes input decks and parses output. The
  `psi4` executable comes from a pixi environment of its own
  (`pixi install -e psi4`); tautomator never imports Psi4.
  **No analytical PCM gradients**, so `refinement.solvation = "optimize"` is
  refused; it minimizes in gas phase and adds one PCM single point on that
  geometry. Offers all five 3c composites (wB97X-3c, r2SCAN-3c, B97-3c,
  PBEh-3c, HF-3c).
- **OrbMol-v2 (refinement only).** The machine-learned potential from
  orb-models (Apache-2.0 code and weights), trained on gas-phase
  wB97M-V/def2-TZVPD data (OMol25 + OPoly26), with total charge and spin as
  inputs. Selected by naming it as the refinement *method*
  (`[refinement] method = "OrbMol-v2"`); `compute.driver` stays psi4 or pyscf
  and still runs the solvation single point and scoring. There is no basis. It
  has **no implicit solvent**, so `solvation = "optimize"` is refused. Measured
  against wB97M-V (807 training-set structures; 29 geometry pairs):
  conformer/tautomer energies within a charge state MAE 0.11 kcal/mol
  (wB97X-3c: 0.31), gas-phase deprotonation energies MAE 0.14 kcal/mol vs
  def2-TZVPD, and its minima are lower in wB97M-V energy than wB97X-3c's in
  29/29 cases. Dianions are weaker. CPU only, about 1 GB of RAM, seconds per
  optimization. See `examples/glycine_orbmol.toml`.

DFT integration grids are hardcoded: (99, 590) for the SCF grid, (50, 194)
for the VV10 NLC grid. These aren't user-tunable on purpose.

## Configuration

One TOML file, five sections (`[molecule]`, `[sampling]`, `[refinement]`,
`[scoring]`, `[compute]`), loaded by `config.load_config`. The config says
which *physics* goes into each energy term; the runners decide how to get it
(analytic vs finite-difference Hessians, grids, cavity construction).

A top-level `protocol` key fills in every level of theory and window at once;
anything written in the file wins over the preset.

| | `high_quality` | `fast` |
|---|---|---|
| sampling `ewin` | 10 | 15 |
| refinement | wB97X-3c/vDZP | OrbMol-v2 |
| refinement solvation | driver's best (see below) | wB97X-3c/vDZP single point |
| refinement `ewin` | 8 | 6 |
| scoring | wB97M-V/def2-QZVPPD | wB97M-V/def2-TZVPPD |
| cost (40-molecule training set) | ~74 d | ~15–21 d |

`fast` agrees with `high_quality` to ~0.05 pKa at the scoring level and ≤0.01
pKa on the refinement window, on training-set molecules. The basis step
QZVPPD → TZVPPD is nearly free (offset −0.20, scatter 0.08 kcal/mol over 22
transitions); `high_quality` keeps QZVPPD because that is what wB97M-V's
authors recommend. Dropping VV10 is *not* free — wB97X-D3BJ scatters 0.74
kcal/mol (0.39 pKa) — which is why neither protocol does.

**Solvation at refinement** (`refinement.solvation`) is the setting that
matters most for the window, because the refinement filter decides what scoring
ever sees:

- `"optimize"` — minimize in implicit solvent. PySCF only; the default there.
- `"single_point"` — minimize in gas phase, then one solvated single point on
  that geometry. The default on Psi4 and for OrbMol. For DFT this is one extra
  job, since the optimizer already produced the gas-phase energy; for OrbMol it
  is a DFT pair (solvated and gas) at `solvation_method`/`solvation_basis`,
  which default to wB97X-3c/vDZP.
- `"none"` — gas phase. Warns when scoring is solvated, because gas and
  solvated rankings genuinely differ: a 4 kcal/mol gas window cost 3.3 pKa in
  testing, and conformers inside a 6 kcal/mol *solvated* window sat up to 9.1
  kcal/mol above the gas-phase minimum.

With solvation in the filter, refinement ranks the way scoring does (Spearman
1.00 against a scoring-level reference) and a 6 kcal/mol window keeps 95% of
the scored set at 0 pKa cost.

`refinement.solvation_prefilter_ewin` is an optional cost control for that
single point: conformers already this far above the charge state's gas-phase
minimum are skipped. It is off unless set, must exceed `ewin`, and warns if it
is within 4 kcal/mol of it. It matters for OrbMol, where optimization takes
seconds and the DFT pair dominates; for DFT refinement the single point is
~10% of one optimization at the same level, and the config says so.

The `3c` composites are a functional *and* the basis they were parameterized
with, so `config.load_config` supplies the basis and refuses any other pairing
(`wB97X-3c` with def2-TZVP is a different method, not wB97X-3c).

`load_config` warns when the scoring basis has no diffuse functions, whatever
the charge range. Scoring energies are compared across charge states, and the
deprotonations between them need diffuse functions even when no anion appears.
At wB97M-V in IEF-PCM water, def2-TZVPP put deprotonation energies above
def2-TZVPPD by 1.3–2.8 kcal/mol for neutral acids, 1.0–2.4 for cations becoming
zwitterions (66–91% of the same acid without the cation) and 0.7–0.9 for
ammonium ions; def2-SVP against def2-SVPD is 5–7 times that. The check reads
the basis *name* (`aug-`, `ma-def2-`, the calendar sets, Pople `+`, def2 `…D`),
so an augmented set it does not recognize warns too. vDZP is exempt: it has no
diffuse shells, but it came within 0.8 kcal/mol of def2-TZVPPD on the same
deprotonations, except two tetrazoles, which it put 1.6–1.9 lower. Refinement
bases are not checked: those energies only rank within a charge state, where
def2-TZVPP and def2-TZVPPD differed by at most 0.7 kcal/mol on a zwitterion
against its neutral tautomer.

`config.provenance()` records what each stage computed; `pipeline.py` writes it
into `ensemble.settings` so a checkpoint on disk says which level produced it.

## Quickstart

```bash
pixi install            # the default environment: everything but Psi4
pixi install -e psi4    # only for compute.driver = "psi4"
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
(`--protocol`, `--driver`, `--resume`).

## Repo layout

Source under `tautomator/`. One-liner per module:

| Module | Role |
|---|---|
| `types.py` | Core dataclasses: `Geometry`, `Conformer`, `Microstate`, `ChargeState`, `Ensemble` |
| `config.py` | TOML loader; driver-specific defaults; cross-field validation |
| `pipeline.py` | Stage orchestration |
| `sampling.py` | Approach 1 + Approach 2 |
| `refinement.py`, `scoring.py` | Backend-agnostic stage runners |
| `pyscf_runner.py`, `psi4_runner.py`, `orbmol_runner.py` | Backend implementations (OrbMol: refinement only) |
| `geometric_logging.py` | Stops geomeTRIC reconfiguring the application's logging (PySCF and OrbMol optimizations) |
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

- **Psi4 has a pixi environment to itself** (`psi4`, beside `default`).
  tautomator only ever launches the `psi4` executable, so the two environments
  are solved separately and share no versions. In one environment they
  could not move: Psi4 after 1.10.0 is built against pybind11-abi 11 and
  PySCF's xcfun against abi 4, which held Psi4 at 1.10.0 and libxc at 7.0.
  `psi4_runner.psi4_executable` takes the `psi4` in the environment next to
  the running interpreter's (`.pixi/envs/psi4`) before any on `PATH`, so the
  locked one is not shadowed, and launches it with its own `bin` first on
  `PATH`: Psi4 finds `mctc-gcp` there, and r2SCAN-3c fails without it.
  `refine()` and `score()` look the executable up when they pick the driver,
  so a Psi4 that is not installed stops the run instead of failing every
  conformer in turn. Keep it that way round: don't `import psi4` in tautomator.
  Besides the ABI conflict, running Psi4 as a separate program is what keeps
  tautomator's license independent of Psi4's LGPL-3.0.
- **`xtb` is floored at `>=6.7.1` in `pixi.toml`.** Pressure on `mctc-lib`
  has made the solver fall back to the ancient `xtb` 6.4.1 (the last release
  with no `mctc-lib` dependency) before; that was `gcp-correction` 2.3.2
  (build 2), which now lives in the `psi4` environment. xtb 6.7.1 build >=4
  also fixes a Fortran format-string bug
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
- **OrbMol-v2 is loaded on CPU, in float32-high, from a checksummed file.**
  orb-models 0.7.0 cannot target MPS (it calls `.cuda()` for any non-CPU
  device, and its Warp neighbour list has no MPS backend), and at ~20 atoms CPU
  is faster anyway. The precision was float64, on the theory that geomeTRIC's
  step control and the finite-difference Hessian both need a smooth surface;
  measured, that costs 1.8x and buys nothing — 3% more gradient calls to
  converge, energies within 0.042 kcal/mol, geometries within 0.02 Å,
  frequencies within 3.4 cm⁻¹, no spurious imaginary modes either way, and
  quasi-RRHO within 0.056 kcal/mol (≤0.009 pKa on a measured deprotonation).
  orb-models sets torch's global default dtype, so it is one precision per
  process. The weights come from Hugging Face rather than orb-models' default
  S3 URL (same file, ~1000x faster here) and their sha256 is checked on every
  load; `TAUTOMATOR_ORBMOL_WEIGHTS` points at a local copy. Once downloaded they
  are read from the cache without asking the server again — the checksum
  already identifies the file, and a rate-limited revalidation request used to
  fail the load with the weights on disk. Refinement loads the model before
  its first conformer, so a model that cannot be loaded stops the run instead
  of being recorded as every conformer's optimization failure.
- **orb-models' dependencies are bent in `pixi.toml`.** It pins
  `dm-tree==0.1.8`, which has no py3.13 wheel and whose conda-forge builds
  conflict with current pytorch, so the pin is overridden (orb-models only uses
  `tree.map_structure`/`flatten`). torch comes from conda-forge and is held at
  2.8/2.9 because newer builds need pybind11-abi 11 while PySCF's xcfun needs
  abi 4; on linux-64 it is the MKL build, which moves libblas to MKL.
- **On osx-arm64, BLAS/LAPACK is Apple's Accelerate, selected in `pixi.toml`.**
  conda-forge defaults to OpenBLAS; `libblas = { build = "*_newaccelerate" }`
  swaps it for every package in an environment at once, because numpy, scipy,
  PySCF, Psi4, xtb and torch all link the generic `libblas`/`liblapack`. The
  selector and its macOS requirement appear twice, once per environment: the
  psi4 environment leaves the default feature out and inherits neither.
  Measured on an M4 Pro at 4 threads on a 20-atom molecule: OrbMol-v2 gradients
  3.7x faster, a PySCF wB97M-V/def2-TZVPPD Fock build 1.4x, PySCF wB97X-3c
  1.1x, and PySCF at def2-QZVPPD 1.05x — above 800 basis functions PySCF's XC
  quadrature runs its own non-BLAS code, and its PCM integrals and VV10 never
  used BLAS. A Psi4 1.11 wB97X-3c gradient job is 1.1-1.5x faster, more the
  busier the machine is, for 30% less CPU time. Energies move by 1e-12 Ha
  (PySCF) and 2e-7 Ha (Psi4). This build needs macOS 15.5, hence
  `[system-requirements]`; `*_accelerate` is the same BLAS with reference
  LAPACK and no floor, and measured the same. BLIS is not an option (PySCF gets
  a NaN Fock matrix), nor pthreads OpenBLAS (pytorch excludes it). Accelerate
  ignores `OMP_NUM_THREADS` — its own threads answer to
  `VECLIB_MAXIMUM_THREADS` — but left alone a job given N threads used at most
  8% more than N cores. linux-64 is untouched.
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

## License

Apache-2.0; see [LICENSE](LICENSE).

That covers tautomator only. Its dependencies are installed separately (pixi
fetches them from conda-forge and PyPI) and keep their own licenses. Psi4, xtb
and CREST (LGPL-3.0) run as separate executables. Of the libraries tautomator
imports, most are Apache-2.0, BSD or MIT (PySCF, RDKit, PyTorch, orb-models,
geomeTRIC); ASE is LGPL-2.1, and pyscf-dispersion bundles the LGPL-3.0 dftd4
and s-dftd3 libraries.
