# The nonelectrostatic correction

IEF-PCM is electrostatics only. Against experiment, an electrostatics-only
continuum under-solvates anions by a median 15 kcal/mol and cations by 6 (see
the Roadmap in the README), and it leaves every logP below zero. This document
describes the correction meant to close that gap: what it computes, why it is
built the way it is, and what is still missing. Nothing is fitted yet.

The code is in three modules:

- `tautomator/nonelectrostatic.py`: the integrals, their inputs (`Solute`),
  the parameters (`Params`), the parameter-free `Descriptors` a fit keeps, and
  the file scoring saves them to (`save_descriptors`, `load_descriptors`).
- `tautomator/cavity_quadrature.py`: grids for the inside and the outside of
  the cavity.
- `tautomator/pyscf_nonelectrostatic.py`: `solute_from_mf`, which reads the
  inputs off a converged PySCF PCM mean field, and
  `single_point_with_descriptors`, the PCM single point scoring runs when it
  saves them.

---

## The model

The form follows CMIRS (Pomogaeva & Chipman; equations as collected in Herbert,
arXiv:2203.06846, eqs. 4.18–4.24):

```
G_nonelectrostatic = A * dispersion + B * exchange + short-range field term
```

| Term | Integral | Domain |
|---|---|---|
| dispersion | `rho I / (w (w + sqrt(rho_solvent)))`, with `I(r)` the damped `R^-6` interaction with uniform solvent | solute density inside the cavity, solvent outside |
| exchange | `\|grad rho\|` | outside the cavity |
| short range | `sum_i A_i (c- \|F_i\|^n- if F_i < 0 else c+ F_i^n+)` | the cavity surface |

The third term replaces CMIRS's FESR, `C |F_min|^g + D F_max^g`, which uses only
the two extrema of the normal field `F` on the surface and so describes one
hydrogen-bond acceptor site and one donor site. The surface integral is the
quantity those extrema are a limit of, and counts every strong-field site.

The code returns the parameter-free integrals (`nonelectrostatic.descriptors`);
the coefficients and exponents are applied afterwards (`Descriptors.energy`), so
a fit never recomputes them. CMIRS's published parameters belong to an
isodensity cavity and do not transfer to the switching/Gaussian sphere cavity:
fields on the sphere surface are larger, and CMIRS's FESR coefficients applied
to them give −144 kcal/mol for a glycine zwitterion.

```python
from tautomator import nonelectrostatic
from tautomator.config import solvent_electron_density
from tautomator.pyscf_nonelectrostatic import solute_from_mf
from tautomator.pyscf_runner import _build_mf

_mol, mf = _build_mf(geom, charge, "wB97M-V", "def2-QZVPPD", "IEFPCM", "water")
mf.kernel()
d = nonelectrostatic.descriptors(solute_from_mf(mf), solvent_electron_density("water"))

d.dispersion, d.exchange            # G_disp / A (dimensionless) and G_exch / B (e/bohr)
d.field_extrema()                   # (F_min, F_max), as FESR uses
d.surface_integral(3.2, -1)         # sum A |F|^3.2 over negative-field elements
d.energy(nonelectrostatic.Params(...)).total
```

The dispersion term needs the solvent's mean electron density, the liquid's
molecules per volume times electrons per molecule
(`config.solvent_electron_density`, from measured densities at 298.15 K):

| solvent | density (kg/m³) | ρ_s (e/bohr³) |
|---|---|---|
| water | 997.047 (IAPWS-95) | 0.0494 |
| 1-octanol (`octanol`) | 821.85 (median of eight measurements, 821.5–822.3) | 0.0417 |
| water-saturated 1-octanol (`woctanol`) | 831.7, with 4.9% water by mass | 0.0421 |
| n-hexadecane | 769.8 (Outcalt et al. 2010) | 0.0394 |

CMIRS uses 0.05 for water, the same number rounded; the prototype values below
were computed with it.

MNSol's octanol free energies are fitted in **water-saturated octanol**
(`woctanol`), not dry. MNSol derives most of them from the aqueous values and
log P(o/w) (the manual's worked example is phenol), and octanol–water partition
coefficients are measured between mutually saturated phases; logP, the reason
for fitting octanol at all, refers to the same phase. Water-saturated octanol
holds 4.9% water by mass and is denser than dry (831.7 kg/m³ at 298.15 K; J.
Chem. Eng. Data 57, 2221 (2012)), which puts its ρ_s 1.1% above dry octanol's.
No measured static permittivity of the wet phase turned up, so it keeps dry
octanol's 9.8629, as xtb's ALPB `woctanol` does (its fitted parameters are what
differ from `octanol`'s, and sampling and RRHO in `woctanol` use them). MNSol
does not say which of its octanol values came from dry octanol instead (gas to
octanol partitioning); the manual says most did not. The water side of logP,
octanol-saturated water, holds only a few millimolar octanol and is plain
water here.

At def2-QZVPPD the descriptors cost about a quarter of the PCM SCF they come
from (toluene: 26 s against 107 s, 8 threads).

---

## What the prototype established

These were measured in a standalone prototype before the code moved here, some
of them with SS(V)PE, which the pipeline no longer offers. The conditions are
given with each.

**The field belongs in the surface term, not the surface charge.** The apparent
surface charge density, recast as a field (`nonelectrostatic.charge_field`), is
exact for a charge at the centre of a sphere and tracks the solute's normal
field well on fully exposed elements. Near the creases where spheres meet it
does not:

| formate, HF/6-31G* | IEF-PCM | SS(V)PE |
|---|---|---|
| correlation with the normal field, fully exposed elements | 0.98 | 0.96 |
| correlation, all elements (area-weighted) | 0.96 | 0.41 |
| most negative value (normal field: −0.063) | −0.18 | −0.76 |

Part of that is physical — the continuum's surface charge really is singular
along a re-entrant crease — and for SS(V)PE part is the symmetrized operator,
whose charges are not local quantities there. Raised to a power near 3, those
few elements dominate: at HF/6-31+G*, `sum A |F|^3.2` over negative elements is
0.016 from the field and 0.33 from SS(V)PE's charges. Both are available
(`source="field"` or `"charge"`); the field is the default.

**The volume integrals need a grid that knows where the cavity is.** Cut off
with a mask on a DFT grid, the exchange integral converges only linearly in the
shell spacing and jumps as atoms move. `cavity_quadrature.cavity_grid` cuts each
ray from each atom at the spheres it crosses and puts a separate rule on every
stretch. Values below use CMIRS's water coefficients only to express the
integrals in kcal/mol (HF/6-31+G*, SS(V)PE):

| grid | formate dispersion | formate exchange | acetone dispersion | acetone exchange |
|---|---|---|---|---|
| masked DFT grid (99, 590), as the SCF uses | −3.895 | 11.790 | −6.178 | 15.839 |
| masked DFT grid (400, 2030) | −3.884 | 13.264 | −6.176 | 16.227 |
| cavity grid, Lebedev 29, 30 radial | −3.886 | 12.918 | −6.175 | 16.154 |
| **cavity grid, Lebedev 41, 40 radial (default)** | −3.887 | 12.943 | −6.177 | 16.017 |
| cavity grid, Lebedev 59, 80 radial | −3.887 | 12.934 | −6.177 | 16.045 |

The default is within 0.03 kcal/mol; the remaining error is angular, from the
crease in the surface each ray ends on.

**The integrals are smooth in the geometry; the extrema are not.** Stretching a
C–O bond of formate over ±0.02 Å in 21 steps, with surface elements appearing
and disappearing (706–710), the scatter about a quartic is 9e-7 kcal/mol for
dispersion, 6e-5 for exchange and 5e-8 relative for the surface integral. For
`F_min` it is 5e-4 relative: the extremum hops between surface elements, so the
original FESR term has a discontinuous derivative and the integral does not.

**Psi4 and PySCF agreed** on every input, on the psi4-gaussian-pcm prototype
that preceded psi4-swig-pcm (HF/def2-SVP, SS(V)PE; formate, methylammonium, the
glycine zwitterion): SCF energies to 1e-12 relative, dispersion and exchange
integrals to 1e-7, surface integrals to 6e-7. tautomator has no Psi4 path for
the descriptors yet (below).

---

## Fitting

The reference data are MNSol's absolute solvation free energies in water,
water-saturated 1-octanol and n-hexadecane; `scripts/select_mnsol_set.py` picks
the solutes and `tautomator/mnsol.py` turns the database's geometries into
SMILES. Octanol–water transfer free energies of solutes with no absolute value
in either solvent are held out as a logP test.

Each solute is run once per solvent and once in the gas phase
(`scripts/run_mnsol_set.py`), and `ΔG_solv = G_soln − G_gas` is taken between
the two ensembles. Every run holds the solute to the one species MNSol
measured (`[sampling] fixed_species`), and the gas-phase run is gas phase
throughout, sampling and RRHO included. Within the
conformer and RRHO model that is exact. A single ensemble with PCM single points
is not: gas-phase conformers miss the relaxation in solvent (ΔG too positive),
solution-phase conformers miss the relaxation in the gas phase (ΔG too
negative), and either misses conformers that exist in one phase only. It also
matches how the correction will be used, on conformers minimized in PCM.

The fit should keep each conformer's components and descriptors, not each
solute's final ΔG, and recompute the Boltzmann weights for every trial
parameter set: `G_soln(θ) = −kT ln Σ_j g_j exp(−β (G_j⁰ + ΔG_j(θ)))`. The
correction is applied after the SCF, so this costs no quantum chemistry.

### Saved descriptors

With `[scoring] nonelectrostatic_descriptors = true` (PySCF, with IEF-PCM, C-PCM
or COSMO), scoring reads the descriptors off each conformer's solvated single
point and writes `<output_dir>/nonelectrostatic.npz`; `run_mnsol_set.py` turns
it on for every solvent run. The file holds every conformer scored in solution,
the ones scoring's energy window then drops included, since the correction can
bring one of them back into the window. The ensemble.json beside it has only
those the window kept.

```python
from tautomator.nonelectrostatic import load_descriptors

conformers, settings = load_descriptors(run_dir / "nonelectrostatic.npz")
c = conformers[0]
c.free_energy                  # E_gas + ΔG_PCM + G_RRHO (hartree), no correction
c.states                       # g_j: multiplicity, times 2 for a collapsed enantiomer pair
c.descriptors.energy(params)   # ΔG_j(θ)
settings["solvent_rho"], settings["delta"]   # what the dispersion integral used
```

Each conformer has its geometry, its microstate and charge, its energy
components, the three integrals, and per surface element the area, both
normal fields and the atom whose sphere the element is on. That is about
50 kB per conformer, and on the order of 150 MB for the MNSol set. The
dispersion integral is stored at one damping length (`delta`, CMIRS's 7 bohr)
and one solvent density; changing either would mean rerunning the SCF.
Computing the descriptors adds about a quarter to scoring.

---

## Not done yet

- **Conformers outside the refinement window** have no scoring-level energy, so
  they have no descriptors either; refinement's 8 kcal/mol window decides
  which conformers the fit can reweight.
- **Psi4.** psi4_runner runs Psi4 as a separate program, so the descriptors would
  have to be evaluated inside its input deck from psi4-swig-pcm's cavity and
  charges.
- **Gradients.** The correction is added after the SCF, so its gradient needs
  the response of the density (a Z-vector solve) as well as the explicit
  derivatives of the three integrals. The integrals are smooth in the geometry
  (above), so nothing in the quadrature stands in the way.
