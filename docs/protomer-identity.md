# Protomer identity

Two SMILES can describe the same chemical species. RDKit will canonicalise them
apart anyway, because in its model they are different molecular graphs. When
that happens to a microstate, the pipeline computes the same species twice —
two conformer ensembles, two sets of DFT energies, and a partition function
that double-counts it.

`protomer_key` in `qm_pka/rdkit_utils.py` is the identity used for microstates
in the RDKit-first path. It gives one key per species, whichever Lewis
structure was written for it.

## The problem

A 4-substituted imidazolium carries its cation on a delocalised N–C–N unit.
Both nitrogens bear a hydrogen, and the `+` can be written on either one:

```
[H]c1c(R)n([H])c([H])[n+]1[H]
[H]c1c(R)[n+]([H])c([H])n1[H]
```

These are one species. Canonical SMILES makes them two. In histamine the
enumerator emitted both at q=+1 and again at q=+2; a delocalised tetronic-acid
enolate did the same thing at three charge states, once as a three-way split.

The same shape appears wherever charge is delocalised: imidazolate,
tetrazolate, amidinium, benzimidazolium, enolates of 1,3-dicarbonyls.

## The key

Resonance moves formal charge and bond order, and nothing else. It cannot break
a heavy-atom bond or relocate a hydrogen. Our enumeration is restricted the same
way — it only adds and removes H at heteroatoms. So across every microstate of
one molecule the heavy-atom framework is fixed, and

> (framework, per-atom H count, net charge, stereochemistry)

identifies a species uniquely, whatever Lewis structure it was written as.

`protomer_key` computes exactly that: erase what resonance can change, keep what
it cannot, canonicalise the result.

1. Build the **skeleton** — every bond reduced to single, every formal charge
   zeroed, aromaticity and stereo stripped, per-atom hydrogen counts pinned as
   explicit.
2. Canonicalise it. This gives both the framework string and a canonical atom
   ranking.
3. Append the **net charge** and the **stereo descriptors**, as CIP labels keyed
   by each atom's rank in the skeleton.

Hydrogen counts are what keep genuine tautomers apart: 4-methylimidazole and
5-methylimidazole have the same framework but different per-atom H, so they
never merge. The same holds for two different protonation sites.

### Why the skeleton, and not canonical SMILES

RDKit's canonical atom ranking depends on formal charge. It does *not* depend on
bond order — aromatic and Kekulé forms of one molecule rank identically — but
charge is enough. Two resonance forms of one molecule therefore canonicalise to
two *different* atom orderings.

That matters because it would make the stereo labelling depend on which Lewis
structure was handed in. Ranking on the skeleton avoids it: the skeleton is
byte-identical across every resonance form of a molecule, so the ranking is too.

### Why not a resonance enumeration

The obvious alternative is to enumerate resonance structures with
`Chem.ResonanceMolSupplier` and take a canonical minimum over the orbit. It
works, and it produces the same merges, but it rests on a much larger surface:
the enumeration is order-dependent and incomplete (3-ethylphenolate has four
resonance forms; the supplier returns three, and which three depends on atom
ordering), truncation at `maxStructs` is silent, and the flag choices interact
with obligatory charge-separated groups.

Erasing charge and bond order needs none of that. It also handles the
charge-separated groups *better*: sulfoxide `S=O` ↔ `S⁺–O⁻`, amine oxides, nitro
and phosphine oxides all merge without any special-casing, where a conservative
resonance enumeration leaves them apart.

## Where it runs

`enumerate_charge_state` collapses its BFS output by `protomer_key`, and
`run_approach1` collapses again after tautomer expansion, since both stages can
introduce the duplication. `deduplicate_protomers` chooses the survivor
deliberately rather than by sort order, because it seeds both stereoisomer
enumeration and the ETKDG geometry: fewest formally charged atoms first (the
dominant resonance contributor, which keeps a valid-but-absurd `CC([O-])=[OH+]`
from standing in for acetic acid), then the most perceivable stereo elements
(a delocalised anion drawn with its double bond in one position can expose a
stereogenic bond the other hides), then canonical SMILES for determinism.

The CREST-first path needs none of this. It identifies microstates by
`h_assignment_fingerprint` — hydrogen count per heavy atom, read off the
geometry — which contains no charges and no bond orders, so two resonance forms
produce the same fingerprint by construction.

## Enumeration rules

Every rule in `charge_enumeration.py` pins the hydrogen count *and* the formal
charge on both sides. An RDKit product template inherits any property it does
not state, so a product written `[NH2:1]` keeps the reactant's `+1`:
`[NH3+:1]>>[NH2:1]` yields an `[NH2+]` whose net charge never reaches the
target, and the rule never fires. Every rule that neutralizes an ion has that
shape, so leaving the charge implicit disables exactly the half of the table
needed to walk a charged input back toward neutral — and, through
`[nH:1]>>[n-:1]`, aromatic N–H deprotonation for neutral inputs too.

## Unsupported inputs

`validate_input_smiles` rejects two classes at the pipeline entry rather than
approximating them. Both are refused because nothing downstream would treat them
correctly, and a plausible-looking number is worse than an error.

**Open-shell species (radicals).** `Geometry.multiplicity` assumes the lowest
multiplicity for the electron count, so a triplet would be evaluated as a
singlet and return an energy for the wrong state. *There are no plans to support
them.* The enumerator needs no separate radical guard: pinning the formal charge
on both sides of every rule is what removed the unpaired electrons, which arose
only on products whose charge the template had failed to adjust.

**Multi-component inputs (salts, solvates, mixtures).** Conformer search is
meaningless when fragments translate freely, and the thermodynamic cycle assumes
a single solute. Desalt and submit the component of interest. *There are no plans
to support them.*

Two further limits are **not** checked, because RDKit discards them before the
key is ever computed. Both *may* be supported in future; neither is today.

**Enhanced stereochemistry** (AND/OR stereo groups) is not modelled. Structures
relying on it are treated as having absolute stereo.

**Atropisomerism** is not perceived from SMILES — `FindPotentialStereo` reports
nothing for a tetra-*ortho*-substituted biaryl under either stereo-perception
setting, and `enumerate_stereoisomers` returns a single isomer. It survives only
in CXSMILES (`|wU:7.6|`), which round-trips correctly, but microstate labels are
written as plain canonical SMILES, which discards it. Supporting atropisomers
would require changes to the stereo enumerator and the stored label format, not
just to the key.
