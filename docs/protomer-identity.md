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

## When a proton moves during minimisation

The key above identifies a species from a SMILES. `qm_pka/protomer_geometry.py`
identifies one from *coordinates*, which is what catches a conformer whose proton
migrated between heavy atoms while xTB or DFT was minimising it. It is not rare:
across the first training batch about a quarter of conformers at both stages were
filed under a protomer their geometry no longer matched, and nothing detected it.
A protonated carboxylic acid handing its proton to a free amine is the typical
case, and the resulting ammonium/acid form is the obviously correct structure --
the enumerated label was the absurd one.

Ownership is decided by **nearest heavy atom, with no distance cutoff**. That is
enough because it is a deterministic function of the coordinates, and because
where a proton is genuinely shared the two candidates are chemically alike: when
they are alike by automorphism -- the two oxygens of a carboxylate, the two ends
of maleate -- canonicalising the skeleton absorbs the choice and both assignments
name the same species. Where they are inequivalent the choice is arbitrary and is
left so: `charge_state_free_energy` sums flatly over every conformer of every
microstate, so which microstate holds a conformer moves the answer only through
`includes_enantiomer`, at most kT ln 2 = 0.41 kcal/mol. The margin between first
and second nearest heavy atom is recorded for diagnosis but nothing branches on
it, so there is no threshold to tune.

Repair runs before deduplication at both stages, for the same reason
deduplication precedes the Hessians: a migrated conformer is another microstate's
structure under the wrong label, and re-filing it first lets it collapse against
that microstate's own conformers instead of buying a Hessian, and later a DFT
optimization, for a duplicate.

### Why every protomer shares a heavy-atom order

`rdkit_utils.frame_atom_order` ranks heavy atoms canonically on the *frame* --
the heavy-atom graph with hydrogen counts, charges, bond orders, aromaticity and
stereo all erased -- which is byte-identical for every protomer of a molecule.
`smiles_to_3d` embeds in that order. Without it each microstate inherits the
canonical order of its own SMILES, and those diverge, because RDKit's canonical
ranking depends on formal charge and hydrogen count; re-filing a conformer would
then require discovering an atom correspondence first. Hydrogens still interleave
differently, since SMILES writes each one attached to its heavy atom, so a moved
conformer's hydrogens are regrouped -- through the *H-pinned* skeleton's ranking,
which is what decides which automorphic site the proton sits on.

### What is not repaired

Three outcomes end in `excluded_conformers` rather than being dropped, so a run
records what it computed and could not place:

**A detached hydrogen.** The energy is real but belongs to a fragmented species.
Two conformers in the first batch had hydrogens 5-6 A from every heavy atom, both
carrying Boltzmann weight 1.0.

**No matching microstate.** Approach 1's microstate set is *prescribed* by the
enumerator, and a geometry outside it cannot be labelled without perceiving bond
orders from coordinates. C-protonated arenium is the observed case. Unlike every
other exclusion in the pipeline this one discards a valid energy for a real
species; `multiplicity` records the size of the loss. Approach 2 has no such case:
its microstates are *discovered* from the hydrogen distribution, so an unseen
distribution opens a new microstate instead.

**An ambiguous destination.** Several microstates share a protonation key and
the geometry does not settle which -- see below. Rare: nothing in the first
training batch reached this.

### Stereochemistry is a tie-break, not part of the key

The skeleton cannot hold stereo. Every bond is single, so a double bond has no
configuration to annotate, and erasing charge and bond order can remove a
stereocentre outright by making two substituents identical -- which is the same
erasure that merges resonance forms, working as intended:

```
C[C@@H](C(=O)O)C(=O)[O-]     real stereo elements 1, skeleton 0
C[C@H](/C=C/C)/C=C\C         real stereo elements 3, skeleton 0
```

`protomer_key` handles this by appending CIP labels taken from the real molecule
and keyed by skeleton rank. That works because it only ever compares *resonance
forms*, which have identical hydrogen counts and therefore identical skeletons,
so the ranks mean the same thing on both sides.

The geometry path cannot do the same. It compares *different protomers*, whose
skeletons differ, so skeleton ranks are not comparable between them -- and there
is no CIP descriptor to read from bare coordinates in any case. So stereo stays
out of the protonation key and is applied only when the key leaves several
candidates, by `resolve_stereo_candidates`: impose each candidate's bond orders
on the coordinates, ask `AssignStereochemistryFrom3D` what configuration they
show, and keep the candidate that agrees with itself. Each hypothesis is tested
on its own bond orders, so the migration having invalidated the *source's* bond
orders does not matter.

Which stereo elements enter that comparison is the whole difficulty. Three
treatments, and each is load-bearing:

**Decisive** -- the elements the candidates disagree on. Real values kept; these
are what answer the question.

**Non-discriminating** -- elements every candidate specifies identically. They
cannot tell the candidates apart, and comparing them actively harms, in two ways.
A microstate that stands for a collapsed enantiomeric pair
(`includes_enantiomer`) writes one configuration but means both, so the geometry
may legitimately show the other. And a migration can *create* a stereocentre:
`N/C(=C/C(F)(F)F)C([OH2+])[OH2+]` has two identical substituents on that carbon,
so it is not a centre until a proton leaves, and the source geometry never had an
opinion about it. Demanding a match there rejects every candidate.

Erasing them is the obvious move and it is wrong. A 1,4-ring carbon has two
constitutionally identical branches, so it is a stereocentre only *relative to
its partner*; erase the partner and RDKit drops the surviving tag when writing
the SMILES, leaving cis and trans byte-identical. So they are **neutralised** --
both sides forced to one arbitrary common tag -- which cancels them from the
comparison while keeping the pair writable.

**Everything else** -- erased, because `AssignStereochemistryFrom3D` annotates
whatever the coordinates support, including bonds no label constrains: a
protonated carbonyl is a stereogenic double bond, and it would match nothing.

One filter applies to the decisive bonds too: a bond that is *single* under a
given candidate's bond orders is free to rotate and cannot testify, so it is
dropped from that candidate's comparison rather than scored as a mismatch.

Ring cis/trans is worth calling out because it is not a classical stereocentre,
yet the two diastereomers have identical hydrogen counts and therefore share a
protonation key. Only this tie-break tells them apart. Note also that the *frame*
used for atom ordering erases hydrogen counts as well, so it cannot distinguish a
saturated ring from an aromatic one at all -- harmless, since every protomer
shares the frame and so shares its tie-break, but the H-pinned skeleton is what
keeps such branches apart for identity.

Deriving bond orders from the geometry instead, by searching for a resonance
structure that accommodates the observed hydrogen pattern, would be a much larger
problem than this one: the enumerator has already produced the candidate Lewis
structures, so the task is to choose from a known list rather than to construct
one. It would only help for **no matching microstate**, where the species is
outside the model anyway.

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
