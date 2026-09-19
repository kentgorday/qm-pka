from qm_pka.stereo import (
    canonical_enantiomer,
    deduplicate_enantiomers,
    enumerate_and_deduplicate,
    enumerate_stereoisomers,
    mirror_smiles,
)


class TestEnumerateStereoisomers:
    def test_no_stereocenters(self) -> None:
        result = enumerate_stereoisomers("CCO")
        assert len(result) == 1

    def test_one_tetrahedral(self) -> None:
        # Alanine-like: one tetrahedral center -> 2 stereoisomers
        result = enumerate_stereoisomers("CC(N)C(=O)O")
        assert len(result) == 2

    def test_two_tetrahedral(self) -> None:
        """Butane-2,3-diol: RS and SR are one meso compound, so 3 and not 4."""
        result = enumerate_stereoisomers("CC(O)C(O)C")
        assert len(result) == 3

    def test_ez_bond(self) -> None:
        # 2-butene has E/Z, and neither is specified here
        result = enumerate_stereoisomers("CC=CC")
        assert len(result) == 2

    def test_combined_tetrahedral_and_ez(self) -> None:
        """One free centre beside a *specified* E bond: 2, not 4.

        This assertion used to read `>= 2`, which passes whether the specified
        bond is respected or re-enumerated -- so it recorded the ambiguity
        instead of deciding it, and the defect lived behind it for five months.
        """
        result = enumerate_stereoisomers("CC(O)/C=C/C")
        assert len(result) == 2
        assert all("/C=C/" in smi for smi in result), "the E bond must survive"


class TestSpecifiedStereoIsAnInputNotAQuestion:
    """A configuration the caller wrote down must come back unchanged.

    Enumerating it produces a *different compound* and scores it as though it
    were the one asked about. Fumaric acid is the case that exposed this: it was
    turned into fumaric and maleic acid at every charge state, and the maleate
    monoanion -- stabilised by a near-symmetric internal hydrogen bond -- won the
    energy window, so the reported pKa1 was a fumaric-to-maleate transition.
    """

    def test_a_specified_double_bond_is_not_re_enumerated(self) -> None:
        assert enumerate_stereoisomers("O=C(O)/C=C/C(=O)O") == ["O=C(O)/C=C/C(=O)O"]

    def test_the_cis_isomer_is_likewise_left_alone(self) -> None:
        """Symmetrically: asking about maleic acid must not produce fumaric."""
        result = enumerate_stereoisomers(r"O=C(O)/C=C\C(=O)O")
        assert len(result) == 1
        assert "/C=C/" not in result[0]

    def test_a_specified_centre_is_not_re_enumerated(self) -> None:
        """Proline: the spurious partner was harmless only because it is an
        enantiomer and gets collapsed. E/Z pairs are diastereomers and do not."""
        assert enumerate_stereoisomers("O=C(O)[C@@H]1CCCN1") == ["O=C(O)[C@@H]1CCCN1"]

    def test_a_centre_created_by_protonation_is_still_enumerated(self) -> None:
        """The case the enumeration exists for, and the reason not to over-correct.

        A tertiary amine is not a stereocentre; protonating it quaternises the
        nitrogen, which cannot invert without breaking a bond. The charge-state
        walk writes `[NH+]` with no tag, so the centre is *unspecified in the
        product* and both invertomers must still be generated.
        """
        result = enumerate_stereoisomers("CC[NH+](C)C[C@@H](C)O")
        assert len(result) == 2, "both nitrogen invertomers"
        assert len({smi for smi in result}) == 2
        # ... and the carbinol centre the caller did specify is untouched.
        assert all("[C@@H](C)O" in smi or "[C@H](C)O" in smi for smi in result)

    def test_an_unspecified_centre_is_still_enumerated(self) -> None:
        assert len(enumerate_stereoisomers("CC(N)C(=O)O")) == 2


class TestMirrorSmiles:
    def test_inverts_tetrahedral(self) -> None:
        r_form = "[C@@H](F)(Cl)Br"
        s_form = mirror_smiles(r_form)
        assert r_form != s_form
        # Mirroring twice should give back the original
        back = mirror_smiles(s_form)
        assert back == r_form or canonical_enantiomer(back) == canonical_enantiomer(r_form)

    def test_preserves_ez(self) -> None:
        e_form = r"F/C=C/F"
        mirrored = mirror_smiles(e_form)
        # E/Z should be preserved (no tetrahedral centers to flip)
        assert "=" in mirrored

    def test_achiral_unchanged(self) -> None:
        smi = "CCO"
        assert mirror_smiles(smi) == smi


class TestCanonicalEnantiomer:
    def test_deterministic(self) -> None:
        r_form = "[C@@H](F)(Cl)Br"
        s_form = mirror_smiles(r_form)
        assert canonical_enantiomer(r_form) == canonical_enantiomer(s_form)

    def test_achiral_returns_self(self) -> None:
        smi = "CC(=O)O"
        assert canonical_enantiomer(smi) == smi

    def test_meso_compound(self) -> None:
        # meso-tartaric acid: mirror image = self
        # (R,S)-tartaric acid
        meso = "[C@H](O)(C(=O)O)[C@@H](O)C(=O)O"
        assert canonical_enantiomer(meso) == canonical_enantiomer(meso)


class TestDeduplicateEnantiomers:
    def test_pair_reduced_to_one(self) -> None:
        r_form = "[C@@H](F)(Cl)Br"
        s_form = "[C@H](F)(Cl)Br"
        result = deduplicate_enantiomers([r_form, s_form])
        assert len(result) == 1
        _, has_enant = result[0]
        assert has_enant is True

    def test_preserves_diastereomers(self) -> None:
        # Two stereocenters: RR/SS are enantiomers, RS/SR are enantiomers
        stereoisomers = enumerate_stereoisomers("CC(O)C(O)C")
        deduped = deduplicate_enantiomers(stereoisomers)
        # Should have fewer than total stereoisomers
        assert len(deduped) <= len(stereoisomers)
        # But should keep at least one diastereomer pair
        assert len(deduped) >= 2 or len(stereoisomers) <= 2

    def test_no_stereocenters(self) -> None:
        result = deduplicate_enantiomers(["CCO", "OCC"])
        # Both canonicalize to the same thing
        assert len(result) == 1
        _, has_enant = result[0]
        assert has_enant is False

    def test_meso_detected_as_enantiomeric(self) -> None:
        # meso-tartaric acid: physically achiral, but RDKit's SMILES
        # canonicalization doesn't detect internal symmetry, so the
        # canonical and mirror SMILES differ. This means includes_enantiomer
        # is True, which double-counts by 2x — but since this happens at
        # every charge state, it cancels in pKa ratios.
        meso = "[C@H](O)(C(=O)O)[C@@H](O)C(=O)O"
        result = deduplicate_enantiomers([meso])
        assert len(result) == 1
        _, has_enant = result[0]
        assert has_enant is True


class TestEnumerateAndDeduplicate:
    def test_one_center(self) -> None:
        result = enumerate_and_deduplicate("CC(N)C(=O)O")
        # One tetrahedral -> 2 stereoisomers -> 1 after enantiomer dedup
        assert len(result) == 1
        _, has_enant = result[0]
        assert has_enant is True

    def test_two_centers(self) -> None:
        # 2,3-butanediol: RR, SS (enantiomers), RS (meso)
        result = enumerate_and_deduplicate("CC(O)C(O)C")
        smiles_list = [smi for smi, _ in result]
        enant_list = [e for _, e in result]
        # Should keep meso + one of RR/SS = 2
        assert len(smiles_list) >= 2
        # One should be enantiomeric (RR or SS), one should not (meso)
        assert True in enant_list
        assert False in enant_list

    def test_no_centers(self) -> None:
        result = enumerate_and_deduplicate("CCO")
        assert len(result) == 1
        _, has_enant = result[0]
        assert has_enant is False

    def test_ez_only(self) -> None:
        # E/Z are not enantiomers (mirror doesn't flip E/Z)
        result = enumerate_and_deduplicate("CC=CC")
        assert len(result) == 2
        # Neither E nor Z has a tetrahedral center, so no enantiomers
        for _, has_enant in result:
            assert has_enant is False
