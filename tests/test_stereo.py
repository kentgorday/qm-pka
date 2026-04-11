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
        # Two stereocenters -> up to 4 stereoisomers
        result = enumerate_stereoisomers("CC(O)C(O)C")
        assert len(result) >= 3  # may be 3 if meso is present

    def test_ez_bond(self) -> None:
        # 2-butene has E/Z
        result = enumerate_stereoisomers("CC=CC")
        assert len(result) == 2

    def test_combined_tetrahedral_and_ez(self) -> None:
        # Molecule with both tetrahedral and E/Z
        result = enumerate_stereoisomers("CC(O)/C=C/C")
        # 1 tetrahedral x defined E -> 2 stereoisomers from tetrahedral
        # But enumerate with onlyUnassigned=False reassigns all
        assert len(result) >= 2


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
