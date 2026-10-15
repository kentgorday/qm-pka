"""Tests for experimental pKa label parsing and run-directory lookup."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tautomator.training_set import (
    compound_dir_name,
    compound_id,
    find_compound_dir,
    parse_pka_column,
)


class TestParsePkaColumn:
    def test_an_acid_ladder_walks_down_from_neutral(self) -> None:
        got = parse_pka_column("pKa1=2.847;pKa2=5.696")
        assert [(v.higher_charge, v.lower_charge) for v in got] == [(0, -1), (-1, -2)]

    def test_a_base_ladder_walks_down_from_the_cation(self) -> None:
        got = parse_pka_column("pKaH1=9.925;pKaH2=6.854")
        assert [(v.higher_charge, v.lower_charge) for v in got] == [(1, 0), (2, 1)]

    def test_a_mixed_entry_is_assigned_by_label_not_position(self) -> None:
        """The case that made positional pairing wrong.

        perrin3287 lists `pKa1` before `pKaH1`, so pairing by order sends a
        q=0 -> q=-1 measurement to the q=+1 -> q=0 transition.
        """
        got = parse_pka_column("pKa1=10.640;pKaH1=1.952")
        by_label = {v.label: (v.higher_charge, v.lower_charge) for v in got}
        assert by_label == {"pKa1": (0, -1), "pKaH1": (1, 0)}
        assert got[0].label == "pKa1", "order is preserved, but no longer meaningful"

    def test_every_transition_drops_exactly_one_proton(self) -> None:
        for column in ("pKa1=1;pKa2=2;pKa3=3", "pKaH1=1;pKaH2=2", "pKa1=1;pKaH1=2"):
            for v in parse_pka_column(column):
                assert v.lower_charge == v.higher_charge - 1

    def test_blank_and_trailing_separators_are_tolerated(self) -> None:
        assert parse_pka_column("") == []
        assert len(parse_pka_column("pKa1=3.1;")) == 1

    @pytest.mark.parametrize("bad", ["pKb1=3.0", "pKa=3.0", "pKa0=3.0", "pKa1", "3.0"])
    def test_an_unrecognised_label_raises(self, bad: str) -> None:
        """Guessing would be indistinguishable from a bad prediction downstream."""
        with pytest.raises(ValueError):
            parse_pka_column(bad)

    def test_the_real_training_set_parses(self) -> None:
        path = Path(__file__).parent.parent / "training_set.csv"
        if not path.exists():
            pytest.skip("training_set.csv not present")
        rows = list(csv.DictReader(path.open()))
        mixed = 0
        for r in rows:
            values = parse_pka_column(r["pkas"])
            assert values, f"{r['unique_ID']} has no parseable pKa"
            kinds = {v.label.startswith("pKaH") for v in values}
            if len(kinds) > 1:
                mixed += 1
        assert mixed == 2, "perrin3294 and perrin3287 mix label types"


class TestCompoundDirs:
    def test_a_new_molecule_gets_a_directory_named_for_its_id(self, tmp_path: Path) -> None:
        assert find_compound_dir(tmp_path, "perrin_supp5572") == tmp_path / "perrin_supp5572"

    def test_a_prefixed_directory_is_found_at_any_row_number(self, tmp_path: Path) -> None:
        """The row a molecule sits on moves whenever the CSV is regenerated."""
        old = tmp_path / "004_perrin3020"
        old.mkdir()
        assert find_compound_dir(tmp_path, "perrin3020") == old

    def test_the_id_must_match_whole(self, tmp_path: Path) -> None:
        (tmp_path / "022_perrin_supp5572").mkdir()
        (tmp_path / "003_serjeant25500").mkdir()
        assert find_compound_dir(tmp_path, "supp5572") == tmp_path / "supp5572"
        assert find_compound_dir(tmp_path, "serjeant2550") == tmp_path / "serjeant2550"

    def test_the_unprefixed_directory_wins(self, tmp_path: Path) -> None:
        (tmp_path / "004_perrin3020").mkdir()
        (tmp_path / "perrin3020").mkdir()
        assert find_compound_dir(tmp_path, "perrin3020") == tmp_path / "perrin3020"

    def test_two_prefixed_directories_for_one_id_raise(self, tmp_path: Path) -> None:
        (tmp_path / "003_perrin3020").mkdir()
        (tmp_path / "004_perrin3020").mkdir()
        with pytest.raises(ValueError, match="more than one run directory"):
            find_compound_dir(tmp_path, "perrin3020")

    def test_an_unsafe_id_is_sanitised_the_same_way_both_times(self, tmp_path: Path) -> None:
        assert compound_dir_name("a/b c") == "a_b_c"
        (tmp_path / "001_a_b_c").mkdir()
        assert find_compound_dir(tmp_path, "a/b c") == tmp_path / "001_a_b_c"

    @pytest.mark.parametrize(
        "dir_name", ["022_perrin_supp5572", "perrin_supp5572"], ids=["prefixed", "plain"]
    )
    def test_the_id_is_read_back_from_either_naming(self, dir_name: str) -> None:
        assert compound_id(dir_name) == "perrin_supp5572"
