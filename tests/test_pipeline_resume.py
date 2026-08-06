"""Tests for stage-level resumption in run_pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qm_pka.ensemble import serialize_ensemble
from qm_pka.pipeline import _find_resume_point
from qm_pka.types import ChargeState, Conformer, Ensemble, Geometry, Microstate


def _ensemble(smiles: str = "CCO") -> Ensemble:
    geom = Geometry(symbols=("O", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]))
    conf = Conformer(geometry=geom, electronic_energy=-1.0)
    ms = Microstate(tautomer_id="abc123", conformers=[conf], smiles=smiles)
    ens = Ensemble(input_smiles=smiles)
    ens.charge_states[0] = ChargeState(charge=0, microstates=[ms])
    return ens


class TestFindResumePoint:
    def test_none_when_no_checkpoints(self, tmp_path: Path) -> None:
        assert _find_resume_point(tmp_path, "CCO") is None

    def test_finds_sampling(self, tmp_path: Path) -> None:
        serialize_ensemble(_ensemble(), tmp_path / "sampling")
        found = _find_resume_point(tmp_path, "CCO")
        assert found is not None
        _ens, stage = found
        assert stage == "sampling"

    def test_refinement_supersedes_sampling(self, tmp_path: Path) -> None:
        # both present -> resume from the later stage, not the earlier one
        serialize_ensemble(_ensemble(), tmp_path / "sampling")
        serialize_ensemble(_ensemble(), tmp_path / "refinement")
        found = _find_resume_point(tmp_path, "CCO")
        assert found is not None
        _ens, stage = found
        assert stage == "refinement"

    def test_conformers_round_trip(self, tmp_path: Path) -> None:
        serialize_ensemble(_ensemble(), tmp_path / "refinement")
        found = _find_resume_point(tmp_path, "CCO")
        assert found is not None
        ens, _stage = found
        conf = ens.charge_states[0].microstates[0].conformers[0]
        assert conf.electronic_energy == -1.0
        assert conf.geometry.symbols == ("O", "H")

    def test_smiles_mismatch_is_refused(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Output dirs are numbered by CSV row; editing the training set shifts
        # them, so a checkpoint can belong to a different molecule entirely.
        serialize_ensemble(_ensemble("CCO"), tmp_path / "refinement")
        assert _find_resume_point(tmp_path, "CCN") is None
        assert "but this config asks for" in caplog.text
