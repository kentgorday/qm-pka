"""What the pipeline hands its stages: solvent, species and descriptor settings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tautomator import pipeline
from tautomator.config import load_config
from tautomator.ensemble import serialize_ensemble
from tautomator.nonelectrostatic import DESCRIPTOR_FILE
from tautomator.types import ChargeState, Conformer, Ensemble, Geometry, Microstate


class _Stop(Exception):
    """Raised by the stubbed sampling stage once it has seen its arguments."""


@pytest.mark.parametrize(
    ("scoring", "expected"), [("", "woctanol"), ('solvent_model = "none"', None)]
)
def test_a_gas_phase_run_samples_in_the_gas_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scoring: str, expected: str | None
) -> None:
    """A run scored in the gas phase samples without ALPB, so its conformers are
    gas-phase minima and its free energy a gas-phase one throughout."""
    seen: dict[str, object] = {}

    def fake_approach1(**kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(pipeline, "run_approach1", fake_approach1)
    toml = tmp_path / "config.toml"
    toml.write_text(
        f"""protocol = "high_quality"
[molecule]
smiles = "CC(=O)[O-]"
charge_min = -1
charge_max = -1
solvent = "woctanol"
[sampling]
fixed_species = true
[scoring]
{scoring}
[compute]
output_dir = "{tmp_path / "out"}"
"""
    )
    with pytest.raises(_Stop):
        pipeline.run_pipeline(load_config(toml))
    assert seen["solvent"] == expected
    assert seen["fixed_species"] is True


@pytest.mark.parametrize(
    ("scoring", "saved"), [("", False), ("nonelectrostatic_descriptors = true", True)]
)
def test_scoring_saves_descriptors_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scoring: str, saved: bool
) -> None:
    """The descriptors go next to the run's ensemble.json, under their fixed name."""
    seen: dict[str, object] = {}

    def fake_score(*args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(pipeline, "score", fake_score)
    out = tmp_path / "out"
    geom = Geometry(("O", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]))
    ens = Ensemble("[OH-]")
    ens.charge_states[-1] = ChargeState(-1, [Microstate("[OH-]", [Conformer(geom, -75.0)])])
    serialize_ensemble(ens, out / "refinement")
    toml = tmp_path / "config.toml"
    toml.write_text(
        f"""protocol = "high_quality"
[molecule]
smiles = "[OH-]"
charge_min = -1
charge_max = -1
[sampling]
fixed_species = true
[scoring]
{scoring}
[compute]
output_dir = "{out}"
"""
    )
    with pytest.raises(_Stop):
        pipeline.run_pipeline(load_config(toml), resume=True)
    assert seen["descriptors_path"] == (out / DESCRIPTOR_FILE if saved else None)
