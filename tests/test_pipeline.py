"""What the pipeline hands its sampling stage: solvent and species settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from tautomator import pipeline
from tautomator.config import load_config


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
