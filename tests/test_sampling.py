"""Tests for the sampling stage."""

from __future__ import annotations

import numpy as np
import pytest

from qm_pka import sampling
from qm_pka.types import Conformer, Geometry


def test_a_failed_hessian_does_not_evict_the_microstate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conformer with no RRHO term cannot be compared: free_energy sums
    non-None components, so it is short a term worth 11-114 kcal/mol against an
    ensemble whose real spread is a few kcal/mol. Left in the window it becomes
    e_min and evicts every genuine conformer of the microstate."""
    geom = Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    )
    healthy = [Conformer(geometry=geom, electronic_energy=-76.0 - i * 0.0016) for i in range(5)]
    broken = Conformer(geometry=geom, electronic_energy=-76.002)

    calls = {"n": 0}

    def fake_frequencies(*args: object, **kwargs: object) -> list[float]:
        calls["n"] += 1
        if calls["n"] == 6:  # the last conformer's Hessian fails
            raise RuntimeError("xtb --hess produced no frequencies")
        return [100.0, 200.0, 300.0]

    monkeypatch.setattr(sampling, "frequencies", fake_frequencies)
    monkeypatch.setattr(sampling, "quasi_rrho_free_energy", lambda f: 0.12)
    monkeypatch.setattr(sampling, "deduplicate_conformers", lambda c, e, ethr_kcal=None: c)

    out = sampling._dedupe_add_rrho_and_filter([*healthy, broken], 0, "water", 6.0)

    # every genuine conformer survives ...
    assert all(any(c is o for o in out) for c in healthy)
    # ... and the one with no RRHO is carried to refinement, which recomputes it
    assert any(broken is o for o in out)
    assert broken.rrho_correction is None
