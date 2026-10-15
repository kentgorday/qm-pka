"""Reading a training-set CSV: experimental pKa labels and run directories.

The `pkas` column mixes two label conventions, and which transition a value
refers to is carried by the *label*, never by the column order or by
`acid_base`:

* ``pKa{n}``  is the n-th proton lost starting from the neutral molecule, so
  ``pKa1`` is q=0 -> q=-1 and ``pKa2`` is q=-1 -> q=-2.
* ``pKaH{n}`` is the n-th proton lost starting from the fully protonated form,
  so ``pKaH1`` is q=+1 -> q=0 and ``pKaH2`` is q=+2 -> q=+1.

Two of the forty training entries mix them -- ``perrin3294``
(``pKa1=9.208;pKaH1=2.186``) and ``perrin3287``
(``pKa1=10.640;pKaH1=1.952``) -- and for those, pairing values with transitions
by position assigns a q=0 -> q=-1 measurement to the q=+1 -> q=0 transition and
reports a number that means nothing. Parse the label.

A batch run keeps each molecule's output in a directory named for its
``unique_ID`` alone, so results stay attached to the molecule when the CSV is
regenerated and its rows move. Runs made before that carry a row-number prefix
(``003_serjeant2550``); `find_compound_dir` still finds them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_LABEL = re.compile(r"^pKa(H?)(\d+)$")
_ROW_PREFIX = re.compile(r"^\d{3}_")


@dataclass(frozen=True)
class PkaValue:
    """One experimental pKa, with the charge-state transition it measures."""

    label: str
    value: float
    higher_charge: int
    """The protonated side of the equilibrium."""
    lower_charge: int
    """The deprotonated side; always ``higher_charge - 1``."""


def parse_pka_column(column: str) -> list[PkaValue]:
    """Parse a ``pkas`` cell into values tagged with their transitions.

    Raises on an unrecognised label rather than guessing, since a silent
    mis-assignment is indistinguishable from a bad prediction downstream.
    """
    out: list[PkaValue] = []
    for field in column.split(";"):
        field = field.strip()
        if not field:
            continue
        label, _, raw = field.partition("=")
        match = _LABEL.match(label.strip())
        if match is None or not raw:
            raise ValueError(f"unrecognised pKa label: {field!r}")
        protonated, index = match.group(1) == "H", int(match.group(2))
        if index < 1:
            raise ValueError(f"pKa index must start at 1: {field!r}")
        # pKaH{n}: +n -> +(n-1).   pKa{n}: -(n-1) -> -n.
        higher = index if protonated else -(index - 1)
        out.append(PkaValue(label.strip(), float(raw), higher, higher - 1))
    return out


def compound_dir_name(unique_id: str) -> str:
    """Filesystem-safe directory name for a molecule's ``unique_ID``."""
    return "".join(c if c.isalnum() else "_" for c in unique_id)[:50]


def compound_id(dir_name: str) -> str:
    """The ID a run directory is named for, without any row-number prefix.

    An ID that itself begins with three digits and an underscore cannot be told
    apart from a prefixed one; go from ID to directory with `find_compound_dir`
    where that matters.
    """
    return _ROW_PREFIX.sub("", dir_name)


def find_compound_dir(output_root: Path, unique_id: str) -> Path:
    """Where a molecule's output lives under ``output_root``, or should be created.

    Prefers the directory named for the ID, then one from an older run that
    prefixed the row number, whatever that number was.
    """
    name = compound_dir_name(unique_id)
    plain = output_root / name
    if plain.is_dir():
        return plain
    prefixed = sorted(output_root.glob(f"[0-9][0-9][0-9]_{name}"))
    if len(prefixed) > 1:
        found = ", ".join(p.name for p in prefixed)
        raise ValueError(f"more than one run directory for {unique_id!r}: {found}")
    return prefixed[0] if prefixed else plain
