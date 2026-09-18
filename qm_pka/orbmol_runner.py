"""Refinement with the OrbMol-v2 machine-learned potential (orb-models).

OrbMol-v2 is a conservative graph network (forces and Hessians are exact
derivatives of its energy) trained on OMol25 + OPoly26: gas-phase
wB97M-V/def2-TZVPD total energies and forces.  It takes the total charge and
spin multiplicity as inputs and adds a long-range Coulomb term over learned
per-atom charges.  It has no implicit solvent, so it is a refinement-only
driver: config.load_config refuses it for scoring and refuses a refinement
solvent model.  Every energy here is a gas-phase energy in Hartree, on the same
absolute scale as wB97M-V/def2-TZVPD.

The functions mirror the psi4/pyscf runners so refinement.py can call any
driver the same way.  ``method`` selects the model (see MODELS); ``basis`` and
the solvent arguments exist only for that shared signature.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qm_pka.config import DEFAULT_MEMORY_GB
from qm_pka.types import Geometry

log = logging.getLogger(__name__)

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


@dataclass(frozen=True)
class _Model:
    loader: str  # constructor name in orb_models.forcefield.pretrained
    url: str
    sha256: str


# Weights are fetched on first use and cached by cached_path under
# ~/.cache/cached_path; set QM_PKA_ORBMOL_WEIGHTS to a local copy for offline
# use.  Hugging Face serves the same file orb-models' default S3 URL does
# (identical sha256) but, in testing, ~1000x faster.  The checksum is verified
# on every load, so a changed or truncated file cannot silently change results.
MODELS: dict[str, _Model] = {
    "orbmol-v2": _Model(
        loader="orbmol_v2",
        url=(
            "https://huggingface.co/Orbital-Materials/orbmol-v2/resolve/main/"
            "orbmol-v2-teqabfhg-20260523.ckpt"
        ),
        sha256="3dbef70b5fdc9124392181b0d8686c79f1c02bf5fefcb878a4eaa51d03e4f5e8",
    ),
}

# float64 throughout: geomeTRIC's energy-based step control and the
# finite-difference Hessian both want a smooth surface, and orb-models itself
# recommends against "float32-high" for second derivatives.  orb-models sets
# torch's *global* default dtype when it loads a model, so one precision per
# process is all that is safe anyway.
_PRECISION = "float64"

# Central-difference step for the Hessian (Angstrom).
_HESSIAN_STEP = 5e-3


def _model_spec(method: str) -> _Model:
    spec = MODELS.get(method.lower())
    if spec is None:
        raise ValueError(f"Unknown OrbMol model {method!r} (known: {', '.join(MODELS)})")
    return spec


def _verified_weights(path: str, sha256: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise RuntimeError(
            f"OrbMol weights at {path} have sha256 {digest.hexdigest()}, expected "
            f"{sha256}; the file is corrupt or not the model this code was validated with."
        )
    return path


@cache
def _load(method: str) -> tuple[Any, Any]:
    """Load a model on CPU, once per process.

    CPU only: orb-models 0.7.0 cannot target MPS (its architecture builder calls
    ``.cuda()`` for any non-CPU device, and the Warp neighbour list has no MPS
    backend), and at ~20 atoms CPU was faster than MPS with workarounds anyway.
    ``torch.compile`` is off: it buys nothing at this size.
    """
    spec = _model_spec(method)

    import warp
    from cached_path import cached_path
    from orb_models.forcefield import pretrained

    warp.config.quiet = True  # no initialisation banner on stdout
    local = os.environ.get("QM_PKA_ORBMOL_WEIGHTS") or str(cached_path(spec.url))
    weights = _verified_weights(local, spec.sha256)
    log.info(f"Loading {method} from {weights}")
    with warnings.catch_warnings():
        # Both expected: float64 is chosen deliberately (see _PRECISION).
        warnings.filterwarnings("ignore", message="Setting global torch default dtype")
        warnings.filterwarnings("ignore", message="Consider passing 'precision=float32-high'")
        model, adapter = getattr(pretrained, spec.loader)(
            weights_path=weights, device="cpu", precision=_PRECISION, compile=False
        )
    return model, adapter


def _energy_gradient(
    geom: Geometry, charge: int, method: str, threads: int = 1
) -> tuple[float, NDArray[np.float64]]:
    """Return (energy in Hartree, gradient in Hartree/Bohr with shape (n, 3))."""
    import torch
    from ase import Atoms

    torch.set_num_threads(threads)
    model, adapter = _load(method.lower())
    atoms = Atoms(symbols=list(geom.symbols), positions=np.asarray(geom.coords))
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(geom.multiplicity(charge))
    out = model.predict(adapter.from_ase_atoms(atoms, device="cpu"), split=False)
    energy_ev = float(out["energy"].detach().double().reshape(-1)[0])
    forces_ev_ang = out["grad_forces"].detach().double().cpu().numpy().reshape(-1, 3)
    gradient = -forces_ev_ang * BOHR_TO_ANG / HARTREE_TO_EV
    return energy_ev / HARTREE_TO_EV, gradient


def _refuse_solvent(solvent_model: str | None) -> None:
    if solvent_model is not None:
        raise ValueError(
            "OrbMol-v2 is a gas-phase potential with no implicit solvent; "
            f"solvent_model={solvent_model!r} cannot be honoured."
        )


def single_point(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str = "",
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> float:
    """Gas-phase energy in Hartree."""
    _refuse_solvent(solvent_model)
    return _energy_gradient(geom, charge, method, threads=threads)[0]


def optimize(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str = "",
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> tuple[Geometry, float, bool]:
    """Gas-phase geometry optimization with geomeTRIC.

    Same strategy and convergence criteria as the pyscf driver: TRIC internal
    coordinates for 100 steps, then Cartesian coordinates from the last
    geometry for another 100.  Returns (geometry, energy_hartree, converged);
    the geometry and energy are the last step's whether or not it converged.
    """
    import geometric.engine
    import geometric.molecule
    import geometric.optimize
    from geometric.errors import GeomOptNotConvergedError

    _refuse_solvent(solvent_model)

    class _Engine(geometric.engine.Engine):  # type: ignore[misc]
        def __init__(self, start: Geometry) -> None:
            m = geometric.molecule.Molecule()
            m.elem = list(start.symbols)
            m.xyzs = [np.asarray(start.coords, dtype=float)]
            super().__init__(m)
            self.last: tuple[NDArray[np.float64], float] | None = None

        def calc_new(self, coords: NDArray[np.float64], dirname: str) -> dict[str, Any]:
            xyz = coords.reshape(-1, 3) * BOHR_TO_ANG
            e, g = _energy_gradient(
                Geometry(symbols=geom.symbols, coords=xyz), charge, method, threads=threads
            )
            self.last = (xyz, e)
            return {"energy": e, "gradient": g.ravel()}

    def _run(start: Geometry, coordsys: str) -> tuple[Geometry, float, bool]:
        engine = _Engine(start)
        with tempfile.TemporaryDirectory(prefix="orbmol_opt_") as tmpdir:
            try:
                progress = geometric.optimize.run_optimizer(
                    customengine=engine,
                    input=os.path.join(tmpdir, str(uuid.uuid4())),
                    coordsys=coordsys,
                    maxiter=100,
                    logIni=_quiet_log_ini(tmpdir),
                )
                xyz = np.asarray(progress.xyzs[-1], dtype=float)
                energy = float(progress.qm_energies[-1])
                converged = True
            except GeomOptNotConvergedError:
                assert engine.last is not None
                xyz, energy = engine.last
                converged = False
        return Geometry(symbols=start.symbols, coords=xyz), energy, converged

    opt_geom, energy, converged = _run(geom, "tric")
    if converged:
        return opt_geom, energy, True
    log.info(
        "Internal-coord opt did not converge in 100 steps; "
        "retrying from last geometry with Cartesian coords"
    )
    return _run(opt_geom, "cart")


def _quiet_log_ini(tmpdir: str) -> str:
    """A geomeTRIC logging config that prints nothing.

    Its default config writes every step to stdout, which would drown the
    pipeline log across hundreds of conformers.
    """
    path = os.path.join(tmpdir, "log.ini")
    with open(path, "w") as f:
        f.write(
            "[loggers]\nkeys=root\n\n[handlers]\nkeys=null\n\n[formatters]\nkeys=\n\n"
            "[logger_root]\nlevel=WARNING\nhandlers=null\n\n"
            "[handler_null]\nclass=NullHandler\nargs=()\n"
        )
    return path


def frequencies(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str = "",
    solvent_model: str | None = None,
    solvent: str | None = None,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> list[float]:
    """Harmonic frequencies in cm^-1 (imaginary ones negative), translations and
    rotations projected out.

    The Hessian is a central difference of the model's analytic gradients (6N
    gradient calls).  The harmonic analysis is PySCF's -- the one the pyscf
    driver uses -- so masses and projection match that driver exactly.
    """
    from pyscf import gto
    from pyscf.hessian.thermo import harmonic_analysis

    _refuse_solvent(solvent_model)
    n = geom.n_atoms
    x0 = np.asarray(geom.coords, dtype=float)
    hess = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        dx = np.zeros(3 * n)
        dx[k] = _HESSIAN_STEP
        grad = [
            _energy_gradient(
                Geometry(geom.symbols, x0 + sign * dx.reshape(-1, 3)),
                charge,
                method,
                threads=threads,
            )[1]
            for sign in (1.0, -1.0)
        ]
        hess[k] = (grad[0] - grad[1]).ravel() / (2 * _HESSIAN_STEP / BOHR_TO_ANG)
    hess = 0.5 * (hess + hess.T)

    mol = gto.M(
        atom=[(s, tuple(c)) for s, c in zip(geom.symbols, x0, strict=True)],
        basis="sto-3g",  # unused: harmonic_analysis only needs masses and coordinates
        charge=charge,
        spin=geom.n_electrons(charge) % 2,
        verbose=0,
    )
    hess_atoms = hess.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    results = harmonic_analysis(mol, hess_atoms, imaginary_freq=False)
    return [float(f) for f in results["freq_wavenumber"]]
