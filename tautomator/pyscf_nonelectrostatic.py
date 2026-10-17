"""The inputs to the nonelectrostatic terms, from a converged PySCF PCM calculation.

`solute_from_mf` takes the mean field `pyscf_runner._build_mf` builds for one of
the analytical PCM models (IEF-PCM, C-PCM, COSMO), FastPCM or PySCF's own, once
its SCF has converged, and returns the backend-neutral `nonelectrostatic.Solute`:

* the PCM surface and its apparent surface charges for the converged density,
  with the sphere radii the cavity was built from (`_build_mf` sets them; an
  ECP atom's radius goes by its true nuclear charge, as in
  `_patch_pcm_ecp_cavity`);
* the solute's own electric field (nuclei and electrons, no reaction field)
  normal to each surface element;
* the electron density and its gradient on `cavity_quadrature` grids for the
  inside and the outside of the cavity.

An ECP leaves only the valence density for the dispersion and exchange
integrals, and puts the ECP-reduced nuclear charges in the field.  At
def2-QZVPPD that affects iodine and heavier elements only.

`single_point_with_descriptors` is `pyscf_runner.single_point` with the
descriptors read off its mean field before it is discarded; scoring calls it
when [scoring] nonelectrostatic_descriptors is on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyscf import df, dft, gto

from tautomator import pyscf_runner
from tautomator.cavity_quadrature import LEBEDEV_ORDER, N_RADIAL, cavity_grid
from tautomator.config import DEFAULT_MEMORY_GB, PCM_METHODS, solvent_electron_density
from tautomator.nonelectrostatic import DensityGrid, Descriptors, Solute, Surface, descriptors
from tautomator.types import Geometry

Array = NDArray[np.float64]

# Grid points per block when evaluating the density, and the memory (bytes) one
# block of field integrals may take.
_GRID_BLOCK = 20_000
_FIELD_BLOCK_BYTES = 200e6


def _charge_scale(method: str, eps: float) -> float:
    """How PySCF's PCM model scales a conductor's surface charges, for a point
    charge at the centre of a sphere (see `nonelectrostatic.Surface`).

    COSMO's f(eps) has 1/2 where C-PCM's has 0; IEF-PCM keeps (eps-1)/(eps+1)
    in its K and R matrices, but its charges come out at (eps-1)/eps there.
    """
    if method.upper() == "COSMO":
        return (eps - 1.0) / (eps + 0.5)
    return (eps - 1.0) / eps


def _surface(mol: Any, ws: Any) -> Surface:
    """The PCM surface and the apparent surface charges the solvent object holds."""
    from pyscf.data import radii
    from pyscf.solvent import pcm

    table = ws.radii_table
    if table is None:
        table = ws.vdw_scale * pcm.modified_Bondi + ws.r_probe / radii.BOHR
    # Sphere radii go by true nuclear charge, also for ECP atoms.
    true_z = [gto.charge(mol.atom_symbol(ia)) for ia in range(mol.natm)]
    # Each atom's elements are one contiguous slice.
    sizes = [p1 - p0 for p0, p1 in ws.surface["gslice_by_atom"]]
    eps = float(ws.eps)
    return Surface(
        coords=np.asarray(ws.surface["grid_coords"], dtype=np.float64),
        normals=np.asarray(ws.surface["norm_vec"], dtype=np.float64),
        areas=np.asarray(ws.surface["area"], dtype=np.float64),
        charges=np.asarray(ws._intermediates["q"], dtype=np.float64),
        atom_coords=np.asarray(mol.atom_coords(unit="B"), dtype=np.float64),
        atom_radii=np.asarray([table[z] for z in true_z], dtype=np.float64),
        element_atoms=np.repeat(np.arange(mol.natm), sizes),
        eps=eps,
        charge_scale=_charge_scale(ws.method, eps),
    )


def density_on_grid(mol: Any, dm: Array, coords: Array, weights: Array) -> DensityGrid:
    """Electron density and its gradient at the given quadrature points."""
    ni = dft.numint.NumInt()
    rho = np.empty((4, len(coords)))
    for p0 in range(0, len(coords), _GRID_BLOCK):
        ao = ni.eval_ao(mol, coords[p0 : p0 + _GRID_BLOCK], deriv=1)
        rho[:, p0 : p0 + _GRID_BLOCK] = ni.eval_rho(mol, ao, dm, xctype="GGA", hermi=1)
    return DensityGrid(
        coords=coords, weights=weights, rho=rho[0], grad=np.ascontiguousarray(rho[1:].T)
    )


def electric_field(mol: Any, dm: Array, points: Array) -> Array:
    """The solute's electric field (nuclei + electrons) at each point, shape (n, 3).

    The electronic potential at s is -sum D_mn (mn|s), and by translational
    invariance its gradient with respect to s is the integral with the
    derivative moved onto the orbitals, so E_elec = 2 sum D_mn (grad m n|s).
    Nuclei enter with the charges the electrons see (ECP-reduced, if any).
    """
    sep = points[:, None, :] - mol.atom_coords(unit="B")[None, :, :]
    dist = np.linalg.norm(sep, axis=2)
    field: Array = np.einsum("a,pax->px", mol.atom_charges(), sep / dist[:, :, None] ** 3)

    intor = mol._add_suffix("int3c2e_ip1")
    block = max(1, int(_FIELD_BLOCK_BYTES / (3 * 8 * mol.nao**2)))
    for p0 in range(0, len(points), block):
        fakemol = gto.fakemol_for_charges(points[p0 : p0 + block])
        ints = df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s1", comp=3)
        field[p0 : p0 + block] += 2.0 * np.einsum("xijp,ij->px", ints, dm)
    return field


def solute_from_mf(
    mf: Any, lebedev_order: int = LEBEDEV_ORDER, n_radial: int = N_RADIAL
) -> Solute:
    """Gather surface, density and normal field from a converged PCM mean field.

    ``lebedev_order`` and ``n_radial`` set the cavity quadrature the density is
    evaluated on (see `cavity_quadrature.cavity_grid`), not the PCM surface.
    """
    if not mf.converged:
        raise ValueError("the SCF is not converged")
    mol, ws = mf.mol, mf.with_solvent
    if not ws.surface or "q" not in (ws._intermediates or {}):
        raise ValueError("the PCM solvent object holds no converged surface charges")
    dm = np.asarray(mf.make_rdm1(), dtype=np.float64)
    if dm.ndim == 3:  # unrestricted: total density
        dm = dm[0] + dm[1]
    # The charges for this density.  Those the SCF left behind belong to the
    # last density it evaluated, which a Newton retry need not end on.
    ws._get_vind(dm)
    surface = _surface(mol, ws)
    field = electric_field(mol, dm, surface.coords)
    centres, radii = surface.atom_coords, surface.atom_radii
    inside = cavity_grid(centres, radii, "inside", lebedev_order, n_radial)
    outside = cavity_grid(centres, radii, "outside", lebedev_order, n_radial)
    return Solute(
        surface=surface,
        interior=density_on_grid(mol, dm, *inside),
        exterior=density_on_grid(mol, dm, *outside),
        normal_field=np.einsum("px,px->p", field, surface.normals),
    )


def single_point_with_descriptors(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    solvent_model: str,
    solvent: str,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> tuple[float, Descriptors]:
    """A PCM single point, as `pyscf_runner.single_point` runs it, and its descriptors.

    Returns (total energy in hartree, descriptors at the default cavity
    quadrature, with the solvent's electron density in the dispersion term).
    """
    if solvent_model.upper().replace("-", "") not in PCM_METHODS:
        raise ValueError(
            f"The nonelectrostatic descriptors need an analytical PCM model "
            f"({', '.join(sorted(set(PCM_METHODS.values())))}), not {solvent_model!r}"
        )
    _mol, mf = pyscf_runner._build_mf(
        geom,
        charge,
        method,
        basis,
        solvent_model,
        solvent,
        pcm_hydrogen_radius,
        threads,
        memory_gb,
    )
    energy = pyscf_runner._run_scf_robust(mf)
    return energy, descriptors(solute_from_mf(mf), solvent_electron_density(solvent))
