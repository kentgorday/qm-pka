"""Quasi-RRHO thermochemistry from harmonic frequencies.

Implements Grimme's quasi-rigid-rotor-harmonic-oscillator (quasi-RRHO)
approach (Grimme, Chem. Eur. J. 2012, 18, 9955) for computing free
energy corrections from vibrational frequencies. Low-frequency modes
are interpolated between harmonic oscillator and free rotor treatments
using a damping function, avoiding the divergent entropy contributions
that plague standard RRHO at low frequencies.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Physical constants
KB = 1.380649e-23  # Boltzmann constant, J/K
H_PLANCK = 6.62607015e-34  # Planck constant, J·s
C_LIGHT = 2.99792458e10  # speed of light, cm/s
R_GAS = 8.314462618  # gas constant, J/(mol·K)
HARTREE_TO_JOULE = 4.3597447222071e-18
CAL_TO_JOULE = 4.184
AVOGADRO = 6.02214076e23

# Default cutoff for quasi-RRHO damping (cm⁻¹)
QRRHO_CUTOFF = 100.0


def quasi_rrho_free_energy(
    frequencies_cm: list[float],
    temperature: float = 298.15,
    cutoff: float = QRRHO_CUTOFF,
    pressure: float = 101325.0,
) -> float:
    """Compute the vibrational free energy correction using quasi-RRHO.

    Uses Grimme's interpolation between harmonic oscillator and free
    rotor for low-frequency modes (below `cutoff` cm⁻¹).

    Args:
        frequencies_cm: Vibrational frequencies in cm⁻¹. The first 6
            (or 5 for linear) should be near-zero translational/rotational
            modes — these are automatically excluded (|freq| < 10 cm⁻¹).
            Imaginary frequencies (negative values) are treated as real
            using their absolute value.
        temperature: Temperature in Kelvin.
        cutoff: Frequency cutoff for quasi-RRHO damping in cm⁻¹.
        pressure: Pressure in Pa (for translational entropy, unused here
            since we only compute vibrational contributions).

    Returns:
        Vibrational free energy correction in Hartree.
        This is G_vib = ZPE + H_vib(T) - T*S_vib(T), where S_vib
        uses the quasi-RRHO treatment for low-frequency modes.
    """
    kbt = KB * temperature
    beta = 1.0 / kbt

    g_vib = 0.0
    for freq in frequencies_cm:
        # Skip translational/rotational modes and imaginary frequencies
        if abs(freq) < 10.0:
            continue
        # Treat imaginary frequencies as real (use absolute value).
        # Small imaginary frequencies at a nominally optimized geometry are
        # typically numerical artifacts, not true saddle points.
        if freq < 0:
            log.warning(
                f"Imaginary frequency ({freq:.1f} cm⁻¹) treated as real "
                f"({abs(freq):.1f} cm⁻¹) for free energy calculation"
            )
            freq = abs(freq)

        # Convert frequency to energy
        hv = H_PLANCK * C_LIGHT * freq  # energy of one quantum, Joules

        # ZPE contribution
        zpe = 0.5 * hv

        # Thermal enthalpy: hv / (exp(hv/kT) - 1)
        x = hv * beta
        h_vib = hv / (np.exp(x) - 1.0)

        # Entropy: quasi-RRHO interpolation
        s_ho = _entropy_ho(freq, temperature)
        s_fr = _entropy_free_rotor(freq, temperature)
        w = _grimme_weight(freq, cutoff)
        s_vib = w * s_ho + (1.0 - w) * s_fr

        # G = ZPE + H_thermal - T*S  (per mode, in Joules)
        g_mode = zpe + h_vib - temperature * s_vib
        g_vib += g_mode

    # Convert from Joules (per molecule) to Hartree
    return g_vib / HARTREE_TO_JOULE


def _entropy_ho(freq: float, temperature: float) -> float:
    """Harmonic oscillator entropy for a single mode (J/K per molecule)."""
    hv = H_PLANCK * C_LIGHT * freq
    x = hv / (KB * temperature)
    # S_HO = kB * [x/(e^x - 1) - ln(1 - e^{-x})]
    return float(KB * (x / (np.exp(x) - 1.0) - np.log(1.0 - np.exp(-x))))


def _entropy_free_rotor(freq: float, temperature: float) -> float:
    """Free rotor entropy for a single mode (J/K per molecule).

    Uses the Grimme (2012) expression with an effective moment of inertia
    derived from the frequency: mu' = h/(8*pi^2*freq*c).
    """
    # Effective moment of inertia
    mu_prime = H_PLANCK / (8.0 * np.pi**2 * C_LIGHT * freq)
    # S_FR = R * (1/2 + ln(sqrt(8*pi^3*mu'*kT/h^2)))
    # Per molecule (divide R by Avogadro = kB):
    arg = 8.0 * np.pi**3 * mu_prime * KB * temperature / H_PLANCK**2
    return float(KB * (0.5 + 0.5 * np.log(arg)))


def _grimme_weight(freq: float, cutoff: float) -> float:
    """Grimme damping function: w(v) = 1 / (1 + (cutoff/v)^4).

    Returns 1 for frequencies >> cutoff (pure HO),
    0 for frequencies << cutoff (pure free rotor).
    """
    return 1.0 / (1.0 + (cutoff / freq) ** 4)
