from qm_pka.thermo import (
    QRRHO_CUTOFF,
    _grimme_weight,
    quasi_rrho_free_energy,
)


class TestGrimmeWeight:
    def test_high_frequency_is_ho(self) -> None:
        # Far above cutoff -> weight ~ 1 (harmonic oscillator)
        w = _grimme_weight(1000.0, QRRHO_CUTOFF)
        assert w > 0.999

    def test_low_frequency_is_free_rotor(self) -> None:
        # Far below cutoff -> weight ~ 0 (free rotor)
        w = _grimme_weight(10.0, QRRHO_CUTOFF)
        assert w < 0.01

    def test_at_cutoff_is_half(self) -> None:
        # At the cutoff frequency, w = 1/(1 + 1) = 0.5
        w = _grimme_weight(QRRHO_CUTOFF, QRRHO_CUTOFF)
        assert abs(w - 0.5) < 1e-10

    def test_monotonically_increasing(self) -> None:
        freqs = [10, 50, 100, 200, 500, 1000]
        weights = [_grimme_weight(f, QRRHO_CUTOFF) for f in freqs]
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]


class TestQuasiRRHOFreeEnergy:
    def test_single_high_frequency(self) -> None:
        # A single high-frequency mode (like a C-H stretch ~3000 cm⁻¹)
        # should give a positive free energy (dominated by ZPE)
        g = quasi_rrho_free_energy([3000.0])
        assert g > 0  # ZPE contribution is positive

    def test_near_zero_modes_excluded(self) -> None:
        # Translational/rotational modes (< 10 cm⁻¹) should be skipped
        g_with = quasi_rrho_free_energy([3000.0, 5.0, 3.0, 0.1])
        g_without = quasi_rrho_free_energy([3000.0])
        assert abs(g_with - g_without) < 1e-12

    def test_imaginary_frequencies_treated_as_real(self) -> None:
        # Imaginary frequencies (negative values) are treated as real
        # using their absolute value
        g_imag = quasi_rrho_free_energy([3000.0, -100.0])
        g_real = quasi_rrho_free_energy([3000.0, 100.0])
        assert abs(g_imag - g_real) < 1e-12

    def test_water_frequencies(self) -> None:
        # Water has 3 vibrational modes: ~1595, ~3657, ~3756 cm⁻¹
        # Plus 3 translational + 3 rotational (near zero)
        freqs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1595.0, 3657.0, 3756.0]
        g = quasi_rrho_free_energy(freqs, temperature=298.15)
        # ZPE of water: ~0.5 * (1595 + 3657 + 3756) cm⁻¹ ≈ 4504 cm⁻¹
        # In Hartree: ~0.0205
        # Total G_vib should be in this ballpark (ZPE dominates)
        assert 0.015 < g < 0.025

    def test_temperature_dependence(self) -> None:
        freqs = [500.0, 1000.0, 1500.0]
        g_low = quasi_rrho_free_energy(freqs, temperature=200.0)
        g_high = quasi_rrho_free_energy(freqs, temperature=400.0)
        # Higher temperature -> lower free energy (larger -TS term)
        assert g_high < g_low

    def test_empty_returns_zero(self) -> None:
        g = quasi_rrho_free_energy([])
        assert g == 0.0

    def test_all_near_zero_returns_zero(self) -> None:
        g = quasi_rrho_free_energy([0.0, 0.0, 0.0, 5.0, 3.0, 1.0])
        assert g == 0.0

    def test_low_frequency_damped(self) -> None:
        # A very low frequency (50 cm⁻¹) should use mostly free-rotor
        # entropy, giving a more negative -TS contribution than pure HO
        g_low_freq = quasi_rrho_free_energy([50.0])
        g_high_freq = quasi_rrho_free_energy([500.0])
        # Low freq mode has smaller ZPE but more favorable entropy
        # -> lower free energy
        assert g_low_freq < g_high_freq
