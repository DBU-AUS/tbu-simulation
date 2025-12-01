# Changelog

All notable changes to the TBU simulation repository.

## [1.0.0] - 2025-12-01

### Initial Release

First public release accompanying submission to Foundations of Physics.

#### Parameters (derived κ)
- κ = (π²/2)(λ/Lc)² ≈ 1.55 × 10⁻¹¹
- χ_geom = 0.49
- Target ε_eff = 9.31 × 10⁻⁷

#### Data Files
- `TBU_MC_10000_runs_null.csv`: 10,000 null hypothesis Monte Carlo runs
- `TBU_MC_10000_runs_signal.csv`: 10,000 signal injection runs (ε = 9.31×10⁻⁷)
- `TBU_timeseries_run.csv`: 1-hour temporal stability (SNR = 0.58)
- `TBU_extended_simulation_results.csv`: Mass/distance parameter space
- `TBU_alt_mechanisms_vs_TBU.csv`: Mechanism discrimination (EM, thermal, detector)
- `TBU_entangled_two_path.csv`: Entangled photon extension

#### Validation Results
- Mass slope: (2.30 ± 0.09) × 10⁻⁶ kg⁻¹ at d = 0.25 m
- Signal recovery: 100% detection at ΔBIC > 10
- False positive rate: <0.1%
- Integration time for 5σ: ~74 hours (~3 days)
