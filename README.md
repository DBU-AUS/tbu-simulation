# TBU Simulation Repository

Monte Carlo validation suite for **"\title{Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions}"** 

<a href="https://doi.org/10.5281/zenodo.17807599"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17807599.svg" alt="DOI"></a>

## Quick Start

```bash
pip install numpy pandas scipy
python tbu_simulations_v3.py
```

## Key Results

| Prediction | Theory | Simulation |
|------------|--------|------------|
| Target ε_eff | 9.3 × 10⁻⁷ | (9.31 ± 0.07) × 10⁻⁷ |
| Mass scaling slope | 2.30 × 10⁻⁶ kg⁻¹ | (2.30 ± 0.09) × 10⁻⁶ kg⁻¹ |
| Integration for 5σ | ~74 hours | ~74 hours (3 days) |
| Total photons | 1.2 × 10¹² | 1.2 × 10¹² |

## Framework Parameters

From the derived κ (Appendix L):

| Parameter | Value | Source |
|-----------|-------|--------|
| κ | 1.55 × 10⁻¹¹ | (π²/2)(λ/Lc)² |
| χ_geom | 0.49 | Geometric overlap |
| λ_opt | 532 nm | Green laser |
| L_c | 0.30 m | Coherence length |

## Repository Structure

```
├── tbu_simulations_v3.py          # Main simulation script
├── simulation_config.json          # All parameters
├── data/
│   ├── TBU_MC_10000_runs_null.csv      # 10k null hypothesis runs
│   ├── TBU_MC_10000_runs_signal.csv    # 10k signal injection runs  
│   ├── TBU_timeseries_run.csv          # 1-hour temporal stability
│   ├── TBU_extended_simulation_results.csv  # Mass/distance scaling
│   ├── TBU_alt_mechanisms_vs_TBU.csv   # Mechanism discrimination
│   └── TBU_entangled_two_path.csv      # Entangled extension
└── README.md
```

## Data Files

### TBU_MC_10000_runs_signal.csv
Signal injection Monte Carlo (10,000 runs). Validates detection of ε_eff = 9.31 × 10⁻⁷ in realistic noise.

- `epsilon_phys`: 9.31 × 10⁻⁷ (injected signal)
- `beta_hat`: Recovered correlation coefficient
- `deltaBIC_dbu_vs_null`: Model comparison (>10 indicates strong TBU evidence)

### TBU_MC_10000_runs_null.csv  
Null hypothesis Monte Carlo (10,000 runs). Confirms no false positives when signal absent.

- `beta_hat`: Should be ~0 with no systematic bias
- False positive rate: <0.1%

### TBU_timeseries_run.csv
One-hour continuous observation (3,600 seconds). Demonstrates per-hour SNR = 0.58.

- `eps_hat`: Measured correlation per second
- Mean: 9.31 × 10⁻⁷
- Std: 1.6 × 10⁻⁶
- SNR: 0.58

### TBU_extended_simulation_results.csv
Parameter space exploration across mass (0–2 kg) and distance (0.1–0.5 m).

Key validation: Mass slope at d = 0.25 m
- Slope: (2.30 ± 0.09) × 10⁻⁶ kg⁻¹
- R² = 0.9973
- Intercept consistent with zero
- **Note on uncertainties:** `eps_measured` represents the noiseless theoretical prediction. Quoted uncertainties in the paper (e.g., ±0.09 × 10⁻⁶ kg⁻¹) derive from regressions on `eps_noisy` (which includes 8% measurement noise) and repeated Monte Carlo runs.

### TBU_alt_mechanisms_vs_TBU.csv
Mechanism discrimination test (Section 7.6.1). Distinguishes TBU from systematics:

| Mechanism | Mass Dependence | Signature |
|-----------|-----------------|-----------|
| TBU | Linear in M | ε ∝ M_ext |
| EM cross-talk | Independent | Constant |
| Thermal drift | Independent | Correlated noise |
| Detector mismatch | Independent | Static bias |

### TBU_entangled_two_path.csv
Two-path entangled photon predictions. Joint correlation: ε_joint = √(ε_A × ε_B).

## Detection Feasibility

For 5σ detection of ε = 9.3 × 10⁻⁷:

```
Standard error required: SE = 9.3×10⁻⁷ / 5 = 1.86×10⁻⁷
Hourly scatter: σ ≈ 1.6×10⁻⁶  
Hours needed: N = (σ/SE)² = (1.6×10⁻⁶ / 1.86×10⁻⁷)² ≈ 74 hours
Total photons: 74 hr × 3600 s × 4.5×10⁶/s ≈ 1.2×10¹²
```

## Reproducing Results

All simulations are deterministic with `np.random.seed(42)`:

```python
python tbu_simulations_v3.py
```

Verify key outputs:
```python
import pandas as pd
df = pd.read_csv('TBU_timeseries_run.csv')
print(f"Mean: {df.eps_hat.mean():.3e}")  # → 9.31e-7
print(f"Std: {df.eps_hat.std():.3e}")    # → 1.6e-6
print(f"SNR: {df.eps_hat.mean()/df.eps_hat.std():.2f}")  # → 0.58
```

## Citation

If using this code or data:

```bibtex
@article{TBU2025,
  title={\title{Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions}},
  author={Artz, Gavin},
  year={2025}
}
```

## License

MIT License. See LICENSE file.
