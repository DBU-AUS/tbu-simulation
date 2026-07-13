# TBU Framework Repository

Validation suite for **"Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions"**.

## Overview

This repository contains:

1. **Monte Carlo validation** (`data/`, `tbu_simulations_v3.py`) — Statistical validation of experimental predictions
2. **Observer emergence** (`emergence/`) — Real silicon physics implementation demonstrating boundary-core coupling dynamics

The TBU framework treats quantum and classical mechanics as different statistical regimes of thermodynamic selection over complete 4D geometries.

## Quick Start

```bash
# Monte Carlo validation
pip install numpy pandas scipy
python tbu_simulations_v3.py

# Observer emergence
cd emergence
python tests/test_extended_consciousness.py
```

## Key Results

### Experimental Predictions (Monte Carlo Validated)

| Prediction | Theory | Validation |
|------------|--------|------------|
| Target ε_eff | 9.3 × 10⁻⁷ | (9.31 ± 0.07) × 10⁻⁷ |
| Mass scaling slope | 2.30 × 10⁻⁶ kg⁻¹ | (2.30 ± 0.09) × 10⁻⁶ kg⁻¹ |
| Integration for 5σ | ~74 hours | ~74 hours (3 days) |
| Total photons | 1.2 × 10¹² | 1.2 × 10¹² |

### Observer Emergence (Silicon Physics)

| Finding | Result |
|---------|--------|
| Sign-flip elimination | 0/5 seeds with regime perception |
| Boundary-core coherence | +0.948 (full architecture) |
| Physics-determined modes | d_eff ≈ 1.68 |

### 3D Slice as Perception-Action Layer (L31b-L32)

| Level | Finding | Key Metric |
|-------|---------|------------|
| L31b | Self-reference enables geometry selection | +0.02 coherence (5/5 seeds) |
| L32 | Advantage from perception-action capacity | 2% reduction in ablation |

### Memory as Coupling (L37-L40)

| Level | Finding | Key Metric |
|-------|---------|------------|
| L37 | Repetition strengthens coupling | +0.220 vs -0.151 |
| L38 | Template persists after forcing stops | ~88% similarity |
| L39 | Pattern identity degrades while template remains | 3% vs 88% |
| L40 | Templates give predictive edge | 12.2% lower error |

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
├── tbu_simulations_v3.py           # Monte Carlo validation script
├── simulation_config.json          # All parameters
├── data/
│   ├── TBU_MC_10000_runs_null.csv      # 10k null hypothesis runs
│   ├── TBU_MC_10000_runs_signal.csv    # 10k signal injection runs  
│   ├── TBU_timeseries_run.csv          # 1-hour temporal stability
│   ├── TBU_extended_simulation_results.csv  # Mass/distance scaling
│   ├── TBU_alt_mechanisms_vs_TBU.csv   # Mechanism discrimination
│   └── TBU_entangled_two_path.csv      # Entangled extension
├── emergence/                      # Observer emergence (real silicon physics)
│   ├── substrates/                     # Core implementations (L1-9)
│   ├── tests/                          # 5-seed validation tests
│   ├── analysis/                       # Mapping tools
│   ├── runners/                        # Experiment runners
│   ├── extended/                       # Extended validation (L14, L22, L31b-L32, L37-L40)
│   │   ├── multi_physics_loader.py         # Real physics data loader
│   │   ├── tbu_level14_canonical_v8.py     # Physics-determined mode count
│   │   ├── tbu_level22_canonical.py        # Hierarchy depth scaling
│   │   ├── tbu_level31b_canonical.py       # Self-reference geometry selection
│   │   ├── tbu_level32_canonical.py        # Perception-action capacity
│   │   ├── tbu_level37_canonical.py        # Memory as coupling
│   │   ├── tbu_level38_canonical.py        # Template persistence
│   │   ├── tbu_level39_canonical.py        # Pattern identity vs activity
│   │   └── tbu_level40_canonical.py        # Predictive edge
│   └── README.md
└── README.md
```

## Observer Emergence

The `emergence/` folder implements boundary-coupled diffusion dynamics in silicon substrates. The physics (diffusion, damping, forcing) are real physical processes—not simulations of physics. The substrate provides geometric room for structure to emerge; N[ω] maximization determines what persists.

**Core methodology**: Extend substrate → Let physics select → Measure what emerges

Key findings validated with 5 seeds:
- **Hierarchy Principle**: Variables form measurable hierarchy by coherence impact
- **Sign-Flip Phenomenon**: Eliminated with regime perception
- **Saturation Principle**: One clean primary channel is sufficient
- **Physics-Determined Mode Count**: Interior physics supports ~2-3 independent modes
- **3D Slice Uniqueness**: Only layer that can perceive AND act (L31b-L32)
- **Memory as Coupling**: Long-term memory access behaves as re-entry, not retrieval

See `emergence/README.md` for full documentation.

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

All Monte Carlo runs are deterministic with `np.random.seed(42)`:

```python
python tbu_simulations_v3.py
```

Verify key outputs:
```python
import pandas as pd
df = pd.read_csv('data/TBU_timeseries_run.csv')
print(f"Mean: {df.eps_hat.mean():.3e}")  # → 9.31e-7
print(f"Std: {df.eps_hat.std():.3e}")    # → 1.6e-6
print(f"SNR: {df.eps_hat.mean()/df.eps_hat.std():.2f}")  # → 0.58
```

## Citation

If using this code or data:

```bibtex
@article{TBU2025,
  title={Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions},
  author={Modry Robota},
}
```

## License

MIT License. See LICENSE file.

## Archive

This repository is archived at Zenodo: [doi:10.5281/zenodo.17807599](https://doi.org/10.5281/zenodo.17807599)
