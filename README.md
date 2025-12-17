# TBU Simulation

Simulation code and datasets for:
> **"Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions"**
> 
> Gavin Artz (2025)

## Repository Structure

```
tbu-simulation/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CHANGELOG.md
├── simulation_config.json
├── tbu_simulations_v3.py        # Monte Carlo validation (Appendix D)
├── data/                        # Simulation datasets
│   ├── TBU_MC_10000_runs_null.csv
│   ├── TBU_MC_10000_runs_signal.csv
│   └── ...
└── emergence/                   # Emergence validation (Appendix R)
    ├── README.md
    ├── tbu_honest.py            # Honest substrate (Levels 1–2)
    ├── tbu_honest_extended.py   # Extended substrate (Level 3)
    ├── tbu_honest_action.py     # Action substrate (Levels 4–5)
    ├── tbu_honest_boundary.py   # Complete boundary (Level 8)
    ├── tbu_honest_multiagent.py # Multi-pattern validation
    ├── tbu_honest_reproduce_paper.py  # Reproduction script
    ├── volatility_aware_substrate.py  # Constraint perception (Section R.6)
    ├── random_channel_substrate.py    # Null test control
    ├── run_null_test.py               # Three-way comparison
    ├── run_vol_aware_replication.py   # 5-seed replication
    ├── tbu_reconditioning_scanner.py  # Fingerprint scanner
    ├── tbu_honest_boundary_reconditioning_logger.py  # CSV logger
    └── results/
```

## Two Validation Components

### 1. Monte Carlo Experimental Validation (Appendix D)

Tests whether the predicted ε_eff ~ 10⁻⁶ correlations are detectable with proposed methodology.

- **Code**: `tbu_simulations_v3.py`
- **Data**: `data/` folder (10,000 realizations)
- **Result**: Detection methodology validated at 5σ significance

### 2. Emergence Validation (Appendix R)

Tests whether observer-like structure emerges from N[ω] selection alone.

- **Code**: `emergence/` folder
- **Methodology**: "Honest substrate"—M is diagnostic only, never fed back into dynamics
- **Result**: Hierarchy, self-reference, coherence, and autonomy all emerge from uniform initial conditions

| Level | File | Key Result |
|-------|------|------------|
| 1–2 | `tbu_honest.py` | 9,091× differentiation, χ = 1/(1+M) with R² = 0.995 |
| 3 | `tbu_honest_extended.py` | Self-reference localisation (89% vs 12%) |
| 4–5 | `tbu_honest_action.py` | Agency and sensorimotor closure |
| 8 | `tbu_honest_boundary.py` | Sustainable action (67% consumption reduction under stress) |
| 8+ | `volatility_aware_substrate.py` | Constraint perception eliminates sign-flip (Section R.6) |
| — | `tbu_honest_multiagent.py` | Shared equilibrium across multiple patterns |

### 3. Constraint Perception Validation (Section R.6)

Tests whether the sign-flip interferometric fingerprint is diagnostic of regime blindness.

| Condition | r_pooled | Sign-flip | Interpretation |
|-----------|----------|-----------|----------------|
| Baseline (6 ch, regime-blind) | +0.01 | YES | Regime blindness → washout |
| Volatility-Aware (7 ch, has info) | +0.92 | no | Perception → unified coupling |
| Random Channel (7 ch, NO info) | −0.00 | YES | Noise doesn't help |

**Key finding:** It's the *information* that matters. Opening a perceptual channel for a constraint variable eliminates the interferometric signature. Adding noise does not.

See [emergence/README.md](emergence/README.md) for details.

## Quick Start

### Monte Carlo Validation

```bash
python tbu_simulations_v3.py
```

### Emergence Validation

```bash
cd emergence

# Core emergence (Levels 1–2)
python tbu_honest.py

# Extended validation (Level 3)
python tbu_honest_extended.py

# Full substrate ladder
python tbu_honest_action.py
python tbu_honest_boundary.py
python tbu_honest_multiagent.py

# Reproduce all Appendix R claims
python tbu_honest_reproduce_paper.py --all
```

### Constraint Perception Validation (Section R.6)

```bash
cd emergence

# Three-way comparison (baseline vs volatility vs random)
python run_null_test.py

# 5-seed replication
python run_vol_aware_replication.py
```

## Requirements

```
numpy>=1.20
scipy>=1.7
matplotlib>=3.4
```

Optional for constraint perception analysis:
```
pandas
statsmodels
```

## Citation

```bibtex
@article{artz2025tbu,
  title={Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions},
  author={Artz, Gavin},
  year={2025},
}
```

## License

MIT License - see [LICENSE](LICENSE)

## Contact

- Email: gavinartz@gmail.com
- ORCID: 0009-0006-5089-8447

## Archive

This repository is archived at Zenodo: [doi:10.5281/zenodo.17807599](https://doi.org/10.5281/zenodo.17807599)
