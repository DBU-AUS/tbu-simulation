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
    ├── tbu_scalable_v2.py
    ├── run_emergence.py
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
- **Result**: Hierarchy, self-reference, coherence, and autonomy all emerge from uniform initial conditions

See [`emergence/README.md`](emergence/README.md) for details.

## Quick Start

### Monte Carlo Validation
```bash
python tbu_simulations_v3.py
```

### Emergence Validation
```bash
cd emergence
python run_emergence.py --single
```

## Requirements

```
numpy>=1.20
scipy>=1.7
matplotlib>=3.4
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
