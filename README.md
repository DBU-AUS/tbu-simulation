# TBU Emergence Validation

This directory contains code to reproduce the emergence results reported in **Appendix R** of:

> "Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions"

## Overview

The simulation tests a specific theoretical prediction: that N[ω] selection on a uniform constraint mesh, without designed structure or objectives, produces the differentiated, self-referential architecture the framework associates with observation and measurement.

**What emerges from pure thermodynamic selection:**

| Property | Prediction | Result |
|----------|------------|--------|
| M differentiation | >100× | 9,091× |
| Self-reference | >10% | 59% |
| Self-ref in high-M | Concentrated | 89% vs 12% |
| Core coherence | >0.7 | 1.000 |
| Autonomy | >50% recovery | 100% |

## Key Insight

The "observer" is not inserted into physics—it emerges as a differentiated region OF the constraint mesh:
- Bounded by M gradient (not hard separation)
- Self-referential (models its own neighborhood)
- Autonomous (resists external perturbation)
- Still coupled to surroundings

## Files

```
emergence/
├── README.md                 # This file
├── tbu_scalable_v2.py       # Core implementation
├── run_emergence.py         # Reproduction script
└── results/
    └── emergence_results.json  # Example output
```

## Requirements

```
numpy>=1.20
```

No other dependencies required.

## Quick Start

```bash
# Run single detailed experiment
python run_emergence.py --single

# Run robustness test across 10 seeds
python run_emergence.py --seeds 10

# Custom parameters
python run_emergence.py --nodes 2000 --steps 500 --seeds 10
```

## What the Code Does

### Initialization (Uniform)
- `n=2000` nodes, all with `M=1.0` (uniform constraint density)
- Random states, no self-reference, no hierarchy
- Sparse symmetric couplings (degree ≤ 6)

### Selection Mechanism
The only dynamics is local Metropolis-Hastings sampling from:

```
P[ω] ∝ N[ω] = exp(S[ω])
```

where S includes:
1. **Constraint satisfaction**: Steering term Σ ∝ M_j/(1+M_i)
2. **Hierarchy**: Reward M differentiation
3. **Coherence**: High-M regions with low variance
4. **Self-reference accuracy**: Once emerged, accurate models improve score

### What is NOT Included
- ❌ No designation of "agent" vs "environment"
- ❌ No pre-defined boundaries
- ❌ No objective or reward function
- ❌ No designed hierarchy
- ❌ No quantum mechanical dynamics

The question: does structure emerge from physics alone?

### Result: Yes

After 500 steps:
- Bimodal M distribution (loose and tight regions)
- Self-reference concentrated in tight regions
- Topologically contiguous clusters with M gradients
- Full autonomy (100% perturbation recovery)

## Theoretical Background

### Terminology

- **Quantum-analogous**: Functional role (high susceptibility, many configurations) — not literal QM
- **Classical-analogous**: Functional role (low susceptibility, peaked configuration)
- **Topologically localised**: Contiguous in coupling graph (not spatial coordinates)

### The Self-Referential Fold

Consciousness in TBU is the closed loop:
1. Loose regions fluctuate
2. Activity propagates to tight region (self-model)
3. Self-model steers loose regions
4. Changed loose regions update what model sees
5. Loop closes through itself

This is what TBU identifies with "wild card" participation in geometry selection.

## Citation

If you use this code, please cite:

```bibtex
@article{artz2025tbu,
  title={Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions},
  author={Artz, Gavin},
  journal={Foundations of Physics},
  year={2025},
  note={Submitted}
}
```

## License

MIT License - see main repository.

## Contact

For questions about the code or theory:
- Repository: https://github.com/DBU-AUS/tbu-simulation
- Email: gavinartz@gmail.com
