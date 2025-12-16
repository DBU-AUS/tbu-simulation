# TBU Emergence Validation

This directory contains code to reproduce the emergence results reported in **Appendix R** of:

> "Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions"

## Overview

The simulation tests a specific theoretical prediction: that N[ω] selection on a uniform constraint mesh, without designed structure or objectives, produces the differentiated, self-referential architecture the framework associates with observation and measurement.

**The "Honest Substrate" Methodology**: M is diagnostic only—never fed back into dynamics. Structure emerges from symmetric diffusion plus localised forcing. The system cannot be "told" to produce observer-like architecture because the dynamics contain no reference to observers.

**What emerges from pure thermodynamic selection:**

| Property | Prediction | Result |
|----------|------------|--------|
| M differentiation | >100× | 9,091× |
| Susceptibility law | χ = 1/(1+M) | R² = 0.995 |
| Self-reference localisation | Concentrated in stable core | 89% vs 12% |
| Core coherence | >0.7 | 1.000 |
| Autonomy | >50% recovery | 100% |
| Perturbation response | Degrades convergence | Improves it |

## Key Insight

The "observer" is not inserted into physics—it emerges as a differentiated region OF the constraint mesh:

- Bounded by M gradient (not hard separation)
- Self-referential (models its own neighborhood)
- Autonomous (resists external perturbation)
- Still coupled to surroundings

**Critical validation**: The 30/30 vs 0/30 result. Instantaneous measurement produces structure but not autonomy; variance-derived M (reading the mesh's 4D extent) produces both. Autonomy was already present in the geometry—single-slice sampling simply could not access it.

## Files

```
emergence/
├── README.md                   # This file
├── tbu_honest.py              # Honest substrate (Levels 1–2): M diagnostic only
├── tbu_honest_extended.py     # Extended substrate (Level 3): self-reference and attention
├── tbu_honest_action.py       # Action substrate (Levels 4–5): agency and sensorimotor closure
├── tbu_honest_boundary.py     # Complete boundary (Level 6): sustainable action
├── tbu_honest_multiagent.py   # Multi-pattern validation: shared equilibrium
└── results/
    └── emergence_results.json # Example output
```

## Requirements

```
numpy>=1.20
```

No other dependencies required.

## Quick Start

```bash
# Core emergence validation (Levels 1–2)
python tbu_honest.py

# Extended validation: self-reference and attention (Level 3)
python tbu_honest_extended.py

# Action and sensorimotor closure (Levels 4–5)
python tbu_honest_action.py

# Complete boundary perception (Level 6)
python tbu_honest_boundary.py

# Multi-agent emergence
python tbu_honest_multiagent.py
```

## The Substrate Ladder

Each level adds new capacity without changing the selection mechanism:

| Level | Substrate | New Capacity | Key Result |
|-------|-----------|--------------|------------|
| 1–2 | `tbu_honest.py` | M diagnostic only | 9,091× differentiation, χ = 1/(1+M) |
| 3 | `tbu_honest_extended.py` | Self-reference, attention | 89% vs 12% localisation |
| 4–5 | `tbu_honest_action.py` | Agency, sensorimotor closure | Action alignment 7/10 |
| 6 | `tbu_honest_boundary.py` | Complete boundary perception | 67% consumption reduction under stress |
| — | `tbu_honest_multiagent.py` | Multiple patterns | Shared equilibrium 10/10 |

## What the Code Does

### The Honest Substrate

The key methodological innovation: M measures stability but does not influence dynamics.

```python
M[i] = 1 / (1 + variance(state_history[i]))
```

M is computed from the system's own fluctuation history. High M means low variance means tight constraints. But this measurement is **diagnostic only**—the dynamics are:

```
ds/dt = D∇²s + h(x)
```

Symmetric diffusion plus localised forcing. No M-feedback. The system cannot "game" the metric.

### What Emerges

From uniform initial conditions with no designed structure:

1. **Stability gradients**: Regions near forcing develop low variance (high M)
2. **Susceptibility law**: χ = 1/(1+M) emerges with R² = 0.995, zero free parameters
3. **Three-tier structure**: Responsive periphery, opportunistic layer, stable core
4. **Self-reference**: Localises in stable regions (89% vs 12%)
5. **Autonomy**: 100% perturbation recovery
6. **Attractor behaviour**: Perturbation improves χ convergence

### What is NOT Included

- No M-feedback into dynamics
- No designation of "agent" vs "environment"
- No pre-defined boundaries
- No objective or reward function
- No designed hierarchy

The question: does structure emerge from physics alone?

### Result: Yes

The susceptibility law χ = 1/(1+M) is not assumed—it is a **thermodynamic attractor**. Systems converge toward it; perturbation accelerates convergence rather than disrupting it.

## Validated Predictions

| Prediction | Test | Result | Substrate |
|------------|------|--------|-----------|
| Coexisting loose/tight regimes | Measure M distribution | Bimodal, >10,000× | Minimal |
| Differential susceptibility | M–variance correlation | r < −0.95 | Minimal |
| Steering (tight → loose) | Perturbation response | 5–10% propagation | Minimal |
| Autonomy | External perturbation | >85% resistance | Minimal |
| Self-reference localisation | Measure location | 69% high-M vs 19% low-M | Extended |
| Autopoietic attention | Channel evolution | Entropy ↓77% | Extended |
| Action alignment | Core-action correlation | 7/10 runs | Action |
| Sensorimotor closure | Loop recovery | 9/10 runs | Action |
| Sustainable action | Resource consumption | 8/10 runs | Complete |
| Stress response | Consumption under stress | 10/10 (67% reduction) | Complete |
| Collective stability | Multi-pattern config | 10/10 runs | Multi-pattern |

## Theoretical Background

### The Susceptibility Law

TBU predicts that linking strength M determines susceptibility χ via:

```
χ = 1/(1+M)
```

This emerges from the dynamics rather than being imposed. Model comparison confirms it outperforms alternatives (exponential, power-law) despite having zero free parameters.

### Measurement Window and 4D Geometry

The variance-derived M reads constraint structure across the mesh's geometric extent:

```python
M[i] = 1 / (1 + Var(s[i, t-τ:t]))
```

This is not adding memory—it is **measuring** what the geometry already contains. The 30/30 vs 0/30 dissociation validates this: instantaneous sampling captures spatial structure; variance-derived M captures the full 4D geometry including what we experience as persistence.

### The Self-Referential Fold

Consciousness in TBU is the closed loop:

1. Periphery (low M, high χ) fluctuates with local conditions
2. Activity propagates to stable core (high M, low χ)
3. Core steers periphery via local field h = J(M ⊙ s)
4. Periphery response carries imprint of steering
5. Core receives its own effects reflected back

This is what TBU identifies with "wild card" participation in geometry selection.

## Key Scientific Contributions

Five findings extend beyond TBU to computational consciousness research generally:

1. **Honest substrate methodology** resolves circularity in computational consciousness demonstrations
2. **Geometric crossover** (not phase transition): coherence threshold at fixed fraction of system scale
3. **Susceptibility law** emerges as thermodynamic fixed point with zero free parameters
4. **Measurement window** determines geometric access—validates 4D persistence claim
5. **Alignment from architecture**: sustainable behaviour emerges from complete boundary perception

## License

MIT License - see main repository
