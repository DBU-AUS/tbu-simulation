# TBU Emergence Validation

Computational validation of observer structure emergence from N[ω] selection, as described in Appendix R of "Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions"

## Overview

This repository demonstrates that observer-like architecture emerges from minimal substrate without explicit selection laws.

**Key finding:** M is not a dynamical variable requiring an update rule. M is a functional of state history:

$$M_i(t) \propto \frac{1}{\mathrm{Var}(s_i(t-\tau \ldots t))}$$

Stable regions (low variance) have high M → steering authority → alignment → more stability. The loop closes through state persistence.

## Repository Structure

```
emergence/
├── substrates/                 # Core implementations (L1-9)
│   ├── tbu_honest.py              # Honest boundary substrate
│   └── tbu_honest_boundary.py     # Boundary-coupled dynamics
├── tests/                      # 5-seed validation tests
│   └── test_extended_consciousness.py
├── analysis/                   # Mapping and analysis tools
│   ├── map_constraint_mesh.py
│   └── mesh_connectivity.py
├── runners/                    # Experiment runners
├── extended/                   # Extended validation (L14, L22, L37-L40)
│   ├── multi_physics_loader.py        # Real physics data loader
│   ├── tbu_level14_canonical_v8.py    # Physics-determined mode count
│   ├── observe_level14_saturation.py  # Shell saturation
│   ├── tbu_level22_canonical.py       # Hierarchy depth scaling
│   ├── tbu_reconditioning_capacity.py # RC metric
│   ├── tbu_level37_canonical.py       # Memory as coupling
│   ├── tbu_level38_canonical.py       # Template persistence
│   ├── tbu_level39_canonical.py       # Pattern identity vs activity
│   └── tbu_level40_canonical.py       # Predictive edge
└── README.md
```

## Key Results

### Core Observer Properties (L1-9)

| Property | Description | Typical Result |
|----------|-------------|----------------|
| Differentiation | M range > 1000× | ✓ |
| χ emergence | M-variance correlation < -0.5 | ✓ |
| Steering | Periphery follows core perturbation | ✓ |
| Autonomy | Core resists periphery perturbation | ✓ |
| Development | Structure maintained over time | ✓ |
| Resilience | χ survives noise stress | ✓ |

### Extended Validation (L14, L22, L37-L40)

| Level | Finding | Key Metric |
|-------|---------|------------|
| L14 | Physics-determined mode count | d_eff ≈ 1.68 (relaxation) vs 2.61 (forced) |
| L22 | Hierarchy depth scales with reconditioning | Optimal depth matches mode count |
| L37 | Repetition strengthens coupling | +0.220 vs -0.151 |
| L38 | Template persists after forcing stops | ~88% similarity |
| L39 | Pattern identity degrades while template remains | 3% vs 88% |
| L40 | Templates give predictive edge | 12.2% lower error |

### Memory as Coupling (L37-L40)

Long-term memory access behaves as coupling/re-entry to existing 4D structure, not retrieval from internal store.

**The two-hop limit**: Indirect re-entry via template can sustain coherent experience but cannot preserve full geometric identity. Only direct intersection restores identity.

| System | Location | τ | Function |
|--------|----------|---|----------|
| Internal (3D) | Buffer/cores | 36-47 | Continuity, specious present |
| External (4D) | Mesh geometry | 77 | Long-term patterns |

**Lock sentences**:
- **L37**: Long-term memory access behaves as coupling/re-entry, not retrieval from internal store.
- **L38**: Remembering is not re-experiencing the past; it is re-entering a learned alignment that only stabilises when it still matches real geometry.
- **L39**: Memory works because templates persist longer than patterns—but memory fails because templates are not patterns.
- **L40**: Templates do not make the system quieter—they make it more accurate.

## Quick Start

```bash
# Core emergence tests
python tests/test_extended_consciousness.py

# Extended validation
cd extended

# Level 14: Physics-determined mode count
python tbu_level14_canonical_v8.py

# Level 22: Hierarchy depth scaling
python tbu_level22_canonical.py --test sanity --steps 10000 --seeds 5

# Level 37-40: Memory as coupling
python tbu_level37_canonical.py
python tbu_level38_canonical.py
python tbu_level39_canonical.py
python tbu_level40_canonical.py
```

## Requirements

```bash
pip install numpy scipy psutil

# Optional (for real physics data in L37-L40):
pip install requests  # NDBC ocean buoy data
pip install obspy     # IRIS seismic data
```

## What Is and Is Not Designed

**Minimal substrate:**
- DESIGNED: State space, graph, boundary, M-weighted steering
- EMERGENT: M differentiation, χ relationship, core/periphery structure, autonomy

**Extended substrate:**
- DESIGNED: Self-model capacity, attention capacity, drift mechanisms
- EMERGENT: Where self-models localize, which channels get attended

**Memory experiments (L37-L40):**
- DESIGNED: Substrate physics (diffusion, damping), observer architecture (EMA, cores)
- EMERGENT: Template persistence, two-hop limit, predictive edge

The substrate creates a **channel** through which N[ω] selection expresses as structure. The resulting patterns are discovered, not coded.

## Relationship to Paper

| Paper Section | Validation |
|---------------|------------|
| Appendix R.1-R.5 | Core emergence (L1-9) |
| Appendix R.6 | Physics-determined modes (L14) |
| Appendix R.7 | Hierarchy absorption (L22) |
| Appendix R.8 | Memory as coupling (L37-L40) |
| Section 2.8.3 | Ceremony grounding via L37-L40 |

The memory findings (L37-L40) provide empirical grounding for the ceremony section, establishing that:
- Place-based knowledge irreducibility is architectural, not cultural
- The two-hop limit explains why "being on Country" cannot be replaced by remembering
- Templates are model-alignment devices, not storage mechanisms

## Citation

```bibtex
@article{artz2025tbu,
  title={Entropy Maximisation under Conservation Constraints on 4D Geometries: Testable Predictions},
  author={Artz, Gavin},
  
}
```

## License

MIT License

## Contact

Gavin Artz - gavinartz@gmail.com

Repository: https://github.com/DBU-AUS/tbu-simulation
