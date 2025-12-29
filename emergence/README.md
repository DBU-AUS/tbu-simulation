# TBU Observer Emergence

Simulation code for observer emergence in the Thermodynamic Block Universe framework.

## Directory Structure

```
emergence/
├── substrates/          # Core substrate implementations (Levels 1-9)
├── tests/               # Validation tests (5-seed)
├── analysis/            # Mapping and analysis tools
├── runners/             # Experiment runners
├── results/             # Output data
└── README.md
```

## Substrates (Levels 1-9)

| File | Level | Description |
|------|-------|-------------|
| `tbu_honest.py` | 1-4 | Base substrate with diffusion, damping, self-models |
| `tbu_honest_action.py` | 5-6 | Adds action-perception coupling |
| `tbu_honest_boundary.py` | 7-8 | Adds environment interaction |
| `tbu_honest_extended.py` | 8+ | Extended boundary substrate |
| `tbu_honest_multiagent.py` | 8+ | Multi-agent extension |
| `volatility_aware_substrate.py` | 8+ | Adds volatility perception (t-tangent) |
| `random_channel_substrate.py` | — | Null test control (random channel) |
| `tbu_extended_consciousness.py` | 9 | Full architecture (primary + secondary + redundancy) |

## Tests (Appendix T Validation)

All tests validated with 5 seeds (42-46).

| File | Finding | Key Result |
|------|---------|------------|
| `test_saturation.py` | Primary channels saturate | One sufficient |
| `test_secondary_stacking.py` | Secondaries compound with primary | +0.022 benefit |
| `test_noise_tolerance.py` | 25% danger zone | Sign-flip transition |
| `test_redundancy.py` | Regime-aligned rescue | 100% from shared |
| `test_extended_consciousness.py` | Full architecture | +0.948 coherence |
| `test_robustness.py` | Multi-config validation | All findings confirmed |
| `test_primary_classification.py` | Variable classification | Hierarchy emerges |
| `test_secondary_classification.py` | Secondary behavior | Context-dependent |
| `test_multi_perception.py` | Multi-channel perception | Mesh connectivity |

## Analysis Tools

| File | Purpose |
|------|---------|
| `map_constraint_mesh.py` | Map constraint connectivity |
| `mesh_connectivity.py` | Analyze mesh topology |
| `tbu_reconditioning_scanner.py` | Scan for reconditioning signatures |
| `tbu_honest_boundary_reconditioning_logger.py` | Log reconditioning events |

## Runners

| File | Purpose |
|------|---------|
| `run_null_test.py` | Three-way comparison (baseline/volatility/random) |
| `run_vol_aware_replication.py` | 5-seed volatility-aware replication |
| `tbu_honest_reproduce_paper.py` | Reproduce paper figures |

## Key Findings

### Hierarchy Principle
Variables form measurable hierarchy by coherence impact:
- **Primary (regime):** volatility, Δ = +0.92
- **Secondary (content):** health, resources, Δ ≈ -0.25 alone, +0.02 with primary

### Sign-Flip Phenomenon
Without regime perception, boundary-core correlation inverts under stress.
With volatility perception, sign-flip eliminated (0/5 seeds).

### Saturation Principle
Primary channels saturate—one clean primary is sufficient.
Additional primaries add noise, not information.

### Redundancy Lemma
Under 25% noise, regime-aligned redundancy provides full rescue.
Independent geometric content provides only 35% alone.

### Physics-Determined Mode Count
Interior physics supports ~2-3 independent modes (d_eff ≈ 1.68).
Additional perception channels cannot access nonexistent structure.

## Usage

```bash
# Run from emergence directory
cd emergence

# Test full architecture
python tests/test_extended_consciousness.py

# Run robustness validation
python tests/test_robustness.py

# Run null test comparison
python runners/run_null_test.py
```

## Requirements

- Python 3.8+
- NumPy
- SciPy (for some analysis tools)

## Citation

If using this code, please cite:
```
Artz, G. (2025). The Everywhen: Unifying Quantum and Classical Mechanics 
through 4D Thermodynamics. Foundations of Physics (submitted).
```

## License

MIT License - see repository root.
