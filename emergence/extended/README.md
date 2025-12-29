# TBU Extended Validation (Levels 12-40)

Extended validation code supporting interpretive claims in the paper.

## Overview

These levels extend beyond the core observer emergence (L1-9) to validate:
- Physics-determined mode count (L14)
- Hierarchy as reconditioning absorption (L22)
- Memory as coupling (L37)

## Files

| File | Level | Finding | Paper Reference |
|------|-------|---------|-----------------|
| `tbu_level14_canonical_v8.py` | 14 | d_eff ≈ 1.68 (relaxation) vs 2.61 (forced) | Physics-determined mode count |
| `observe_level14_saturation.py` | 14 | Shell redundancy saturation | Supports N*=2 finding |
| `tbu_level22_canonical.py` | 22 | Depth scales with reconditioning modes | Hierarchy absorption |
| `tbu_reconditioning_capacity.py` | 22 | RC metric formalization | Reconditioning capacity |
| `tbu_level37_canonical.py` | 37 | Coupling +0.220 / -0.151 | Memory as coupling |

## Key Findings

### Level 14: Physics-Determined Mode Count

The interior physics supports only ~2-3 truly independent modes:

| Regime | d_eff | Interpretation |
|--------|-------|----------------|
| Forced | 2.61 | Apparent complexity (noise-inflated) |
| Relaxation | 1.68 | True independent modes |

Additional perception channels beyond ~4-5 shells provide diminishing returns.

### Level 22: Hierarchy as Reconditioning Absorption

Hierarchy depth scales with unresolved reconditioning modes:

- Flat environment: depth degrades performance (no reconditioning to absorb)
- With reconditioning: optimal depth matches mode count
- Sign-flip = insufficient hierarchy to absorb regime ambiguity

> "Hierarchy prevents sign-flip by giving contradiction somewhere to go."

### Level 37: Memory as Coupling

Long-term memory access behaves as coupling/re-entry, not retrieval:

| Condition | Coupling Change |
|-----------|-----------------|
| Repetition | +0.220 (strengthens) |
| Absence | -0.151 (degrades) |

> "Memory is not storage—memory is coupling."

## Usage

```bash
# From emergence/extended directory
cd emergence/extended

# Level 14: Mode count analysis
python tbu_level14_canonical_v8.py

# Level 22: Depth scaling
python tbu_level22_canonical.py --test sanity --steps 10000 --seeds 5

# Level 22: Reconditioning capacity
python tbu_reconditioning_capacity.py

# Level 37: Coupling dynamics
python tbu_level37_canonical.py
```

## Dependencies

These scripts depend on substrates in `../substrates/`:
- `tbu_honest_boundary.py`
- `tbu_honest.py`

## Note

These levels represent engineering extensions that inform interpretation of the core physics findings. They are not required to reproduce the main experimental results (Sections 7, R.6) but support the interpretive discussion in the paper.
