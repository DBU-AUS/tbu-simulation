# TBU Extended Validation (Levels 12-40)

Extended validation code supporting interpretive claims in the paper.

## Overview

These levels extend beyond the core observer emergence (L1-9) to validate:
- Physics-determined mode count (L14)
- Hierarchy as reconditioning absorption (L22)
- Memory as coupling (L37-L40)

## Files

| File | Level | Finding | Paper Reference |
|------|-------|---------|-----------------|
| `multi_physics_loader.py` | — | Real physics data from NDBC, NOAA, ACE, IRIS | Data source for L37-L40 |
| `tbu_level14_canonical_v8.py` | 14 | d_eff ≈ 1.68 (relaxation) vs 2.61 (forced) | Physics-determined mode count |
| `observe_level14_saturation.py` | 14 | Shell redundancy saturation | Supports N*=2 finding |
| `tbu_level22_canonical.py` | 22 | Depth scales with reconditioning modes | Hierarchy absorption |
| `tbu_reconditioning_capacity.py` | 22 | RC metric formalization | Reconditioning capacity |
| `tbu_level37_canonical.py` | 37 | Coupling +0.220 / -0.151 | Memory as coupling |
| `tbu_level38_canonical.py` | 38 | Template persists at ~88% after forcing stops | Template persistence |
| `tbu_level39_canonical.py` | 39 | Pattern → 3%, template → 88% | Pattern identity vs activity |
| `tbu_level40_canonical.py` | 40 | 12.2% predictive advantage | Predictive edge |

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

### Levels 37-40: Memory as Coupling

Long-term memory access behaves as coupling/re-entry, not retrieval from internal store.

| Level | Finding | Key Metric |
|-------|---------|------------|
| L37 | Repetition strengthens coupling | +0.220 vs -0.151 |
| L38 | Template persists after forcing stops | ~88% similarity |
| L39 | Pattern identity degrades while template remains | 3% vs 88% |
| L40 | Templates give predictive edge | 12.2% lower error |

**The two-hop limit**: Indirect re-entry via template can sustain coherent experience but cannot preserve full geometric identity. Only direct intersection restores identity.

**Lock sentences**:
- **L37**: Long-term memory access behaves as coupling/re-entry, not retrieval from internal store.
- **L38**: Remembering is not re-experiencing the past; it is re-entering a learned alignment that only stabilises when it still matches real geometry.
- **L39**: Memory works because templates persist longer than patterns—but memory fails because templates are not patterns.
- **L40**: Templates do not make the system quieter—they make it more accurate.

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

# Level 38: Template persistence
python tbu_level38_canonical.py

# Level 39: Pattern identity
python tbu_level39_canonical.py

# Level 40: Predictive edge
python tbu_level40_canonical.py
```

## Dependencies

These scripts depend on:

**Substrates** (in `../substrates/`):
- `tbu_honest_boundary.py`
- `tbu_honest.py`

**Data loader** (in this directory):
- `multi_physics_loader.py` — fetches real ocean/magnetometer/solar data from public APIs; falls back to synthetic data if network unavailable

**Optional Python packages** (for real data):
- `requests` — NDBC ocean buoy, NOAA magnetometer, ACE solar wind
- `obspy` — IRIS seismic data
- `h5py` — LIGO auxiliary channels (requires local files)

## Note

These levels represent engineering extensions that inform interpretation of the core physics findings. They are not required to reproduce the main experimental results (Sections 7, R.6) but support the interpretive discussion in the paper.

The memory findings (L37-L40) provide empirical grounding for the ceremony section (Section 2.8.3), establishing that:
- Place-based knowledge irreducibility is architectural, not cultural
- The two-hop limit explains why "being on Country" cannot be replaced by remembering
- Templates are model-alignment devices, not storage mechanisms
