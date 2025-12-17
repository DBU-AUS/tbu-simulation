#!/usr/bin/env python3
"""
================================================================================
TBU NULL TEST: Random Channel vs Volatility Channel
================================================================================

This test compares three conditions:
1. Baseline (Level 8) - regime blind
2. Volatility-Aware - perceives volatility (regime information)  
3. Random Channel - perceives noise (no regime information)

Expected results if the finding is real:
- Baseline: Sign-flip YES
- Volatility-Aware: Sign-flip NO (information enables coherence)
- Random Channel: Sign-flip YES (noise doesn't help)

If random channel also eliminates sign-flip, the finding is an artifact
of adding any extra channel, not specific to volatility information.

Results from manuscript Section R.6:
  Condition                              r_high   r_low   r_pool  Sign-flip
  Baseline (6 channels)                  +0.47   -0.00    +0.01       YES
  Volatility-Aware (7 ch, has info)      +0.88   +0.96    +0.92        no
  Random Channel (7 ch, NO info)         +0.17   -0.00    -0.00       YES

Conclusion: It's the INFORMATION that matters.

Usage:
  python run_null_test.py

================================================================================
"""

import numpy as np
from tbu_honest_boundary import HonestBoundarySubstrate
from volatility_aware_substrate import VolatilityAwareSubstrate
from random_channel_substrate import RandomChannelSubstrate


def run_and_analyze(substrate_class, name, n_steps=5000, warmup=500, seed=42):
    """Run substrate and return sign-flip analysis."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    substrate = substrate_class(seed=seed)
    
    log = []
    prev_boundary = None
    
    for step in range(n_steps):
        substrate.step_physics()
        
        if step >= warmup:
            grid = substrate.grid
            boundary = grid[substrate.forcing_coords]
            core = grid[~substrate.forcing_mask]
            
            if prev_boundary is not None:
                volatility = float(np.mean(np.abs(boundary - prev_boundary)))
            else:
                volatility = 0.0
            prev_boundary = boundary.copy()
            
            log.append({
                'boundary_std': float(boundary.std()),
                'core_std': float(core.std()),
                'volatility': volatility,
            })
        
        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}/{n_steps}")
    
    # Analyze
    boundary_std = np.array([d['boundary_std'] for d in log])
    core_std = np.array([d['core_std'] for d in log])
    volatility = np.array([d['volatility'] for d in log])
    
    vol_thresh = np.median(volatility)
    high_vol = volatility >= vol_thresh
    low_vol = ~high_vol
    
    r_high = np.corrcoef(boundary_std[high_vol], core_std[high_vol])[0, 1]
    r_low = np.corrcoef(boundary_std[low_vol], core_std[low_vol])[0, 1]
    r_pooled = np.corrcoef(boundary_std, core_std)[0, 1]
    sign_flip = (r_high * r_low < 0)
    
    return {
        'name': name,
        'r_high': r_high,
        'r_low': r_low,
        'r_pooled': r_pooled,
        'sign_flip': sign_flip,
    }


def main():
    print("=" * 70)
    print("NULL TEST: Random Channel vs Volatility Channel")
    print("=" * 70)
    print()
    print("Question: Is it the INFORMATION that matters, or just adding a channel?")
    print()
    print("Expected if finding is real:")
    print("  - Baseline:         Sign-flip YES (regime blind)")
    print("  - Volatility-Aware: Sign-flip NO  (has regime info)")
    print("  - Random Channel:   Sign-flip YES (no regime info)")
    print()
    
    # Run all three conditions
    baseline = run_and_analyze(
        HonestBoundarySubstrate, 
        "Baseline (Level 8, 6 channels)"
    )
    
    vol_aware = run_and_analyze(
        VolatilityAwareSubstrate, 
        "Volatility-Aware (7 channels, has regime info)"
    )
    
    random_ch = run_and_analyze(
        RandomChannelSubstrate, 
        "Random Channel (7 channels, NO regime info)"
    )
    
    # Results
    print()
    print("=" * 70)
    print("NULL TEST RESULTS")
    print("=" * 70)
    print()
    print(f"{'Condition':<45} {'r_high':>8} {'r_low':>8} {'r_pool':>8} {'Sign-flip':>10}")
    print("-" * 80)
    
    for r in [baseline, vol_aware, random_ch]:
        flip_str = "YES" if r['sign_flip'] else "no"
        print(f"{r['name']:<45} {r['r_high']:>+8.3f} {r['r_low']:>+8.3f} "
              f"{r['r_pooled']:>+8.3f} {flip_str:>10}")
    
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    
    # Check the pattern
    baseline_flip = baseline['sign_flip']
    vol_flip = vol_aware['sign_flip']
    random_flip = random_ch['sign_flip']
    
    if baseline_flip and not vol_flip and random_flip:
        print("SUCCESS: Null test PASSED")
        print()
        print("  - Baseline shows sign-flip (regime blind)")
        print("  - Volatility channel eliminates sign-flip (has info)")
        print("  - Random channel preserves sign-flip (no info)")
        print()
        print("CONCLUSION: It's the INFORMATION that matters.")
        print("Adding volatility perception enables coherence.")
        print("Adding noise does not.")
    
    elif baseline_flip and not vol_flip and not random_flip:
        print("WARNING: Random channel also eliminates sign-flip")
        print()
        print("This suggests the effect might be an artifact of")
        print("adding any extra channel, not specific to volatility.")
        print("Further investigation needed.")
    
    elif not baseline_flip:
        print("UNEXPECTED: Baseline shows no sign-flip")
        print("Cannot interpret null test without baseline sign-flip.")
    
    else:
        print(f"UNEXPECTED PATTERN:")
        print(f"  Baseline sign-flip: {baseline_flip}")
        print(f"  Volatility sign-flip: {vol_flip}")
        print(f"  Random sign-flip: {random_flip}")


if __name__ == "__main__":
    main()
