#!/usr/bin/env python3
"""
================================================================================
TBU 5-SEED REPLICATION: Volatility-Aware Substrate
================================================================================

Replicates the key finding across 5 independent seeds to confirm robustness.

Key finding being tested:
  Sign-flip elimination when volatility is perceptible.

Results from manuscript Section R.6:
  Seed    r_high    r_low   r_pooled  Sign-flip
  42      +0.878   +0.961    +0.924       No
  43      +0.877   +0.956    +0.923       No
  44      +0.878   +0.963    +0.925       No
  45      +0.878   +0.961    +0.924       No
  46      +0.879   +0.961    +0.925       No
  Mean    +0.878   +0.960    +0.924      0/5

Conclusion: Coupling unification is robust across seeds.

Usage:
  python run_vol_aware_replication.py

================================================================================
"""

import numpy as np
from tbu_honest_boundary import HonestBoundarySubstrate
from volatility_aware_substrate import VolatilityAwareSubstrate


def run_seed(substrate_class, seed, n_steps=5000, warmup=500):
    """Run one seed, return sign-flip analysis."""
    sub = substrate_class(seed=seed)
    
    log = []
    prev_boundary = None
    
    for step in range(n_steps):
        sub.step_physics()
        
        if step >= warmup:
            grid = sub.grid
            boundary = grid[sub.forcing_coords]
            core = grid[~sub.forcing_mask]
            
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
    
    # Analyze
    boundary_std = np.array([d['boundary_std'] for d in log])
    core_std = np.array([d['core_std'] for d in log])
    volatility = np.array([d['volatility'] for d in log])
    
    thresh = np.median(volatility)
    hi = volatility >= thresh
    lo = ~hi
    
    r_hi = np.corrcoef(boundary_std[hi], core_std[hi])[0, 1]
    r_lo = np.corrcoef(boundary_std[lo], core_std[lo])[0, 1]
    r_pool = np.corrcoef(boundary_std, core_std)[0, 1]
    sign_flip = (r_hi * r_lo < 0)
    
    return {
        'r_high': r_hi,
        'r_low': r_lo,
        'r_pooled': r_pool,
        'sign_flip': sign_flip,
    }


def main():
    seeds = [42, 43, 44, 45, 46]
    
    print("=" * 70)
    print("5-SEED REPLICATION: Volatility-Aware Substrate")
    print("=" * 70)
    print()
    
    results = []
    for seed in seeds:
        print(f"Running seed {seed}...", end=" ", flush=True)
        r = run_seed(VolatilityAwareSubstrate, seed)
        results.append(r)
        flip_str = "YES" if r['sign_flip'] else "no"
        print(f"r_hi={r['r_high']:+.3f}, r_lo={r['r_low']:+.3f}, "
              f"r_pool={r['r_pooled']:+.3f}, sign_flip={flip_str}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Seed':<8} {'r_high':>10} {'r_low':>10} {'r_pooled':>10} {'Sign-flip':>12}")
    print("-" * 52)
    for seed, r in zip(seeds, results):
        flip_str = "Yes" if r['sign_flip'] else "No"
        print(f"{seed:<8} {r['r_high']:>+10.3f} {r['r_low']:>+10.3f} "
              f"{r['r_pooled']:>+10.3f} {flip_str:>12}")
    
    # Means
    mean_hi = np.mean([r['r_high'] for r in results])
    mean_lo = np.mean([r['r_low'] for r in results])
    mean_pool = np.mean([r['r_pooled'] for r in results])
    n_flip = sum(1 for r in results if r['sign_flip'])
    
    print("-" * 52)
    print(f"{'Mean':<8} {mean_hi:>+10.3f} {mean_lo:>+10.3f} "
          f"{mean_pool:>+10.3f} {n_flip}/5")
    print()
    
    if n_flip == 0:
        print("SUCCESS: Sign-flip eliminated in ALL seeds")
        print("Coupling unification is robust")
    else:
        print(f"Sign-flip present in {n_flip}/5 seeds")


if __name__ == "__main__":
    main()
