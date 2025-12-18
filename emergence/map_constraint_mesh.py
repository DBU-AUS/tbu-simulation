#!/usr/bin/env python3
"""
================================================================================
TBU CONSTRAINT MESH MAPPING
================================================================================

Maps the extent of hidden constraint structure in the substrate:
- How many variable pairs show reconditioning fingerprints?
- Which conditioning variables reveal which relationships?
- Is there hierarchy/structure in the constraints?
- What's the total "hidden information" in the mesh?

Key findings:
- 129 sign-flip relationships (20.7% of pairs)
- 229 relationships show >50% washout
- Mesh is EXTENSIVE - constraint structure pervades the system

Usage:
  python map_constraint_mesh.py

================================================================================
"""

import numpy as np
from typing import Dict, List, Any
from itertools import combinations

from tbu_honest_boundary import HonestBoundarySubstrate


def run_substrate_and_log(n_steps: int = 6000, warmup: int = 500, seed: int = 42) -> List[Dict]:
    """Run substrate and log comprehensive state."""
    print(f"Running substrate for {n_steps} steps...")
    
    substrate = HonestBoundarySubstrate(seed=seed)
    
    log = []
    prev_boundary = substrate.grid[substrate.forcing_coords].copy()
    
    # Warmup
    for _ in range(warmup):
        substrate.step_physics()
    
    prev_boundary = substrate.grid[substrate.forcing_coords].copy()
    
    for step in range(n_steps - warmup):
        substrate.step_physics()
        
        grid = substrate.grid
        boundary = grid[substrate.forcing_coords]
        core = grid[~substrate.forcing_mask]
        
        # Boundary volatility
        dy = boundary - prev_boundary
        boundary_volatility = float(np.mean(np.abs(dy)))
        prev_boundary = boundary.copy()
        
        report = substrate.report()
        
        # Attention distribution
        mean_attn = substrate.attention.mean(axis=0)
        
        log.append({
            # Grid statistics
            'grid_mean': float(grid.mean()),
            'grid_std': float(grid.std()),
            'core_mean': float(core.mean()),
            'core_std': float(core.std()),
            'boundary_mean': float(boundary.mean()),
            'boundary_std': float(boundary.std()),
            
            # Dynamics
            'boundary_volatility': boundary_volatility,
            'boundary_dy_mean': float(np.mean(dy)),
            'boundary_dy_std': float(np.std(dy)),
            
            # M and coherence
            'M_ratio': float(report['M_ratio']),
            'core_coherence': float(report['core_coherence']),
            
            # Environment
            'env_health': float(report['env_health']),
            'env_resources': float(report['env_resources']),
            'env_stability': float(report['env_stability']),
            'obs_sigma': float(report['obs_sigma']),
            
            # Attention
            'attention_predictable': float(mean_attn[0] + mean_attn[3]),
            'attention_entropy': float(mean_attn[1]),
            'attention_health': float(mean_attn[4] + mean_attn[5]),
            
            # Actions
            'action_mean': float(report['action_mean']),
            'action_std': float(report['action_std']),
            'action_stability': float(report['action_stability']),
        })
        
        if (step + 1) % 1000 == 0:
            print(f"  Step {step + warmup + 1}/{n_steps}")
    
    return log


def analyze_pair_conditioned(log: List[Dict], 
                              var_x: str, 
                              var_y: str, 
                              var_z: str,
                              min_samples: int = 100) -> Dict:
    """Analyze one variable pair conditioned on one regime variable."""
    
    x = np.array([d[var_x] for d in log])
    y = np.array([d[var_y] for d in log])
    z = np.array([d[var_z] for d in log])
    
    # Check for valid data
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    
    if len(x) < min_samples * 2:
        return None
    
    # Median split
    z_thresh = np.median(z)
    high = z >= z_thresh
    low = ~high
    
    if high.sum() < min_samples or low.sum() < min_samples:
        return None
    
    # Correlations
    r_pooled = np.corrcoef(x, y)[0, 1]
    r_high = np.corrcoef(x[high], y[high])[0, 1]
    r_low = np.corrcoef(x[low], y[low])[0, 1]
    
    if not (np.isfinite(r_pooled) and np.isfinite(r_high) and np.isfinite(r_low)):
        return None
    
    # Metrics
    best = max(abs(r_high), abs(r_low))
    washout_ratio = 1.0 - abs(r_pooled) / best if best > 0.01 else 0
    sign_flip = (r_high * r_low < 0) and abs(r_high) > 0.05 and abs(r_low) > 0.05
    interaction = abs(r_high - r_low)
    
    return {
        'var_x': var_x,
        'var_y': var_y,
        'var_z': var_z,
        'r_pooled': r_pooled,
        'r_high': r_high,
        'r_low': r_low,
        'washout_ratio': washout_ratio,
        'sign_flip': sign_flip,
        'interaction': interaction,
        'n_high': int(high.sum()),
        'n_low': int(low.sum()),
    }


def map_constraint_mesh(log: List[Dict]) -> List[Dict]:
    """Comprehensive scan of all variable pairs and conditioning relationships."""
    
    # Observable variables (X, Y candidates)
    observables = [
        'grid_mean', 'grid_std',
        'core_mean', 'core_std',
        'boundary_mean', 'boundary_std',
        'boundary_volatility', 'boundary_dy_mean', 'boundary_dy_std',
        'attention_predictable', 'attention_entropy', 'attention_health',
        'action_mean', 'action_std',
    ]
    
    # Conditioning variables (Z candidates)
    conditions = [
        'M_ratio',
        'core_coherence', 
        'boundary_volatility',
        'env_health',
        'env_resources',
        'env_stability',
        'obs_sigma',
        'action_stability',
    ]
    
    results = []
    
    # All pairs of observables
    pairs = list(combinations(observables, 2))
    total = len(pairs) * len(conditions)
    
    print(f"\nScanning {len(pairs)} variable pairs × {len(conditions)} conditions = {total} tests")
    
    count = 0
    for var_x, var_y in pairs:
        for var_z in conditions:
            # Skip if Z is same as X or Y
            if var_z == var_x or var_z == var_y:
                continue
                
            result = analyze_pair_conditioned(log, var_x, var_y, var_z)
            if result:
                results.append(result)
            
            count += 1
            if count % 200 == 0:
                print(f"  {count}/{total}...")
    
    return results


def summarize_mesh(results: List[Dict]) -> Dict:
    """Summarize the constraint mesh structure."""
    
    print("\n" + "=" * 80)
    print("CONSTRAINT MESH SUMMARY")
    print("=" * 80)
    
    # Basic stats
    n_total = len(results)
    n_sign_flip = sum(1 for r in results if r['sign_flip'])
    n_high_washout = sum(1 for r in results if r['washout_ratio'] > 0.5)
    n_strong_interaction = sum(1 for r in results if r['interaction'] > 0.3)
    
    print(f"\nTotal relationships tested: {n_total}")
    print(f"Sign flips detected: {n_sign_flip} ({100*n_sign_flip/n_total:.1f}%)")
    print(f"High washout (>50%): {n_high_washout} ({100*n_high_washout/n_total:.1f}%)")
    print(f"Strong interaction (>0.3): {n_strong_interaction} ({100*n_strong_interaction/n_total:.1f}%)")
    
    # Sign flips - these are the most interesting
    print("\n" + "-" * 80)
    print("SIGN FLIP FINGERPRINTS (strongest reconditioning signatures)")
    print("-" * 80)
    
    sign_flips = [r for r in results if r['sign_flip']]
    sign_flips.sort(key=lambda r: -r['washout_ratio'])
    
    if sign_flips:
        print(f"\n{'X':<20} {'Y':<20} {'Z':<18} {'r_pool':>8} {'r_hi':>8} {'r_lo':>8} {'wash':>6}")
        print("-" * 90)
        for r in sign_flips[:20]:
            print(f"{r['var_x']:<20} {r['var_y']:<20} {r['var_z']:<18} "
                  f"{r['r_pooled']:>+8.3f} {r['r_high']:>+8.3f} {r['r_low']:>+8.3f} "
                  f"{r['washout_ratio']:>6.2f}")
    else:
        print("  No sign flips detected")
    
    # Conditioning variable effectiveness
    print("\n" + "-" * 80)
    print("CONDITIONING VARIABLE EFFECTIVENESS")
    print("-" * 80)
    print("\nWhich regime variables reveal the most hidden structure?")
    
    by_condition = {}
    for r in results:
        z = r['var_z']
        if z not in by_condition:
            by_condition[z] = {'count': 0, 'sign_flips': 0, 'total_washout': 0}
        by_condition[z]['count'] += 1
        if r['sign_flip']:
            by_condition[z]['sign_flips'] += 1
        by_condition[z]['total_washout'] += r['washout_ratio']
    
    print(f"\n{'Condition':<20} {'Tests':>6} {'Sign-flips':>12} {'Avg Washout':>12}")
    print("-" * 55)
    
    sorted_conds = sorted(by_condition.items(), key=lambda x: -x[1]['sign_flips'])
    
    for z, stats in sorted_conds:
        avg_wash = stats['total_washout'] / stats['count']
        print(f"{z:<20} {stats['count']:>6} {stats['sign_flips']:>12} {avg_wash:>12.3f}")
    
    return {
        'n_total': n_total,
        'n_sign_flip': n_sign_flip,
        'n_high_washout': n_high_washout,
        'by_condition': by_condition,
        'sign_flips': sign_flips,
    }


def main():
    print("=" * 80)
    print("TBU CONSTRAINT MESH MAPPING")
    print("=" * 80)
    print()
    print("Mapping the extent of hidden constraint structure")
    print()
    
    # Run substrate
    log = run_substrate_and_log(n_steps=6000, warmup=500, seed=42)
    print(f"\nLogged {len(log)} steps")
    
    # Comprehensive scan
    results = map_constraint_mesh(log)
    
    # Summarize
    summary = summarize_mesh(results)
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print(f"1. {summary['n_sign_flip']} relationships show sign-flip fingerprint")
    print(f"2. {summary['n_high_washout']} relationships show >50% washout")
    print(f"3. Constraint mesh is {'EXTENSIVE' if summary['n_sign_flip'] > 10 else 'SPARSE'}")
    print()
    print("This determines how much 'hidden information' exists that perception could access.")


if __name__ == "__main__":
    main()
