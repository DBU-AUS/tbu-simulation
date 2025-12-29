#!/usr/bin/env python3
"""
================================================================================
TBU CONSTRAINT MESH MAPPING
================================================================================

Before extending perception, map the extent of the constraint mesh:
- How many variable pairs show reconditioning fingerprints?
- Which conditioning variables reveal which relationships?
- Is there hierarchy/structure in the constraints?
- What's the total "hidden information" in the mesh?

This is reconnaissance before expansion.

================================================================================
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Any
from itertools import combinations

sys.path.insert(0, '/mnt/project')
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


def map_constraint_mesh(log: List[Dict]) -> Dict:
    """
    Comprehensive scan of all variable pairs and conditioning relationships.
    """
    
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


def summarize_mesh(results: List[Dict]):
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
    
    # High washout without sign flip
    print("\n" + "-" * 80)
    print("HIGH WASHOUT WITHOUT SIGN FLIP (hidden correlations)")
    print("-" * 80)
    
    high_washout = [r for r in results if r['washout_ratio'] > 0.5 and not r['sign_flip']]
    high_washout.sort(key=lambda r: -r['washout_ratio'])
    
    if high_washout:
        print(f"\n{'X':<20} {'Y':<20} {'Z':<18} {'r_pool':>8} {'r_hi':>8} {'r_lo':>8} {'wash':>6}")
        print("-" * 90)
        for r in high_washout[:15]:
            print(f"{r['var_x']:<20} {r['var_y']:<20} {r['var_z']:<18} "
                  f"{r['r_pooled']:>+8.3f} {r['r_high']:>+8.3f} {r['r_low']:>+8.3f} "
                  f"{r['washout_ratio']:>6.2f}")
    
    # Conditioning variable effectiveness
    print("\n" + "-" * 80)
    print("CONDITIONING VARIABLE EFFECTIVENESS")
    print("-" * 80)
    print("\nWhich regime variables reveal the most hidden structure?")
    
    by_condition = {}
    for r in results:
        z = r['var_z']
        if z not in by_condition:
            by_condition[z] = {'count': 0, 'sign_flips': 0, 'total_washout': 0, 'total_interaction': 0}
        by_condition[z]['count'] += 1
        if r['sign_flip']:
            by_condition[z]['sign_flips'] += 1
        by_condition[z]['total_washout'] += r['washout_ratio']
        by_condition[z]['total_interaction'] += r['interaction']
    
    print(f"\n{'Condition':<20} {'Tests':>6} {'Sign-flips':>12} {'Avg Washout':>12} {'Avg Interact':>12}")
    print("-" * 65)
    
    sorted_conds = sorted(by_condition.items(), 
                          key=lambda x: -x[1]['sign_flips'])
    
    for z, stats in sorted_conds:
        avg_wash = stats['total_washout'] / stats['count']
        avg_int = stats['total_interaction'] / stats['count']
        print(f"{z:<20} {stats['count']:>6} {stats['sign_flips']:>12} "
              f"{avg_wash:>12.3f} {avg_int:>12.3f}")
    
    # Variable pair analysis
    print("\n" + "-" * 80)
    print("VARIABLE PAIR ANALYSIS")
    print("-" * 80)
    print("\nWhich variable pairs have the most hidden structure?")
    
    by_pair = {}
    for r in results:
        pair = (r['var_x'], r['var_y'])
        if pair not in by_pair:
            by_pair[pair] = {'sign_flips': 0, 'max_washout': 0, 'conditions': []}
        if r['sign_flip']:
            by_pair[pair]['sign_flips'] += 1
        if r['washout_ratio'] > by_pair[pair]['max_washout']:
            by_pair[pair]['max_washout'] = r['washout_ratio']
            by_pair[pair]['best_condition'] = r['var_z']
        by_pair[pair]['conditions'].append(r['var_z'])
    
    # Pairs with most hidden structure
    interesting_pairs = [(p, s) for p, s in by_pair.items() if s['sign_flips'] > 0 or s['max_washout'] > 0.5]
    interesting_pairs.sort(key=lambda x: (-x[1]['sign_flips'], -x[1]['max_washout']))
    
    print(f"\n{'X':<20} {'Y':<20} {'Sign-flips':>10} {'Max Wash':>10} {'Best Z':>20}")
    print("-" * 85)
    for (x, y), stats in interesting_pairs[:15]:
        best_z = stats.get('best_condition', '-')
        print(f"{x:<20} {y:<20} {stats['sign_flips']:>10} "
              f"{stats['max_washout']:>10.2f} {best_z:>20}")
    
    return {
        'n_total': n_total,
        'n_sign_flip': n_sign_flip,
        'n_high_washout': n_high_washout,
        'by_condition': by_condition,
        'by_pair': by_pair,
        'sign_flips': sign_flips,
    }


def analyze_mesh_topology(results: List[Dict], summary: Dict):
    """
    Analyze the topology of constraint relationships.
    """
    
    print("\n" + "=" * 80)
    print("MESH TOPOLOGY ANALYSIS")
    print("=" * 80)
    
    # Build adjacency structure
    # Node = variable, Edge = exists hidden relationship
    
    variables = set()
    edges = {}  # (x, y) -> list of conditioning variables that reveal it
    
    for r in results:
        if r['washout_ratio'] > 0.3 or r['sign_flip']:
            x, y, z = r['var_x'], r['var_y'], r['var_z']
            variables.add(x)
            variables.add(y)
            
            pair = tuple(sorted([x, y]))
            if pair not in edges:
                edges[pair] = []
            edges[pair].append({
                'condition': z,
                'washout': r['washout_ratio'],
                'sign_flip': r['sign_flip'],
                'r_high': r['r_high'],
                'r_low': r['r_low'],
            })
    
    print(f"\nVariables with hidden relationships: {len(variables)}")
    print(f"Variable pairs with hidden structure: {len(edges)}")
    
    # Find clusters
    print("\n" + "-" * 80)
    print("RELATIONSHIP CLUSTERS")
    print("-" * 80)
    
    # Group by what condition reveals them
    by_revealer = {}
    for pair, conditions in edges.items():
        for c in conditions:
            z = c['condition']
            if z not in by_revealer:
                by_revealer[z] = []
            by_revealer[z].append((pair, c))
    
    for z, pairs in sorted(by_revealer.items(), key=lambda x: -len(x[1])):
        sign_flip_pairs = [p for p, c in pairs if c['sign_flip']]
        print(f"\n{z} reveals {len(pairs)} relationships ({len(sign_flip_pairs)} sign-flips):")
        for (x, y), c in pairs[:5]:
            flip = "FLIP" if c['sign_flip'] else ""
            print(f"  {x} <-> {y}: r_hi={c['r_high']:+.2f}, r_lo={c['r_low']:+.2f} {flip}")
        if len(pairs) > 5:
            print(f"  ... and {len(pairs) - 5} more")
    
    # Hierarchy check: do some conditions subsume others?
    print("\n" + "-" * 80)
    print("CONDITIONING HIERARCHY")
    print("-" * 80)
    print("\nDo some regime variables reveal supersets of others?")
    
    condition_sets = {z: set(tuple(sorted(p)) for p, _ in pairs) 
                      for z, pairs in by_revealer.items()}
    
    for z1 in condition_sets:
        for z2 in condition_sets:
            if z1 != z2:
                s1, s2 = condition_sets[z1], condition_sets[z2]
                if s1 > s2 and len(s1) > len(s2):  # Strict superset
                    print(f"  {z1} ({len(s1)}) ⊃ {z2} ({len(s2)})")
                elif s1 == s2 and len(s1) > 0:
                    print(f"  {z1} ≡ {z2} ({len(s1)} relationships)")
    
    return edges


def main():
    print("=" * 80)
    print("TBU CONSTRAINT MESH MAPPING")
    print("=" * 80)
    print()
    print("Mapping the extent of hidden constraint structure before expansion")
    print()
    
    # Run substrate
    log = run_substrate_and_log(n_steps=6000, warmup=500, seed=42)
    print(f"\nLogged {len(log)} steps")
    
    # Comprehensive scan
    results = map_constraint_mesh(log)
    
    # Summarize
    summary = summarize_mesh(results)
    
    # Topology
    edges = analyze_mesh_topology(results, summary)
    
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
