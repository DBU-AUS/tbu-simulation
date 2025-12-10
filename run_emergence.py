#!/usr/bin/env python3
"""
TBU Emergence - Reproduction Script

This script reproduces the emergence results reported in Appendix R of:
"Entropy Maximisation under Conservation Constraints on 4D Geometries"

Run this script to verify that observer-like structure emerges from
N[ω] selection on a uniform constraint mesh.

Usage:
    python run_emergence.py [--seeds N] [--nodes N] [--steps N]

Arguments:
    --seeds: Number of random seeds to test (default: 10)
    --nodes: Number of nodes in mesh (default: 2000)
    --steps: Number of selection steps (default: 500)

Author: Gavin Artz
Date: December 2025
"""

import argparse
import json
import numpy as np
from datetime import datetime
from tbu_scalable_v2 import TBUEmergence, run_emergence_experiment


def run_robustness_test(n_seeds: int = 10, 
                        n_nodes: int = 2000, 
                        n_steps: int = 500) -> dict:
    """
    Run emergence experiment across multiple seeds to verify robustness.
    """
    print("=" * 70)
    print("TBU EMERGENCE ROBUSTNESS TEST")
    print("=" * 70)
    print(f"Testing {n_seeds} random seeds")
    print(f"Nodes: {n_nodes}, Steps: {n_steps}")
    print()
    
    results = []
    
    for i, seed in enumerate(range(n_seeds)):
        print(f"Seed {seed} ({i+1}/{n_seeds})...", end=" ")
        
        tbu = TBUEmergence(n_nodes=n_nodes, seed=seed)
        
        for _ in range(n_steps):
            tbu.select_step()
        
        analysis = tbu.get_full_analysis()
        autonomy = tbu.test_autonomy()
        
        result = {
            'seed': seed,
            'M_range': analysis['M_range'],
            'self_ref_pct': analysis['self_ref_pct'],
            'self_ref_high_M': analysis['self_ref_high_M'],
            'self_ref_low_M': analysis['self_ref_low_M'],
            'core_coherence': analysis['core_coherence'],
            'autonomy_recovery': autonomy['recovery_pct'],
        }
        results.append(result)
        
        # Check emergence criteria
        passed = (
            result['M_range'] > 100 and
            result['self_ref_pct'] > 10 and
            result['core_coherence'] > 0.7 and
            result['autonomy_recovery'] > 50
        )
        
        print(f"M={result['M_range']:.0f}x, SR={result['self_ref_pct']:.0f}%, "
              f"C={result['core_coherence']:.2f}, A={result['autonomy_recovery']:.0f}% "
              f"{'✓' if passed else '✗'}")
    
    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    M_ranges = [r['M_range'] for r in results]
    self_refs = [r['self_ref_pct'] for r in results]
    coherences = [r['core_coherence'] for r in results]
    autonomies = [r['autonomy_recovery'] for r in results]
    
    passed_count = sum(1 for r in results if 
                       r['M_range'] > 100 and 
                       r['self_ref_pct'] > 10 and
                       r['core_coherence'] > 0.7 and
                       r['autonomy_recovery'] > 50)
    
    summary = {
        'n_seeds': n_seeds,
        'n_nodes': n_nodes,
        'n_steps': n_steps,
        'passed_count': passed_count,
        'passed_pct': passed_count / n_seeds * 100,
        'M_range': {
            'mean': np.mean(M_ranges),
            'std': np.std(M_ranges),
            'min': np.min(M_ranges),
            'max': np.max(M_ranges),
        },
        'self_ref_pct': {
            'mean': np.mean(self_refs),
            'std': np.std(self_refs),
            'min': np.min(self_refs),
            'max': np.max(self_refs),
        },
        'core_coherence': {
            'mean': np.mean(coherences),
            'std': np.std(coherences),
            'min': np.min(coherences),
            'max': np.max(coherences),
        },
        'autonomy_recovery': {
            'mean': np.mean(autonomies),
            'std': np.std(autonomies),
            'min': np.min(autonomies),
            'max': np.max(autonomies),
        },
        'individual_results': results,
    }
    
    print(f"""
Emergence across {n_seeds} seeds:
  
  M differentiation:
    Mean: {summary['M_range']['mean']:.0f}x ± {summary['M_range']['std']:.0f}
    Range: [{summary['M_range']['min']:.0f}, {summary['M_range']['max']:.0f}]
    
  Self-reference:
    Mean: {summary['self_ref_pct']['mean']:.1f}% ± {summary['self_ref_pct']['std']:.1f}
    Range: [{summary['self_ref_pct']['min']:.1f}%, {summary['self_ref_pct']['max']:.1f}%]
    
  Core coherence:
    Mean: {summary['core_coherence']['mean']:.3f} ± {summary['core_coherence']['std']:.3f}
    Range: [{summary['core_coherence']['min']:.3f}, {summary['core_coherence']['max']:.3f}]
    
  Autonomy (recovery):
    Mean: {summary['autonomy_recovery']['mean']:.0f}% ± {summary['autonomy_recovery']['std']:.0f}
    Range: [{summary['autonomy_recovery']['min']:.0f}%, {summary['autonomy_recovery']['max']:.0f}%]
    
  PASSED: {passed_count}/{n_seeds} ({summary['passed_pct']:.0f}%)
""")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='TBU Emergence Reproduction Script'
    )
    parser.add_argument('--seeds', type=int, default=10,
                        help='Number of random seeds to test')
    parser.add_argument('--nodes', type=int, default=2000,
                        help='Number of nodes in mesh')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of selection steps')
    parser.add_argument('--single', action='store_true',
                        help='Run single detailed experiment instead of robustness test')
    parser.add_argument('--output', type=str, default='emergence_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    if args.single:
        # Run single detailed experiment
        results = run_emergence_experiment(
            n_nodes=args.nodes,
            n_steps=args.steps,
            seed=42,
            verbose=True
        )
    else:
        # Run robustness test
        results = run_robustness_test(
            n_seeds=args.seeds,
            n_nodes=args.nodes,
            n_steps=args.steps
        )
    
    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'script': 'run_emergence.py',
        'version': '1.0',
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
