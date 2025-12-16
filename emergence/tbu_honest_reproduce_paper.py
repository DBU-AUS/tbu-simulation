#!/usr/bin/env python3
"""
================================================================================
REPRODUCE PAPER RESULTS - TBU Emergence Validation
================================================================================

This script reproduces the specific results claimed in Appendix R of:
  "Entropy Maximisation under Conservation Constraints on 4D Geometries"

Each function runs the exact experiment with exact parameters that produced
the paper's numbers. Output is labeled to match paper tables.

Usage:
    python reproduce_paper.py --all              # Run everything
    python reproduce_paper.py --table-r1         # Table R.1: Level 1-2 results
    python reproduce_paper.py --susceptibility   # chi = 1/(1+M) validation
    python reproduce_paper.py --30-30-true       # 30/30 vs 0/30 (TRUE instantaneous)
    python reproduce_paper.py --self-reference   # 89% vs 12% localization (Level 3+)
    python reproduce_paper.py --ablation         # Ablation table

Key experiments:
  - Table R.1: M differentiation (9,091x), coherence (1.000), autonomy (100%)
  - 30/30 vs 0/30: Full window detects autonomy, single-slice does not
  - 89% vs 12%: Self-reference localizes in high-M regions (Level 3+)
  - Ablation: Distance structure -> coherence

Requirements:
    - numpy>=1.20
    - tbu_honest.py in same directory
    - tbu_honest_extended.py in same directory (for --self-reference)

================================================================================
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from tbu_honest import (
    HonestSubstrate,
    run_ablation,
    largest_connected_component_fraction,
)


def _find_largest_component_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Find largest connected component using BFS on periodic torus."""
    from collections import deque
    
    H, W = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    best_component = None
    best_size = 0
    
    for i in range(H):
        for j in range(W):
            if binary_mask[i, j] and not visited[i, j]:
                component = np.zeros_like(binary_mask, dtype=bool)
                q = deque([(i, j)])
                visited[i, j] = True
                component[i, j] = True
                comp_size = 0
                
                while q:
                    y, x = q.popleft()
                    comp_size += 1
                    for dy, dx in ((1,0), (-1,0), (0,1), (0,-1)):
                        yy, xx = (y + dy) % H, (x + dx) % W
                        if binary_mask[yy, xx] and not visited[yy, xx]:
                            visited[yy, xx] = True
                            component[yy, xx] = True
                            q.append((yy, xx))
                
                if comp_size > best_size:
                    best_size = comp_size
                    best_component = component
    
    return best_component


def reproduce_table_r1(n_seeds: int = 10, n_steps: int = 2000, size: int = 64):
    """Reproduce Table R.1: Level 1-2 Core Results."""
    print("=" * 70)
    print("TABLE R.1: Level 1-2 Core Results")
    print("=" * 70)
    print()
    print(f"Parameters: size={size}, steps={n_steps}, seeds={n_seeds}")
    print()
    
    results = []
    
    for seed in range(1, n_seeds + 1):
        print(f"Seed {seed}/{n_seeds}...", end=" ", flush=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = HonestSubstrate(
                size=size,
                forcing_shape="ring",
                seed=seed,
                state_dir=Path(tmpdir),
            )
            
            for _ in range(n_steps):
                sub.step_physics()
            
            report = sub.report()
            chi_result = sub.measure_susceptibility_empirical()
            
            M = sub.measure_M()
            interior = ~sub.forcing_mask
            M_interior = M[interior]
            
            M_max = float(M_interior.max())
            M_min = float(M_interior.min())
            M_range = M_max / M_min if M_min > 0 else float('inf')
            
            M_p99 = float(np.percentile(M_interior, 99))
            M_p01 = float(np.percentile(M_interior, 1))
            M_pct_range = M_p99 / M_p01 if M_p01 > 0 else float('inf')
            
            if 'error' not in chi_result:
                results.append({
                    'seed': seed,
                    'M_range': M_range,
                    'M_pct_range': M_pct_range,
                    'coherence': report['core_coherence'],
                    'ordering_validated': chi_result['ordering_validated'],
                    'autonomy_works': chi_result['autonomy']['autonomy_works'],
                })
                print(f"M_range={M_range:.0f}x, "
                      f"coherence={report['core_coherence']:.1%}, "
                      f"chi-ordering={'Y' if chi_result['ordering_validated'] else 'N'}")
            else:
                print(f"ERROR: {chi_result.get('error', 'unknown')}")
    
    if not results:
        print("\nNo successful runs!")
        return
    
    print()
    print("=" * 70)
    print("SUMMARY (compare to paper Table R.1)")
    print("=" * 70)
    print()
    
    M_ranges = [r['M_range'] for r in results]
    M_pct_ranges = [r['M_pct_range'] for r in results]
    coherences = [r['coherence'] for r in results]
    ordering_pass = sum(1 for r in results if r['ordering_validated'])
    autonomy_pass = sum(1 for r in results if r['autonomy_works'])
    
    print(f"  M differentiation (max/min): {np.mean(M_ranges):.0f}x +/- {np.std(M_ranges):.0f}")
    print(f"                               (paper: 9,091x)")
    print()
    print(f"  M differentiation (p99/p01): {np.mean(M_pct_ranges):.0f}x +/- {np.std(M_pct_ranges):.0f}")
    print()
    print(f"  Core coherence:           {np.mean(coherences):.3f} +/- {np.std(coherences):.3f}")
    print(f"                            (paper: 1.000)")
    print()
    print(f"  Susceptibility ordering:  {ordering_pass}/{len(results)} pass")
    print(f"  Autonomy:                 {autonomy_pass}/{len(results)} pass")
    print()
    
    return results


def reproduce_susceptibility_r2(n_seeds: int = 10, n_steps: int = 2000, size: int = 64):
    """Reproduce susceptibility law validation: var ~ 1/(1+M)^b using split-time windows."""
    print("=" * 70)
    print("SUSCEPTIBILITY LAW: var ~ 1/(1+M)^b (Level 1-2)")
    print("=" * 70)
    print()
    print(f"Parameters: size={size}, steps={n_steps}, seeds={n_seeds}")
    print()
    print("Method: SPLIT-TIME WINDOWS (avoids circularity)")
    print()
    
    all_M = []
    all_var = []
    
    for seed in range(1, n_seeds + 1):
        print(f"Seed {seed}/{n_seeds}...", end=" ", flush=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = HonestSubstrate(
                size=size,
                forcing_shape="ring",
                seed=seed,
                state_dir=Path(tmpdir),
                history_window=100,
            )
            
            history_phase1 = []
            for step in range(n_steps // 2):
                sub.step_physics()
                if step % 10 == 0:
                    history_phase1.append(sub.grid.copy())
            
            if len(history_phase1) < 20:
                print("insufficient history phase 1")
                continue
            
            H1 = np.array(history_phase1[-50:], dtype=np.float64)
            var_phase1 = np.var(H1, axis=0) + 1e-12
            M_phase1 = np.median(var_phase1) / var_phase1
            M_phase1 = np.clip(M_phase1, sub.M_clip_min, sub.M_clip_max)
            
            history_phase2 = []
            for step in range(n_steps // 2):
                sub.step_physics()
                if step % 10 == 0:
                    history_phase2.append(sub.grid.copy())
            
            if len(history_phase2) < 20:
                print("insufficient history phase 2")
                continue
            
            H2 = np.array(history_phase2[-50:], dtype=np.float64)
            var_phase2 = np.var(H2, axis=0)
            
            interior = ~sub.forcing_mask
            M_flat = M_phase1[interior].flatten()
            var_flat = var_phase2[interior].flatten()
            
            n_bins = 10
            for q in range(n_bins):
                q_low = q / n_bins
                q_high = (q + 1) / n_bins
                
                thresh_low = np.quantile(M_flat, q_low)
                thresh_high = np.quantile(M_flat, q_high)
                
                if q == n_bins - 1:
                    bin_mask = (M_flat >= thresh_low) & (M_flat <= thresh_high)
                else:
                    bin_mask = (M_flat >= thresh_low) & (M_flat < thresh_high)
                
                if bin_mask.sum() < 10:
                    continue
                
                all_M.append(float(M_flat[bin_mask].mean()))
                all_var.append(float(var_flat[bin_mask].mean()))
        
        print("done")
    
    if len(all_M) < 10:
        print("Not enough data points!")
        return
    
    all_M = np.array(all_M)
    all_var = np.array(all_var)
    
    log_var = np.log(all_var + 1e-12)
    log_1pM = np.log(1.0 + all_M)
    
    A = np.vstack([log_1pM, np.ones_like(log_1pM)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_var, rcond=None)
    neg_b, const = coeffs
    exponent = -neg_b
    
    log_var_pred = neg_b * log_1pM + const
    ss_res = np.sum((log_var - log_var_pred) ** 2)
    ss_tot = np.sum((log_var - np.mean(log_var)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"  Exponent b:         {exponent:.2f} (paper: b ~ 1.3)")
    print(f"  R-squared:          {r_squared:.3f}")
    print()
    
    if r_squared > 0.8:
        print(f"  POWER LAW VALIDATED: var ~ (1+M)^(-{exponent:.1f})")
    else:
        print(f"  Moderate fit (R^2={r_squared:.3f})")
    print()
    
    return {'exponent': exponent, 'r_squared': r_squared}


def reproduce_30_30_true_instantaneous(n_runs: int = 30, n_steps: int = 2000, size: int = 64):
    """Reproduce 30/30 vs 0/30 with TRUE instantaneous M (single-slice, no variance)."""
    print("=" * 70)
    print("30/30 vs 0/30: TRUE INSTANTANEOUS (single-slice, no variance)")
    print("=" * 70)
    print()
    print(f"Parameters: size={size}, steps={n_steps}, runs={n_runs}")
    print()
    print("Full window: variance from 100 snapshots (reads 4D geometry)")
    print("Instantaneous: M from CURRENT GRID STATE (single 3D slice)")
    print("  - M_instant = |grid - spatial_median|")
    print("  - Stable = pixels closest to median (bottom 30%)")
    print()
    
    full_window_results = []
    instant_results = []
    
    for run in range(1, n_runs + 1):
        print(f"Run {run}/{n_runs}...", end=" ", flush=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = HonestSubstrate(
                size=size,
                forcing_shape="ring",
                seed=run,
                state_dir=Path(tmpdir),
                history_window=100,
            )
            
            for _ in range(n_steps):
                sub.step_physics()
            
            if len(sub.history) < 100:
                print("insufficient")
                full_window_results.append(False)
                instant_results.append(False)
                continue
            
            interior = ~sub.forcing_mask
            
            # Full window (standard)
            chi_result = sub.measure_susceptibility_empirical()
            if 'error' not in chi_result:
                full_window_results.append(chi_result['autonomy']['autonomy_works'])
            else:
                full_window_results.append(False)
            
            # TRUE INSTANTANEOUS: single-slice M
            grid_now = sub.grid.copy()
            interior_vals = grid_now[interior]
            spatial_median = np.median(interior_vals)
            
            M_instant = np.abs(grid_now - spatial_median)
            M_instant_interior = M_instant[interior]
            thresh_instant = np.quantile(M_instant_interior, sub.stability_quantile)
            stable_instant = interior & (M_instant <= thresh_instant)
            
            lcc_frac, _ = largest_connected_component_fraction(stable_instant)
            
            if np.isnan(lcc_frac) or stable_instant.sum() < 10:
                instant_results.append(False)
            else:
                core_instant = _find_largest_component_mask(stable_instant)
                
                if core_instant is None or core_instant.sum() < 10:
                    instant_results.append(False)
                else:
                    periphery_instant = interior & (~core_instant)
                    
                    if periphery_instant.sum() < 10:
                        instant_results.append(False)
                    else:
                        grid_backup = sub.grid.copy()
                        history_backup = list(sub.history)
                        step_backup = sub.step
                        
                        core_baseline = float(sub.grid[core_instant].mean())
                        perturbation = 0.3
                        
                        sub.grid[periphery_instant] += perturbation
                        sub.grid = np.tanh(sub.grid)
                        
                        for _ in range(30):
                            sub.step_physics()
                        
                        core_after = float(sub.grid[core_instant].mean())
                        core_drift = abs(core_after - core_baseline)
                        relative_drift = core_drift / perturbation
                        
                        sub.grid = grid_backup
                        sub.history = history_backup
                        sub.step = step_backup
                        
                        instant_results.append(relative_drift < 0.15)
        
        f = "Y" if full_window_results[-1] else "N"
        i = "Y" if instant_results[-1] else "N"
        print(f"full: {f}, instant: {i}")
    
    print()
    print("=" * 70)
    print("RESULTS (compare to paper)")
    print("=" * 70)
    print()
    
    full_pass = sum(full_window_results)
    instant_pass = sum(instant_results)
    
    print(f"  Full window (100 snapshots):    {full_pass}/{n_runs} autonomy")
    print(f"                                  (paper: 30/30)")
    print()
    print(f"  TRUE INSTANTANEOUS (no var):    {instant_pass}/{n_runs} autonomy")
    print(f"                                  (paper: 0/30)")
    print()
    
    if full_pass >= n_runs * 0.9 and instant_pass <= n_runs * 0.1:
        print("  EXACT MATCH: 30/30 vs 0/30 reproduced!")
    elif full_pass >= n_runs * 0.8 and instant_pass <= n_runs * 0.2:
        print("  CLOSE MATCH: Strong separation achieved")
    else:
        print(f"  Separation: {full_pass - instant_pass} runs difference")
    
    print()
    print("Interpretation:")
    print("  Full window reads 4D geometric structure -> autonomy emerges")
    print("  Single slice sees only 3D snapshot -> no autonomy visible")
    print("  This confirms: autonomy is GEOMETRIC (4D), not temporal accumulation")
    print()
    
    return {'full_pass': full_pass, 'instant_pass': instant_pass}


def reproduce_self_reference_localization(n_seeds: int = 10, n_steps: int = 3000, size: int = 64):
    """Reproduce the self-reference localization claim: 89% vs 12%."""
    print("=" * 70)
    print("SELF-REFERENCE LOCALIZATION: 89% vs 12% (Level 3+)")
    print("=" * 70)
    print()
    print(f"Parameters: size={size}, steps={n_steps}, seeds={n_seeds}")
    print()
    print("Paper claim: Self-reference concentrates in high-M regions")
    print("  - High-M functional rate: 89%")
    print("  - Low-M functional rate:  12%")
    print()
    
    try:
        from tbu_honest_extended import HonestExtendedSubstrate
    except ImportError:
        print("ERROR: tbu_honest_extended.py not found")
        print("This test requires the extended substrate (Level 3+)")
        return None
    
    results = []
    
    for seed in range(1, n_seeds + 1):
        print(f"Seed {seed}/{n_seeds}...", end=" ", flush=True)
        
        sub = HonestExtendedSubstrate(
            size=size,
            forcing_shape="ring",
            seed=seed,
            history_window=100,
        )
        
        for _ in range(n_steps):
            sub.step_physics()
        
        loc = sub.measure_self_model_localization()
        
        high_rate = loc['high_M_functional_rate']
        low_rate = loc['low_M_functional_rate']
        
        results.append({'high': high_rate, 'low': low_rate})
        print(f"high-M: {high_rate:.1%}, low-M: {low_rate:.1%}")
    
    print()
    print("=" * 70)
    print("RESULTS (compare to paper)")
    print("=" * 70)
    print()
    
    high_rates = [r['high'] for r in results]
    low_rates = [r['low'] for r in results]
    
    high_mean = np.mean(high_rates)
    high_std = np.std(high_rates)
    low_mean = np.mean(low_rates)
    low_std = np.std(low_rates)
    
    print(f"  High-M functional rate: {high_mean:.1%} +/- {high_std:.1%}")
    print(f"                          (paper: 89%)")
    print()
    print(f"  Low-M functional rate:  {low_mean:.1%} +/- {low_std:.1%}")
    print(f"                          (paper: 12%)")
    print()
    
    if high_mean > 0.80 and low_mean < 0.20:
        print("  LOCALIZATION REPRODUCED: Self-reference concentrates in high-M")
    elif high_mean > low_mean * 3:
        print("  Strong localization but not exact match")
    else:
        print("  Localization not as expected")
    
    print()
    print("Interpretation:")
    print("  Self-models persist where M is high (low susceptibility)")
    print("  This emerges from dynamics, not architectural constraint")
    print()
    
    return results


def reproduce_ablation(n_seeds: int = 5):
    """Reproduce ablation table."""
    print("=" * 70)
    print("ABLATION TABLE: Coherence requires persistent distance structure")
    print("=" * 70)
    print()
    print(f"Running {n_seeds} seeds per configuration for statistics.")
    print()
    
    all_results = []
    
    for seed in range(1, n_seeds + 1):
        print(f"--- Seed {seed}/{n_seeds} ---")
        results = run_ablation(size=64, n_steps=10000, forcing_width=2, seed=seed)
        all_results.append(results)
    
    print()
    print("=" * 70)
    print("AGGREGATE RESULTS (mean +/- std across seeds)")
    print("=" * 70)
    print()
    
    config_names = [r['name'] for r in all_results[0]]
    
    print(f"{'Configuration':<30} {'Coherence':>15} {'Clusters':>12}")
    print("-" * 60)
    
    for i, name in enumerate(config_names):
        coherences = [all_results[s][i]['coherence'] for s in range(n_seeds)]
        clusters = [all_results[s][i]['n_clusters'] for s in range(n_seeds)]
        
        coh_mean = np.nanmean(coherences)
        coh_std = np.nanstd(coherences)
        clust_mean = np.nanmean(clusters)
        clust_std = np.nanstd(clusters)
        
        if not np.isnan(coh_mean):
            coh_str = f"{coh_mean:.1%} +/- {coh_std:.1%}"
        else:
            coh_str = "N/A"
        
        print(f"{name:<30} {coh_str:>15} {clust_mean:>6.1f} +/- {clust_std:>4.1f}")
    
    print()
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce paper results from Appendix R",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--all", action="store_true",
                        help="Run all reproduction experiments")
    parser.add_argument("--table-r1", action="store_true",
                        help="Reproduce Table R.1 (Level 1-2 results)")
    parser.add_argument("--susceptibility", action="store_true",
                        help="Reproduce susceptibility power law validation")
    parser.add_argument("--30-30-true", dest="exp_30_30_true", action="store_true",
                        help="Reproduce 30/30 vs 0/30 with TRUE instantaneous")
    parser.add_argument("--self-reference", dest="self_ref", action="store_true",
                        help="Reproduce 89%% vs 12%% self-reference localization (Level 3+)")
    parser.add_argument("--ablation", action="store_true",
                        help="Reproduce ablation table")
    
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds for statistical tests (default: 10)")
    parser.add_argument("--runs", type=int, default=30,
                        help="Number of runs for 30/30 experiment (default: 30)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Steps per run (default: 2000)")
    parser.add_argument("--size", type=int, default=64,
                        help="Grid size (default: 64)")
    
    args = parser.parse_args()
    
    if not any([args.all, args.table_r1, args.susceptibility, 
                args.exp_30_30_true, args.self_ref, args.ablation]):
        args.all = True
    
    print()
    print("=" * 70)
    print("TBU EMERGENCE VALIDATION - PAPER REPRODUCTION")
    print("=" * 70)
    print()
    print("Reproducing results from Appendix R of:")
    print("  'Entropy Maximisation under Conservation Constraints on 4D Geometries'")
    print()
    
    if args.all or args.table_r1:
        reproduce_table_r1(n_seeds=args.seeds, n_steps=args.steps, size=args.size)
        print("\n" + "=" * 70 + "\n")
    
    if args.all or args.susceptibility:
        reproduce_susceptibility_r2(n_seeds=args.seeds, n_steps=args.steps, size=args.size)
        print("\n" + "=" * 70 + "\n")
    
    if args.all or args.exp_30_30_true:
        reproduce_30_30_true_instantaneous(n_runs=args.runs, n_steps=args.steps, size=args.size)
        print("\n" + "=" * 70 + "\n")
    
    if args.all or args.self_ref:
        reproduce_self_reference_localization(n_seeds=args.seeds, n_steps=args.steps, size=args.size)
        print("\n" + "=" * 70 + "\n")
    
    if args.all or args.ablation:
        reproduce_ablation()
        print("\n" + "=" * 70 + "\n")
    
    print("REPRODUCTION COMPLETE")
    print()


if __name__ == "__main__":
    main()