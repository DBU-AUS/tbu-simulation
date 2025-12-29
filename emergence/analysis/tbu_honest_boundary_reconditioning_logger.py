#!/usr/bin/env python3
"""
================================================================================
TBU RECONDITIONING LOGGER - Wrapper for tbu_honest_boundary.py (Level 8)
================================================================================

This script wraps tbu_honest_boundary.py (Level 8: Environmental health sensing)
to produce timestep-level CSV logs suitable for the reconditioning fingerprint
scanner.

Why Level 8?
  The sign-flip reconditioning fingerprint (opposite-signed correlations in
  different regimes that cancel when pooled) requires sufficient internal
  complexity. Level 8 substrates have:
    - Attention mechanisms that learn from prediction errors
    - Actions that consume resources and disturb stability
    - Environmental health coupling to observation quality

Key prediction being tested:
  core_std <-> boundary_std | boundary_volatility shows SIGN FLIP:
    - High volatility regime: r = +0.9 (positive)
    - Low volatility regime: r = -0.3 (negative)
    - Pooled: r = -0.08 (near zero - washout!)

Usage:
  python tbu_reconditioning_logger_boundary.py --steps 6000 --csv run_001.csv
  python tbu_reconditioning_logger_boundary.py --steps 6000 --csv run_002.csv --seed 42 --check
  
Then analyze with:
  python tbu_reconditioning_scanner.py --csv run_001.csv --top 20

Requirements:
  - numpy>=1.20
  - tbu_honest_boundary.py in same directory (or PYTHONPATH)
================================================================================
"""

from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from tbu_honest_boundary import HonestBoundarySubstrate
except ImportError:
    print("ERROR: tbu_honest_boundary.py must be in the same directory or PYTHONPATH")
    sys.exit(1)


def run_with_logging(
    n_steps: int, csv_path: Path, size: int = 64, seed: Optional[int] = None,
    warmup: int = 500, log_interval: int = 1, report_interval: int = 500, verbose: bool = True,
) -> List[Dict[str, Any]]:
    if verbose:
        print("=" * 70)
        print("TBU RECONDITIONING LOGGER (Level 8: Boundary Substrate)")
        print("=" * 70)
        print(f"Steps: {n_steps} (warmup: {warmup}), Output: {csv_path}, Seed: {seed}\n")
    
    sub = HonestBoundarySubstrate(size=size, seed=seed, history_window=100)
    prev_boundary = sub.grid[sub.forcing_coords].copy()
    
    if verbose:
        print(f"\nWarmup phase ({warmup} steps)...")
    for i in range(warmup):
        sub.step_physics()
        if verbose and (i + 1) % report_interval == 0:
            print(f"  warmup step {i + 1}/{warmup}")
    
    prev_boundary = sub.grid[sub.forcing_coords].copy()
    if verbose:
        print(f"Warmup complete. Starting logged run...\n")
    
    log_data: List[Dict[str, Any]] = []
    last_M_ratio = 1.0
    
    fieldnames = ['step', 'core_std', 'boundary_std', 'M_ratio', 'boundary_volatility',
                  'env_health', 'obs_sigma', 'attention_predictable', 'attention_entropy',
                  'attention_health', 'action_std', 'interior_std', 'grid_std', 'core_coherence']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(n_steps - warmup):
            sub.step_physics()
            
            if i % 10 == 0:
                M = sub.measure_M()
                forcing_M = float(M[sub.forcing_mask].mean())
                non_forcing_M = float(M[~sub.forcing_mask].mean())
                last_M_ratio = non_forcing_M / (forcing_M + 1e-12)
            
            if (i + 1) % log_interval == 0:
                grid = sub.grid
                boundary = grid[sub.forcing_coords]
                core = grid[~sub.forcing_mask]
                
                dy = boundary - prev_boundary
                boundary_volatility = float(np.mean(np.abs(dy)))
                prev_boundary = boundary.copy()
                
                health = sub.environment.get_health_signal()
                env_health = float(health["health"])
                obs_sigma = (sub.probe.env_noise_floor + sub.probe.env_noise_scale * (1.0 - env_health)
                            if sub.env_noise_enabled else sub.probe.env_noise_floor)
                
                mean_attention = sub.attention.mean(axis=0)
                report = sub.report()
                
                metrics = {
                    'step': int(sub.step),
                    'core_std': float(core.std()),
                    'boundary_std': float(boundary.std()),
                    'M_ratio': float(last_M_ratio),
                    'boundary_volatility': boundary_volatility,
                    'env_health': env_health,
                    'obs_sigma': float(obs_sigma),
                    'attention_predictable': float(mean_attention[0] + mean_attention[3]),
                    'attention_entropy': float(mean_attention[1]),
                    'attention_health': float(mean_attention[4] + mean_attention[5]),
                    'action_std': float(np.std(sub.actions)),
                    'interior_std': float(core.std()),
                    'grid_std': float(report['grid_std']),
                    'core_coherence': float(report['core_coherence']),
                }
                writer.writerow(metrics)
                log_data.append(metrics)
                if len(log_data) % 1000 == 0: f.flush()
            
            if verbose and (i + 1) % report_interval == 0:
                m = sub.report()
                print(f"[{sub.step:>8}] M_ratio={last_M_ratio:.2f}x  coherence={m['core_coherence']:.1%}  "
                      f"env_health={sub.environment.get_health_signal()['health']:.2f}  logged={len(log_data)}")
    
    if verbose:
        print(f"\nComplete. Logged {len(log_data)} timesteps to {csv_path}\n")
        if log_data:
            M_ratios = [d['M_ratio'] for d in log_data if np.isfinite(d['M_ratio'])]
            volatilities = [d['boundary_volatility'] for d in log_data if np.isfinite(d['boundary_volatility'])]
            print(f"Summary: M_ratio mean={np.mean(M_ratios):.2f}, boundary_volatility mean={np.mean(volatilities):.4f}")
    return log_data


def check_fingerprints(log_data: List[Dict[str, Any]], warmup_fraction: float = 0.1):
    n = len(log_data)
    start = int(n * warmup_fraction)
    data = log_data[start:]
    
    X = np.array([d['boundary_std'] for d in data])
    Y = np.array([d['core_std'] for d in data])
    Z1 = np.array([d['M_ratio'] for d in data])
    Z2 = np.array([d['boundary_volatility'] for d in data])
    
    def analyze(X, Y, Z, name):
        thr = np.median(Z)
        hi, lo = Z >= thr, Z < thr
        r_pool = np.corrcoef(X, Y)[0, 1]
        r_hi = np.corrcoef(X[hi], Y[hi])[0, 1]
        r_lo = np.corrcoef(X[lo], Y[lo])[0, 1]
        best = max(abs(r_hi), abs(r_lo))
        washout_ratio = 1.0 - abs(r_pool) / best if best > 0 else 0
        
        print(f"\nFingerprint: boundary_std <-> core_std | {name}")
        print(f"  r_pool = {r_pool:+.4f}")
        print(f"  r_hi   = {r_hi:+.4f}  (n={hi.sum()})")
        print(f"  r_lo   = {r_lo:+.4f}  (n={lo.sum()})")
        print(f"  washout_ratio = {washout_ratio:.2f}")
        if r_hi * r_lo < 0:
            print(f"  *** SIGN FLIP DETECTED ***")
    
    print("\n" + "=" * 60)
    print("RECONDITIONING FINGERPRINT CHECK")
    print("=" * 60)
    analyze(X, Y, Z1, "M_ratio")
    analyze(X, Y, Z2, "boundary_volatility")


def main():
    ap = argparse.ArgumentParser(description="TBU Reconditioning Logger for Level 8 Boundary Substrate")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--csv", type=Path, default=Path("tbu_boundary_log.csv"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--log-interval", type=int, default=1)
    ap.add_argument("--report-interval", type=int, default=500)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--check", action="store_true", help="Run fingerprint check after logging")
    args = ap.parse_args()
    
    log_data = run_with_logging(
        n_steps=args.steps, csv_path=args.csv, size=args.size, seed=args.seed,
        warmup=args.warmup, log_interval=args.log_interval,
        report_interval=args.report_interval, verbose=not args.quiet,
    )
    
    if args.check and log_data:
        check_fingerprints(log_data)


if __name__ == "__main__":
    main()