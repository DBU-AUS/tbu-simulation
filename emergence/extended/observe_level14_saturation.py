#!/usr/bin/env python3
"""
================================================================================
LEVEL 14: OBSERVING REDUNDANCY SATURATION
================================================================================

Question: Does redundancy have a saturation point like primaries do?

METHOD:
  - Create substrate with configurable number of geometric view channels
  - Each channel reads from a different "shell depth" in the core
  - Measure learning benefit (coherence improvement across shocks)
  - Test across multiple seeds for statistical stability

OBSERVATION ONLY:
  - We measure what happens with 0-6 redundancy channels
  - No engineering of "optimal" behavior
  - Let the system show us if/where saturation occurs

================================================================================
"""

import sys
import os
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# Add substrates to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'substrates'))
from tbu_honest_boundary import HonestBoundarySubstrate


def compute_distance_from_boundary(forcing_mask: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance of each pixel from nearest boundary."""
    H, W = forcing_mask.shape
    distance = np.zeros((H, W), dtype=np.float64)
    
    by, bx = np.where(forcing_mask)
    core_mask = ~forcing_mask
    cy, cx = np.where(core_mask)
    
    if len(cy) == 0 or len(by) == 0:
        return distance
    
    # Process in chunks to avoid memory issues
    chunk_size = 500
    for start in range(0, len(cy), chunk_size):
        end = min(start + chunk_size, len(cy))
        chunk_cy = cy[start:end, np.newaxis]
        chunk_cx = cx[start:end, np.newaxis]
        
        # Toroidal distance
        dy = np.minimum(np.abs(by - chunk_cy), H - np.abs(by - chunk_cy))
        dx = np.minimum(np.abs(bx - chunk_cx), W - np.abs(bx - chunk_cx))
        
        min_dist = np.min(dy + dx, axis=1)
        distance[cy[start:end], cx[start:end]] = min_dist
    
    return distance


class ShellRedundancySubstrate(HonestBoundarySubstrate):
    """
    Substrate with shell-based redundancy channels.
    
    Each redundancy channel reads from a different concentric shell
    within the core, providing multiple geometric views.
    """
    
    def __init__(self, n_shells: int = 1, primary_noise: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        
        self.n_shells = n_shells
        self.primary_noise = primary_noise
        
        # Compute distance field for shell masks
        self.distance_field = compute_distance_from_boundary(self.forcing_mask)
        self.max_distance = float(self.distance_field.max())
        
        # Create shell masks
        self.shell_masks = []
        core = ~self.forcing_mask
        
        if self.max_distance > 0 and n_shells > 0:
            for i in range(n_shells):
                rho_lo = i / n_shells
                rho_hi = (i + 1) / n_shells
                d_lo = rho_lo * self.max_distance
                d_hi = rho_hi * self.max_distance
                
                shell = (self.distance_field > d_lo) & (self.distance_field <= d_hi) & core
                if shell.sum() > 0:
                    self.shell_masks.append(shell)
        
        self.actual_shells = len(self.shell_masks)
        
        # Extend channel array: base + primary + shells
        n_base = self.n_channels
        n_new = 1 + self.actual_shells  # primary + shells
        self.n_channels = n_base + n_new
        
        # Reinitialize attention with new size
        self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
        
        # Track for t-tangent
        self.prev_boundary_tangent = self.grid[self.forcing_coords].copy()
        self.prev_channels = None
    
    def _compute_t_tangent(self) -> float:
        """Primary channel: local temporal slope."""
        current = self.grid[self.forcing_coords]
        slope = np.mean(np.abs(current - self.prev_boundary_tangent))
        self.prev_boundary_tangent = current.copy()
        return float(np.tanh(slope * 50))
    
    def _compute_shell_value(self, idx: int) -> float:
        """Redundancy channel: shell's activity."""
        if idx >= len(self.shell_masks):
            return 0.0
        vals = self.grid[self.shell_masks[idx]]
        return float(np.tanh(vals.std() * 10))
    
    def step_physics(self):
        """Physics step with extended channels."""
        n_base = 6  # base channels from parent
        channels = np.zeros((self.n_forcing, self.n_channels))
        
        # Base channels
        boundary = self.grid[self.forcing_coords]
        for i in range(min(n_base, self.n_channels)):
            channels[:, i] = boundary + self.rng.normal(0, 0.01, self.n_forcing)
        
        # Primary channel (with noise)
        t_val = self._compute_t_tangent()
        noise = self.rng.normal(0, self.primary_noise, self.n_forcing)
        channels[:, n_base] = t_val + noise
        
        # Shell channels
        for i in range(self.actual_shells):
            shell_val = self._compute_shell_value(i)
            channels[:, n_base + 1 + i] = shell_val + self.rng.normal(0, 0.02, self.n_forcing)
        
        # Update attention via prediction error
        self._update_attention_hebbian(channels)
        
        # Execute physics
        self._execute_physics(channels)
        
        self.prev_channels = channels.copy()
    
    def _update_attention_hebbian(self, channels: np.ndarray):
        """Hebbian attention learning."""
        eta = 0.01
        boundary_now = self.grid[self.forcing_coords]
        
        if not hasattr(self, '_prev_b_att'):
            self._prev_b_att = boundary_now.copy()
            self._prev_c_att = channels.copy()
            return
        
        # Compute actual change
        dy = boundary_now - self._prev_b_att
        
        # Compute prediction from previous channels
        if self._prev_c_att.shape == channels.shape:
            prediction = np.sum(self.attention * self._prev_c_att, axis=1)
            error = dy - prediction
            
            # Update attention proportional to error × channel
            self.attention += eta * (error[:, None] * self._prev_c_att)
            self.attention = np.clip(self.attention, 1e-3, None)
            self.attention /= self.attention.sum(axis=1, keepdims=True)
        
        self._prev_b_att = boundary_now.copy()
        self._prev_c_att = channels.copy()
    
    def _execute_physics(self, channels: np.ndarray):
        """Execute physics with attention-weighted forcing."""
        # Attention-weighted input
        weighted = np.sum(self.attention * channels, axis=1)
        
        # Apply forcing
        self.grid[self.forcing_coords] += 0.1 * weighted
        
        # Diffusion
        lap = (
            np.roll(self.grid, 1, 0) + np.roll(self.grid, -1, 0) +
            np.roll(self.grid, 1, 1) + np.roll(self.grid, -1, 1) -
            4 * self.grid
        )
        self.grid += 0.2 * lap
        
        # Damping
        self.grid *= 0.99
        
        # Saturation
        self.grid = np.tanh(self.grid)
        
        self.step += 1


def measure_learning_benefit(
    n_shells: int,
    seed: int,
    n_shocks: int = 5,
    shock_noise: float = 0.35,
    shock_duration: int = 400,
    recovery_duration: int = 200,
    warmup: int = 1000,
) -> Dict:
    """Measure learning benefit for given shell count."""
    
    substrate = ShellRedundancySubstrate(
        n_shells=n_shells,
        primary_noise=0.05,
        seed=seed,
    )
    
    # Warmup
    for _ in range(warmup):
        substrate.step_physics()
    
    # Repeated shocks
    coherences = []
    
    for shock_idx in range(n_shocks):
        b_stds = []
        c_stds = []
        
        # Shock period
        for _ in range(shock_duration):
            substrate.primary_noise = shock_noise
            substrate.step_physics()
            b_stds.append(substrate.grid[substrate.forcing_coords].std())
            c_stds.append(substrate.grid[~substrate.forcing_mask].std())
        
        # Compute coherence
        b = np.array(b_stds)
        c = np.array(c_stds)
        if b.std() > 1e-6 and c.std() > 1e-6:
            coh = float(np.corrcoef(b, c)[0, 1])
        else:
            coh = 0.0
        coherences.append(coh)
        
        # Recovery
        for _ in range(recovery_duration):
            substrate.primary_noise = 0.05
            substrate.step_physics()
    
    # Compute improvement
    first = coherences[0]
    last = coherences[-1]
    improvement = last - first
    
    # Compute trend
    if len(coherences) > 2:
        trend = float(np.corrcoef(range(n_shocks), coherences)[0, 1])
    else:
        trend = 0.0
    
    return {
        'n_shells': n_shells,
        'actual_shells': substrate.actual_shells,
        'coherences': coherences,
        'first': first,
        'last': last,
        'improvement': improvement,
        'trend': trend,
    }


def observe_saturation(
    shell_counts: List[int] = [0, 1, 2, 3, 4, 5, 6],
    seeds: List[int] = [42, 43, 44],
) -> Dict:
    """Observe whether redundancy saturates across multiple seeds."""
    
    print("=" * 70)
    print("LEVEL 14: OBSERVING REDUNDANCY SATURATION")
    print("=" * 70)
    print()
    print(f"Shell counts to test: {shell_counts}")
    print(f"Seeds: {seeds}")
    print()
    
    all_results = {n: [] for n in shell_counts}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for n_shells in shell_counts:
            result = measure_learning_benefit(n_shells, seed)
            all_results[n_shells].append(result)
            print(f"  Shells={n_shells}: Δ={result['improvement']:+.3f}, trend={result['trend']:+.3f}")
    
    return all_results


def analyze_saturation(results: Dict) -> Dict:
    """Analyze saturation behavior from results."""
    
    print("\n" + "=" * 70)
    print("SATURATION ANALYSIS")
    print("=" * 70)
    
    # Aggregate across seeds
    shell_counts = sorted(results.keys())
    
    mean_improvements = []
    std_improvements = []
    mean_trends = []
    
    for n in shell_counts:
        improvements = [r['improvement'] for r in results[n]]
        trends = [r['trend'] for r in results[n]]
        
        mean_improvements.append(np.mean(improvements))
        std_improvements.append(np.std(improvements))
        mean_trends.append(np.mean(trends))
    
    # Print table
    print(f"\n{'Shells':>8} {'Mean Δ':>12} {'Std Δ':>10} {'Mean Trend':>12}")
    print("-" * 50)
    
    for i, n in enumerate(shell_counts):
        print(f"{n:>8} {mean_improvements[i]:>+12.3f} {std_improvements[i]:>10.3f} {mean_trends[i]:>+12.3f}")
    
    # Find peak
    peak_idx = np.argmax(mean_improvements)
    peak_shells = shell_counts[peak_idx]
    peak_improvement = mean_improvements[peak_idx]
    
    # Check for saturation
    print("\n" + "-" * 50)
    print("SATURATION CHECK:")
    
    # Compute marginal improvements
    marginals = np.diff(mean_improvements)
    print(f"\nMarginal improvements: {[f'{m:+.3f}' for m in marginals]}")
    
    # Look for where marginal goes negative or close to zero
    saturation_point = None
    for i, m in enumerate(marginals):
        if m < 0.05:  # Threshold for "negligible improvement"
            saturation_point = shell_counts[i + 1]
            break
    
    if saturation_point is not None:
        print(f"\n→ Saturation begins around {saturation_point} shells")
        print(f"   (marginal improvement becomes negligible)")
    else:
        print(f"\n→ No clear saturation in tested range")
        print(f"   More shells might still help")
    
    print(f"\nPeak improvement at {peak_shells} shells: {peak_improvement:+.3f}")
    
    return {
        'shell_counts': shell_counts,
        'mean_improvements': mean_improvements,
        'std_improvements': std_improvements,
        'mean_trends': mean_trends,
        'peak_shells': peak_shells,
        'peak_improvement': peak_improvement,
        'saturation_point': saturation_point,
        'marginals': list(marginals),
    }


def main():
    results = observe_saturation(
        shell_counts=[0, 1, 2, 3, 4, 5, 6],
        seeds=[42, 43, 44, 45, 46],  # 5 seeds for stability
    )
    
    analysis = analyze_saturation(results)
    
    print("\n" + "=" * 70)
    print("EMERGENT FINDING")
    print("=" * 70)
    
    if analysis['saturation_point'] is not None:
        print(f"""
  REDUNDANCY SATURATES at ~{analysis['saturation_point']} geometric views
  
  Unlike primaries which saturate immediately with noise,
  redundancy provides compounding benefits up to a point,
  then marginal returns become negligible.
  
  This suggests an OPTIMAL substrate architecture:
  - Not too few views (insufficient redundancy)
  - Not too many views (wasted resources)
  
  The saturation point likely depends on:
  - Noise level in primary channel
  - Correlation between shell views
  - Learning rate and adaptation timescale
""")
    else:
        print(f"""
  NO SATURATION OBSERVED in tested range (0-6 shells)
  
  Redundancy continues providing benefits even at 6 geometric views.
  This suggests either:
  - True saturation occurs beyond tested range
  - System can exploit arbitrary numbers of views
  
  Peak improvement was at {analysis['peak_shells']} shells.
""")
    
    return analysis


if __name__ == "__main__":
    main()
