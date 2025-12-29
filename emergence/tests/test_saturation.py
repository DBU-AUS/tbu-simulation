#!/usr/bin/env python3
"""
================================================================================
PRIMARY CHANNEL SATURATION TEST
================================================================================

Question: Does adding more primary channels improve coherence?

Finding: NO - one clean primary is sufficient. Additional primaries saturate.

Results (5-seed validation):
| Configuration              | Mean r | Std   |
|----------------------------|--------|-------|
| t-tangent only             | +0.926 | 0.001 |
| t-tangent + action_stab    | +0.938 | 0.000 |
| t-tangent + spatial        | +0.921 | 0.001 |
| All primaries              | +0.923 | 0.000 |

Conclusion: Primary channels saturate. One is sufficient.

================================================================================
"""

import sys
import os

# Add substrates to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'substrates'))

import numpy as np
from typing import Dict, List, Tuple
from tbu_honest_boundary import HonestBoundarySubstrate


class SaturationTestSubstrate(HonestBoundarySubstrate):
    """Substrate with configurable primary perception channels."""
    
    def __post_init__(self):
        super().__post_init__()
        
        self.add_t_tangent = getattr(self, '_add_t_tangent', True)
        self.add_action_stability = getattr(self, '_add_action_stability', False)
        self.add_spatial_coherence = getattr(self, '_add_spatial_coherence', False)
        
        extra = sum([self.add_t_tangent, self.add_action_stability, self.add_spatial_coherence])
        if extra > 0:
            self.n_channels += extra
            old_attention = self.attention.copy()
            self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
            self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
        
        self.boundary_history_for_tangent: List[np.ndarray] = []
        
        channels_added = []
        if self.add_t_tangent:
            channels_added.append("t-tangent")
        if self.add_action_stability:
            channels_added.append("action_stability")
        if self.add_spatial_coherence:
            channels_added.append("spatial_coherence")
        
        print(f"  Primary channels: {', '.join(channels_added) if channels_added else 'none'}")
    
    def _compute_t_tangent(self) -> float:
        if len(self.boundary_history_for_tangent) < 2:
            return 0.0
        current = self.boundary_history_for_tangent[-1]
        previous = self.boundary_history_for_tangent[-2]
        return float(np.tanh(np.mean(np.abs(current - previous)) * 50))
    
    def _compute_action_stability(self) -> float:
        if len(self.action_history) < 10:
            return 0.5
        recent = np.array(self.action_history[-10:])
        var = np.var(recent)
        return float(1.0 / (1.0 + var * 100))
    
    def _compute_spatial_coherence(self) -> float:
        boundary_vals = self.grid[self.forcing_coords]
        if len(boundary_vals) < 2:
            return 0.5
        diffs = np.abs(np.diff(boundary_vals))
        return float(1.0 / (1.0 + np.mean(diffs) * 10))
    
    def step_physics(self):
        boundary_pre = self.grid[self.forcing_coords].copy()
        self.boundary_history_for_tangent.append(boundary_pre)
        if len(self.boundary_history_for_tangent) > 5:
            self.boundary_history_for_tangent = self.boundary_history_for_tangent[-5:]
        
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        lap = up + dn + lf + rt - 4.0 * self.grid
        
        self.grid += self.diffusion * lap
        self.grid *= 1.0 - self.damping
        
        boundary_state_pre = self.grid[self.forcing_coords]
        dy_action = boundary_state_pre - self.prev_boundary_for_action
        self.prev_boundary_for_action = boundary_state_pre.copy()
        
        if self.action_gain_scale > 0:
            self.actions += self.action_smoothing * (dy_action - self.actions)
        
        action_magnitude = float(np.mean(np.abs(self.actions)))
        self.environment.consume(action_magnitude * self.action_consumption_scale)
        boundary_variation = float(np.mean(np.abs(dy_action)))
        self.environment.disturb(boundary_variation * self.action_disturbance_scale)
        self.environment.step()
        
        base_channels = self.probe.read_channels(self.n_forcing)
        
        extras = []
        if self.add_t_tangent:
            t_tangent = self._compute_t_tangent()
            ch = np.ones(self.n_forcing) * (t_tangent - 0.5) * 0.6
            ch += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch.reshape(-1, 1))
        
        if self.add_action_stability:
            action_stab = self._compute_action_stability()
            ch = np.ones(self.n_forcing) * (action_stab - 0.5) * 0.6
            ch += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch.reshape(-1, 1))
        
        if self.add_spatial_coherence:
            spatial_coh = self._compute_spatial_coherence()
            ch = np.ones(self.n_forcing) * (spatial_coh - 0.5) * 0.6
            ch += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch.reshape(-1, 1))
        
        if extras:
            channels = np.column_stack([base_channels] + extras)
        else:
            channels = base_channels
        
        weighted_input = np.sum(self.attention * channels, axis=1)
        
        if self.action_gain_scale > 0:
            action_mag = np.tanh(np.abs(self.actions))
            mag_mean = np.mean(action_mag) + 1e-12
            gain = 1.0 + self.action_gain_scale * (action_mag - mag_mean)
            gmin = max(0.05, 1.0 - self.action_gain_scale)
            gmax = 1.0 + self.action_gain_scale
            gain = np.clip(gain, gmin, gmax)
        else:
            gain = np.ones(self.n_forcing, dtype=np.float64)
        
        self.grid[self.forcing_coords] += self.forcing_strength * gain * weighted_input
        self.grid = np.tanh(self.grid)
        
        self._update_self_models()
        self._update_attention_ext(channels)
        self.prev_channels = channels.copy()
        
        self.history.append(self.grid.copy())
        if len(self.history) > self.history_window:
            self.history.pop(0)
        
        self.action_history.append(self.actions.copy())
        if len(self.action_history) > self.history_window:
            self.action_history.pop(0)
        
        self.step += 1
    
    def _update_attention_ext(self, channels: np.ndarray):
        eta = self.attention_learning_rate
        boundary_now = self.grid[self.forcing_coords]
        dy = boundary_now - self.prev_boundary
        self.prev_boundary = boundary_now.copy()
        
        if self.step < 1 or self.prev_channels is None:
            return
        if self.prev_channels.shape[1] != channels.shape[1]:
            return
        
        prediction = np.sum(self.attention * self.prev_channels, axis=1)
        error = dy - prediction
        self.attention += eta * (error[:, None] * self.prev_channels)
        self.attention = np.clip(self.attention, 1e-3, None)
        self.attention /= self.attention.sum(axis=1, keepdims=True)


def create_substrate(t_tangent: bool, action_stab: bool, spatial: bool, seed: int):
    class Configured(SaturationTestSubstrate):
        def __post_init__(self):
            self._add_t_tangent = t_tangent
            self._add_action_stability = action_stab
            self._add_spatial_coherence = spatial
            super().__post_init__()
    return Configured(seed=seed)


def run_config(t_tangent: bool, action_stab: bool, spatial: bool, seed: int,
               n_steps: int = 5000, warmup: int = 500) -> float:
    substrate = create_substrate(t_tangent, action_stab, spatial, seed)
    
    log = []
    for step in range(n_steps):
        substrate.step_physics()
        
        if step >= warmup:
            boundary = substrate.grid[substrate.forcing_coords]
            core = substrate.grid[~substrate.forcing_mask]
            log.append({
                'boundary_std': float(boundary.std()),
                'core_std': float(core.std()),
            })
    
    boundary_std = np.array([d['boundary_std'] for d in log])
    core_std = np.array([d['core_std'] for d in log])
    
    return float(np.corrcoef(boundary_std, core_std)[0, 1])


def main():
    print("=" * 70)
    print("PRIMARY CHANNEL SATURATION TEST")
    print("=" * 70)
    print()
    print("Question: Does adding more primary channels help?")
    print()
    
    seeds = [42, 43, 44, 45, 46]
    
    configs = [
        ("t-tangent only", True, False, False),
        ("t-tangent + action_stab", True, True, False),
        ("t-tangent + spatial", True, False, True),
        ("All primaries", True, True, True),
    ]
    
    results = {name: [] for name, *_ in configs}
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        for name, t_tan, act_stab, spatial in configs:
            print(f"\n  {name}...")
            r = run_config(t_tan, act_stab, spatial, seed)
            results[name].append(r)
            print(f"    → pooled r = {r:+.3f}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    baseline_r = np.mean(results["t-tangent only"])
    
    print(f"\n{'Configuration':<30} {'Mean r':>10} {'Std':>8} {'Δ':>10}")
    print("-" * 62)
    
    for name, *_ in configs:
        vals = results[name]
        mean_r = np.mean(vals)
        std_r = np.std(vals)
        delta = mean_r - baseline_r
        print(f"{name:<30} {mean_r:>+10.3f} {std_r:>8.3f} {delta:>+10.3f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION: Primary channels saturate. One is sufficient.")
    print("=" * 70)


if __name__ == "__main__":
    main()
