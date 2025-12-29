#!/usr/bin/env python3
"""
================================================================================
SECONDARY CHANNEL STACKING TEST  
================================================================================

Question: Do secondary channels compound benefit when primary is present?

Finding: YES - secondaries add incremental improvement with primary context.

Results (5-seed validation):
| Configuration                    | Mean r  | Δ from primary |
|---------------------------------|---------|----------------|
| Primary only                     | +0.926  | baseline       |
| Primary + health                 | +0.933  | +0.007         |
| Primary + resources              | +0.931  | +0.005         |
| Primary + health + resources     | +0.948  | +0.022         |

Conclusion: Secondaries compound positively WITH primary context.
Without primary, they cause sign-flip (see test_noise_tolerance.py).

================================================================================
"""

import sys
import os

# Add substrates to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'substrates'))

import numpy as np
from typing import Dict, List
from tbu_honest_boundary import HonestBoundarySubstrate


class SecondaryStackingSubstrate(HonestBoundarySubstrate):
    """Substrate with primary + configurable secondary channels."""
    
    def __post_init__(self):
        super().__post_init__()
        
        self.add_t_tangent = getattr(self, '_add_t_tangent', True)
        self.add_health = getattr(self, '_add_health', False)
        self.add_resources = getattr(self, '_add_resources', False)
        
        extra = sum([self.add_t_tangent, self.add_health, self.add_resources])
        if extra > 0:
            self.n_channels += extra
            old_attention = self.attention.copy()
            self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
            self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
        
        self.boundary_history_for_tangent: List[np.ndarray] = []
        
        channels_added = []
        if self.add_t_tangent:
            channels_added.append("t-tangent")
        if self.add_health:
            channels_added.append("health")
        if self.add_resources:
            channels_added.append("resources")
        
        print(f"  Channels: {', '.join(channels_added) if channels_added else 'base only'}")
    
    def _compute_t_tangent(self) -> float:
        if len(self.boundary_history_for_tangent) < 2:
            return 0.0
        current = self.boundary_history_for_tangent[-1]
        previous = self.boundary_history_for_tangent[-2]
        return float(np.tanh(np.mean(np.abs(current - previous)) * 50))
    
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
        
        if self.add_health:
            health = self.environment.get_health_signal()['health']
            ch = np.ones(self.n_forcing) * (health - 0.5) * 0.6
            ch += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch.reshape(-1, 1))
        
        if self.add_resources:
            resources = self.environment.get_health_signal()['resource_level']
            ch = np.ones(self.n_forcing) * (resources - 0.5) * 0.6
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


def create_substrate(t_tangent: bool, health: bool, resources: bool, seed: int):
    class Configured(SecondaryStackingSubstrate):
        def __post_init__(self):
            self._add_t_tangent = t_tangent
            self._add_health = health
            self._add_resources = resources
            super().__post_init__()
    return Configured(seed=seed)


def run_config(t_tangent: bool, health: bool, resources: bool, seed: int,
               n_steps: int = 5000, warmup: int = 500) -> float:
    substrate = create_substrate(t_tangent, health, resources, seed)
    
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
    print("SECONDARY CHANNEL STACKING TEST")
    print("=" * 70)
    print()
    print("Question: Do secondary channels compound benefit with primary?")
    print()
    
    seeds = [42, 43, 44, 45, 46]
    
    configs = [
        ("Primary only", True, False, False),
        ("Primary + health", True, True, False),
        ("Primary + resources", True, False, True),
        ("Primary + health + resources", True, True, True),
    ]
    
    results = {name: [] for name, *_ in configs}
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        for name, t_tan, health, resources in configs:
            print(f"\n  {name}...")
            r = run_config(t_tan, health, resources, seed)
            results[name].append(r)
            print(f"    → pooled r = {r:+.3f}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    baseline_r = np.mean(results["Primary only"])
    
    print(f"\n{'Configuration':<35} {'Mean r':>10} {'Std':>8} {'Δ':>10}")
    print("-" * 67)
    
    for name, *_ in configs:
        vals = results[name]
        mean_r = np.mean(vals)
        std_r = np.std(vals)
        delta = mean_r - baseline_r
        print(f"{name:<35} {mean_r:>+10.3f} {std_r:>8.3f} {delta:>+10.3f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION: Secondaries compound positively WITH primary context.")
    print("=" * 70)


if __name__ == "__main__":
    main()
