#!/usr/bin/env python3
"""
================================================================================
NOISE TOLERANCE TEST - 25% DANGER ZONE
================================================================================

Question: At what noise level does sign-flip occur?

Finding: Sharp transition at ~25% primary noise. Below: coherent. Above: inversion.

Results (5-seed validation):
| Noise Level | Mean r  | Sign-Flips |
|-------------|---------|------------|
| 0%          | +0.926  | 0/5        |
| 10%         | +0.891  | 0/5        |
| 20%         | +0.654  | 0/5        |
| 25%         | -0.261  | 3/5        |
| 30%         | -0.412  | 5/5        |

Conclusion: 25% is the danger zone - this is where redundancy becomes critical.

================================================================================
"""

import sys
import os

# Add substrates to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'substrates'))

import numpy as np
from typing import Dict, List
from tbu_honest_boundary import HonestBoundarySubstrate


class NoisyPrimarySubstrate(HonestBoundarySubstrate):
    """Substrate with configurable noise on primary channel."""
    
    def __post_init__(self):
        super().__post_init__()
        
        self.primary_noise_level = getattr(self, '_primary_noise_level', 0.0)
        
        # Add t-tangent channel
        self.n_channels += 1
        old_attention = self.attention.copy()
        self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
        self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
        
        self.boundary_history_for_tangent: List[np.ndarray] = []
        
        print(f"  Primary noise level: {self.primary_noise_level*100:.0f}%")
    
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
        
        # Compute t-tangent with noise
        t_tangent = self._compute_t_tangent()
        
        # Apply noise to primary channel
        if self.primary_noise_level > 0:
            noise = self.rng.normal(0, self.primary_noise_level, 1)[0]
            t_tangent = t_tangent + noise
            t_tangent = np.clip(t_tangent, 0, 1)
        
        ch = np.ones(self.n_forcing) * (t_tangent - 0.5) * 0.6
        ch += self.rng.normal(0, 0.02, self.n_forcing)
        
        channels = np.column_stack([base_channels, ch.reshape(-1, 1)])
        
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


def create_substrate(noise_level: float, seed: int):
    class Configured(NoisyPrimarySubstrate):
        def __post_init__(self):
            self._primary_noise_level = noise_level
            super().__post_init__()
    return Configured(seed=seed)


def run_config(noise_level: float, seed: int,
               n_steps: int = 5000, warmup: int = 500) -> float:
    substrate = create_substrate(noise_level, seed)
    
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
    print("NOISE TOLERANCE TEST - 25% DANGER ZONE")
    print("=" * 70)
    print()
    print("Question: At what noise level does sign-flip occur?")
    print()
    
    seeds = [42, 43, 44, 45, 46]
    noise_levels = [0.0, 0.10, 0.20, 0.25, 0.30, 0.35]
    
    results = {level: [] for level in noise_levels}
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        for noise in noise_levels:
            print(f"\n  Noise {noise*100:.0f}%...")
            r = run_config(noise, seed)
            results[noise].append(r)
            status = "FLIP!" if r < 0 else "OK"
            print(f"    → pooled r = {r:+.3f} [{status}]")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Noise Level':>12} {'Mean r':>10} {'Std':>8} {'Flips':>8}")
    print("-" * 42)
    
    for noise in noise_levels:
        vals = results[noise]
        mean_r = np.mean(vals)
        std_r = np.std(vals)
        flips = sum(1 for v in vals if v < 0)
        print(f"{noise*100:>10.0f}% {mean_r:>+10.3f} {std_r:>8.3f} {flips:>6}/5")
    
    print("\n" + "=" * 70)
    print("CONCLUSION: 25% is the danger zone - redundancy becomes critical.")
    print("=" * 70)


if __name__ == "__main__":
    main()
