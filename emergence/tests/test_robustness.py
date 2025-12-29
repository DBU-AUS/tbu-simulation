#!/usr/bin/env python3
"""
================================================================================
ROBUSTNESS TEST - MULTI-CONFIGURATION VALIDATION
================================================================================

Validates key findings across multiple configurations and seeds.

Tests:
1. Baseline (no primary) - should show sign-flip
2. With t-tangent - should eliminate sign-flip
3. With t-tangent + secondary - should improve further
4. Under 25% noise without redundancy - should show sign-flip
5. Under 25% noise with redundancy - should rescue

Results confirm:
- Primary perception eliminates sign-flip
- Secondary channels compound with primary
- Redundancy rescues under noise

================================================================================
"""

import sys
import os

# Add substrates to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'substrates'))

import numpy as np
from typing import Dict, List, Tuple
from tbu_honest_boundary import HonestBoundarySubstrate


class RobustnessTestSubstrate(HonestBoundarySubstrate):
    """Flexible substrate for robustness testing."""
    
    def __post_init__(self):
        super().__post_init__()
        
        self.add_t_tangent = getattr(self, '_add_t_tangent', False)
        self.add_secondary = getattr(self, '_add_secondary', False)
        self.add_redundancy = getattr(self, '_add_redundancy', False)
        self.primary_noise = getattr(self, '_primary_noise', 0.0)
        
        extra = sum([self.add_t_tangent, self.add_secondary, self.add_redundancy])
        if extra > 0:
            self.n_channels += extra
            old_attention = self.attention.copy()
            self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
            self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
        
        self.boundary_history_for_tangent: List[np.ndarray] = []
    
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
            if self.primary_noise > 0:
                t_tangent += self.rng.normal(0, self.primary_noise)
                t_tangent = np.clip(t_tangent, 0, 1)
            ch = np.ones(self.n_forcing) * (t_tangent - 0.5) * 0.6
            ch += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch.reshape(-1, 1))
        
        if self.add_secondary:
            health = self.environment.get_health_signal()['health']
            ch = np.ones(self.n_forcing) * (health - 0.5) * 0.6
            ch += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch.reshape(-1, 1))
        
        if self.add_redundancy:
            t_tangent_clean = self._compute_t_tangent()
            ch = np.ones(self.n_forcing) * (t_tangent_clean - 0.5) * 0.6
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
        
        if self.prev_channels is not None and self.prev_channels.shape == channels.shape:
            eta = self.attention_learning_rate
            boundary_now = self.grid[self.forcing_coords]
            dy = boundary_now - self.prev_boundary
            self.prev_boundary = boundary_now.copy()
            prediction = np.sum(self.attention * self.prev_channels, axis=1)
            error = dy - prediction
            self.attention += eta * (error[:, None] * self.prev_channels)
            self.attention = np.clip(self.attention, 1e-3, None)
            self.attention /= self.attention.sum(axis=1, keepdims=True)
        
        self.prev_channels = channels.copy()
        
        self.history.append(self.grid.copy())
        if len(self.history) > self.history_window:
            self.history.pop(0)
        
        self.action_history.append(self.actions.copy())
        if len(self.action_history) > self.history_window:
            self.action_history.pop(0)
        
        self.step += 1


def create_substrate(t_tangent: bool, secondary: bool, redundancy: bool, 
                    noise: float, seed: int):
    class Configured(RobustnessTestSubstrate):
        def __post_init__(self):
            self._add_t_tangent = t_tangent
            self._add_secondary = secondary
            self._add_redundancy = redundancy
            self._primary_noise = noise
            super().__post_init__()
    return Configured(seed=seed)


def run_config(t_tangent: bool, secondary: bool, redundancy: bool,
               noise: float, seed: int, n_steps: int = 5000, warmup: int = 500) -> float:
    substrate = create_substrate(t_tangent, secondary, redundancy, noise, seed)
    
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
    print("ROBUSTNESS TEST - MULTI-CONFIGURATION VALIDATION")
    print("=" * 70)
    
    seeds = [42, 43, 44, 45, 46]
    
    # (name, t_tangent, secondary, redundancy, noise, expected_sign)
    configs = [
        ("Baseline (no primary)", False, False, False, 0.0, "flip"),
        ("With t-tangent", True, False, False, 0.0, "positive"),
        ("t-tangent + secondary", True, True, False, 0.0, "positive"),
        ("25% noise, no redundancy", True, False, False, 0.25, "flip"),
        ("25% noise + redundancy", True, False, True, 0.25, "positive"),
    ]
    
    results = {}
    
    for name, t_tan, sec, red, noise, expected in configs:
        print(f"\n{'='*60}")
        print(f"CONFIG: {name}")
        print(f"{'='*60}")
        
        vals = []
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ")
            r = run_config(t_tan, sec, red, noise, seed)
            vals.append(r)
            status = "FLIP" if r < 0 else "OK"
            print(f"r = {r:+.3f} [{status}]")
        
        results[name] = {
            'values': vals,
            'mean': np.mean(vals),
            'std': np.std(vals),
            'flips': sum(1 for v in vals if v < 0),
            'expected': expected,
        }
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Configuration':<30} {'Mean r':>10} {'Flips':>8} {'Expected':>12} {'Pass':>6}")
    print("-" * 70)
    
    all_pass = True
    for name, data in results.items():
        expected = data['expected']
        if expected == "flip":
            passed = data['flips'] >= 3  # Majority should flip
        else:
            passed = data['flips'] <= 1  # At most 1 flip
        
        all_pass = all_pass and passed
        pass_str = "✓" if passed else "✗"
        
        print(f"{name:<30} {data['mean']:>+10.3f} {data['flips']:>6}/5 {expected:>12} {pass_str:>6}")
    
    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED - Findings validated across seeds")
    else:
        print("SOME TESTS FAILED - Review configurations")
    print("=" * 70)


if __name__ == "__main__":
    main()
