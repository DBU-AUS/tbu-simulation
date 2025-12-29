#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT 2: SECONDARY CONFIRMATION TEST
================================================================================

Verify all secondary variables are harmful or neutral when perceived alone.

Hypothesis: Secondary variables without primary context degrade coherence.

Configs:
- Baseline (6ch)              ← reference
- env_health (7ch)            ← known harmful (-0.246)
- env_resources (7ch)         ← test (should be ≡ env_health)
- obs_sigma (7ch)             ← test (should be ≡ env_health)
- action_stability (7ch)      ← test

Prediction: All secondaries should have Δ ≤ 0 from baseline.

================================================================================
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'substrates'))  # Assume tbu_honest_boundary.py in same directory
from tbu_honest_boundary import HonestBoundarySubstrate


class SecondaryChannelSubstrate(HonestBoundarySubstrate):
    """Substrate with one secondary perception channel."""
    
    def __post_init__(self):
        super().__post_init__()
        
        self.extra_channel = getattr(self, '_extra_channel', None)
        
        if self.extra_channel:
            self.n_channels += 1
            
            old_attention = self.attention.copy()
            self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
            self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
            
            print(f"Extra perception: {self.extra_channel} ({self.n_channels} total channels)")
        else:
            print(f"Baseline: {self.n_channels} channels (no extra perception)")
        
        self.current_extra_value = 0.0
        self.action_stability_window = 20
        self.action_history_for_stability: List[float] = []
    
    def _compute_env_health(self) -> float:
        return float(self.environment.get_health_signal()['health'])
    
    def _compute_env_resources(self) -> float:
        return float(self.environment.resources / self.environment.max_resources)
    
    def _compute_obs_sigma(self) -> float:
        # obs_sigma is already computed in probe, but we need raw value
        h = self._compute_env_health()
        obs_sigma = 0.02 + 0.25 * (1.0 - h)
        # Invert so high sigma = low value (for consistency)
        return float(1.0 - min(obs_sigma, 1.0))
    
    def _compute_action_stability(self) -> float:
        if len(self.action_history_for_stability) < 2:
            return 0.5
        
        recent = self.action_history_for_stability[-self.action_stability_window:]
        if len(recent) < 2:
            return 0.5
        
        # Stability = inverse of variance
        var = np.var(recent)
        stability = 1.0 / (1.0 + var * 100)  # Scale factor
        return float(stability)
    
    def step_physics(self):
        # Track action for stability
        self.action_history_for_stability.append(float(np.mean(np.abs(self.actions))))
        if len(self.action_history_for_stability) > self.action_stability_window + 10:
            self.action_history_for_stability = self.action_history_for_stability[-self.action_stability_window:]
        
        # Compute extra channel value
        if self.extra_channel == 'env_health':
            self.current_extra_value = self._compute_env_health()
        elif self.extra_channel == 'env_resources':
            self.current_extra_value = self._compute_env_resources()
        elif self.extra_channel == 'obs_sigma':
            self.current_extra_value = self._compute_obs_sigma()
        elif self.extra_channel == 'action_stability':
            self.current_extra_value = self._compute_action_stability()
        
        # Standard physics
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        lap = up + dn + lf + rt - 4.0 * self.grid
        
        self.grid += self.diffusion * lap
        self.grid *= 1.0 - self.damping
        
        # Actions
        boundary_state_pre = self.grid[self.forcing_coords]
        dy_action = boundary_state_pre - self.prev_boundary_for_action
        self.prev_boundary_for_action = boundary_state_pre.copy()
        
        if self.action_gain_scale > 0:
            self.actions += self.action_smoothing * (dy_action - self.actions)
        
        # Environment
        action_magnitude = float(np.mean(np.abs(self.actions)))
        self.environment.consume(action_magnitude * self.action_consumption_scale)
        boundary_variation = float(np.mean(np.abs(dy_action)))
        self.environment.disturb(boundary_variation * self.action_disturbance_scale)
        self.environment.step()
        
        # Read base channels
        base_channels = self.probe.read_channels(self.n_forcing)
        
        # Add extra channel if configured
        if self.extra_channel:
            extra = np.ones(self.n_forcing) * (self.current_extra_value - 0.5) * 0.6
            extra += self.rng.normal(0, 0.02, self.n_forcing)
            channels = np.column_stack([base_channels, extra.reshape(-1, 1)])
        else:
            channels = base_channels
        
        # Attention-weighted input
        weighted_input = np.sum(self.attention * channels, axis=1)
        
        # Gain gating
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
        
        # Updates
        self._update_self_models()
        self._update_attention_extended(channels)
        self.prev_channels = channels.copy()
        
        # History
        if len(self.history) >= 20:
            M = self.measure_M()
            core_mask = (M >= np.median(M[~self.forcing_mask])) & (~self.forcing_mask)
            core_mean = float(self.grid[core_mask].mean()) if core_mask.any() else 0.0
            self.core_mean_series.append(core_mean)
            self.action_mean_series.append(float(np.mean(np.abs(self.actions))))
            if len(self.core_mean_series) > self.history_window * 2:
                self.core_mean_series = self.core_mean_series[-self.history_window:]
                self.action_mean_series = self.action_mean_series[-self.history_window:]
        
        self.history.append(self.grid.copy())
        if len(self.history) > self.history_window:
            self.history.pop(0)
        
        self.self_model_history.append(self.self_models.copy())
        if len(self.self_model_history) > self.history_window:
            self.self_model_history.pop(0)
        
        self.attention_history.append(self.attention.copy())
        if len(self.attention_history) > self.history_window:
            self.attention_history.pop(0)
        
        self.action_history.append(self.actions.copy())
        if len(self.action_history) > self.history_window:
            self.action_history.pop(0)
        
        health = self.environment.get_health_signal()['health']
        self.health_history.append(health)
        if len(self.health_history) > self.history_window:
            self.health_history.pop(0)
        
        self.step += 1
    
    def _update_attention_extended(self, channels: np.ndarray):
        eta = self.attention_learning_rate
        
        boundary_now = self.grid[self.forcing_coords]
        dy = boundary_now - self.prev_boundary
        self.prev_boundary = boundary_now.copy()
        
        if self.step < 1:
            return
        
        if self.prev_channels is None or self.prev_channels.shape[1] != channels.shape[1]:
            return
        
        prediction = np.sum(self.attention * self.prev_channels, axis=1)
        error = dy - prediction
        self.attention += eta * (error[:, None] * self.prev_channels)
        
        self.attention = np.clip(self.attention, 1e-3, None)
        self.attention /= self.attention.sum(axis=1, keepdims=True)


def create_substrate(extra_channel: Optional[str], seed: int) -> SecondaryChannelSubstrate:
    """Factory to create substrate with specified extra channel."""
    
    class ConfiguredSubstrate(SecondaryChannelSubstrate):
        def __post_init__(self):
            self._extra_channel = extra_channel
            super().__post_init__()
    
    return ConfiguredSubstrate(seed=seed)


def run_and_analyze(name: str, extra_channel: Optional[str], 
                    seed: int = 42, n_steps: int = 5000, warmup: int = 500) -> Dict:
    """Run one configuration and return analysis."""
    
    print(f"\n{'='*60}")
    print(f"Configuration: {name}")
    print(f"{'='*60}")
    
    substrate = create_substrate(extra_channel, seed)
    
    log = []
    prev_boundary = None
    
    for step in range(n_steps):
        substrate.step_physics()
        
        if step >= warmup:
            grid = substrate.grid
            boundary = grid[substrate.forcing_coords]
            core = grid[~substrate.forcing_mask]
            
            if prev_boundary is not None:
                volatility = float(np.mean(np.abs(boundary - prev_boundary)))
            else:
                volatility = 0.0
            prev_boundary = boundary.copy()
            
            log.append({
                'boundary_std': float(boundary.std()),
                'core_std': float(core.std()),
                'volatility': volatility,
            })
        
        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}/{n_steps}")
    
    # Analyze
    boundary_std = np.array([d['boundary_std'] for d in log])
    core_std = np.array([d['core_std'] for d in log])
    volatility = np.array([d['volatility'] for d in log])
    
    # Pooled correlation
    r_pooled = np.corrcoef(boundary_std, core_std)[0, 1]
    
    # Sign-flip check
    thresh = np.median(volatility)
    high = volatility >= thresh
    low = ~high
    
    r_high = np.corrcoef(boundary_std[high], core_std[high])[0, 1]
    r_low = np.corrcoef(boundary_std[low], core_std[low])[0, 1]
    sign_flip = (r_high * r_low < 0) and abs(r_high) > 0.05 and abs(r_low) > 0.05
    
    return {
        'name': name,
        'channel': extra_channel,
        'pooled_r': r_pooled,
        'r_high': r_high,
        'r_low': r_low,
        'sign_flip': sign_flip,
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 2: SECONDARY CONFIRMATION TEST")
    print("=" * 70)
    print()
    print("Hypothesis: Secondary variables without primary context degrade coherence")
    print()
    
    configs = [
        ("Baseline (6ch)", None),
        ("env_health (7ch)", "env_health"),
        ("env_resources (7ch)", "env_resources"),
        ("obs_sigma (7ch)", "obs_sigma"),
        ("action_stability (7ch)", "action_stability"),
    ]
    
    results = []
    for name, channel in configs:
        r = run_and_analyze(name, channel, seed=42)
        results.append(r)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    baseline = results[0]['pooled_r']
    
    print(f"\n{'Configuration':<30} {'Pooled r':>10} {'Δ':>10} {'Sign-flip':>12} {'Secondary?':>12}")
    print("-" * 80)
    
    for r in results:
        delta = r['pooled_r'] - baseline
        flip = "YES" if r['sign_flip'] else "no"
        secondary = "CONFIRMED" if delta <= 0 else "NOT secondary"
        print(f"{r['name']:<30} {r['pooled_r']:>+10.3f} {delta:>+10.3f} {flip:>12} {secondary:>12}")
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION")
    print("=" * 70)
    
    secondaries = []
    for r in results[1:]:  # Skip baseline
        delta = r['pooled_r'] - baseline
        if delta < -0.1:
            secondaries.append(r['channel'])
            print(f"  {r['channel']}: SECONDARY - HARMFUL (Δ = {delta:+.3f})")
        elif delta <= 0:
            secondaries.append(r['channel'])
            print(f"  {r['channel']}: SECONDARY - NEUTRAL (Δ = {delta:+.3f})")
        else:
            print(f"  {r['channel']}: NOT SECONDARY (Δ = {delta:+.3f}) - helps alone!")
    
    print()
    
    # Check equivalence
    print("=" * 70)
    print("EQUIVALENCE CHECK")
    print("=" * 70)
    
    env_health_r = results[1]['pooled_r']
    env_resources_r = results[2]['pooled_r']
    obs_sigma_r = results[3]['pooled_r']
    
    print(f"\n  env_health:    {env_health_r:+.3f}")
    print(f"  env_resources: {env_resources_r:+.3f}")
    print(f"  obs_sigma:     {obs_sigma_r:+.3f}")
    
    if abs(env_health_r - env_resources_r) < 0.05 and abs(env_health_r - obs_sigma_r) < 0.05:
        print("\n  ✓ CONFIRMED: env_health ≡ env_resources ≡ obs_sigma")
        print("  These reveal the same constraint dimension")
    else:
        print("\n  × NOT equivalent - different effects")
    
    print()
    
    if len(secondaries) == 4:
        print("✓ All four candidates confirmed as SECONDARY")
        print("  (harmful or neutral when perceived alone)")
    else:
        print(f"  {len(secondaries)}/4 candidates confirmed as secondary")


if __name__ == "__main__":
    main()
