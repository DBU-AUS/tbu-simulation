#!/usr/bin/env python3
"""
================================================================================
MULTI-CHANNEL PERCEPTION TEST
================================================================================

Question: Does perceiving multiple constraint dimensions compound the effect?

Test design:
1. Baseline - no extra perception (6 channels)
2. Volatility only - add volatility perception (7 channels)
3. Env_health only - add env_health perception (7 channels)  
4. Both - add volatility AND env_health perception (8 channels)

If mesh mapping is correct:
- Volatility reveals 38 relationships (29 sign-flips)
- Env_health reveals 49 relationships (16 sign-flips)
- These are DIFFERENT relationships
- Both should reveal MORE than either alone

Measure:
- Sign-flip presence on multiple conditioning variables
- Pooled correlation (higher = more unified)
- Overall coherence metrics

================================================================================
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

sys.path.insert(0, '/mnt/project')
from tbu_honest_boundary import HonestBoundarySubstrate, Environment, MultiChannelProbeWithHealth


class MultiPerceptionSubstrate(HonestBoundarySubstrate):
    """
    Substrate with configurable perception channels.
    
    Can add:
    - Volatility perception (Channel 6)
    - Environment health perception (Channel 7)
    - Both
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Configuration flags (set before calling this)
        self.perceive_volatility = getattr(self, '_perceive_volatility', False)
        self.perceive_env_health = getattr(self, '_perceive_env_health', False)
        
        # Count extra channels
        extra_channels = 0
        if self.perceive_volatility:
            extra_channels += 1
        if self.perceive_env_health:
            extra_channels += 1
        
        if extra_channels > 0:
            self.n_channels += extra_channels
            
            # Expand attention
            old_attention = self.attention.copy()
            self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
            self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
        
        # Volatility tracking
        self.boundary_history_for_volatility: List[np.ndarray] = []
        self.volatility_window = 20
        self.current_volatility = 0.0
        
        # Environment health tracking
        self.current_env_health = 1.0
        
        channels_str = []
        if self.perceive_volatility:
            channels_str.append("volatility")
        if self.perceive_env_health:
            channels_str.append("env_health")
        
        if channels_str:
            print(f"Extra perception: {', '.join(channels_str)} ({self.n_channels} total channels)")
        else:
            print(f"Baseline: {self.n_channels} channels (no extra perception)")
    
    def _compute_boundary_volatility(self) -> float:
        if len(self.boundary_history_for_volatility) < 2:
            return 0.0
        
        recent = self.boundary_history_for_volatility[-self.volatility_window:]
        if len(recent) < 2:
            return 0.0
        
        diffs = [np.mean(np.abs(recent[i] - recent[i-1])) 
                 for i in range(1, len(recent))]
        
        raw_volatility = np.mean(diffs)
        normalized = np.tanh(raw_volatility * 50)
        
        return float(normalized)
    
    def step_physics(self):
        # Store boundary for volatility
        boundary_pre = self.grid[self.forcing_coords].copy()
        self.boundary_history_for_volatility.append(boundary_pre)
        if len(self.boundary_history_for_volatility) > self.volatility_window + 10:
            self.boundary_history_for_volatility = self.boundary_history_for_volatility[-self.volatility_window:]
        
        # Compute current values
        self.current_volatility = self._compute_boundary_volatility()
        self.current_env_health = self.environment.get_health_signal()['health']
        
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
        
        # Environment effects
        action_magnitude = float(np.mean(np.abs(self.actions)))
        self.environment.consume(action_magnitude * self.action_consumption_scale)
        
        boundary_variation = float(np.mean(np.abs(dy_action)))
        self.environment.disturb(boundary_variation * self.action_disturbance_scale)
        
        self.environment.step()
        
        # Read base channels
        base_channels = self.probe.read_channels(self.n_forcing)
        
        # Build full channel array
        channel_list = [base_channels]
        
        if self.perceive_volatility:
            vol_channel = np.ones(self.n_forcing) * (self.current_volatility - 0.5) * 0.6
            vol_channel += self.rng.normal(0, 0.02, self.n_forcing)
            channel_list.append(vol_channel.reshape(-1, 1))
        
        if self.perceive_env_health:
            health_channel = np.ones(self.n_forcing) * (self.current_env_health - 0.5) * 0.6
            health_channel += self.rng.normal(0, 0.02, self.n_forcing)
            channel_list.append(health_channel.reshape(-1, 1))
        
        channels = np.column_stack(channel_list) if len(channel_list) > 1 else base_channels
        
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
        
        # Logging
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


def create_substrate(perceive_volatility: bool, perceive_env_health: bool, seed: int) -> MultiPerceptionSubstrate:
    """Factory to create substrate with specified perception."""
    
    class ConfiguredSubstrate(MultiPerceptionSubstrate):
        def __post_init__(self):
            self._perceive_volatility = perceive_volatility
            self._perceive_env_health = perceive_env_health
            super().__post_init__()
    
    return ConfiguredSubstrate(seed=seed)


def run_and_analyze(name: str, perceive_volatility: bool, perceive_env_health: bool, 
                    seed: int = 42, n_steps: int = 5000, warmup: int = 500) -> Dict:
    """Run one configuration and return analysis."""
    
    print(f"\n{'='*60}")
    print(f"Configuration: {name}")
    print(f"{'='*60}")
    
    substrate = create_substrate(perceive_volatility, perceive_env_health, seed)
    
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
            
            report = substrate.report()
            
            log.append({
                'boundary_std': float(boundary.std()),
                'core_std': float(core.std()),
                'volatility': volatility,
                'M_ratio': report['M_ratio'],
                'env_health': report['env_health'],
                'core_coherence': report['core_coherence'],
            })
        
        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}/{n_steps}")
    
    # Analyze - check sign-flip on multiple conditions
    boundary_std = np.array([d['boundary_std'] for d in log])
    core_std = np.array([d['core_std'] for d in log])
    volatility = np.array([d['volatility'] for d in log])
    M_ratio = np.array([d['M_ratio'] for d in log])
    env_health = np.array([d['env_health'] for d in log])
    core_coherence = np.array([d['core_coherence'] for d in log])
    
    results = {'name': name}
    
    # Test multiple conditioning variables
    for z_name, z_data in [('volatility', volatility), ('M_ratio', M_ratio), 
                            ('env_health', env_health), ('core_coherence', core_coherence)]:
        thresh = np.median(z_data)
        high = z_data >= thresh
        low = ~high
        
        r_high = np.corrcoef(boundary_std[high], core_std[high])[0, 1]
        r_low = np.corrcoef(boundary_std[low], core_std[low])[0, 1]
        r_pooled = np.corrcoef(boundary_std, core_std)[0, 1]
        
        sign_flip = (r_high * r_low < 0) and abs(r_high) > 0.05 and abs(r_low) > 0.05
        
        results[f'{z_name}_r_high'] = r_high
        results[f'{z_name}_r_low'] = r_low
        results[f'{z_name}_r_pooled'] = r_pooled
        results[f'{z_name}_sign_flip'] = sign_flip
    
    # Overall pooled correlation (measure of unification)
    results['pooled_r'] = np.corrcoef(boundary_std, core_std)[0, 1]
    
    # Count sign-flips
    results['n_sign_flips'] = sum(1 for k, v in results.items() if k.endswith('_sign_flip') and v)
    
    return results


def main():
    print("=" * 70)
    print("MULTI-CHANNEL PERCEPTION TEST")
    print("=" * 70)
    print()
    print("Question: Does perceiving multiple constraints compound the effect?")
    print()
    
    configs = [
        ("Baseline (6ch)", False, False),
        ("Volatility only (7ch)", True, False),
        ("Env_health only (7ch)", False, True),
        ("Both (8ch)", True, True),
    ]
    
    results = []
    for name, vol, health in configs:
        r = run_and_analyze(name, vol, health, seed=42)
        results.append(r)
    
    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Configuration':<25} {'Pooled r':>10} {'Sign-flips':>12}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<25} {r['pooled_r']:>+10.3f} {r['n_sign_flips']:>12}")
    
    # Detailed sign-flip analysis
    print("\n" + "-" * 70)
    print("SIGN-FLIP ANALYSIS (boundary_std <-> core_std)")
    print("-" * 70)
    
    conditions = ['volatility', 'M_ratio', 'env_health', 'core_coherence']
    
    print(f"\n{'Config':<20}", end="")
    for c in conditions:
        print(f" {c[:8]:>10}", end="")
    print()
    print("-" * 65)
    
    for r in results:
        print(f"{r['name'][:20]:<20}", end="")
        for c in conditions:
            flip = "FLIP" if r[f'{c}_sign_flip'] else "no"
            print(f" {flip:>10}", end="")
        print()
    
    # Detailed correlations
    print("\n" + "-" * 70)
    print("CORRELATION DETAILS (r_high / r_low)")
    print("-" * 70)
    
    for c in conditions:
        print(f"\nConditioned on {c}:")
        print(f"  {'Config':<25} {'r_high':>10} {'r_low':>10} {'r_pool':>10}")
        print(f"  {'-'*55}")
        for r in results:
            print(f"  {r['name'][:25]:<25} {r[f'{c}_r_high']:>+10.3f} "
                  f"{r[f'{c}_r_low']:>+10.3f} {r[f'{c}_r_pooled']:>+10.3f}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    baseline = results[0]
    vol_only = results[1]
    health_only = results[2]
    both = results[3]
    
    print(f"\nPooled correlation (unification measure):")
    print(f"  Baseline:      {baseline['pooled_r']:+.3f}")
    print(f"  Volatility:    {vol_only['pooled_r']:+.3f} (Δ = {vol_only['pooled_r'] - baseline['pooled_r']:+.3f})")
    print(f"  Env_health:    {health_only['pooled_r']:+.3f} (Δ = {health_only['pooled_r'] - baseline['pooled_r']:+.3f})")
    print(f"  Both:          {both['pooled_r']:+.3f} (Δ = {both['pooled_r'] - baseline['pooled_r']:+.3f})")
    
    print(f"\nSign-flips remaining:")
    print(f"  Baseline:      {baseline['n_sign_flips']}/4")
    print(f"  Volatility:    {vol_only['n_sign_flips']}/4")
    print(f"  Env_health:    {health_only['n_sign_flips']}/4")
    print(f"  Both:          {both['n_sign_flips']}/4")
    
    # Test for compounding
    vol_improvement = vol_only['pooled_r'] - baseline['pooled_r']
    health_improvement = health_only['pooled_r'] - baseline['pooled_r']
    both_improvement = both['pooled_r'] - baseline['pooled_r']
    expected_additive = vol_improvement + health_improvement
    
    print(f"\nCompounding test:")
    print(f"  Vol improvement:    {vol_improvement:+.3f}")
    print(f"  Health improvement: {health_improvement:+.3f}")
    print(f"  Expected additive:  {expected_additive:+.3f}")
    print(f"  Actual (both):      {both_improvement:+.3f}")
    
    if both_improvement > expected_additive * 1.1:
        print(f"\n  → SUPER-ADDITIVE: Combined effect exceeds sum of parts")
    elif both_improvement > max(vol_improvement, health_improvement) * 1.1:
        print(f"\n  → COMPOUNDING: Combined effect exceeds either alone")
    elif both_improvement < min(vol_improvement, health_improvement):
        print(f"\n  → INTERFERENCE: Combined effect worse than either alone")
    else:
        print(f"\n  → ROUGHLY ADDITIVE: Combined effect ≈ sum of parts")


if __name__ == "__main__":
    main()
