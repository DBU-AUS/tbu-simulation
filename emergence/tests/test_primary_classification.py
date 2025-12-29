#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT 1: PRIMARY CLASSIFICATION TEST
================================================================================

Test each candidate primary variable alone (7 channels each).

Hypothesis: All primary variables should improve coherence when perceived alone.

Configs:
- Baseline (6ch)              ← reference
- boundary_volatility (7ch)   ← known primary
- core_coherence (7ch)        ← test
- M_ratio (7ch)               ← test

Prediction: All three primaries should show positive Δ from baseline.

================================================================================
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'substrates'))  # Assume tbu_honest_boundary.py in same directory
from tbu_honest_boundary import HonestBoundarySubstrate


class SingleChannelSubstrate(HonestBoundarySubstrate):
    """Substrate with one extra perception channel (configurable)."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Which channel to add (set before calling)
        self.extra_channel = getattr(self, '_extra_channel', None)
        
        if self.extra_channel:
            self.n_channels += 1
            
            # Expand attention
            old_attention = self.attention.copy()
            self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
            self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
            
            print(f"Extra perception: {self.extra_channel} ({self.n_channels} total channels)")
        else:
            print(f"Baseline: {self.n_channels} channels (no extra perception)")
        
        # Tracking for derived values
        self.boundary_history_for_volatility: List[np.ndarray] = []
        self.volatility_window = 20
        self.current_extra_value = 0.0
    
    def _compute_boundary_volatility(self) -> float:
        if len(self.boundary_history_for_volatility) < 2:
            return 0.0
        recent = self.boundary_history_for_volatility[-self.volatility_window:]
        if len(recent) < 2:
            return 0.0
        diffs = [np.mean(np.abs(recent[i] - recent[i-1])) for i in range(1, len(recent))]
        return float(np.tanh(np.mean(diffs) * 50))
    
    def _compute_core_coherence(self) -> float:
        if len(self.history) < 20:
            return 0.5
        M = self.measure_M()
        core_mask = (M >= np.median(M[~self.forcing_mask])) & (~self.forcing_mask)
        if not core_mask.any():
            return 0.5
        
        # Coherence = correlation of core region over recent history
        recent = self.history[-20:]
        core_series = [h[core_mask].mean() for h in recent]
        if np.std(core_series) < 1e-10:
            return 0.5
        
        # Autocorrelation at lag 1
        s = np.array(core_series)
        r = np.corrcoef(s[:-1], s[1:])[0, 1]
        return float((r + 1) / 2) if np.isfinite(r) else 0.5  # Map to [0, 1]
    
    def _compute_M_ratio(self) -> float:
        report = self.report()
        M_ratio = report.get('M_ratio', 1.0)
        # Normalize to roughly [0, 1]
        return float(np.tanh(M_ratio / 100))
    
    def step_physics(self):
        # Store boundary for volatility
        boundary_pre = self.grid[self.forcing_coords].copy()
        self.boundary_history_for_volatility.append(boundary_pre)
        if len(self.boundary_history_for_volatility) > self.volatility_window + 10:
            self.boundary_history_for_volatility = self.boundary_history_for_volatility[-self.volatility_window:]
        
        # Compute extra channel value
        if self.extra_channel == 'boundary_volatility':
            self.current_extra_value = self._compute_boundary_volatility()
        elif self.extra_channel == 'core_coherence':
            self.current_extra_value = self._compute_core_coherence()
        elif self.extra_channel == 'M_ratio':
            self.current_extra_value = self._compute_M_ratio()
        
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


def create_substrate(extra_channel: Optional[str], seed: int) -> SingleChannelSubstrate:
    """Factory to create substrate with specified extra channel."""
    
    class ConfiguredSubstrate(SingleChannelSubstrate):
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
    
    # Sign-flip check on volatility
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
    print("EXPERIMENT 1: PRIMARY CLASSIFICATION TEST")
    print("=" * 70)
    print()
    print("Hypothesis: All primary variables should improve coherence alone")
    print()
    
    configs = [
        ("Baseline (6ch)", None),
        ("boundary_volatility (7ch)", "boundary_volatility"),
        ("core_coherence (7ch)", "core_coherence"),
        ("M_ratio (7ch)", "M_ratio"),
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
    
    print(f"\n{'Configuration':<30} {'Pooled r':>10} {'Δ':>10} {'Sign-flip':>12} {'Primary?':>10}")
    print("-" * 75)
    
    for r in results:
        delta = r['pooled_r'] - baseline
        flip = "YES" if r['sign_flip'] else "no"
        primary = "YES" if delta > 0.1 else ("maybe" if delta > 0 else "NO")
        print(f"{r['name']:<30} {r['pooled_r']:>+10.3f} {delta:>+10.3f} {flip:>12} {primary:>10}")
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION")
    print("=" * 70)
    
    primaries = []
    for r in results[1:]:  # Skip baseline
        delta = r['pooled_r'] - baseline
        if delta > 0.1:
            primaries.append(r['channel'])
            print(f"  {r['channel']}: PRIMARY (Δ = {delta:+.3f})")
        elif delta > 0:
            print(f"  {r['channel']}: WEAK PRIMARY (Δ = {delta:+.3f})")
        else:
            print(f"  {r['channel']}: NOT PRIMARY (Δ = {delta:+.3f})")
    
    print()
    if len(primaries) == 3:
        print("✓ All three candidates confirmed as PRIMARY")
    else:
        print(f"  {len(primaries)}/3 candidates confirmed as primary")


if __name__ == "__main__":
    main()
