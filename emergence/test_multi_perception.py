#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT 3: MULTI-CHANNEL PERCEPTION (HIERARCHY PRINCIPLE)
================================================================================

Tests the hierarchy principle: secondary variables become beneficial only
when combined with primary variables.

METHODOLOGY (HONEST):
- We extend the Level 8 HonestBoundarySubstrate with perception channels
- Channels carry information computed FROM the substrate's own state
- NO objectives, rewards, or conditional logic are added
- We observe whether coupling structure changes

This is NOT engineering - we are opening windows, not pushing buttons.

HYPOTHESIS:
- Volatility alone: large positive Δ (primary works alone)
- Env_health alone: negative Δ (harmful without context)
- Both together: larger positive Δ than volatility alone (super-additive)

The super-additive effect would demonstrate the hierarchy principle:
secondary variables are harmful alone but beneficial with primary context.

CONFIGURATIONS:
- Baseline (6 channels): No extra perception
- Volatility only (7 ch): Primary variable alone
- Env_health only (7 ch): Secondary variable alone  
- Both (8 ch): Primary + Secondary together

================================================================================
"""

import sys
import numpy as np
from typing import Dict, List, Optional, Set

from tbu_honest_boundary import HonestBoundarySubstrate


class MultiChannelSubstrate(HonestBoundarySubstrate):
    """
    Level 8+ substrate with configurable extra perception channels.
    
    Channels carry information computed from the substrate's own state.
    No objectives or rewards are modified - only perceptual access changes.
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Which channels to add (set via factory)
        self.extra_channels: Set[str] = getattr(self, '_extra_channels', set())
        
        n_extra = len(self.extra_channels)
        if n_extra > 0:
            self.n_channels += n_extra
            
            old_attention = self.attention.copy()
            self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
            self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
            
            print(f"[HONEST] Extra perception: {self.extra_channels} ({self.n_channels} total channels)")
            print(f"[HONEST] No objectives, rewards, or conditional logic added")
        else:
            print(f"[HONEST] Baseline: {self.n_channels} channels (no extra perception)")
        
        # Tracking
        self.boundary_history_for_volatility: List[np.ndarray] = []
        self.volatility_window = 20
        self.current_volatility = 0.0
        self.current_env_health = 0.0
    
    def _compute_boundary_volatility(self) -> float:
        """Compute boundary volatility from recent history (NOT injected)."""
        if len(self.boundary_history_for_volatility) < 2:
            return 0.0
        recent = self.boundary_history_for_volatility[-self.volatility_window:]
        if len(recent) < 2:
            return 0.0
        diffs = [np.mean(np.abs(recent[i] - recent[i-1])) for i in range(1, len(recent))]
        return float(np.tanh(np.mean(diffs) * 50))
    
    def _compute_env_health(self) -> float:
        """Read environment health (NOT injected - reading existing state)."""
        return float(self.environment.get_health_signal()['health'])
    
    def step_physics(self):
        # Store boundary for volatility computation
        boundary_pre = self.grid[self.forcing_coords].copy()
        self.boundary_history_for_volatility.append(boundary_pre)
        if len(self.boundary_history_for_volatility) > self.volatility_window + 10:
            self.boundary_history_for_volatility = self.boundary_history_for_volatility[-self.volatility_window:]
        
        # Compute channel values FROM OWN STATE
        self.current_volatility = self._compute_boundary_volatility()
        self.current_env_health = self._compute_env_health()
        
        # === STANDARD PHYSICS ===
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
        
        # === PERCEPTION ===
        base_channels = self.probe.read_channels(self.n_forcing)
        channels = base_channels
        
        # Add extra channels in consistent order
        if 'boundary_volatility' in self.extra_channels:
            extra = np.ones(self.n_forcing) * (self.current_volatility - 0.5) * 0.6
            extra += self.rng.normal(0, 0.02, self.n_forcing)
            channels = np.column_stack([channels, extra.reshape(-1, 1)])
        
        if 'env_health' in self.extra_channels:
            extra = np.ones(self.n_forcing) * (self.current_env_health - 0.5) * 0.6
            extra += self.rng.normal(0, 0.02, self.n_forcing)
            channels = np.column_stack([channels, extra.reshape(-1, 1)])
        
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


def create_substrate(extra_channels: Set[str], seed: int) -> MultiChannelSubstrate:
    """Factory to create substrate with specified extra channels."""
    
    class ConfiguredSubstrate(MultiChannelSubstrate):
        def __post_init__(self):
            self._extra_channels = extra_channels
            super().__post_init__()
    
    return ConfiguredSubstrate(seed=seed)


def run_and_analyze(name: str, extra_channels: Set[str],
                    seed: int = 42, n_steps: int = 5000, warmup: int = 500) -> Dict:
    """Run one configuration and return analysis."""
    
    print(f"\n{'='*60}")
    print(f"Configuration: {name}")
    print(f"{'='*60}")
    
    substrate = create_substrate(extra_channels, seed)
    
    log = []
    prev_boundary = None
    
    for step in range(n_steps):
        substrate.step_physics()
        
        if step >= warmup:
            grid = substrate.grid
            boundary = grid[substrate.forcing_coords]
            core = grid[~substrate.forcing_mask]
            
            M = substrate.measure_M()
            core_mask = (M >= np.median(M[~substrate.forcing_mask])) & (~substrate.forcing_mask)
            
            if prev_boundary is not None:
                volatility = float(np.mean(np.abs(boundary - prev_boundary)))
            else:
                volatility = 0.0
            prev_boundary = boundary.copy()
            
            # Compute M_ratio for conditioning
            core_M = M[core_mask].mean() if core_mask.any() else 1.0
            boundary_M = M[substrate.forcing_mask].mean()
            M_ratio = core_M / (boundary_M + 1e-10)
            
            log.append({
                'boundary_std': float(boundary.std()),
                'core_std': float(core.std()),
                'volatility': volatility,
                'M_ratio': M_ratio,
                'env_health': substrate.current_env_health,
            })
        
        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}/{n_steps}")
    
    # Analyze
    boundary_std = np.array([d['boundary_std'] for d in log])
    core_std = np.array([d['core_std'] for d in log])
    volatility = np.array([d['volatility'] for d in log])
    M_ratio = np.array([d['M_ratio'] for d in log])
    
    # Pooled correlation
    r_pooled = np.corrcoef(boundary_std, core_std)[0, 1]
    
    # Check sign-flip on multiple conditioners
    sign_flips = {}
    for cond_name, cond_var in [('volatility', volatility), ('M_ratio', M_ratio)]:
        thresh = np.median(cond_var)
        high = cond_var >= thresh
        low = ~high
        
        if high.sum() > 10 and low.sum() > 10:
            r_high = np.corrcoef(boundary_std[high], core_std[high])[0, 1]
            r_low = np.corrcoef(boundary_std[low], core_std[low])[0, 1]
            sign_flip = (r_high * r_low < 0) and abs(r_high) > 0.05 and abs(r_low) > 0.05
            sign_flips[cond_name] = sign_flip
        else:
            sign_flips[cond_name] = None
    
    return {
        'name': name,
        'channels': extra_channels,
        'n_channels': 6 + len(extra_channels),
        'pooled_r': r_pooled,
        'sign_flips': sign_flips,
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 3: MULTI-CHANNEL PERCEPTION (HIERARCHY PRINCIPLE)")
    print("=" * 70)
    print()
    print("METHODOLOGY: Open perception channels, observe coupling changes")
    print("NO engineering, NO injection, NO reward modification")
    print()
    print("Hypothesis: Secondary variables become beneficial with primary context")
    print()
    
    configs = [
        ("Baseline (6ch)", set()),
        ("Volatility only (7ch)", {"boundary_volatility"}),
        ("Env_health only (7ch)", {"env_health"}),
        ("Both (8ch)", {"boundary_volatility", "env_health"}),
    ]
    
    results = []
    for name, channels in configs:
        r = run_and_analyze(name, channels, seed=42)
        results.append(r)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    baseline_r = results[0]['pooled_r']
    
    print(f"\n{'Configuration':<25} {'Channels':>8} {'Pooled r':>10} {'Δ':>10} {'Sign-flips':>15}")
    print("-" * 75)
    
    for r in results:
        delta = r['pooled_r'] - baseline_r
        flips = sum(1 for v in r['sign_flips'].values() if v)
        print(f"{r['name']:<25} {r['n_channels']:>8} {r['pooled_r']:>+10.3f} {delta:>+10.3f} {flips:>15}")
    
    # Compounding analysis
    print("\n" + "=" * 70)
    print("COMPOUNDING ANALYSIS")
    print("=" * 70)
    
    vol_delta = results[1]['pooled_r'] - baseline_r
    health_delta = results[2]['pooled_r'] - baseline_r
    both_delta = results[3]['pooled_r'] - baseline_r
    expected_additive = vol_delta + health_delta
    
    print(f"\n  Volatility improvement:    {vol_delta:+.3f}")
    print(f"  Env_health improvement:    {health_delta:+.3f}")
    print(f"  Expected if additive:      {expected_additive:+.3f}")
    print(f"  Actual (both):             {both_delta:+.3f}")
    print(f"  Difference:                {both_delta - expected_additive:+.3f}")
    
    if both_delta > expected_additive + 0.05:
        print("\n  → SUPER-ADDITIVE: Combined effect exceeds sum of parts")
        print("  → Hierarchy principle CONFIRMED")
    elif both_delta > expected_additive - 0.05:
        print("\n  → ADDITIVE: Combined effect equals sum of parts")
    else:
        print("\n  → SUB-ADDITIVE: Combined effect less than sum of parts")
    
    # Sign-flip analysis
    print("\n" + "=" * 70)
    print("SIGN-FLIP ANALYSIS")
    print("=" * 70)
    
    print(f"\n{'Configuration':<25} {'volatility':>12} {'M_ratio':>12}")
    print("-" * 50)
    
    for r in results:
        vol_flip = "FLIP" if r['sign_flips'].get('volatility') else "no"
        M_flip = "FLIP" if r['sign_flips'].get('M_ratio') else "no"
        print(f"{r['name']:<25} {vol_flip:>12} {M_flip:>12}")
    
    # Hierarchy principle
    print("\n" + "=" * 70)
    print("HIERARCHY PRINCIPLE TEST")
    print("=" * 70)
    
    if health_delta < 0:
        print("\n  ✓ Env_health alone is HARMFUL (Δ < 0)")
    else:
        print("\n  × Env_health alone is NOT harmful")
    
    if vol_delta > 0.5:
        print("  ✓ Volatility alone achieves coherence (Δ > 0.5)")
    else:
        print("  × Volatility alone does NOT achieve coherence")
    
    if both_delta > vol_delta:
        print("  ✓ Both together exceeds volatility alone")
        print("\n  → Secondary (env_health) becomes beneficial with primary context")
    else:
        print("  × Both together does NOT exceed volatility alone")


if __name__ == "__main__":
    main()
