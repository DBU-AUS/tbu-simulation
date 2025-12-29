#!/usr/bin/env python3
"""
================================================================================
TBU VOLATILITY-AWARE SUBSTRATE
================================================================================

Extends HonestBoundarySubstrate with boundary volatility as a perception channel.

The insight from Appendix S: boundary_volatility is the conditioning variable
that reveals the sign-flip fingerprint. If we make it perceivable (not as an
objective, just as sensory input), the system can potentially learn to navigate
differently in different regimes.

This is NOT engineering an objective. We're opening a perceptual channel and
observing whether behavior naturally differentiates by regime.

Key addition:
  - Channel 6: Boundary volatility (recent |Δboundary| smoothed)
  
The system can now perceive which regime it's in. We observe whether:
  1. Attention shifts to this channel in different conditions
  2. Action strategy differs between high/low volatility periods
  3. The system implicitly "knows" the sign-flip structure

Results (Section R.6 of manuscript):
  - Sign-flip eliminated when volatility is perceptible (0/5 seeds)
  - Pooled correlation: 0.01 → 0.92
  - Null test confirms: random channel preserves sign-flip

================================================================================
"""

import sys
from typing import Dict, Any, List, Optional
import numpy as np

from tbu_honest_boundary import HonestBoundarySubstrate


class VolatilityAwareSubstrate(HonestBoundarySubstrate):
    """
    Level 8+ substrate with boundary volatility perception.
    
    Single addition: Channel 6 reports recent boundary volatility.
    No objectives, no rewards - just perception.
    """
    
    def __post_init__(self):
        # Call parent initialization
        super().__post_init__()
        
        # Add volatility channel
        self.n_channels += 1  # Now 7 channels
        
        # Expand attention to include new channel
        # Initialize with equal weight to existing channels
        old_attention = self.attention.copy()
        self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
        self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
        
        # Volatility tracking
        self.boundary_history_for_volatility: List[np.ndarray] = []
        self.volatility_window = 20  # Steps to compute volatility over
        self.current_volatility = 0.0
        
        print(f"Volatility perception: Channel 6 added (window={self.volatility_window})")
        print(f"Total channels: {self.n_channels}")
    
    def _compute_boundary_volatility(self) -> float:
        """
        Compute recent boundary volatility from history.
        Returns normalized [0, 1] volatility estimate.
        """
        if len(self.boundary_history_for_volatility) < 2:
            return 0.0
        
        # Use recent history
        recent = self.boundary_history_for_volatility[-self.volatility_window:]
        if len(recent) < 2:
            return 0.0
        
        # Compute mean absolute differences
        diffs = [np.mean(np.abs(recent[i] - recent[i-1])) 
                 for i in range(1, len(recent))]
        
        raw_volatility = np.mean(diffs)
        
        # Normalize to roughly [0, 1] using empirical scaling
        # (based on typical boundary_std range ~0.05-0.15)
        normalized = np.tanh(raw_volatility * 50)  # Saturates around 0.02
        
        return float(normalized)
    
    def step_physics(self):
        """
        Extended step with volatility channel.
        """
        # --- Store boundary for volatility computation ---
        boundary_pre = self.grid[self.forcing_coords].copy()
        self.boundary_history_for_volatility.append(boundary_pre)
        if len(self.boundary_history_for_volatility) > self.volatility_window + 10:
            self.boundary_history_for_volatility = self.boundary_history_for_volatility[-self.volatility_window:]
        
        # --- Compute current volatility ---
        self.current_volatility = self._compute_boundary_volatility()
        
        # --- Standard physics (diffusion, damping) ---
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        lap = up + dn + lf + rt - 4.0 * self.grid
        
        self.grid += self.diffusion * lap
        self.grid *= 1.0 - self.damping
        
        # --- LEVEL 7: actions track dy ---
        boundary_state_pre = self.grid[self.forcing_coords]
        dy_action = boundary_state_pre - self.prev_boundary_for_action
        self.prev_boundary_for_action = boundary_state_pre.copy()
        
        if self.action_gain_scale > 0:
            self.actions += self.action_smoothing * (dy_action - self.actions)
        
        # --- LEVEL 8: actions affect environment ---
        action_magnitude = float(np.mean(np.abs(self.actions)))
        self.environment.consume(action_magnitude * self.action_consumption_scale)
        
        boundary_variation = float(np.mean(np.abs(dy_action)))
        self.environment.disturb(boundary_variation * self.action_disturbance_scale)
        
        self.environment.step()
        
        # --- Read base channels (0-5) ---
        base_channels = self.probe.read_channels(self.n_forcing)
        
        # --- Add volatility channel (6) ---
        # Broadcast volatility to all forcing pixels with small spatial noise
        vol_channel = np.ones(self.n_forcing) * (self.current_volatility - 0.5) * 0.6
        vol_channel += self.rng.normal(0, 0.02, self.n_forcing)
        
        # Combine into full channel array
        channels = np.column_stack([base_channels, vol_channel])
        
        # --- Attention-weighted input ---
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
        
        # --- Saturation ---
        self.grid = np.tanh(self.grid)
        
        # --- Level 5: self-model update ---
        self._update_self_models()
        
        # --- Level 6: attention update (now 7 channels) ---
        self._update_attention_with_volatility(channels)
        self.prev_channels = channels.copy()
        
        # --- Log time-series ---
        if len(self.history) >= 20:
            M = self.measure_M()
            core_mask = (M >= np.median(M[~self.forcing_mask])) & (~self.forcing_mask)
            core_mean = float(self.grid[core_mask].mean()) if core_mask.any() else 0.0
            self.core_mean_series.append(core_mean)
            self.action_mean_series.append(float(np.mean(np.abs(self.actions))))
            if len(self.core_mean_series) > self.history_window * 2:
                self.core_mean_series = self.core_mean_series[-self.history_window:]
                self.action_mean_series = self.action_mean_series[-self.history_window:]
        
        # --- Record state ---
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
    
    def _update_attention_with_volatility(self, channels: np.ndarray):
        """
        Update attention weights including volatility channel.
        Same predictive learning rule as Level 6, extended to 7 channels.
        """
        eta = self.attention_learning_rate
        
        # Current boundary state
        boundary_now = self.grid[self.forcing_coords]
        dy = boundary_now - self.prev_boundary
        self.prev_boundary = boundary_now.copy()
        
        if self.step < 1:
            return
        
        if self.prev_channels is None or self.prev_channels.shape[1] != channels.shape[1]:
            return
        
        # Hebbian-style update: channels that predict dy get more attention
        prediction = np.sum(self.attention * self.prev_channels, axis=1)
        error = dy - prediction
        self.attention += eta * (error[:, None] * self.prev_channels)
        
        # Ensure valid distribution
        self.attention = np.clip(self.attention, 1e-3, None)
        self.attention /= self.attention.sum(axis=1, keepdims=True)
    
    def report(self) -> Dict[str, Any]:
        """Extended report including volatility channel attention."""
        base_report = super().report()
        
        # Add volatility-specific metrics
        mean_attention = self.attention.mean(axis=0)
        base_report['attention_volatility'] = float(mean_attention[6]) if len(mean_attention) > 6 else 0.0
        base_report['current_volatility'] = float(self.current_volatility)
        
        return base_report


if __name__ == "__main__":
    print("=" * 70)
    print("VOLATILITY-AWARE SUBSTRATE TEST")
    print("=" * 70)
    
    substrate = VolatilityAwareSubstrate(seed=42)
    
    for step in range(1000):
        substrate.step_physics()
        if (step + 1) % 200 == 0:
            report = substrate.report()
            print(f"Step {step+1}: volatility={substrate.current_volatility:.3f}, "
                  f"attn_vol={report['attention_volatility']:.3f}")
    
    print("\nTest complete.")
