#!/usr/bin/env python3
"""
================================================================================
TBU RANDOM CHANNEL SUBSTRATE (NULL TEST CONTROL)
================================================================================

Same as VolatilityAwareSubstrate but Channel 6 is random noise
instead of volatility information.

If sign-flip persists with random channel but disappears with
volatility channel, this confirms it's the INFORMATION that matters,
not just adding an extra channel.

Results (Section R.6 of manuscript):
  - Baseline (6 channels): Sign-flip YES, r_pooled ≈ 0
  - Volatility-Aware (7 channels, has info): Sign-flip NO, r_pooled = +0.92
  - Random Channel (7 channels, NO info): Sign-flip YES, r_pooled ≈ 0

Conclusion: It's the information that matters. Adding volatility perception
enables coherence. Adding noise does not.

================================================================================
"""

from typing import Dict, Any, List
import numpy as np

from tbu_honest_boundary import HonestBoundarySubstrate


class RandomChannelSubstrate(HonestBoundarySubstrate):
    """
    Level 8 substrate with RANDOM channel (no regime information).
    
    Identical to VolatilityAwareSubstrate except Channel 6 is noise.
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Add random channel (same structure as volatility-aware)
        self.n_channels += 1  # Now 7 channels
        
        old_attention = self.attention.copy()
        self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
        self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
        
        print(f"Random channel: Channel 6 added (NOISE - no regime info)")
        print(f"Total channels: {self.n_channels}")
    
    def step_physics(self):
        """Extended step with random channel instead of volatility."""
        
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
        
        # --- RANDOM channel (6) - NO regime information ---
        # Just noise, same scale as volatility channel would have
        random_channel = self.rng.normal(0, 0.3, self.n_forcing)
        
        # Combine into full channel array
        channels = np.column_stack([base_channels, random_channel])
        
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
        self._update_attention_random(channels)
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
    
    def _update_attention_random(self, channels: np.ndarray):
        """Same attention update as volatility-aware."""
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


if __name__ == "__main__":
    print("=" * 70)
    print("RANDOM CHANNEL SUBSTRATE TEST (NULL CONTROL)")
    print("=" * 70)
    
    substrate = RandomChannelSubstrate(seed=42)
    
    for step in range(1000):
        substrate.step_physics()
        if (step + 1) % 200 == 0:
            report = substrate.report()
            print(f"Step {step+1}: M_ratio={report['M_ratio']:.2f}")
    
    print("\nTest complete.")
