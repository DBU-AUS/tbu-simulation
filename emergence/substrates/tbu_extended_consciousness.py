#!/usr/bin/env python3
"""
================================================================================
TBU EXTENDED CONSCIOUSNESS SUBSTRATE (Level 9)
================================================================================

Full architecture with:
- t-tangent perception (regime identification)
- Secondary channels (health, resources) 
- Redundancy (second perception channel)
- Action-perception coupling

This is the complete Level 9 substrate combining all validated components.

Results (5-seed validation, Appendix T):
  - Coherence with perception: +0.948 ± 0.023
  - Sign-flip eliminated: 0/5 seeds
  - Rescue under 25% noise: +1.04

================================================================================
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from tbu_honest_boundary import HonestBoundarySubstrate


class ExtendedConsciousnessSubstrate(HonestBoundarySubstrate):
    """
    Level 9: Full architecture with primary, secondary, and redundancy channels.
    
    Channel layout:
      0-5: Base channels from probe
      6: t-tangent (primary - regime identification)
      7: Secondary channel 1 (health)
      8: Secondary channel 2 (resources)  
      9: Redundancy channel (regime-aligned)
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Configuration flags
        self.add_t_tangent = getattr(self, '_add_t_tangent', True)
        self.add_secondary = getattr(self, '_add_secondary', True)
        self.add_redundancy = getattr(self, '_add_redundancy', True)
        
        # Count extra channels
        extra = 0
        if self.add_t_tangent:
            extra += 1
        if self.add_secondary:
            extra += 2  # health + resources
        if self.add_redundancy:
            extra += 1
        
        if extra > 0:
            self.n_channels += extra
            old_attention = self.attention.copy()
            self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
            self.attention[:, :old_attention.shape[1]] = old_attention * (old_attention.shape[1] / self.n_channels)
        
        # History for t-tangent computation
        self.boundary_history_for_tangent: List[np.ndarray] = []
        
        # Track channel indices
        self.channel_names = ['probe_' + str(i) for i in range(6)]
        idx = 6
        if self.add_t_tangent:
            self.channel_names.append('t_tangent')
            self.t_tangent_idx = idx
            idx += 1
        if self.add_secondary:
            self.channel_names.append('health')
            self.health_idx = idx
            idx += 1
            self.channel_names.append('resources')
            self.resources_idx = idx
            idx += 1
        if self.add_redundancy:
            self.channel_names.append('redundancy')
            self.redundancy_idx = idx
            idx += 1
        
        config = []
        if self.add_t_tangent:
            config.append("t-tangent")
        if self.add_secondary:
            config.append("secondary(health,resources)")
        if self.add_redundancy:
            config.append("redundancy")
        print(f"  Extended consciousness: {', '.join(config)}")
        print(f"  Total channels: {self.n_channels}")
    
    def _compute_t_tangent(self) -> float:
        """Compute t-tangent from boundary rate of change."""
        if len(self.boundary_history_for_tangent) < 2:
            return 0.0
        current = self.boundary_history_for_tangent[-1]
        previous = self.boundary_history_for_tangent[-2]
        return float(np.tanh(np.mean(np.abs(current - previous)) * 50))
    
    def _get_health_channel(self) -> float:
        """Get current health from environment."""
        return self.environment.get_health_signal()['health']
    
    def _get_resources_channel(self) -> float:
        """Get current resources from environment."""
        return self.environment.get_health_signal()['resource_level']
    
    def _get_redundancy_channel(self) -> float:
        """
        Redundancy channel: regime-aligned information.
        Correlated with t-tangent but from different measurement.
        """
        # Use action stability as redundant regime indicator
        if len(self.action_history) < 10:
            return 0.5
        recent = np.array(self.action_history[-10:])
        var = np.var(recent)
        return float(1.0 / (1.0 + var * 100))
    
    def step_physics(self):
        """Full Level 9 physics step."""
        
        # Store boundary for t-tangent
        boundary_pre = self.grid[self.forcing_coords].copy()
        self.boundary_history_for_tangent.append(boundary_pre)
        if len(self.boundary_history_for_tangent) > 5:
            self.boundary_history_for_tangent = self.boundary_history_for_tangent[-5:]
        
        # Standard physics
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        lap = up + dn + lf + rt - 4.0 * self.grid
        
        self.grid += self.diffusion * lap
        self.grid *= 1.0 - self.damping
        
        # Actions track dy
        boundary_state_pre = self.grid[self.forcing_coords]
        dy_action = boundary_state_pre - self.prev_boundary_for_action
        self.prev_boundary_for_action = boundary_state_pre.copy()
        
        if self.action_gain_scale > 0:
            self.actions += self.action_smoothing * (dy_action - self.actions)
        
        # Actions affect environment
        action_magnitude = float(np.mean(np.abs(self.actions)))
        self.environment.consume(action_magnitude * self.action_consumption_scale)
        boundary_variation = float(np.mean(np.abs(dy_action)))
        self.environment.disturb(boundary_variation * self.action_disturbance_scale)
        self.environment.step()
        
        # Read base channels
        base_channels = self.probe.read_channels(self.n_forcing)
        
        # Build extended channels
        extras = []
        
        if self.add_t_tangent:
            t_tangent = self._compute_t_tangent()
            ch = np.ones(self.n_forcing) * (t_tangent - 0.5) * 0.6
            ch += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch.reshape(-1, 1))
        
        if self.add_secondary:
            # Health channel
            health = self._get_health_channel()
            ch_h = np.ones(self.n_forcing) * (health - 0.5) * 0.6
            ch_h += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch_h.reshape(-1, 1))
            
            # Resources channel
            resources = self._get_resources_channel()
            ch_r = np.ones(self.n_forcing) * (resources - 0.5) * 0.6
            ch_r += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch_r.reshape(-1, 1))
        
        if self.add_redundancy:
            redundancy = self._get_redundancy_channel()
            ch = np.ones(self.n_forcing) * (redundancy - 0.5) * 0.6
            ch += self.rng.normal(0, 0.02, self.n_forcing)
            extras.append(ch.reshape(-1, 1))
        
        if extras:
            channels = np.column_stack([base_channels] + extras)
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
        
        # Self-model update
        self._update_self_models()
        
        # Attention update
        self._update_attention_extended(channels)
        self.prev_channels = channels.copy()
        
        # Record history
        self.history.append(self.grid.copy())
        if len(self.history) > self.history_window:
            self.history.pop(0)
        
        self.action_history.append(self.actions.copy())
        if len(self.action_history) > self.history_window:
            self.action_history.pop(0)
        
        self.step += 1
    
    def _update_attention_extended(self, channels: np.ndarray):
        """Attention update for extended channels."""
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
    
    def report(self) -> Dict[str, Any]:
        """Extended report with channel-specific attention."""
        base_report = super().report()
        
        mean_attention = self.attention.mean(axis=0)
        for i, name in enumerate(self.channel_names):
            if i < len(mean_attention):
                base_report[f'attention_{name}'] = float(mean_attention[i])
        
        return base_report


def create_configured_substrate(
    t_tangent: bool = True,
    secondary: bool = True,
    redundancy: bool = True,
    seed: int = 42
) -> ExtendedConsciousnessSubstrate:
    """Factory function to create configured substrate."""
    
    class Configured(ExtendedConsciousnessSubstrate):
        def __post_init__(self):
            self._add_t_tangent = t_tangent
            self._add_secondary = secondary
            self._add_redundancy = redundancy
            super().__post_init__()
    
    return Configured(seed=seed)


if __name__ == "__main__":
    print("=" * 70)
    print("EXTENDED CONSCIOUSNESS SUBSTRATE TEST")
    print("=" * 70)
    
    substrate = create_configured_substrate(seed=42)
    
    for step in range(1000):
        substrate.step_physics()
        if (step + 1) % 200 == 0:
            report = substrate.report()
            print(f"Step {step+1}: M_ratio={report['M_ratio']:.2f}")
    
    print("\nTest complete.")
