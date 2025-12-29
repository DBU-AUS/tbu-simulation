#!/usr/bin/env python3
"""
================================================================================
TBU HONEST BOUNDARY - Level 8: Environmental Health Sensing
================================================================================

Builds on tbu_honest_action.py (Levels 1-7) by adding environmental sensing.

BASE PHYSICS (grid dynamics unchanged):
  - Torus diffusion + damping + saturation
  - Forcing at B via attention-weighted channel mixture
  - M is diagnostic only (never fed back)

LEVELS 5-7 (unchanged):
  - Self-model: all pixels track neighbor mean (stability-modulated rate)
  - Attention: boundary pixels learn via prediction error minimization
  - Action: smoothed boundary dy gates forcing gain (symmetric, sign-free)

LEVEL 8 (minimal extension - opening channels):
  - Environment: external resource/stability field the system couples to
  - Geometric coupling: configurations with large |a| associate with lower E
    * consumption: proportional to action magnitude
    * disturbance: proportional to |dy| (pre-forcing boundary variation)
  - Health channels: GLOBAL field E sampled at boundary (not local sensing)
  - Attention can learn health channel weights by predictive value only
  
LOAD-BEARING COUPLING (makes E matter to optimizer):
  - Low environment health → increased observation noise on predictable channels
  - This is NOT a "prefer healthy" rule — it's physics: degraded E = degraded sensing
  - The optimizer will discover that maintaining E keeps the boundary predictable
  - obs_sigma = env_noise_floor + env_noise_scale * (1 - health)
  
TBU FRAMING:
  - "Steps" are solver indices, not time (gauge choice)
  - Environment field E lives alongside boundary field
  - Health channels open information about E to attention learning
  - Correlations are across solver-indexed slices of the block

WHAT IS DESIGNED vs WHAT EMERGES:
  - DESIGNED: Environment fields, action-environment coupling, health→noise coupling
  - EMERGENT: If sustainable patterns arise, it's because the optimizer discovers
    that maintaining E keeps the boundary predictable (not because of explicit rule)

NO:
  - M-feedback into dynamics
  - Explicit "prefer healthy environment" rule
  - Constraints on action based on environment state
  - Any modification to base physics

================================================================================
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from collections import deque

import numpy as np


# -----------------------------------------------------------------------------
# Utilities (same as previous levels)
# -----------------------------------------------------------------------------

def compute_ring_cardinality(size: int, width: int) -> int:
    """Compute natural |B| for a ring of given width on a size x size torus."""
    mask = np.zeros((size, size), dtype=bool)
    w = int(width)
    mask[0:w, :] = True
    mask[-w:, :] = True
    mask[:, 0:w] = True
    mask[:, -w:] = True
    return int(mask.sum())


def sample_from_region(region_mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample exactly n pixels uniformly from within a region."""
    eligible = np.flatnonzero(region_mask.ravel())
    if len(eligible) < n:
        raise RuntimeError(f"Region has {len(eligible)} pixels but need {n}")

    chosen = rng.choice(eligible, size=n, replace=False)
    result = np.zeros_like(region_mask, dtype=bool)
    ys, xs = np.unravel_index(chosen, region_mask.shape)
    result[ys, xs] = True
    return result


def largest_connected_component_fraction(mask: np.ndarray):
    """LCC fraction (4-neighbor, periodic). Returns (fraction, n_components)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    total = int(mask.sum())
    if total == 0:
        return float("nan"), 0

    best = 0
    n_components = 0

    for i in range(H):
        for j in range(W):
            if not mask[i, j] or seen[i, j]:
                continue

            q = deque([(i, j)])
            seen[i, j] = True
            size = 0

            while q:
                y, x = q.popleft()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = (y + dy) % H, (x + dx) % W
                    if mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))

            n_components += 1
            best = max(best, size)

    return best / total, n_components


# -----------------------------------------------------------------------------
# Environment (Level 8 addition)
# -----------------------------------------------------------------------------

@dataclass
class Environment:
    """
    External resource/stability pool that the system depends on.
    
    This represents the "substrate" or "Country" — the larger system
    within which the consciousness exists. The system's actions affect
    the environment, and the environment's health affects available signals.
    
    NO M-feedback: environment state is purely external.
    """
    
    # Resource pool (like energy/compute available)
    resources: float = 100.0
    max_resources: float = 100.0
    
    # Stability (like thermal/process stability)
    stability: float = 1.0
    
    # Natural regeneration rate (when not over-exploited)
    regen_rate: float = 0.1  # Lower so environment is more dynamic
    
    # History for trends
    resource_history: List[float] = field(default_factory=list)
    stability_history: List[float] = field(default_factory=list)
    
    def consume(self, amount: float):
        """Actions consume resources (amount determined by action magnitude)."""
        self.resources = max(0, self.resources - amount)
    
    def disturb(self, amount: float):
        """Boundary variation disturbs stability (lower-bounded at 0.1 for numerical stability)."""
        self.stability = max(0.1, self.stability - amount)
    
    def regenerate(self):
        """Environment naturally regenerates (proportional to current stability)."""
        # Resources regenerate proportional to stability
        regen = self.regen_rate * self.stability
        self.resources = min(self.max_resources, self.resources + regen)
        
        # Stability recovers slowly
        self.stability = min(1.0, self.stability + 0.02)
    
    def step(self):
        """One solver iteration: regenerate and record history."""
        self.regenerate()
        self.resource_history.append(self.resources)
        self.stability_history.append(self.stability)
        
        # Trim history
        if len(self.resource_history) > 200:
            self.resource_history = self.resource_history[-100:]
            self.stability_history = self.stability_history[-100:]
    
    def get_health_signal(self) -> Dict[str, float]:
        """
        Return signals about environment health.
        
        These become additional input channels for the boundary.
        
        TBU framing: These are field values at this slice of the geometry,
        not "current state" in a temporal sense.
        """
        # Current levels (normalized 0-1)
        resource_level = self.resources / self.max_resources
        stability_level = self.stability
        
        return {
            'resource_level': resource_level,
            'stability_level': stability_level,
            'health': resource_level * stability_level,  # combined health [0,1]
        }


# -----------------------------------------------------------------------------
# Multi-channel environment with health signals (Level 8 extension)
# -----------------------------------------------------------------------------

class MultiChannelProbeWithHealth:
    """
    Extended probe with environment health as additional channels.
    
    Channels 0-3: Same as before (oscillations, entropy, system-coupled)
    Channels 4-5: Environment health signals (resource level, stability)
    
    LOAD-BEARING COUPLING (makes E matter to optimizer):
    Low environment health increases observation noise on predictable channels.
    This is NOT a "prefer healthy" rule — it's physics: degraded environment
    = degraded sensing quality. The optimizer will discover that maintaining
    E keeps the boundary predictable.
    
    Control ablation: set health_enabled=False to zero health channels
    (shows any health weighting is learned, not hard-coded).
    """
    
    def __init__(self, n_base_channels: int = 4, environment: Optional[Environment] = None,
                 seed: Optional[int] = None, health_enabled: bool = True,
                 env_noise_scale: float = 0.25, env_noise_floor: float = 0.02,
                 env_noise_enabled: bool = True):
        self.n_base_channels = n_base_channels
        self.n_health_channels = 2  # resource, stability
        self.n_channels = n_base_channels + self.n_health_channels
        self.environment = environment
        self.health_enabled = health_enabled  # False = control ablation
        self.env_noise_scale = float(env_noise_scale)
        self.env_noise_floor = float(env_noise_floor)
        self.env_noise_enabled = bool(env_noise_enabled)  # False = remove ALL E-sensing
        self.solver_index = 0  # Not "time" - solver iteration index
        self.rng = np.random.default_rng(seed)
    
    def read_channels(self, n: int) -> np.ndarray:
        """Read n samples from all channels. Returns: (n, n_channels) array"""
        self.solver_index += 1
        t = self.solver_index  # Local alias for formulae
        
        # --- LOAD-BEARING: environment-dependent observation noise ---
        # Low health → high noise → harder to predict → optimizer "cares" about E
        # When env_noise_enabled=False, health is treated as 1.0 (no E-sensing via noise)
        if (self.environment is not None) and self.env_noise_enabled:
            h = float(self.environment.get_health_signal()["health"])  # [0,1]
        else:
            h = 1.0  # No E-sensing: assume perfect health (no noise modulation)
        # When health is low, noise goes up. When health is high, noise is near floor.
        # Clamp to sane max to prevent extreme parameter choices from swamping signal
        obs_sigma = self.env_noise_floor + self.env_noise_scale * (1.0 - h)
        obs_sigma = min(obs_sigma, 1.0)
        
        channels = []
        
        # --- BASE CHANNELS (now with env-dependent noise on predictable ones) ---
        
        # Channel 0: Time oscillation (predictable, but corrupted by env-dependent noise)
        ch0 = np.array([np.sin(t * 0.3 + i * 0.1) * 0.3 for i in range(n)])
        ch0 += self.rng.normal(0, obs_sigma, n)
        channels.append(ch0)
        
        # Channel 1: Hardware entropy (unpredictable - unchanged)
        raw = os.urandom(4 * n)
        u = np.frombuffer(raw, dtype=np.uint32).astype(np.float64)
        ch1 = (u / (2**32) - 0.5) * 2.0
        channels.append(ch1)
        
        # Channel 2: System-coupled (moderate, also gets env-dependent noise)
        base = np.sin(t * 0.05) * 0.2
        noise = self.rng.normal(0, 0.1 + obs_sigma, n)
        ch2 = base + noise
        channels.append(ch2)
        
        # Channel 3: Slow oscillation (predictable, corrupted by env-dependent noise)
        ch3 = np.array([np.sin(t * 0.1 + i * 0.05) * 0.2 for i in range(n)])
        ch3 += self.rng.normal(0, obs_sigma, n)
        channels.append(ch3)
        
        # --- HEALTH CHANNELS (Level 8 addition) ---
        # Health is a GLOBAL field sampled at the boundary (not spatially resolved).
        # Every boundary pixel sees the same E value (plus small noise for numerical reasons).
        # This is "opening a channel" to environment information, not local sensing.
        
        if self.environment is not None and self.health_enabled:
            health = self.environment.get_health_signal()
            
            # Channel 4: Resource level (global, broadcast with spatial noise)
            ch4 = np.ones(n) * (health['resource_level'] - 0.5) * 0.6
            ch4 += self.rng.normal(0, 0.02, n)
            channels.append(ch4)
            
            # Channel 5: Stability level (global, broadcast with spatial noise)
            ch5 = np.ones(n) * (health['stability_level'] - 0.5) * 0.6
            ch5 += self.rng.normal(0, 0.02, n)
            channels.append(ch5)
        else:
            # No environment or control ablation: health channels are zero
            channels.append(np.zeros(n))
            channels.append(np.zeros(n))
        
        return np.stack(channels, axis=1)


# -----------------------------------------------------------------------------
# Boundary Substrate (Level 8)
# -----------------------------------------------------------------------------

@dataclass
class HonestBoundarySubstrate:
    """
    Level 8: Environmental health sensing on honest action foundation.
    
    BASE + LEVELS 5-7:
      - Torus diffusion + damping + forcing via attention + saturation
      - Self-model tracking, attention via prediction error learning
      - Action output with symmetric gain gating
    
    LEVEL 8 EXTENSION (minimal):
      - Environment: external resource/stability pool
      - Actions consume resources and disturb stability
      - Health signals as additional input channels (4→6 total)
      - Attention can learn to weight health channels
    
    If sustainable behavior emerges, it's because attention discovers
    that health-correlated channels help predict boundary dynamics.
    """
    
    size: int = 64
    
    # Forcing geometry
    forcing_shape: str = "ring"
    forcing_n_pixels: Optional[int] = None
    forcing_ring_width: int = 2
    
    # Base physics
    diffusion: float = 0.18
    damping: float = 0.001
    forcing_strength: float = 0.12
    
    # Self-model parameters (Level 5)
    self_model_rate: float = 0.1
    self_model_stability_scale: float = 10.0
    
    # Attention parameters (Level 6, now with 6 channels)
    n_base_channels: int = 4
    n_health_channels: int = 2
    attention_learning_rate: float = 0.05
    
    # Action parameters (Level 7)
    action_smoothing: float = 0.1
    action_gain_scale: float = 0.5
    
    # Environment parameters (Level 8)
    # Note: action magnitude is small (~0.003), |dy| is larger (~0.006)
    action_consumption_scale: float = 30.0   # how much actions consume resources
    action_disturbance_scale: float = 1.0    # how much |dy| disturbs stability (reduced)
    health_channels_enabled: bool = True     # False = control ablation (zero health channels)
    
    # Environment → observation quality coupling (NO reward, just physics of sensing)
    # Low health increases channel noise, making prediction harder
    env_noise_scale: float = 0.25   # extra channel noise when env_health is low
    env_noise_floor: float = 0.02   # baseline noise always present
    env_noise_enabled: bool = True  # False = remove env→noise coupling (no E-sensing)
    
    # Diagnostics
    history_window: int = 100
    history_stride: int = 1
    stability_quantile: float = 0.3
    M_clip_min: float = 0.01
    M_clip_max: float = 100.0
    
    # Housekeeping
    report_interval: int = 200
    seed: Optional[int] = None
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        
        # Level 8: Environment
        self.environment = Environment()
        
        # Probe now includes health channels
        self.n_channels = self.n_base_channels + self.n_health_channels
        self.probe = MultiChannelProbeWithHealth(
            n_base_channels=self.n_base_channels,
            environment=self.environment,
            seed=self.seed,
            health_enabled=self.health_channels_enabled,
            env_noise_scale=self.env_noise_scale,
            env_noise_floor=self.env_noise_floor,
            env_noise_enabled=self.env_noise_enabled
        )
        
        if self.forcing_n_pixels is None:
            self.forcing_n_pixels = compute_ring_cardinality(self.size, self.forcing_ring_width)
        
        # Grid
        self.grid = self.rng.uniform(-0.05, 0.05, (self.size, self.size))
        self.step = 0
        
        # Histories
        self.history: List[np.ndarray] = []
        self.self_model_history: List[np.ndarray] = []
        self.attention_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []
        self.health_history: List[float] = []
        
        # Forcing mask
        self._init_forcing_mask()
        self._update_forcing_coords()
        
        # Level 5: self-models
        self.self_models = self.rng.uniform(-0.05, 0.05, (self.size, self.size))
        
        # Level 6: attention (now 6 channels)
        self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
        self.prev_boundary = self.grid[self.forcing_coords].copy()
        self.prev_channels = self.probe.read_channels(self.n_forcing)
        
        # Level 7: actions
        self.actions = np.zeros(self.n_forcing, dtype=np.float64)
        self.prev_boundary_for_action = self.grid[self.forcing_coords].copy()
        
        # Time-series for coherence
        self.core_mean_series: List[float] = []
        self.action_mean_series: List[float] = []
        
        print(f"Boundary substrate: {self.size}x{self.size} grid")
        print(f"Forcing: {self.forcing_shape} (|B|={self.n_forcing})")
        print(f"Channels: {self.n_base_channels} base + {self.n_health_channels} health = {self.n_channels} total")
        print(f"Environment: resources={self.environment.resources}, stability={self.environment.stability}")
        print(f"E-sensing: health_channels={self.health_channels_enabled}, env_noise={self.env_noise_enabled}")
        if self.env_noise_enabled:
            print(f"Noise coupling: σ = {self.env_noise_floor} + {self.env_noise_scale}*(1-health)")
    
    def _init_forcing_mask(self):
        self.forcing_mask = np.zeros((self.size, self.size), dtype=bool)
        
        if self.forcing_shape == "ring":
            w = int(self.forcing_ring_width)
            self.forcing_mask[0:w, :] = True
            self.forcing_mask[-w:, :] = True
            self.forcing_mask[:, 0:w] = True
            self.forcing_mask[:, -w:] = True
        
        elif self.forcing_shape == "random":
            idx = self.rng.choice(self.size * self.size, size=self.forcing_n_pixels, replace=False)
            ys, xs = np.unravel_index(idx, (self.size, self.size))
            self.forcing_mask[ys, xs] = True
        
        elif self.forcing_shape == "patch":
            side = int(np.ceil(np.sqrt(self.forcing_n_pixels * 1.5)))
            side = min(side, self.size)
            region = np.zeros((self.size, self.size), dtype=bool)
            region[0:side, 0:side] = True
            self.forcing_mask = sample_from_region(region, self.forcing_n_pixels, self.rng)
        
        elif self.forcing_shape == "corners":
            n_per = max(1, self.forcing_n_pixels // 4)
            side = int(np.ceil(np.sqrt(n_per * 1.5)))
            side = min(side, self.size // 2)
            region = np.zeros((self.size, self.size), dtype=bool)
            region[0:side, 0:side] = True
            region[0:side, -side:] = True
            region[-side:, 0:side] = True
            region[-side:, -side:] = True
            self.forcing_mask = sample_from_region(region, self.forcing_n_pixels, self.rng)
    
    def _update_forcing_coords(self):
        self.forcing_coords = np.where(self.forcing_mask)
        self.n_forcing = int(self.forcing_mask.sum())
    
    def _maybe_store_history(self):
        if self.history_stride <= 1 or (self.step % self.history_stride == 0):
            self.history.append(self.grid.copy())
            self.self_model_history.append(self.self_models.copy())
            self.attention_history.append(self.attention.copy())
            self.action_history.append(self.actions.copy())
            self.health_history.append(self.environment.get_health_signal()['health'])
            
            max_len = self.history_window * 2
            if len(self.history) > max_len:
                self.history = self.history[-self.history_window:]
                self.self_model_history = self.self_model_history[-self.history_window:]
                self.attention_history = self.attention_history[-self.history_window:]
                self.action_history = self.action_history[-self.history_window:]
                self.health_history = self.health_history[-self.history_window:]
    
    # =========================================================================
    # Diagnostics (never fed back)
    # =========================================================================
    
    def measure_M(self) -> np.ndarray:
        if len(self.history) < 20:
            return np.ones((self.size, self.size), dtype=np.float64)
        H = np.array(self.history[-self.history_window:], dtype=np.float64)
        var = np.var(H, axis=0) + 1e-12
        M = np.median(var) / var
        return np.clip(M, self.M_clip_min, self.M_clip_max)
    
    def measure_self_model_M(self) -> np.ndarray:
        if len(self.self_model_history) < 20:
            return np.ones((self.size, self.size), dtype=np.float64)
        H = np.array(self.self_model_history[-self.history_window:], dtype=np.float64)
        var = np.var(H, axis=0) + 1e-12
        M = np.median(var) / var
        return np.clip(M, self.M_clip_min, self.M_clip_max)
    
    def measure_action_stability(self) -> np.ndarray:
        if len(self.action_history) < 20:
            return np.ones(self.n_forcing, dtype=np.float64)
        H = np.array(self.action_history[-self.history_window:], dtype=np.float64)
        var = np.var(H, axis=0) + 1e-12
        stability = np.median(var) / var
        return np.clip(stability, self.M_clip_min, self.M_clip_max)
    
    def stable_mask(self) -> np.ndarray:
        if len(self.history) < 20:
            return np.zeros((self.size, self.size), dtype=bool)
        H = np.array(self.history[-self.history_window:], dtype=np.float64)
        var = np.var(H, axis=0)
        thresh = np.quantile(var, self.stability_quantile)
        return var <= thresh
    
    # =========================================================================
    # Physics step
    # =========================================================================
    
    def step_physics(self) -> None:
        """
        One solver iteration.
        
        BASE:
          - diffusion, damping, saturation
          - forcing via attention-weighted channels (6 total)
        
        LEVEL 7:
          - action = smoothed boundary change dy
          - gain = 1 + scale * (tanh(|action|) - spatial_mean)
        
        LEVEL 8:
          - action magnitude affects environment (consumption)
          - pre-forcing |dy| affects environment (disturbance)
          - environment provides health channels (4-5)
          - environment regenerates
        """
        
        # --- Diffusion on torus
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        lap = up + dn + lf + rt - 4.0 * self.grid
        
        self.grid += self.diffusion * lap
        self.grid *= 1.0 - self.damping
        
        # --- LEVEL 7: actions track dy
        boundary_state_pre = self.grid[self.forcing_coords]
        dy_action = boundary_state_pre - self.prev_boundary_for_action
        self.prev_boundary_for_action = boundary_state_pre.copy()
        
        if self.action_gain_scale > 0:
            self.actions += self.action_smoothing * (dy_action - self.actions)
        
        # --- LEVEL 8: actions affect environment (geometric coupling)
        # Action magnitude determines resource consumption
        action_magnitude = float(np.mean(np.abs(self.actions)))
        self.environment.consume(action_magnitude * self.action_consumption_scale)
        
        # Boundary variation (pre-forcing dy) disturbs stability
        # This couples environment to actual boundary dynamics, not action filter
        boundary_variation = float(np.mean(np.abs(dy_action)))
        self.environment.disturb(boundary_variation * self.action_disturbance_scale)
        
        # Environment regenerates
        self.environment.step()
        
        # --- Forcing at B (attention-weighted, now with health channels)
        channels = self.probe.read_channels(self.n_forcing)
        weighted_input = np.sum(self.attention * channels, axis=1)
        
        # Gain gating (same as Level 7)
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
        
        # --- Saturation
        self.grid = np.tanh(self.grid)
        
        # --- Level 5: self-model update
        self._update_self_models()
        
        # --- Level 6: attention update (now 6 channels)
        self._update_attention_predictive()
        self.prev_channels = channels.copy()
        
        # --- Log time-series for coherence
        if len(self.history) >= 20:
            M = self.measure_M()
            core_mask = (M >= np.median(M[~self.forcing_mask])) & (~self.forcing_mask)
            core_mean = float(self.grid[core_mask].mean()) if core_mask.any() else 0.0
            self.core_mean_series.append(core_mean)
            self.action_mean_series.append(float(np.mean(np.abs(self.actions))))
            if len(self.core_mean_series) > self.history_window * 2:
                self.core_mean_series = self.core_mean_series[-self.history_window:]
                self.action_mean_series = self.action_mean_series[-self.history_window:]
        
        self._maybe_store_history()
        self.step += 1
    
    def _update_self_models(self):
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        neighbor_mean = (up + dn + lf + rt) / 4.0
        
        if len(self.self_model_history) >= 20:
            H = np.array(self.self_model_history[-30:], dtype=np.float64)
            sm_var = np.var(H, axis=0) + 1e-8
            stability = 1.0 / (1.0 + self.self_model_stability_scale * sm_var)
        else:
            stability = np.ones((self.size, self.size)) * 0.5
        
        rate = self.self_model_rate * (1.0 - 0.9 * stability)
        self.self_models += rate * (neighbor_mean - self.self_models)
    
    def _update_attention_predictive(self):
        eta = self.attention_learning_rate
        
        y = self.grid[self.forcing_coords]
        dy = y - self.prev_boundary
        self.prev_boundary = y.copy()
        
        if self.step < 1:
            return
        
        prediction = np.sum(self.attention * self.prev_channels, axis=1)
        error = dy - prediction
        self.attention += eta * (error[:, None] * self.prev_channels)
        
        self.attention = np.clip(self.attention, 1e-3, None)
        self.attention /= self.attention.sum(axis=1, keepdims=True)
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    def report(self) -> Dict[str, Any]:
        M = self.measure_M()
        stable = self.stable_mask()
        lcc, n_clusters = largest_connected_component_fraction(stable)
        
        forcing_M = float(M[self.forcing_mask].mean())
        non_forcing_M = float(M[~self.forcing_mask].mean())
        M_ratio = non_forcing_M / (forcing_M + 1e-12)
        
        sm_M = self.measure_self_model_M()
        high_M_mask = M > np.median(M)
        sm_stability_high_M = float(sm_M[high_M_mask].mean()) if high_M_mask.any() else 0.0
        sm_stability_low_M = float(sm_M[~high_M_mask].mean()) if (~high_M_mask).any() else 0.0
        
        mean_attention = self.attention.mean(axis=0)
        
        action_stability = self.measure_action_stability()
        mean_action = float(np.mean(self.actions))
        action_std = float(np.std(self.actions))
        mean_action_stability = float(np.mean(action_stability))
        
        # Environment health
        health = self.environment.get_health_signal()
        
        # Current observation noise level (respects env_noise_enabled)
        if self.env_noise_enabled:
            obs_sigma = self.probe.env_noise_floor + self.probe.env_noise_scale * (1.0 - health['health'])
        else:
            obs_sigma = self.probe.env_noise_floor  # Frozen at floor when E-sensing disabled
        
        return {
            "step": int(self.step),
            "grid_std": float(np.std(self.grid)),
            "M_ratio": float(M_ratio),
            "core_coherence": float(lcc),
            "n_clusters": int(n_clusters),
            "self_model_ratio": float(sm_stability_high_M / (sm_stability_low_M + 1e-8)),
            "attention_predictable": float(mean_attention[0] + mean_attention[3]),
            "attention_entropy": float(mean_attention[1]),
            "attention_health": float(mean_attention[4] + mean_attention[5]),  # Level 8
            "action_mean": float(mean_action),
            "action_std": float(action_std),
            "action_stability": float(mean_action_stability),
            "env_resources": float(health['resource_level']),
            "env_stability": float(health['stability_level']),
            "env_health": float(health['health']),
            "obs_sigma": float(obs_sigma),  # Load-bearing: env↓ → obs_sigma↑
        }
    
    def measure_action_core_coherence(self) -> Dict[str, Any]:
        """Action-core coherence (same as Level 7)."""
        if len(self.core_mean_series) < 30:
            return {"error": "Insufficient history"}
        
        a = np.array(self.action_mean_series[-self.history_window:], dtype=float)
        c = np.array(self.core_mean_series[-self.history_window:], dtype=float)
        
        if np.std(a) < 1e-12 or np.std(c) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a, c)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        
        action_stability = self.measure_action_stability()
        high_stability_fraction = float((action_stability > np.median(action_stability) * 2).mean())
        
        return {
            "action_core_corr": float(corr),
            "action_stability_mean": float(np.mean(action_stability)),
            "high_stability_fraction": float(high_stability_fraction),
        }
    
    def measure_sustainability(self) -> Dict[str, Any]:
        """
        Level 8 specific: Does the system develop sustainable patterns?
        
        This is NOT engineered — it's measured. If sustainable behavior emerges,
        it's because attention learning discovered health-predictability correlations.
        
        TBU framing: These are correlations across solver-indexed slices of the
        sampled block, not causal influences in time.
        
        Note: health_gradient is defined relative to solver index ordering (a gauge
        choice). The "sustainable" label is post-hoc and threshold-based.
        """
        if len(self.health_history) < 30:
            return {"error": "Insufficient history"}
        
        recent_health = self.health_history[-50:]
        
        # Health statistics (geometric: variance across block slices)
        health_mean = float(np.mean(recent_health))
        health_std = float(np.std(recent_health))
        
        # Health gradient across block (difference between early/late slices)
        if len(recent_health) >= 20:
            early = np.mean(recent_health[:10])
            late = np.mean(recent_health[-10:])
            health_gradient = late - early  # "trend" reframed as geometric gradient
        else:
            health_gradient = 0.0
        
        # Attention on health channels
        mean_attention = self.attention.mean(axis=0)
        health_attention = float(mean_attention[4] + mean_attention[5])
        
        # Correlation: health vs action magnitude (energy proxy)
        # Uses mean(|a|) per slice, not spatial variance
        if len(self.action_history) >= 30:
            action_magnitudes = [float(np.mean(np.abs(ah))) for ah in self.action_history[-50:]]
            n = min(len(recent_health), len(action_magnitudes))
            if n >= 20:
                h = recent_health[-n:]
                a = action_magnitudes[-n:]
                if np.std(h) > 1e-12 and np.std(a) > 1e-12:
                    health_action_corr = float(np.corrcoef(h, a)[0, 1])
                    if np.isnan(health_action_corr):
                        health_action_corr = 0.0
                else:
                    health_action_corr = 0.0
            else:
                health_action_corr = 0.0
        else:
            health_action_corr = 0.0
        
        return {
            "health_mean": float(health_mean),
            "health_gradient": float(health_gradient),  # renamed from "trend"
            "health_std": float(health_std),
            "health_attention": float(health_attention),
            "health_action_corr": float(health_action_corr),
            # Post-hoc label (not a core metric)
            "sustainable": bool(health_mean > 0.5 and health_gradient >= -0.1),
        }


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_experiment(n_steps: int = 3000, seed: Optional[int] = None, 
                   health_enabled: bool = True) -> Dict[str, Any]:
    print("=" * 70)
    print("TBU HONEST BOUNDARY - Level 8")
    print("=" * 70)
    print()
    print("BASE (from honest action):")
    print("  - Torus diffusion + damping + saturation")
    print("  - Forcing via attention-weighted channels")
    print("  - M is diagnostic only, never fed back")
    print()
    print("LEVEL 8 CAPACITY ADDED:")
    print("  - Environment: external resource/stability pool")
    print("  - Actions consume resources and disturb stability")
    print("  - Health signals as additional input channels (4→6 total)")
    print("  - NO 'prefer healthy' rule — just sensing capacity")
    if not health_enabled:
        print("  - CONTROL ABLATION: health channels zeroed")
    print()
    print("WHAT SHOULD EMERGE (if it does):")
    print("  - Attention may weight health channels if they help predict dynamics")
    print("  - Sustainable patterns may arise (not engineered)")
    print()
    
    # If we ablate health channels, also ablate env→noise coupling so there is
    # NO remaining information about E (otherwise σ leaks health indirectly).
    sub = HonestBoundarySubstrate(
        seed=seed, 
        health_channels_enabled=health_enabled,
        env_noise_enabled=health_enabled
    )
    
    for step in range(n_steps):
        sub.step_physics()
        
        if step > 0 and step % sub.report_interval == 0:
            r = sub.report()
            print(
                f"[{r['step']:>5}] "
                f"M={r['M_ratio']:.0f}x  "
                f"coh={r['core_coherence']:.0%}  "
                f"att_pred={r['attention_predictable']:.2f}  "
                f"env={r['env_health']:.2f}  "
                f"σ={r['obs_sigma']:.3f}"
            )
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    final = sub.report()
    action_coherence = sub.measure_action_core_coherence()
    sustainability = sub.measure_sustainability()
    
    print("BASE EMERGENCE (inherited):")
    print(f"  M ratio: {final['M_ratio']:.0f}x")
    print(f"  Core coherence: {final['core_coherence']:.1%}")
    print()
    
    print("LEVEL 6 (inherited):")
    print(f"  Attention predictable (ch0+ch3): {final['attention_predictable']:.3f}")
    print(f"  Attention entropy (ch1): {final['attention_entropy']:.3f}")
    print()
    
    print("LEVEL 7 (inherited):")
    print(f"  Action std: {final['action_std']:.4f}")
    if "error" not in action_coherence:
        print(f"  Action-core correlation: {action_coherence['action_core_corr']:.3f}")
    print()
    
    print("LEVEL 8: ENVIRONMENTAL SENSING (load-bearing)")
    print(f"  Attention on health channels (ch4+ch5): {final['attention_health']:.3f}")
    print(f"  Environment resources: {final['env_resources']:.1%}")
    print(f"  Environment stability: {final['env_stability']:.1%}")
    print(f"  Environment health: {final['env_health']:.1%}")
    print(f"  Observation noise (env↓ → σ↑): {final['obs_sigma']:.4f}")
    if "error" not in sustainability:
        print(f"  Health gradient (across block): {sustainability['health_gradient']:+.3f}")
        print(f"  Health-action correlation: {sustainability['health_action_corr']:.3f}")
        print(f"  SUSTAINABLE (post-hoc label): {sustainability['sustainable']}")
    
    return {
        "final": final,
        "action_coherence": action_coherence,
        "sustainability": sustainability,
    }


def main():
    ap = argparse.ArgumentParser(description="TBU Honest Boundary - Level 8")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gain", type=float, default=None, help="Override action_gain_scale")
    ap.add_argument("--no-health", action="store_true", help="Control ablation: zero health channels")
    args = ap.parse_args()
    
    health_enabled = not args.no_health
    
    if args.gain is not None:
        # If health ablated, also ablate env→noise coupling (no E-sensing at all)
        sub = HonestBoundarySubstrate(
            seed=args.seed, 
            action_gain_scale=float(args.gain),
            health_channels_enabled=health_enabled,
            env_noise_enabled=health_enabled
        )
        for step in range(args.steps):
            sub.step_physics()
            if step > 0 and step % sub.report_interval == 0:
                r = sub.report()
                print(
                    f"[{r['step']:>5}] "
                    f"M={r['M_ratio']:.0f}x  "
                    f"att_health={r['attention_health']:.2f}  "
                    f"env={r['env_health']:.2f}  "
                    f"σ={r['obs_sigma']:.3f}"
                )
        final = sub.report()
        sustainability = sub.measure_sustainability()
        print("\nFINAL:", final)
        print("SUSTAINABILITY:", sustainability)
    else:
        run_experiment(n_steps=args.steps, seed=args.seed, health_enabled=health_enabled)


if __name__ == "__main__":
    main()
