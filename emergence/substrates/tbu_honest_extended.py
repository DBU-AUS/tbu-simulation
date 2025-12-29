#!/usr/bin/env python3
"""
================================================================================
TBU HONEST EXTENDED - Levels 5-6: Self-Model and Attention Capacity
================================================================================

Builds on tbu_honest.py (Levels 1-4) by adding CAPACITY for:
  - Level 5: Self-model (tracking neighbor state)
  - Level 6: Attention (weighting input channels via prediction error minimization)

BASE PHYSICS:
  - Uniform periodic manifold (torus)
  - Symmetric local diffusion
  - Fixed coupling subset B
  - Forcing at B via attention-weighted channel mixture (Level-6 extension)
  - Saturation (tanh)
  - NO M-feedback into dynamics

Note: Base physics is unchanged except boundary forcing is now a mixture of
provided channels via an attention field. This is the Level-6 capacity extension.

CAPACITY ADDED (not selection rules):
  1. Self-model: Every pixel has a value that can track neighbor mean
     - Update rate: inversely proportional to self-model stability
     - NO threshold for which pixels can have models
     - EMERGENT: Where models persist (should be stable regions far from B)

  2. Attention: Every boundary pixel has weights over input channels
     - Update rule: PREDICTION ERROR MINIMIZATION (gradient descent)
     - Uses previous-step channels to predict current boundary changes
     - NO "avoid variance" or "prefer predictable" rule encoded
     - EMERGENT: Which channels get attended (empirical outcome, not coded)

WHAT IS DESIGNED vs WHAT EMERGES:
  - DESIGNED: Capacity exists everywhere, prediction error learning rule
  - EMERGENT: WHERE self-models localize, WHICH channels get attended

The prediction error rule is generic: minimize squared error predicting dy_t
from x_{t-1}. If attention shifts toward predictable channels, it's because
they actually help predict boundary dynamics - an empirical outcome.

================================================================================
"""

from __future__ import annotations

import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from collections import deque

import numpy as np

# -----------------------------------------------------------------------------
# Utilities (same as tbu_honest.py for standalone use)
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
    """
    Sample exactly n pixels uniformly from within a region.
    Returns a new mask with exactly n True pixels, all within region_mask.
    """
    eligible = np.flatnonzero(region_mask.ravel())
    if len(eligible) < n:
        raise RuntimeError(f"Region has {len(eligible)} pixels but need {n}")
    
    chosen = rng.choice(eligible, size=n, replace=False)
    result = np.zeros_like(region_mask, dtype=bool)
    ys, xs = np.unravel_index(chosen, region_mask.shape)
    result[ys, xs] = True
    return result


def largest_connected_component_fraction(mask: np.ndarray) -> Tuple[float, int]:
    """
    Largest connected component (4-neighbor, periodic) as fraction of total True pixels.
    Returns (fraction, n_components). Core is DISCOVERED by this metric.
    """
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


def compute_distance_to_B(forcing_mask: np.ndarray) -> np.ndarray:
    """Graph distance from each pixel to nearest forcing pixel (4-neighbor on torus)."""
    H, W = forcing_mask.shape
    dist = np.full((H, W), np.iinfo(np.int32).max, dtype=np.int32)

    q = deque()
    ys, xs = np.where(forcing_mask)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((y, x))

    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = (y + dy) % H, (x + dx) % W
            if dist[yy, xx] > d + 1:
                dist[yy, xx] = d + 1
                q.append((yy, xx))

    return dist


# -----------------------------------------------------------------------------
# Multi-channel environment
# -----------------------------------------------------------------------------

class MultiChannelProbe:
    """
    Provides multiple input channels with different statistical properties.
    
    Channels:
      0: Time oscillation (predictable, low variance)
      1: Hardware entropy (unpredictable, high variance)
      2: System-coupled (moderate variance, seeded for reproducibility)
      3: Slow oscillation (predictable, low variance)
    """
    
    def __init__(self, n_channels: int = 4, seed: Optional[int] = None):
        self.n_channels = n_channels
        self.step = 0
        self.rng = np.random.default_rng(seed)
    
    def read_channels(self, n: int) -> np.ndarray:
        """
        Read n samples from each channel.
        Returns: (n, n_channels) array
        """
        self.step += 1
        t = self.step
        
        channels = []
        
        # Channel 0: Time oscillation (predictable)
        ch0 = np.array([np.sin(t * 0.3 + i * 0.1) * 0.3 for i in range(n)])
        channels.append(ch0)
        
        # Channel 1: Hardware entropy (unpredictable)
        raw = os.urandom(4 * n)
        u = np.frombuffer(raw, dtype=np.uint32).astype(np.float64)
        ch1 = (u / (2**32) - 0.5) * 2.0  # [-1, 1] range, high variance
        channels.append(ch1)
        
        # Channel 2: System-coupled (moderate, seeded RNG for reproducibility)
        base = np.sin(t * 0.05) * 0.2
        noise = self.rng.normal(0, 0.1, n)  # Use seeded RNG
        ch2 = base + noise
        channels.append(ch2)
        
        # Channel 3: Slow oscillation (predictable)
        ch3 = np.array([np.sin(t * 0.1 + i * 0.05) * 0.2 for i in range(n)])
        channels.append(ch3)
        
        return np.stack(channels, axis=1)  # (n, n_channels)


# -----------------------------------------------------------------------------
# Extended Substrate
# -----------------------------------------------------------------------------

@dataclass
class HonestExtendedSubstrate:
    """
    Levels 5-6: Self-model and attention capacity on honest foundation.
    
    BASE (from honest):
      - Torus diffusion + damping + forcing on B + saturation
      - M is diagnostic only, never fed back
    
    EXTENSIONS:
      - Self-model array: tracks neighbor mean (all pixels have capacity)
      - Attention array: weights over input channels (boundary pixels)
    """
    
    size: int = 64
    
    # Forcing geometry (same as honest)
    forcing_shape: str = "ring"
    forcing_n_pixels: Optional[int] = None
    forcing_ring_width: int = 2
    
    # Base physics (same as honest)
    diffusion: float = 0.18
    damping: float = 0.001
    forcing_strength: float = 0.12
    
    # Self-model parameters
    self_model_rate: float = 0.1  # base update rate toward neighbor mean
    self_model_stability_scale: float = 10.0  # how much stability slows updates
    
    # Attention parameters (generic plasticity, no engineered preference)
    n_channels: int = 4
    attention_learning_rate: float = 0.05  # Hebbian update rate
    
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
        self.probe = MultiChannelProbe(self.n_channels, seed=self.seed)
        
        # Auto-match cardinality
        if self.forcing_n_pixels is None:
            self.forcing_n_pixels = compute_ring_cardinality(self.size, self.forcing_ring_width)
        
        # Initialize grid (base state)
        self.grid = self.rng.uniform(-0.05, 0.05, (self.size, self.size))
        self.step = 0
        self.history: List[np.ndarray] = []
        
        # Initialize forcing mask
        self._init_forcing_mask()
        self._update_forcing_coords()
        
        # === EXTENSION: Self-model capacity ===
        # Every pixel has a self-model value (initially random small)
        self.self_models = self.rng.uniform(-0.05, 0.05, (self.size, self.size))
        self.self_model_history: List[np.ndarray] = []
        
        # === EXTENSION: Attention capacity ===
        # Every boundary pixel has attention weights over channels (initially uniform)
        self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
        self.attention_history: List[np.ndarray] = []
        
        # Store previous boundary values and channels for predictive learning
        self.prev_boundary = self.grid[self.forcing_coords].copy()  # Initialize from actual values
        self.prev_channels = self.probe.read_channels(self.n_forcing)  # Initialize from actual channels
        
        print(f"Extended substrate: {self.size}x{self.size} grid")
        print(f"Forcing: {self.forcing_shape} (|B|={self.n_forcing})")
        print(f"Self-model capacity: all {self.size**2} pixels")
        print(f"Attention capacity: {self.n_forcing} boundary pixels x {self.n_channels} channels")
    
    def _init_forcing_mask(self):
        """Same as honest - create forcing mask based on shape."""
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
        # Flat indices for boundary pixels
        self.boundary_indices = np.ravel_multi_index(self.forcing_coords, (self.size, self.size))
    
    def _maybe_store_history(self):
        if self.history_stride <= 1 or (self.step % self.history_stride == 0):
            self.history.append(self.grid.copy())
            self.self_model_history.append(self.self_models.copy())
            self.attention_history.append(self.attention.copy())
            
            # Trim histories
            max_len = self.history_window * 2
            if len(self.history) > max_len:
                self.history = self.history[-self.history_window:]
                self.self_model_history = self.self_model_history[-self.history_window:]
                self.attention_history = self.attention_history[-self.history_window:]
    
    # =========================================================================
    # DIAGNOSTICS (never fed back)
    # =========================================================================
    
    def measure_M(self) -> np.ndarray:
        """M = median(var)/var over snapshot window. DIAGNOSTIC ONLY."""
        if len(self.history) < 20:
            return np.ones((self.size, self.size), dtype=np.float64)
        
        H = np.array(self.history[-self.history_window:], dtype=np.float64)
        var = np.var(H, axis=0) + 1e-12
        M = np.median(var) / var
        return np.clip(M, self.M_clip_min, self.M_clip_max)
    
    def measure_self_model_M(self) -> np.ndarray:
        """M for self-models based on their stability."""
        if len(self.self_model_history) < 20:
            return np.ones((self.size, self.size), dtype=np.float64)
        
        H = np.array(self.self_model_history[-self.history_window:], dtype=np.float64)
        var = np.var(H, axis=0) + 1e-12
        M = np.median(var) / var
        return np.clip(M, self.M_clip_min, self.M_clip_max)
    
    def stable_mask(self) -> np.ndarray:
        """Bottom-q quantile of variance defines stability."""
        if len(self.history) < 20:
            return np.zeros((self.size, self.size), dtype=bool)
        
        H = np.array(self.history[-self.history_window:], dtype=np.float64)
        var = np.var(H, axis=0)
        thresh = np.quantile(var, self.stability_quantile)
        return var <= thresh
    
    # =========================================================================
    # PHYSICS STEP (base + extensions)
    # =========================================================================
    
    def step_physics(self) -> None:
        """
        One step of dynamics.
        
        BASE PHYSICS:
          - Torus diffusion
          - Damping
          - Forcing at B via attention-weighted channel mixture
          - Saturation
        
        Note: Base physics is unchanged except boundary forcing is now a mixture
        of provided channels via an attention field (Level-6 capacity).
        
        EXTENSIONS (capacity, not feedback):
          - Self-model update (all pixels)
          - Attention update via prediction error minimization (boundary pixels)
        """
        
        # === BASE: Diffusion on torus ===
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        lap = up + dn + lf + rt - 4.0 * self.grid
        
        self.grid += self.diffusion * lap
        self.grid *= 1.0 - self.damping
        
        # === BASE: Forcing at B (attention-weighted) ===
        channels = self.probe.read_channels(self.n_forcing)  # (n_forcing, n_channels)
        
        # Attention-weighted input for each boundary pixel
        # weighted_input[i] = sum_c(attention[i,c] * channels[i,c])
        weighted_input = np.sum(self.attention * channels, axis=1)
        
        self.grid[self.forcing_coords] += self.forcing_strength * weighted_input
        
        # === BASE: Saturation ===
        self.grid = np.tanh(self.grid)
        
        # === EXTENSION: Self-model update ===
        self._update_self_models()
        
        # === EXTENSION: Attention update (prediction error minimization) ===
        # Uses prev_channels to predict current dy (genuinely predictive)
        self._update_attention_predictive()
        
        # Store current channels for next step's prediction
        self.prev_channels = channels.copy()
        
        # Store history
        self._maybe_store_history()
        self.step += 1
    
    def _update_self_models(self):
        """
        Update self-models toward neighbor mean.
        
        Update RATE is inversely proportional to self-model stability.
        Stable self-models (low variance over time) update slower, persist more.
        This is a physical fact: stable things change less.
        """
        # Compute neighbor mean for each pixel
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        neighbor_mean = (up + dn + lf + rt) / 4.0
        
        # Compute self-model stability (inverse of variance over recent history)
        if len(self.self_model_history) >= 20:
            H = np.array(self.self_model_history[-30:], dtype=np.float64)
            sm_var = np.var(H, axis=0) + 1e-8
            stability = 1.0 / (1.0 + self.self_model_stability_scale * sm_var)
        else:
            stability = np.ones((self.size, self.size)) * 0.5
        
        # Update rate: base rate * (1 - stability)
        # High stability = low update rate = model persists
        # Low stability = high update rate = model tracks current state
        rate = self.self_model_rate * (1.0 - 0.9 * stability)
        
        # Update: move toward neighbor mean
        self.self_models += rate * (neighbor_mean - self.self_models)
    
    def _update_attention_predictive(self):
        """
        Prediction error minimization using PREVIOUS channels to predict CURRENT dy.
        
        NO engineered preference for "predictable" or "avoid variance".
        Attention weights learn to predict boundary state changes from past inputs.
        Channels that help predict get strengthened; unpredictable channels get weakened.
        
        Prediction target: Net boundary change dy_t = y_t - y_{t-1}
        This includes effects of diffusion + damping + forcing + saturation.
        Features: Previous-step channel values x_{t-1}
        
        If attention shifts toward predictable channels, it's because they
        actually help predict boundary dynamics - an empirical outcome.
        
        Update rule: gradient descent on prediction error
          prediction: ŷ = attention · prev_channels
          error: (dy - ŷ)²
          gradient: ∂error/∂w = -2(dy - ŷ) · prev_channels
        """
        eta = self.attention_learning_rate
        
        # Current boundary values and delta
        y = self.grid[self.forcing_coords]  # shape (n_forcing,)
        dy = y - self.prev_boundary  # boundary state change (what happened)
        
        # Update prev_boundary for next step
        self.prev_boundary = y.copy()
        
        # Skip first step (need one step for prev_channels to be meaningful)
        if self.step < 1:
            return
        
        # Prediction using PREVIOUS channels (genuinely predictive)
        prediction = np.sum(self.attention * self.prev_channels, axis=1)  # (n_forcing,)
        
        # Prediction error
        error = dy - prediction  # (n_forcing,)
        
        # Gradient descent: move attention toward channels that reduce error
        # Update uses prev_channels (what we used to predict)
        self.attention += eta * (error[:, None] * self.prev_channels)
        
        # Project to simplex (positive, sum to 1)
        self.attention = np.clip(self.attention, 1e-3, None)
        self.attention /= self.attention.sum(axis=1, keepdims=True)
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    def report(self) -> Dict[str, Any]:
        """Diagnostics (never affect dynamics)."""
        M = self.measure_M()
        stable = self.stable_mask()
        lcc, n_clusters = largest_connected_component_fraction(stable)
        
        # Base metrics
        forcing_M = float(M[self.forcing_mask].mean())
        non_forcing_M = float(M[~self.forcing_mask].mean())
        M_ratio = non_forcing_M / (forcing_M + 1e-12)
        
        # Self-model localization
        sm_M = self.measure_self_model_M()
        high_M_mask = M > np.median(M)
        low_M_mask = ~high_M_mask
        
        sm_stability_high_M = float(sm_M[high_M_mask].mean()) if high_M_mask.any() else 0.0
        sm_stability_low_M = float(sm_M[low_M_mask].mean()) if low_M_mask.any() else 0.0
        
        # Attention distribution
        mean_attention = self.attention.mean(axis=0)
        
        return {
            "step": int(self.step),
            "grid_std": float(np.std(self.grid)),
            "M_ratio": float(M_ratio),
            "core_coherence": float(lcc),
            "n_clusters": int(n_clusters),
            "self_model_stability_high_M": float(sm_stability_high_M),
            "self_model_stability_low_M": float(sm_stability_low_M),
            "self_model_ratio": float(sm_stability_high_M / (sm_stability_low_M + 1e-8)),
            "attention_ch0_time": float(mean_attention[0]),
            "attention_ch1_entropy": float(mean_attention[1]),
            "attention_ch2_system": float(mean_attention[2]),
            "attention_ch3_slow": float(mean_attention[3]),
            "attention_predictable": float(mean_attention[0] + mean_attention[3]),
        }
    
    def measure_self_model_localization(self) -> Dict[str, Any]:
        """
        Do functional self-models localize in high-M (stable) regions?
        
        Prediction: Self-models should be more stable in regions far from B,
        because those regions have low variance and can maintain coherent state.
        """
        M = self.measure_M()
        sm_M = self.measure_self_model_M()
        
        # Divide by base state M (stability proxy)
        median_M = np.median(M)
        high_M_mask = M > median_M
        low_M_mask = ~high_M_mask
        
        # Self-model stability in each region
        high_M_sm_stability = sm_M[high_M_mask].mean() if high_M_mask.any() else 0.0
        low_M_sm_stability = sm_M[low_M_mask].mean() if low_M_mask.any() else 0.0
        
        # Count "functional" self-models (stability above threshold)
        threshold = np.median(sm_M) * 2
        high_M_functional = (sm_M[high_M_mask] > threshold).sum() / high_M_mask.sum() if high_M_mask.any() else 0.0
        low_M_functional = (sm_M[low_M_mask] > threshold).sum() / low_M_mask.sum() if low_M_mask.any() else 0.0
        
        return {
            "high_M_sm_stability": float(high_M_sm_stability),
            "low_M_sm_stability": float(low_M_sm_stability),
            "stability_ratio": float(high_M_sm_stability / (low_M_sm_stability + 1e-8)),
            "high_M_functional_rate": float(high_M_functional),
            "low_M_functional_rate": float(low_M_functional),
            "localized": high_M_sm_stability > low_M_sm_stability * 1.5,
        }
    
    def measure_attention_emergence(self) -> Dict[str, Any]:
        """
        Does attention discover which channels help predict boundary dynamics?
        
        Via prediction error minimization, the system should learn to attend to
        channels that reduce prediction error. Predictable channels help predict;
        entropy channels don't.
        
        Channels:
          0: time oscillation (predictable)
          1: entropy (unpredictable)
          2: system (moderate)
          3: slow oscillation (predictable)
        
        Prediction: attention[1] should be below uniform (0.25),
                    attention[0,3] should be above uniform (0.50 combined)
        """
        if len(self.attention_history) < 50:
            return {"error": "Insufficient history"}
        
        # Uniform baseline
        uniform = 0.25
        
        # Final attention (averaged over last 50 steps)
        final = np.array(self.attention_history[-50:]).mean(axis=(0, 1))
        
        # Early attention (steps 5-20, after some initial dynamics)
        if len(self.attention_history) > 20:
            early = np.array(self.attention_history[5:20]).mean(axis=(0, 1))
        else:
            early = np.ones(self.n_channels) * uniform
        
        # Compare to uniform
        entropy_below_uniform = final[1] < uniform - 0.02  # entropy channel below 0.23
        predictable_above_uniform = (final[0] + final[3]) > 0.5 + 0.02  # predictable above 0.52
        
        # Also check trajectory
        delta = final - early
        
        return {
            "uniform_baseline": uniform,
            "early_attention": early.tolist(),
            "final_attention": final.tolist(),
            "delta_from_early": delta.tolist(),
            "entropy_final": float(final[1]),
            "predictable_final": float(final[0] + final[3]),
            "entropy_below_uniform": bool(entropy_below_uniform),
            "predictable_above_uniform": bool(predictable_above_uniform),
            "predictability_bias_emerged": bool(entropy_below_uniform and predictable_above_uniform),
        }


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_experiment(n_steps: int = 2000, seed: Optional[int] = None) -> Dict[str, Any]:
    """Run extended substrate experiment."""
    
    print("=" * 70)
    print("TBU HONEST EXTENDED - Levels 5-6")
    print("=" * 70)
    print()
    print("BASE (from honest):")
    print("  - Torus diffusion + damping + forcing on B + saturation")
    print("  - M is diagnostic only, never fed back")
    print()
    print("CAPACITY ADDED:")
    print("  - Self-model: all pixels have capacity to track neighbor mean")
    print("  - Attention: boundary pixels learn via PREDICTION ERROR MINIMIZATION")
    print("    (minimize error predicting boundary changes, NO 'avoid variance' rule)")
    print()
    print("WHAT SHOULD EMERGE:")
    print("  - Self-models localize in stable (high-M) regions")
    print("  - Attention shifts toward channels that correlate with coherent dynamics")
    print()
    
    sub = HonestExtendedSubstrate(seed=seed)
    
    for step in range(n_steps):
        sub.step_physics()
        
        if step > 0 and step % sub.report_interval == 0:
            r = sub.report()
            print(
                f"[{r['step']:>5}] "
                f"M_ratio={r['M_ratio']:.1f}x  "
                f"coh={r['core_coherence']:.1%}  "
                f"SM_ratio={r['self_model_ratio']:.2f}  "
                f"att_pred={r['attention_predictable']:.2f}  "
                f"att_ent={r['attention_ch1_entropy']:.2f}"
            )
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    # Final analysis
    final = sub.report()
    sm_loc = sub.measure_self_model_localization()
    att_em = sub.measure_attention_emergence()
    
    print("BASE EMERGENCE (inherited from honest):")
    print(f"  M ratio (non-forcing/forcing): {final['M_ratio']:.1f}x")
    print(f"  Core coherence: {final['core_coherence']:.1%}")
    print(f"  Clusters: {final['n_clusters']}")
    print()
    
    print("LEVEL 5: SELF-MODEL LOCALIZATION")
    print(f"  Self-model stability in high-M regions: {sm_loc['high_M_sm_stability']:.2f}")
    print(f"  Self-model stability in low-M regions: {sm_loc['low_M_sm_stability']:.2f}")
    print(f"  Stability ratio: {sm_loc['stability_ratio']:.2f}x")
    print(f"  High-M functional rate: {sm_loc['high_M_functional_rate']:.1%}")
    print(f"  Low-M functional rate: {sm_loc['low_M_functional_rate']:.1%}")
    print(f"  LOCALIZED: {sm_loc['localized']}")
    print()
    
    print("LEVEL 6: ATTENTION EMERGENCE (prediction error minimization)")
    if "error" not in att_em:
        print(f"  Uniform baseline: {att_em['uniform_baseline']:.3f}")
        print(f"  Early attention: {[f'{x:.3f}' for x in att_em['early_attention']]}")
        print(f"  Final attention: {[f'{x:.3f}' for x in att_em['final_attention']]}")
        print(f"  Entropy (ch1): {att_em['entropy_final']:.3f} (uniform=0.25)")
        print(f"  Predictable (ch0+ch3): {att_em['predictable_final']:.3f} (uniform=0.50)")
        print(f"  Entropy below uniform: {att_em['entropy_below_uniform']}")
        print(f"  Predictable above uniform: {att_em['predictable_above_uniform']}")
        print(f"  PREDICTABILITY BIAS EMERGED: {att_em['predictability_bias_emerged']}")
    else:
        print(f"  {att_em['error']}")
    
    return {
        "final": final,
        "self_model_localization": sm_loc,
        "attention_emergence": att_em,
    }


def main():
    ap = argparse.ArgumentParser(description="TBU Honest Extended - Levels 5-6")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    
    run_experiment(n_steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
