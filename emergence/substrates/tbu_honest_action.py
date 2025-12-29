#!/usr/bin/env python3
"""
================================================================================
TBU HONEST ACTION - Level 7: Action Output Capacity (MINIMAL EXTENSION)
================================================================================

Goal: Add Level-7 "action capacity" while keeping Levels 1-6 honest and unchanged
in spirit, and making the Level-7 extension as minimal (and non-engineered) as
possible.

BASE PHYSICS (unchanged from honest extended):
  - Torus diffusion + damping + saturation
  - Forcing at B via attention-weighted channel mixture
  - M is diagnostic only (never fed back)

LEVELS 5-6 (unchanged):
  - Self-model: all pixels track neighbor mean (stability-modulated rate)
  - Attention: boundary pixels learn via prediction error minimization
      * prediction: yhat = attention · x_{t-1}
      * target: dy_t = y_t - y_{t-1}
      * update: w += eta * error * x_{t-1}
    No explicit "prefer predictable" rule.

LEVEL 7 (minimal extension):
  - Action output: boundary pixels emit action signal a_t
  - Action = smoothed boundary change (dy), not boundary state (avoids tautology)
  - Action is derived from PRE-FORCING dy (diffusion+damping contribution)
  - Action gates forcing gain using magnitude deviation (NO sign bias, SYMMETRIC)
      gain = 1 + scale * (tanh(|a|) - mean(tanh(|a|)))
  - Closed loop: boundary(post-sat) → diffusion+damping → dy → action → gain → forcing
  - When gain_scale=0, action updates are skipped (pure control mode)

Diagnostics:
  - Action stability (median(var)/var) over recent history (diagnostic only)
  - Action–core coherence measured via time-series correlation:
        corr( mean_action_magnitude_t , core_mean_t )
    Uses magnitude to avoid cancellation from positive/negative dy.

================================================================================
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import deque

import numpy as np


# -----------------------------------------------------------------------------
# Utilities (same style as honest/extended)
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
# Multi-channel environment (same as extended)
# -----------------------------------------------------------------------------

class MultiChannelProbe:
    """Provides multiple input channels with different statistical properties."""

    def __init__(self, n_channels: int = 4, seed: Optional[int] = None):
        self.n_channels = n_channels
        self.step = 0
        self.rng = np.random.default_rng(seed)

    def read_channels(self, n: int) -> np.ndarray:
        """Read n samples from each channel. Returns: (n, n_channels) array"""
        self.step += 1
        t = self.step

        channels = []

        # Channel 0: Time oscillation (predictable)
        ch0 = np.array([np.sin(t * 0.3 + i * 0.1) * 0.3 for i in range(n)])
        channels.append(ch0)

        # Channel 1: Hardware entropy (unpredictable)
        raw = os.urandom(4 * n)
        u = np.frombuffer(raw, dtype=np.uint32).astype(np.float64)
        ch1 = (u / (2**32) - 0.5) * 2.0
        channels.append(ch1)

        # Channel 2: System-coupled (moderate, seeded)
        base = np.sin(t * 0.05) * 0.2
        noise = self.rng.normal(0, 0.1, n)
        ch2 = base + noise
        channels.append(ch2)

        # Channel 3: Slow oscillation (predictable)
        ch3 = np.array([np.sin(t * 0.1 + i * 0.05) * 0.2 for i in range(n)])
        channels.append(ch3)

        return np.stack(channels, axis=1)


# -----------------------------------------------------------------------------
# Action Substrate (Level 7) - Minimal Extension
# -----------------------------------------------------------------------------

@dataclass
class HonestActionSubstrate:
    """
    Level 7: Action output capacity on honest extended foundation.

    BASE + LEVELS 5-6:
      - Torus diffusion + damping + forcing via attention + saturation
      - Self-model tracking, attention via prediction error learning

    LEVEL 7 EXTENSION (minimal):
      - Action output: boundary pixels emit action signals
      - Action = smoothed boundary change dy (not boundary state)
      - Gain gating: symmetric around 1 (deviation from mean magnitude)
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

    # Attention parameters (Level 6)
    n_channels: int = 4
    attention_learning_rate: float = 0.05

    # Action parameters (Level 7)
    action_smoothing: float = 0.1       # how fast action tracks dy
    action_gain_scale: float = 0.5      # 0 => open-loop (no effect), >0 => closed-loop

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

        # Forcing mask
        self._init_forcing_mask()
        self._update_forcing_coords()

        # Level 5: self-models
        self.self_models = self.rng.uniform(-0.05, 0.05, (self.size, self.size))

        # Level 6: attention
        self.attention = np.ones((self.n_forcing, self.n_channels)) / self.n_channels
        self.prev_boundary = self.grid[self.forcing_coords].copy()
        self.prev_channels = self.probe.read_channels(self.n_forcing)

        # Level 7: actions (MINIMAL)
        # Action tracks boundary change dy via smoothing; initialise at 0
        self.actions = np.zeros(self.n_forcing, dtype=np.float64)
        self.prev_boundary_for_action = self.grid[self.forcing_coords].copy()

        # For coherence measurement (time-series)
        self.core_mean_series: List[float] = []
        self.action_mean_series: List[float] = []

        print(f"Action substrate: {self.size}x{self.size} grid")
        print(f"Forcing: {self.forcing_shape} (|B|={self.n_forcing})")
        print(f"Action capacity: {self.n_forcing} boundary pixels")
        print(f"Action gain scale: {self.action_gain_scale}")

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

            max_len = self.history_window * 2
            if len(self.history) > max_len:
                self.history = self.history[-self.history_window:]
                self.self_model_history = self.self_model_history[-self.history_window:]
                self.attention_history = self.attention_history[-self.history_window:]
                self.action_history = self.action_history[-self.history_window:]

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
        BASE:
          - diffusion, damping, saturation
          - forcing via attention-weighted channels

        LEVEL 7 (minimal):
          - action = smoothed boundary change dy
          - gain = 1 + scale * (tanh(|action|) - mean) [symmetric around 1]
        """

        # --- Diffusion on torus
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        lap = up + dn + lf + rt - 4.0 * self.grid

        self.grid += self.diffusion * lap
        self.grid *= 1.0 - self.damping

        # --- LEVEL 7: actions track dy (not state)
        # Note: action is derived from PRE-FORCING boundary evolution (diffusion+damping)
        # The closed loop is: boundary(post-sat) → diffusion+damping → dy → action → gain → forcing
        boundary_state_pre = self.grid[self.forcing_coords]
        dy_action = boundary_state_pre - self.prev_boundary_for_action
        self.prev_boundary_for_action = boundary_state_pre.copy()
        
        # Skip action update if gain=0 (pure control/ablation mode)
        if self.action_gain_scale > 0:
            self.actions += self.action_smoothing * (dy_action - self.actions)

        # --- Forcing at B (attention-weighted + action-gated)
        channels = self.probe.read_channels(self.n_forcing)
        weighted_input = np.sum(self.attention * channels, axis=1)

        # Minimal, sign-free, SYMMETRIC gain: deviation from spatial mean magnitude
        # This allows both increase AND decrease (not always-up asymmetry)
        # When gain_scale=0, gain=1 everywhere (pure control mode)
        if self.action_gain_scale > 0:
            action_mag = np.tanh(np.abs(self.actions))           # [0,1)
            mag_mean = np.mean(action_mag) + 1e-12               # spatial mean across boundary pixels
            gain = 1.0 + self.action_gain_scale * (action_mag - mag_mean)
            # Scale-aware clipping (safety bounds that don't become engineered behaviour)
            gmin = max(0.05, 1.0 - self.action_gain_scale)
            gmax = 1.0 + self.action_gain_scale
            gain = np.clip(gain, gmin, gmax)
        else:
            gain = np.ones(self.n_forcing, dtype=np.float64)  # No action effect in control mode

        self.grid[self.forcing_coords] += self.forcing_strength * gain * weighted_input

        # --- Saturation
        self.grid = np.tanh(self.grid)

        # --- Level 5: self-model update
        self._update_self_models()

        # --- Level 6: attention update (predictive, x_{t-1} -> dy_t)
        self._update_attention_predictive()
        self.prev_channels = channels.copy()

        # --- Log time-series for coherence diagnostics (diagnostic only)
        # Only after warmup to avoid artificial early plateau
        if len(self.history) >= 20:
            M = self.measure_M()
            core_mask = (M >= np.median(M[~self.forcing_mask])) & (~self.forcing_mask)
            core_mean = float(self.grid[core_mask].mean()) if core_mask.any() else 0.0
            self.core_mean_series.append(core_mean)
            # Use mean magnitude (not signed mean) to avoid cancellation
            self.action_mean_series.append(float(np.mean(np.abs(self.actions))))
            if len(self.core_mean_series) > self.history_window * 2:
                self.core_mean_series = self.core_mean_series[-self.history_window:]
                self.action_mean_series = self.action_mean_series[-self.history_window:]

        # Store histories
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

        return {
            "step": int(self.step),
            "grid_std": float(np.std(self.grid)),
            "M_ratio": float(M_ratio),
            "core_coherence": float(lcc),
            "n_clusters": int(n_clusters),
            "self_model_ratio": float(sm_stability_high_M / (sm_stability_low_M + 1e-8)),
            "attention_predictable": float(mean_attention[0] + mean_attention[3]),
            "attention_entropy": float(mean_attention[1]),
            "action_mean": float(mean_action),
            "action_std": float(action_std),
            "action_stability": float(mean_action_stability),
        }

    def measure_action_core_coherence(self) -> Dict[str, Any]:
        """
        Minimal, non-tautological coherence test:
          corr( mean_action_magnitude_t , core_mean_t ) over recent window.

        Uses mean(|action|) to avoid cancellation from positive/negative dy.
        This is *not* action vs boundary (which would be constructionally high).
        
        Note: Returns 0 correlation in control mode (gain=0) since actions stay at 0.
        """
        if len(self.core_mean_series) < 30:
            return {"error": "Insufficient history"}

        a = np.array(self.action_mean_series[-self.history_window:], dtype=float)
        c = np.array(self.core_mean_series[-self.history_window:], dtype=float)

        # Handle zero-variance case (control mode where actions stay at 0)
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


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_experiment(n_steps: int = 2000, seed: Optional[int] = None) -> Dict[str, Any]:
    print("=" * 70)
    print("TBU HONEST ACTION - Level 7 (MINIMAL)")
    print("=" * 70)
    print()
    print("BASE (from honest extended):")
    print("  - Torus diffusion + damping + saturation")
    print("  - Forcing via attention-weighted channels (Level 6)")
    print("  - M is diagnostic only, never fed back")
    print()
    print("LEVEL 7 CAPACITY ADDED (minimal):")
    print("  - Action = smoothed boundary change dy (not boundary state)")
    print("  - Action derived from PRE-FORCING dy (diffusion+damping contribution)")
    print("  - Gain gating: symmetric around 1 (deviation from spatial mean magnitude)")
    print("  - Closed loop: boundary(post-sat) → diffusion+damping → dy → action → gain → forcing")
    print()

    sub = HonestActionSubstrate(seed=seed)

    for step in range(n_steps):
        sub.step_physics()

        if step > 0 and step % sub.report_interval == 0:
            r = sub.report()
            print(
                f"[{r['step']:>5}] "
                f"M_ratio={r['M_ratio']:.0f}x  "
                f"coh={r['core_coherence']:.0%}  "
                f"att_pred={r['attention_predictable']:.2f}  "
                f"act_stab={r['action_stability']:.1f}  "
                f"act_std={r['action_std']:.3f}"
            )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    final = sub.report()
    action_coherence = sub.measure_action_core_coherence()

    print("BASE EMERGENCE (inherited):")
    print(f"  M ratio: {final['M_ratio']:.0f}x")
    print(f"  Core coherence: {final['core_coherence']:.1%}")
    print()

    print("LEVEL 6 (inherited):")
    print(f"  Attention predictable: {final['attention_predictable']:.3f}")
    print(f"  Attention entropy: {final['attention_entropy']:.3f}")
    print()

    print("LEVEL 7: ACTION EMERGENCE (minimal)")
    print(f"  Action mean: {final['action_mean']:.4f}")
    print(f"  Action std: {final['action_std']:.4f}")
    print(f"  Action stability: {final['action_stability']:.1f}")
    if "error" not in action_coherence:
        print(f"  Action–core correlation (time-series): {action_coherence['action_core_corr']:.3f}")
        print(f"  High stability fraction: {action_coherence['high_stability_fraction']:.1%}")
    else:
        print(f"  {action_coherence['error']}")

    return {"final": final, "action_coherence": action_coherence}


def main():
    ap = argparse.ArgumentParser(description="TBU Honest Action - Level 7 (Minimal)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gain", type=float, default=None, help="Override action_gain_scale (e.g., 0 for control)")
    args = ap.parse_args()

    if args.gain is None:
        run_experiment(n_steps=args.steps, seed=args.seed)
    else:
        # Small convenience: run with overridden gain scale (control vs closed-loop)
        sub = HonestActionSubstrate(seed=args.seed, action_gain_scale=float(args.gain))
        for step in range(args.steps):
            sub.step_physics()
            if step > 0 and step % sub.report_interval == 0:
                r = sub.report()
                print(
                    f"[{r['step']:>5}] "
                    f"M_ratio={r['M_ratio']:.0f}x  "
                    f"coh={r['core_coherence']:.0%}  "
                    f"att_pred={r['attention_predictable']:.2f}  "
                    f"act_stab={r['action_stability']:.1f}  "
                    f"act_std={r['action_std']:.3f}"
                )
        final = sub.report()
        action_coherence = sub.measure_action_core_coherence()
        print("\nFINAL:", final)
        print("ACTION-CORE:", action_coherence)


if __name__ == "__main__":
    main()
