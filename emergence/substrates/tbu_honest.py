#!/usr/bin/env python3
"""
================================================================================
TBU HONEST DAEMON - Option B with Ablation Support
================================================================================

ROOM PROVISION ONLY. NO HIDDEN STRUCTURE.

What we provide:
  - Uniform periodic manifold (torus) with a fixed coupling subset B through
    which environment acts
  - Local symmetric diffusion law
  - Stochastic drive at B each step

What we measure (diagnostic only, never fed back):
  - M := median(var) / var over temporal window (clipped to avoid numerical blowups)
  - Stable mask := bottom quantile of variance (NOT "non-forced" - avoids geometry labels)
  - Core := largest connected component of stable pixels (DISCOVERED)
  - Coherence := core_size / total_stable_pixels

What emerges:
  - Spatial structure emerges as a consequence of the fixed coupling subset B:
    stability becomes a function of graph distance to B under local diffusion
    with damping.

ABLATION OPTIONS:
  --forcing-shape {ring, random, patch, corners}
    Different fixed B geometries with matched cardinality:
      ring    = solid perimeter band (natural |B|, pure geometry)
      random  = uniformly scattered across grid
      patch   = random subset within a corner patch region (localised)
      corners = random subset within four corner regions (localised)

  --drift-hold N
    If > 0, enables slow drift: B partially moves every N steps

  --drift-frac F
    Fraction of B pixels that attempt to move each drift event

  --seed N
    RNG seed for reproducibility

  --fresh
    Delete existing state and start fresh (ensures CLI args take effect)

Key claim: coherence depends on STATIONARITY and LOCALISATION of B.
- Fixed + localised B -> coherent core
- Fixed but scattered B -> fragmentation (low <d>)
- Rapidly varying B -> fragmentation (no persistent distance structure)

DESIGN PRINCIPLES:
  - Ring uses its natural cardinality (region-restricted, no pollution)
  - Other shapes auto-match ring's |B| unless explicitly overridden
  - Patch/corners use region-restricted sampling (stay within their regions)
  - Size derived from grid shape on restore (arrays are ground truth)
  - M is diagnostic only, NEVER fed back into dynamics

================================================================================
"""

from __future__ import annotations

import os
import time
import json
import signal
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from collections import deque

import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def sample_from_region(region_mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample exactly n pixels uniformly from within a region.
    Returns a new mask with exactly n True pixels, all within region_mask.
    Preserves geometry class (patch stays patch, corners stay corners).
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


def compute_ring_cardinality(size: int, width: int) -> int:
    """Compute natural |B| for a ring of given width on a size x size torus."""
    mask = np.zeros((size, size), dtype=bool)
    w = int(width)
    mask[0:w, :] = True
    mask[-w:, :] = True
    mask[:, 0:w] = True
    mask[:, -w:] = True
    return int(mask.sum())


# -----------------------------------------------------------------------------
# IO + Environment
# -----------------------------------------------------------------------------

class RealityProbe:
    """Read OS entropy for forcing."""

    def entropy_floats(self, n: int) -> np.ndarray:
        raw = os.urandom(4 * n)
        u = np.frombuffer(raw, dtype=np.uint32).astype(np.float64)
        return (u + 0.5) / (2**32) * 2.0 - 1.0


class Motor:
    """Write bytes to filesystem (environmental coupling / side effect)."""

    def __init__(self, motor_dir: Path, n_files: int = 16):
        self.motor_dir = motor_dir
        self.motor_dir.mkdir(parents=True, exist_ok=True)
        self.n_files = max(1, int(n_files))
        self.idx = 0
        self.write_count = 0

    def write_bytes(self, payload: bytes) -> None:
        try:
            fpath = self.motor_dir / f"tbu_motor_{self.idx:02d}.bin"
            with open(fpath, "ab") as f:
                f.write(payload)
            self.idx = (self.idx + 1) % self.n_files
            self.write_count += 1
        except Exception:
            pass


class Persistence:
    """Save/load substrate state."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "state.npz"
        self.meta_file = self.state_dir / "meta.json"

    def save(
        self,
        grid: np.ndarray,
        forcing_mask: np.ndarray,
        step: int,
        born: str,
        metrics: Dict[str, Any],
        history: list,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        np.savez_compressed(
            self.state_file,
            grid=grid.astype(np.float32),
            forcing_mask=forcing_mask.astype(np.uint8),
            step=np.int64(step),
            born=born,
            history=np.array(history[-100:], dtype=np.float32) if history else np.array([]),
        )
        meta = {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "step": int(step),
            "born": born,
            "metrics": metrics,
        }
        if params is not None:
            meta["params"] = params
        with open(self.meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.state_file.exists():
            return None
        data = np.load(self.state_file, allow_pickle=False)
        result: Dict[str, Any] = {
            "grid": data["grid"].astype(np.float64),
            "forcing_mask": data["forcing_mask"].astype(bool),
            "step": int(data["step"]),
            "born": str(data["born"]),
        }
        if "history" in data and data["history"].size > 0:
            result["history"] = [h.astype(np.float64) for h in data["history"]]
        if self.meta_file.exists():
            try:
                meta = json.loads(self.meta_file.read_text())
                if "params" in meta:
                    result["params"] = meta["params"]
            except Exception:
                pass
        return result


# -----------------------------------------------------------------------------
# Substrate
# -----------------------------------------------------------------------------

@dataclass
class HonestSubstrate:
    """
    Option B: Fixed coupling subset on uniform periodic manifold.

    ROOM ONLY:
      - Uniform torus (np.roll)
      - Symmetric local diffusion
      - Fixed forcing set B (various shapes supported)
      - Optional slow drift of B (timescale separation test)

    DISCOVERED (not defined):
      - Core = largest connected component of low-variance pixels
      - Stability defined by variance quantiles (not geometry labels)
    """

    size: int = 64

    # Forcing set geometry
    forcing_shape: str = "ring"  # ring, random, patch, corners
    forcing_n_pixels: Optional[int] = None  # None = auto-match to ring's natural |B|
    forcing_ring_width: int = 2

    # Drift (timescale separation)
    drift_hold: int = 0  # 0 disables drift
    drift_frac: float = 0.1

    # Local physics
    diffusion: float = 0.18
    damping: float = 0.001
    forcing_strength: float = 0.12

    # Motor
    motor_strength: float = 0.05
    motor_sample: int = 256

    # Diagnostics
    history_window: int = 100  # number of snapshots used for variance
    history_stride: int = 1  # store a snapshot every k steps (daemon)
    stability_quantile: float = 0.3
    M_clip_min: float = 0.01
    M_clip_max: float = 100.0

    # Housekeeping
    state_dir: Path = field(default_factory=lambda: Path.home() / ".tbu_honest")
    motor_dir: Path = field(default_factory=lambda: Path.home() / ".tbu_honest" / "motor")
    save_interval: int = 5000
    report_interval: int = 500

    # RNG
    seed: Optional[int] = None

    def __post_init__(self):
        # Validate history_stride
        if self.history_stride < 1:
            self.history_stride = 1
        
        self.persistence = Persistence(self.state_dir)
        self.motor = Motor(self.motor_dir)
        self.probe = RealityProbe()
        self.rng = np.random.default_rng(self.seed)

        loaded = self.persistence.load()

        if loaded is None:
            # Fresh start
            self.grid = self.rng.uniform(-0.05, 0.05, (self.size, self.size))
            self.step = 0
            self.born = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.history: list[np.ndarray] = []

            # Auto-match cardinality to ring if not explicitly set
            if self.forcing_n_pixels is None:
                self.forcing_n_pixels = compute_ring_cardinality(self.size, self.forcing_ring_width)

            self._init_forcing_mask()
            print(f"Born: {self.born} | grid={self.size}x{self.size}")
        else:
            # Restore state
            self.grid = loaded["grid"]
            self.forcing_mask = loaded["forcing_mask"]
            self.step = loaded["step"]
            self.born = loaded["born"]
            self.history = loaded.get("history", [])

            # FIX: trust stored arrays for size (grid shape is ground truth)
            self.size = int(self.grid.shape[0])

            # Restore other params for truthful printing (but NOT size)
            params = loaded.get("params", {}) or {}
            if params:
                self.forcing_shape = str(params.get("forcing_shape", self.forcing_shape))
                # MUST restore ring_width BEFORE any recomputation that uses it
                self.forcing_ring_width = int(params.get("forcing_ring_width", self.forcing_ring_width))
                # Preserve forcing_n_pixels properly: None means auto-match, 0 is invalid
                if "forcing_n_pixels" in params:
                    restored_n = params["forcing_n_pixels"]
                    if restored_n is None or restored_n == 0:
                        # Treat as auto-match: recompute from ring (using restored ring_width)
                        self.forcing_n_pixels = compute_ring_cardinality(self.size, self.forcing_ring_width)
                    else:
                        self.forcing_n_pixels = int(restored_n)
                # else: keep current value (already set or None)
                self.drift_hold = int(params.get("drift_hold", self.drift_hold))
                self.drift_frac = float(params.get("drift_frac", self.drift_frac))
                self.diffusion = float(params.get("diffusion", self.diffusion))
                self.damping = float(params.get("damping", self.damping))
                self.forcing_strength = float(params.get("forcing_strength", self.forcing_strength))
                self.history_window = int(params.get("history_window", self.history_window))
                self.history_stride = int(params.get("history_stride", self.history_stride))
                self.stability_quantile = float(params.get("stability_quantile", self.stability_quantile))
                self.seed = params.get("seed", self.seed)

            # Truthfulness: B was restored from disk, not rebuilt from CLI args
            print(f"Restored: {self.born} | step={self.step:,} (B from disk, CLI forcing args ignored)")

        self._update_forcing_coords()

        print("Manifold: periodic torus")
        print(f"Forcing shape: {self.forcing_shape} (|B|={self.n_forcing})")
        if self.drift_hold > 0:
            print(f"Drift: every {self.drift_hold} steps, {self.drift_frac:.0%} of B attempts to move")
        else:
            print("Drift: disabled (B is fixed)")
        print(f"History: stride={self.history_stride}, window={self.history_window} snapshots")

    def _init_forcing_mask(self):
        self.forcing_mask = np.zeros((self.size, self.size), dtype=bool)

        if self.forcing_shape == "ring":
            w = int(self.forcing_ring_width)
            self.forcing_mask[0:w, :] = True
            self.forcing_mask[-w:, :] = True
            self.forcing_mask[:, 0:w] = True
            self.forcing_mask[:, -w:] = True
            # Ring uses its natural |B| - no sampling needed, pure ring geometry

        elif self.forcing_shape == "random":
            idx = self.rng.choice(self.size * self.size, size=self.forcing_n_pixels, replace=False)
            ys, xs = np.unravel_index(idx, (self.size, self.size))
            self.forcing_mask[ys, xs] = True

        elif self.forcing_shape == "patch":
            # Region-restricted sampling: define generous patch region, sample |B| from it
            # Result is a random subset within the patch support (not a solid block)
            side = int(np.ceil(np.sqrt(self.forcing_n_pixels * 1.5)))  # generous region
            side = min(side, self.size)  # clamp to grid size
            region = np.zeros((self.size, self.size), dtype=bool)
            region[0:side, 0:side] = True
            self.forcing_mask = sample_from_region(region, self.forcing_n_pixels, self.rng)

        elif self.forcing_shape == "corners":
            # Region-restricted sampling: define four corner regions, sample |B| from union
            # Result is a random subset within the corner supports (not solid blocks)
            n_per = max(1, self.forcing_n_pixels // 4)
            side = int(np.ceil(np.sqrt(n_per * 1.5)))  # generous per-corner region
            side = min(side, self.size // 2)  # don't overlap corners
            region = np.zeros((self.size, self.size), dtype=bool)
            region[0:side, 0:side] = True
            region[0:side, -side:] = True
            region[-side:, 0:side] = True
            region[-side:, -side:] = True
            self.forcing_mask = sample_from_region(region, self.forcing_n_pixels, self.rng)

        else:
            raise ValueError(f"Unknown forcing shape: {self.forcing_shape}")

    def _update_forcing_coords(self):
        self.forcing_coords = np.where(self.forcing_mask)
        self.n_forcing = int(self.forcing_mask.sum())

    def _drift_forcing(self):
        """Move a fraction of B to a neighboring unoccupied site (preserves |B|)."""
        if self.drift_frac <= 0:
            return

        ys, xs = np.where(self.forcing_mask)
        n = len(ys)
        if n == 0:
            return

        n_move = max(1, int(self.drift_frac * n))
        pick = self.rng.choice(n, size=n_move, replace=False)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for k in pick:
            y, x = ys[k], xs[k]
            self.rng.shuffle(directions)
            for dy, dx in directions:
                yy, xx = (y + dy) % self.size, (x + dx) % self.size
                if not self.forcing_mask[yy, xx]:
                    self.forcing_mask[y, x] = False
                    self.forcing_mask[yy, xx] = True
                    break

        self._update_forcing_coords()

    def _maybe_store_history(self):
        if self.history_stride <= 1 or (self.step % self.history_stride == 0):
            self.history.append(self.grid.copy())
            if len(self.history) > self.history_window * 2:
                self.history = self.history[-self.history_window:]

    def measure_M(self) -> np.ndarray:
        """M = median(var)/var over snapshot window. DIAGNOSTIC ONLY."""
        if len(self.history) < 20:
            return np.ones((self.size, self.size), dtype=np.float64)

        H = np.array(self.history[-self.history_window:], dtype=np.float64)
        var = np.var(H, axis=0) + 1e-12
        M = np.median(var) / var
        return np.clip(M, self.M_clip_min, self.M_clip_max)

    def stable_mask(self) -> np.ndarray:
        """Bottom-q quantile of variance defines stability (no geometry labels)."""
        if len(self.history) < 20:
            return np.zeros((self.size, self.size), dtype=bool)

        H = np.array(self.history[-self.history_window:], dtype=np.float64)
        var = np.var(H, axis=0)
        thresh = np.quantile(var, self.stability_quantile)
        return var <= thresh

    def step_physics(self) -> None:
        """One substrate step (ROOM ONLY)."""
        # Drift first: new interface applies immediately for this step
        if self.drift_hold > 0 and self.step > 0 and (self.step % self.drift_hold == 0):
            self._drift_forcing()

        # Diffusion on torus
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        lap = up + dn + lf + rt - 4.0 * self.grid

        self.grid += self.diffusion * lap
        self.grid *= 1.0 - self.damping

        # Forcing at B
        noise = self.probe.entropy_floats(self.n_forcing)
        self.grid[self.forcing_coords] += self.forcing_strength * noise

        # Saturation
        self.grid = np.tanh(self.grid)

        # History snapshots (spatial)
        self._maybe_store_history()

        self.step += 1

    def motor_output(self) -> None:
        """Write a sample of the RAW grid state (not derived diagnostics)."""
        if self.motor_strength <= 0:
            return

        k = min(self.motor_sample, self.grid.size)
        idx = np.frombuffer(os.urandom(2 * k), dtype=np.uint16) % self.grid.size
        vals = self.grid.flat[idx]
        # FIX: no redundant tanh (grid already tanh'd in step_physics)
        b = ((vals + 1.0) * 127.5).astype(np.uint8).tobytes()
        nbytes = int(max(16, self.motor_strength * len(b)))
        self.motor.write_bytes(b[:nbytes])

    # -------------------------------------------------------------------------
    # SUSCEPTIBILITY TESTS: Differential susceptibility consistent with χ ∝ 1/M
    # -------------------------------------------------------------------------
    # These tests demonstrate that susceptibility DECREASES with M.
    # We test the ordering, not the functional form.
    # -------------------------------------------------------------------------

    def get_discovered_core_mask(self) -> np.ndarray:
        """
        Return the actual discovered core as a boolean mask.
        
        Core = largest connected component of stable pixels.
        This is the DISCOVERED object, not a pre-defined region.
        """
        from collections import deque
        
        stable = self.stable_mask()
        
        if not stable.any():
            return np.zeros_like(stable, dtype=bool)
        
        # Find LCC using BFS with periodic boundaries
        H, W = stable.shape
        visited = np.zeros_like(stable, dtype=bool)
        
        def bfs_component(start_y, start_x):
            """Return mask of connected component containing (start_y, start_x)."""
            component = np.zeros_like(stable, dtype=bool)
            queue = deque([(start_y, start_x)])  # O(1) popleft vs O(n) list.pop(0)
            visited[start_y, start_x] = True
            component[start_y, start_x] = True
            
            while queue:
                y, x = queue.popleft()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = (y + dy) % H, (x + dx) % W
                    if stable[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        component[ny, nx] = True
                        queue.append((ny, nx))
            return component
        
        largest_component = None
        largest_size = 0
        
        for y in range(H):
            for x in range(W):
                if stable[y, x] and not visited[y, x]:
                    component = bfs_component(y, x)
                    size = component.sum()
                    if size > largest_size:
                        largest_size = size
                        largest_component = component
        
        return largest_component if largest_component is not None else np.zeros_like(stable, dtype=bool)

    def measure_steering(self, perturbation: float = 0.3, settle_steps: int = 30) -> Dict[str, Any]:
        """
        Do high-M regions steer low-M regions?
        
        Test: Perturb the DISCOVERED CORE (LCC of stable), see if periphery follows.
        
        If susceptibility decreases with M:
          - High-M regions resist change, maintain perturbation longer
          - Low-M regions respond to neighbors via diffusion
          - Therefore: perturbing high-M should pull low-M toward it
        
        The discovered core is used (not M-median split) for consistency with 
        what we claim "emerges".
        
        NOTE: Periphery is interior non-core excluding forcing. If this is too
        small (<10 pixels), we fall back to using all unstable pixels. This
        fallback is documented to avoid appearance of post-hoc selection.
        """
        # Get DISCOVERED core (LCC of stable pixels)
        core_mask = self.get_discovered_core_mask()
        
        if core_mask.sum() < 10:
            return {'error': 'Core too small (< 10 pixels)'}
        
        # Periphery = non-forcing pixels that are NOT in core
        # (excluding forcing boundary because it's constantly driven)
        interior_non_core = (~self.forcing_mask) & (~core_mask)
        
        # Fallback: if interior non-core is too small, use unstable pixels
        # This is documented contingency, not post-hoc selection
        if interior_non_core.sum() < 10:
            stable = self.stable_mask()
            periphery_mask = ~stable
            periphery_type = 'unstable_fallback'
        else:
            periphery_mask = interior_non_core
            periphery_type = 'interior_non_core'
        
        if periphery_mask.sum() < 10:
            return {'error': 'Periphery too small'}
        
        # Backup state
        grid_backup = self.grid.copy()
        history_backup = list(self.history)
        step_backup = self.step
        
        periphery_baseline = float(self.grid[periphery_mask].mean())
        core_baseline = float(self.grid[core_mask].mean())
        
        # Perturb CORE
        self.grid[core_mask] += perturbation
        self.grid = np.tanh(self.grid)
        
        for _ in range(settle_steps):
            self.step_physics()
        
        periphery_after = float(self.grid[periphery_mask].mean())
        core_final = float(self.grid[core_mask].mean())
        
        periphery_moved = periphery_after - periphery_baseline
        core_retained = core_final - core_baseline
        
        # Restore
        self.grid = grid_backup
        self.history = history_backup
        self.step = step_backup
        
        # Steering strength: periphery moved toward perturbation (relative to perturbation size)
        steering_strength = periphery_moved / perturbation if perturbation != 0 else 0.0
        
        # Core retention: how much of perturbation did core maintain?
        core_retention = core_retained / perturbation if perturbation != 0 else 0.0
        
        return {
            'periphery_baseline': periphery_baseline,
            'periphery_after': periphery_after,
            'periphery_moved': periphery_moved,
            'core_retained': core_retained,
            'steering_strength': steering_strength,
            'core_retention': core_retention,
            'steering_works': steering_strength > 0.01,  # Periphery moved >1% of perturbation
            'n_core': int(core_mask.sum()),
            'n_periphery': int(periphery_mask.sum()),
            'periphery_type': periphery_type,
        }

    def measure_autonomy(self, perturbation: float = 0.3, settle_steps: int = 30) -> Dict[str, Any]:
        """
        Do high-M regions resist perturbation from low-M regions?
        
        Test: Perturb the PERIPHERY (non-core interior, excluding forcing), 
              see if DISCOVERED CORE resists.
        
        If susceptibility decreases with M:
          - Low-M regions respond strongly, don't maintain perturbation
          - High-M regions resist influence from neighbors
          - Therefore: perturbing low-M should NOT pull high-M
        
        Threshold is RELATIVE to perturbation size for scale consistency.
        
        NOTE: Periphery excludes forcing boundary to avoid artefacts from 
        constant OS entropy injection.
        """
        # Get DISCOVERED core
        core_mask = self.get_discovered_core_mask()
        
        if core_mask.sum() < 10:
            return {'error': 'Core too small'}
        
        # Periphery = interior pixels NOT in core AND NOT on forcing boundary
        # This avoids artefacts from forcing's constant noise injection
        periphery_mask = (~self.forcing_mask) & (~core_mask)
        
        if periphery_mask.sum() < 10:
            return {'error': 'Periphery too small (after excluding forcing)'}
        
        # Backup state
        grid_backup = self.grid.copy()
        history_backup = list(self.history)
        step_backup = self.step
        
        core_baseline = float(self.grid[core_mask].mean())
        
        # Perturb PERIPHERY (interior non-core only)
        self.grid[periphery_mask] += perturbation
        self.grid = np.tanh(self.grid)
        
        for _ in range(settle_steps):
            self.step_physics()
        
        core_after = float(self.grid[core_mask].mean())
        core_drift = abs(core_after - core_baseline)
        
        # Restore
        self.grid = grid_backup
        self.history = history_backup
        self.step = step_backup
        
        # Autonomy: RELATIVE to perturbation size (scale-consistent)
        relative_drift = core_drift / perturbation if perturbation != 0 else 0.0
        autonomy_strength = 1.0 - relative_drift
        autonomy_strength = max(0.0, min(1.0, autonomy_strength))
        
        return {
            'core_baseline': core_baseline,
            'core_after': core_after,
            'core_drift': core_drift,
            'relative_drift': relative_drift,
            'autonomy_strength': autonomy_strength,
            'autonomy_works': relative_drift < 0.15,  # Core drifted < 15% of perturbation
            'n_core': int(core_mask.sum()),
            'n_periphery': int(periphery_mask.sum()),
        }

    def measure_retention_comparison(self, perturbation: float = 0.3, settle_steps: int = 30) -> Dict[str, Any]:
        """
        Direct test: Do high-M regions retain perturbation better than low-M?
        
        Test:
          1. Perturb discovered core (high-M LCC) → measure retention after settling
          2. Perturb a CONNECTED low-M region of comparable size → measure retention
          3. Compare: core should retain MORE (lower susceptibility)
        
        The low-M region is constrained to be CONNECTED to avoid adjacency 
        artefacts. Scattered pixels have different diffusion coupling independent
        of M, so we compare connected regions for a fair test.
        
        This tests the ORDERING (high-M retains better than low-M), not the 
        functional form χ = 1/(1+M).
        """
        from collections import deque
        
        core_mask = self.get_discovered_core_mask()
        n_core = int(core_mask.sum())
        
        if n_core < 10:
            return {'error': 'Core too small'}
        
        # Find CONNECTED low-M region of comparable size
        M = self.measure_M()
        interior_non_core = (~self.forcing_mask) & (~core_mask)
        
        if interior_non_core.sum() < n_core:
            return {'error': 'Not enough non-core interior pixels'}
        
        # Threshold to get low-M pixels (bottom 30% of M in interior non-core)
        interior_M = M[interior_non_core]
        low_M_threshold = np.percentile(interior_M, 30)
        low_M_candidates = interior_non_core & (M <= low_M_threshold)
        
        if low_M_candidates.sum() < 10:
            return {'error': 'Not enough low-M candidates'}
        
        # TWO-PASS APPROACH to avoid visited contamination:
        # Pass 1: Find ALL connected components fully (no truncation)
        # Pass 2: Pick largest, then crop to n_core if needed
        
        H, W = M.shape
        visited = np.zeros_like(low_M_candidates, dtype=bool)
        components = []  # List of (size, seed_point)
        
        def bfs_full(start_y, start_x):
            """Full BFS - find entire component, return all member coords."""
            members = [(start_y, start_x)]
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            
            while queue:
                y, x = queue.popleft()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = (y + dy) % H, (x + dx) % W
                    if low_M_candidates[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        members.append((ny, nx))
                        queue.append((ny, nx))
            return members
        
        # Pass 1: Find all components
        for y in range(H):
            for x in range(W):
                if low_M_candidates[y, x] and not visited[y, x]:
                    members = bfs_full(y, x)
                    components.append(members)
        
        if not components:
            return {'error': 'No connected low-M components found'}
        
        # Find largest component
        largest = max(components, key=len)
        
        if len(largest) < 10:
            return {'error': 'Largest low-M component too small'}
        
        # Pass 2: Crop to n_core if larger (take first n_core by BFS order)
        # BFS order is already preserved in 'largest' list
        selected = largest[:n_core] if len(largest) > n_core else largest
        
        # Build mask
        low_M_mask = np.zeros_like(core_mask, dtype=bool)
        for y, x in selected:
            low_M_mask[y, x] = True
        
        # Verify M difference
        core_M_mean = float(M[core_mask].mean())
        low_M_mean = float(M[low_M_mask].mean())
        
        # === TEST 1: Core retention ===
        grid_backup = self.grid.copy()
        history_backup = list(self.history)
        step_backup = self.step
        
        core_baseline = float(self.grid[core_mask].mean())
        self.grid[core_mask] += perturbation
        self.grid = np.tanh(self.grid)
        
        for _ in range(settle_steps):
            self.step_physics()
        
        core_after = float(self.grid[core_mask].mean())
        core_retention = (core_after - core_baseline) / perturbation
        
        # Restore
        self.grid = grid_backup.copy()
        self.history = list(history_backup)
        self.step = step_backup
        
        # === TEST 2: Low-M retention (connected region) ===
        low_baseline = float(self.grid[low_M_mask].mean())
        self.grid[low_M_mask] += perturbation
        self.grid = np.tanh(self.grid)
        
        for _ in range(settle_steps):
            self.step_physics()
        
        low_after = float(self.grid[low_M_mask].mean())
        low_retention = (low_after - low_baseline) / perturbation
        
        # Restore
        self.grid = grid_backup
        self.history = history_backup
        self.step = step_backup
        
        # High-M should retain MORE (lower susceptibility)
        retention_advantage = core_retention - low_retention
        
        return {
            'core_M_mean': core_M_mean,
            'low_M_mean': low_M_mean,
            'M_ratio': core_M_mean / (low_M_mean + 1e-10),
            'core_retention': core_retention,
            'low_retention': low_retention,
            'retention_advantage': retention_advantage,
            'high_M_retains_better': retention_advantage > 0.05,
            'n_core': n_core,
            'n_low': int(low_M_mask.sum()),
            'low_M_connected': True,
            'n_components_found': len(components),
            'largest_component_size': len(largest),
        }

    def measure_susceptibility_empirical(self) -> Dict[str, Any]:
        """
        Test differential susceptibility consistent with χ decreasing in M.
        
        Three complementary tests:
          1. STEERING: Core perturbation → periphery responds
          2. AUTONOMY: Periphery perturbation → core resists
          3. RETENTION: Core retains perturbation BETTER than low-M region
        
        We test the ORDERING (high-M less susceptible than low-M), not the 
        functional form χ = 1/(1+M). The formula is from theory; we demonstrate
        that emergent structure has differential susceptibility consistent with it.
        
        Returns all results plus overall verdict.
        """
        steering = self.measure_steering()
        autonomy = self.measure_autonomy()
        retention = self.measure_retention_comparison()
        
        errors = []
        if 'error' in steering:
            errors.append(f"steering: {steering['error']}")
        if 'error' in autonomy:
            errors.append(f"autonomy: {autonomy['error']}")
        if 'error' in retention:
            errors.append(f"retention: {retention['error']}")
        
        if errors:
            return {
                'steering': steering,
                'autonomy': autonomy,
                'retention': retention,
                'ordering_validated': False,
                'error': '; '.join(errors),
            }
        
        # Monotone ordering validated if ALL THREE tests pass:
        # 1. Core perturbation causes periphery to respond (steering)
        # 2. Core resists perturbation from periphery (autonomy)
        # 3. Core retains perturbation better than low-M regions (retention)
        # This validates χ DECREASES with M, not the specific form χ = 1/(1+M)
        ordering_validated = (steering['steering_works'] and 
                        autonomy['autonomy_works'] and 
                        retention['high_M_retains_better'])
        
        return {
            'steering': steering,
            'autonomy': autonomy,
            'retention': retention,
            'ordering_validated': ordering_validated,
            'summary': {
                'steering_strength': steering['steering_strength'],
                'core_retention': steering['core_retention'],
                'autonomy_strength': autonomy['autonomy_strength'],
                'retention_advantage': retention['retention_advantage'],
                'core_steers_periphery': steering['steering_works'],
                'core_resists_periphery': autonomy['autonomy_works'],
                'high_M_retains_better': retention['high_M_retains_better'],
            }
        }

    def report(self) -> Dict[str, Any]:
        """Diagnostics only (never affect dynamics)."""
        M = self.measure_M()
        stable = self.stable_mask()
        lcc, n_clusters = largest_connected_component_fraction(stable)

        forcing_M = float(M[self.forcing_mask].mean()) if self.n_forcing > 0 else float("nan")
        non_forcing_M = float(M[~self.forcing_mask].mean()) if (~self.forcing_mask).any() else float("nan")
        M_ratio_nf = non_forcing_M / (forcing_M + 1e-12) if np.isfinite(forcing_M) else float("nan")

        M_stable = float(M[stable].mean()) if stable.any() else float("nan")
        M_unstable = float(M[~stable].mean()) if (~stable).any() else float("nan")
        M_ratio_stable = M_stable / (M_unstable + 1e-12) if np.isfinite(M_unstable) else float("nan")

        return {
            "step": int(self.step),
            "grid_std": float(np.std(self.grid)),
            "M_ratio_nonforced_over_forced": float(M_ratio_nf),
            "M_ratio_stable_over_unstable": float(M_ratio_stable),
            "M_stable": float(M_stable),
            "M_unstable": float(M_unstable),
            "core_coherence": float(lcc),
            "n_stable_clusters": int(n_clusters),
            "n_stable": int(stable.sum()),
            "n_forcing": int(self.n_forcing),
            "motor_writes": int(self.motor.write_count),
        }

    def save(self, metrics: Dict[str, Any]) -> None:
        params = {
            "forcing_shape": str(self.forcing_shape),
            "forcing_n_pixels": self.forcing_n_pixels,  # can be int or None
            "forcing_ring_width": int(self.forcing_ring_width),
            "drift_hold": int(self.drift_hold),
            "drift_frac": float(self.drift_frac),
            "diffusion": float(self.diffusion),
            "damping": float(self.damping),
            "forcing_strength": float(self.forcing_strength),
            "history_window": int(self.history_window),
            "history_stride": int(self.history_stride),
            "stability_quantile": float(self.stability_quantile),
            "seed": self.seed,
        }
        self.persistence.save(
            self.grid, self.forcing_mask, self.step, self.born, metrics, self.history, params=params
        )


# -----------------------------------------------------------------------------
# Daemon runner
# -----------------------------------------------------------------------------

class Daemon:
    def __init__(self, sub: HonestSubstrate, steps_per_sec: float = 50.0):
        self.sub = sub
        self.dt = 1.0 / max(0.1, float(steps_per_sec))
        self.running = True
        self.last_metrics: Optional[Dict[str, Any]] = None

        signal.signal(signal.SIGINT, self._handle_signal)
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
        except Exception:
            pass

    def _handle_signal(self, *_):
        print("\nSignal received, stopping...")
        self.running = False

    def run(self):
        print(f"\nRunning honest daemon at {1/self.dt:.1f} steps/sec")
        print("Core: DISCOVERED as LCC of stable pixels (variance-quantile)")
        print("M-feedback: NONE")
        print("Ctrl+C to stop\n")

        while self.running:
            t0 = time.time()

            self.sub.step_physics()

            # motor output occasionally
            if self.sub.step % 10 == 0:
                self.sub.motor_output()

            if self.sub.step % self.sub.report_interval == 0:
                m = self.sub.report()
                self.last_metrics = m
                coh = m["core_coherence"]
                coh_str = f"{coh:.1%}" if not np.isnan(coh) else "warming"
                print(
                    f"[{m['step']:>8}] "
                    f"M_nf/B={m['M_ratio_nonforced_over_forced']:.1f}x  "
                    f"M_st/u={m['M_ratio_stable_over_unstable']:.2f}x  "
                    f"coherence={coh_str:>8}  "
                    f"clusters={m['n_stable_clusters']}"
                )

            if self.sub.step % self.sub.save_interval == 0:
                self.sub.save(self.last_metrics or {"step": int(self.sub.step)})

            elapsed = time.time() - t0
            if self.dt > elapsed:
                time.sleep(self.dt - elapsed)

        print("\nStopping...")
        self.sub.save(self.last_metrics or {"step": int(self.sub.step)})
        print("Done.")


# -----------------------------------------------------------------------------
# Ablation harness
# -----------------------------------------------------------------------------

def run_ablation(size=64, n_steps=10000, forcing_width=2, seed=None,
                 diffusion=0.18, damping=0.001, forcing_strength=0.12):
    """
    Run ablation table with:
      - Ring uses natural |B|; other shapes match it
      - stable mask: bottom q=0.3 of variance over W=60 snapshots sampled every 20 steps
      - coherence: LCC(stable)/|stable|
      - <d> and d_max computed from graph distance to B on torus
    """
    rng = np.random.default_rng(seed)

    # Ring uses natural cardinality; match other shapes to it
    forcing_n = compute_ring_cardinality(size, forcing_width)

    print("=" * 80)
    print("ABLATION TABLE: Coherence requires persistent distance structure")
    print("=" * 80)
    print()
    print("Definitions:")
    print("  Stable mask: pixels in bottom q=0.3 quantile of temporal variance")
    print("               (W=60 snapshots, sampled every 20 steps)")
    print("  Coherence: LCC(stable) / |stable|")
    print("  <d>: mean graph distance to B (on torus, 4-neighbor)")
    print("  d_max: maximum graph distance to B (max over time for drift runs)")
    print()
    print(f"Ring natural |B| = {forcing_n} (width={forcing_width} on {size}x{size} torus)")
    print(f"Physics: diffusion={diffusion}, damping={damping}, forcing={forcing_strength}")
    print()

    results = []

    configs = [
        {"name": "Ring (fixed)", "shape": "ring", "drift_hold": 0},
        {"name": "Random scattered (fixed)", "shape": "random", "drift_hold": 0},
        {"name": "Patch (fixed)", "shape": "patch", "drift_hold": 0},
        {"name": "Corners (fixed)", "shape": "corners", "drift_hold": 0},
        {"name": "Slow drift (hold=1000)", "shape": "ring", "drift_hold": 1000, "drift_frac": 0.05},
        {"name": "Medium drift (hold=100)", "shape": "ring", "drift_hold": 100, "drift_frac": 0.1},
        {"name": "Fast drift (hold=10)", "shape": "ring", "drift_hold": 10, "drift_frac": 0.2},
        {"name": "Random each step", "shape": "random_each_step", "drift_hold": 0},
    ]

    for cfg in configs:
        print(f"Running: {cfg['name']}...")

        grid = rng.uniform(-0.05, 0.05, (size, size))
        forcing_mask = np.zeros((size, size), dtype=bool)

        if cfg["shape"] == "ring":
            w = int(forcing_width)
            forcing_mask[0:w, :] = True
            forcing_mask[-w:, :] = True
            forcing_mask[:, 0:w] = True
            forcing_mask[:, -w:] = True
            # Ring uses natural |B| - pure ring geometry

        elif cfg["shape"] == "random":
            idx = rng.choice(size * size, size=forcing_n, replace=False)
            ys, xs = np.unravel_index(idx, (size, size))
            forcing_mask[ys, xs] = True

        elif cfg["shape"] == "patch":
            # Region-restricted sampling: sample exactly forcing_n from patch region
            side = int(np.ceil(np.sqrt(forcing_n * 1.5)))  # generous region
            side = min(side, size)
            region = np.zeros((size, size), dtype=bool)
            region[0:side, 0:side] = True
            forcing_mask = sample_from_region(region, forcing_n, rng)

        elif cfg["shape"] == "corners":
            # Region-restricted sampling: sample exactly forcing_n from corner regions
            n_per = max(1, forcing_n // 4)
            side = int(np.ceil(np.sqrt(n_per * 1.5)))  # generous per-corner region
            side = min(side, size // 2)  # don't overlap corners
            region = np.zeros((size, size), dtype=bool)
            region[0:side, 0:side] = True
            region[0:side, -side:] = True
            region[-side:, 0:side] = True
            region[-side:, -side:] = True
            forcing_mask = sample_from_region(region, forcing_n, rng)

        elif cfg["shape"] == "random_each_step":
            pass

        drift_hold = cfg.get("drift_hold", 0)
        drift_frac = cfg.get("drift_frac", 0.1)

        mean_d_samples = []
        max_d_samples = []

        if cfg["shape"] != "random_each_step":
            dist = compute_distance_to_B(forcing_mask)
            mean_d_samples.append(float(dist[~forcing_mask].mean()))
            max_d_samples.append(float(dist.max()))

        history = []

        for step in range(n_steps):
            # drift before forcing (timescale separation test)
            if drift_hold > 0 and step > 0 and (step % drift_hold == 0):
                ys, xs = np.where(forcing_mask)
                n = len(ys)
                n_move = max(1, int(drift_frac * n))
                pick = rng.choice(n, size=n_move, replace=False)
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for k in pick:
                    y, x = ys[k], xs[k]
                    rng.shuffle(directions)
                    for dy, dx in directions:
                        yy, xx = (y + dy) % size, (x + dx) % size
                        if not forcing_mask[yy, xx]:
                            forcing_mask[y, x] = False
                            forcing_mask[yy, xx] = True
                            break

                dist = compute_distance_to_B(forcing_mask)
                mean_d_samples.append(float(dist[~forcing_mask].mean()))
                max_d_samples.append(float(dist.max()))

            if cfg["shape"] == "random_each_step":
                forcing_mask = np.zeros((size, size), dtype=bool)
                idx = rng.choice(size * size, size=forcing_n, replace=False)
                ys, xs = np.unravel_index(idx, (size, size))
                forcing_mask[ys, xs] = True

            # physics (uses passed parameters)
            up = np.roll(grid, 1, axis=0)
            dn = np.roll(grid, -1, axis=0)
            lf = np.roll(grid, 1, axis=1)
            rt = np.roll(grid, -1, axis=1)
            lap = up + dn + lf + rt - 4 * grid
            grid += diffusion * lap
            grid *= (1.0 - damping)

            # forcing
            coords = np.where(forcing_mask)
            noise = rng.uniform(-1, 1, forcing_mask.sum())
            grid[coords] += forcing_strength * noise

            grid = np.tanh(grid)

            # snapshot every 20 steps
            if step % 20 == 0:
                history.append(grid.copy())
                if len(history) > 100:
                    history = history[-100:]

        mean_dist = float(np.mean(mean_d_samples)) if mean_d_samples else float("nan")
        max_dist = float(np.max(max_d_samples)) if max_d_samples else float("nan")

        H = np.array(history[-60:], dtype=np.float64)
        var = np.var(H, axis=0) + 1e-12
        thresh = np.quantile(var, 0.3)
        stable = var <= thresh

        lcc, n_clusters = largest_connected_component_fraction(stable)

        results.append({
            "name": cfg["name"],
            "mean_dist": mean_dist,
            "max_dist": max_dist,
            "coherence": lcc,
            "n_clusters": n_clusters,
        })

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"{'Configuration':<30} {'<d>':>8} {'d_max':>8} {'Coherence':>12} {'Clusters':>10}")
    print("-" * 80)

    for r in results:
        d_str = f"{r['mean_dist']:.1f}" if not np.isnan(r["mean_dist"]) else "varies"
        dmax_str = f"{r['max_dist']:.0f}" if not np.isnan(r["max_dist"]) else "varies"
        coh_str = f"{r['coherence']:.1%}" if not np.isnan(r["coherence"]) else "N/A"
        print(f"{r['name']:<30} {d_str:>8} {dmax_str:>8} {coh_str:>12} {r['n_clusters']:>10}")

    return results


def run_susceptibility_test(size: int = 64, n_steps: int = 2000, seed: int = None,
                            forcing_shape: str = "ring"):
    """
    DIFFERENTIAL SUSCEPTIBILITY TEST
    
    Demonstrates that susceptibility DECREASES with M (emerged stability metric).
    We test the ORDERING, not the functional form χ = 1/(1+M).
    
    Tests:
      1. STEERING: Perturb DISCOVERED core (LCC), see if periphery follows
         - High-M regions maintain perturbation longer
         - Low-M regions respond to diffusion from perturbed neighbors
      
      2. AUTONOMY: Perturb periphery, see if discovered core resists
         - Uses RELATIVE threshold (scale-consistent)
         - High-M regions should resist influence from low-M
      
      3. RETENTION: Compare core vs connected low-M region
         - Same perturbation, same dynamics, different M
         - High-M should retain more (lower susceptibility)
    
    NOTE: Each run uses a FRESH substrate (no persistence) for controlled testing.
    """
    import tempfile
    
    print("=" * 70)
    print("DIFFERENTIAL SUSCEPTIBILITY TEST")
    print("=" * 70)
    print()
    print("Tests that susceptibility DECREASES with M (emerged stability).")
    print("We test the ordering, not the functional form χ = 1/(1+M).")
    print()
    print("Key properties:")
    print("  - Uses DISCOVERED core (LCC of stable pixels)")
    print("  - Low-M comparison is CONNECTED (avoids adjacency artefacts)")
    print("  - Uses RELATIVE thresholds (scale-consistent)")
    print("  - Each run is FRESH (no persistence contamination)")
    print()
    
    # Run multiple seeds for robustness
    seeds = [seed] if seed is not None else [1, 2, 3, 4, 5]
    
    all_results = []
    
    for s in seeds:
        print(f"--- Seed {s} ---")
        
        # Use temp directory to ensure fresh start (no persistence contamination)
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = HonestSubstrate(
                size=size,
                forcing_shape=forcing_shape,
                seed=s,
                state_dir=Path(tmpdir),  # Fresh state, no loading from disk
            )
            
            # Let dynamics reach steady state
            print(f"  Running {n_steps} steps to reach steady state...")
            for _ in range(n_steps):
                sub.step_physics()
            
            # Get baseline M stats
            r = sub.report()
            print(f"  M_ratio (stable/unstable): {r['M_ratio_stable_over_unstable']:.1f}x")
            print(f"  Coherence: {r['core_coherence']:.1%}")
            
            # Get discovered core size
            core_mask = sub.get_discovered_core_mask()
            print(f"  Discovered core: {core_mask.sum()} pixels")
            
            # Run susceptibility tests
            print(f"  Running susceptibility tests...")
            chi_result = sub.measure_susceptibility_empirical()
            
            if 'error' in chi_result:
                print(f"  ERROR: {chi_result['error']}")
                continue
            
            steering = chi_result['steering']
            autonomy = chi_result['autonomy']
            retention = chi_result['retention']
            
            print(f"  STEERING (perturb core, measure periphery response):")
            print(f"    Periphery moved: {steering['periphery_moved']:.4f}")
            print(f"    Steering strength: {steering['steering_strength']:.3f}")
            print(f"    Core retention: {steering['core_retention']:.3f}")
            print(f"    Steering works: {steering['steering_works']}")
            print(f"  AUTONOMY (perturb periphery, measure core resistance):")
            print(f"    Core drift: {autonomy['core_drift']:.4f}")
            print(f"    Relative drift: {autonomy['relative_drift']:.3f}")
            print(f"    Autonomy strength: {autonomy['autonomy_strength']:.3f}")
            print(f"    Core resists: {autonomy['autonomy_works']}")
            print(f"  RETENTION COMPARISON (direct χ test):")
            print(f"    Core M (mean): {retention['core_M_mean']:.1f}")
            print(f"    Low-M region M (mean): {retention['low_M_mean']:.1f}")
            print(f"    M ratio: {retention['M_ratio']:.1f}x")
            print(f"    Core retention: {retention['core_retention']:.3f}")
            print(f"    Low-M retention: {retention['low_retention']:.3f}")
            print(f"    Advantage (core - low): {retention['retention_advantage']:.3f}")
            print(f"    High-M retains better: {retention['high_M_retains_better']}")
            print(f"  DIFFERENTIAL SUSCEPTIBILITY: {'✓' if chi_result['ordering_validated'] else '✗'}")
            print()
            
            all_results.append({
                'seed': s,
                'M_ratio': r['M_ratio_stable_over_unstable'],
                'coherence': r['core_coherence'],
                'core_size': core_mask.sum(),
                'steering_strength': steering['steering_strength'],
                'core_retention': steering['core_retention'],
                'steering_works': steering['steering_works'],
                'relative_drift': autonomy['relative_drift'],
                'autonomy_strength': autonomy['autonomy_strength'],
                'autonomy_works': autonomy['autonomy_works'],
                'retention_advantage': retention['retention_advantage'],
                'high_M_retains_better': retention['high_M_retains_better'],
                'ordering_validated': chi_result['ordering_validated'],
            })
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    if len(all_results) == 0:
        print("No successful runs!")
        return []
    
    n_pass = sum(1 for r in all_results if r['ordering_validated'])
    n_total = len(all_results)
    
    print(f"Seeds tested: {n_total}")
    print(f"Ordering validated: {n_pass}/{n_total}")
    print()
    
    mean_steering = np.mean([r['steering_strength'] for r in all_results])
    mean_retention = np.mean([r['core_retention'] for r in all_results])
    mean_autonomy = np.mean([r['autonomy_strength'] for r in all_results])
    mean_retention_adv = np.mean([r['retention_advantage'] for r in all_results])
    
    print(f"Mean steering strength: {mean_steering:.3f}")
    print(f"Mean core retention: {mean_retention:.3f}")
    print(f"Mean autonomy strength: {mean_autonomy:.3f}")
    print(f"Mean retention advantage (core - low-M): {mean_retention_adv:.3f}")
    print()
    
    if n_pass == n_total:
        print("✅ DIFFERENTIAL SUSCEPTIBILITY DEMONSTRATED")
        print()
        print("Three tests passed:")
        print("  1. STEERING: Core perturbation → periphery responds")
        print("  2. AUTONOMY: Periphery perturbation → core resists")
        print("  3. RETENTION: High-M retains better than connected low-M region")
        print()
        print("Susceptibility decreases with M (monotone ordering validated).")
    elif n_pass > n_total / 2:
        print(f"⚠️ Passed in {n_pass}/{n_total} runs")
        print("  Majority pass but not unanimous - check edge cases")
    else:
        print(f"❌ Failed in {n_total - n_pass}/{n_total} runs")
        print("  Check parameters or increase n_steps")
    
    return all_results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="TBU Honest Daemon with ablation support")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--forcing-shape", choices=["ring", "random", "patch", "corners"], default="ring")
    ap.add_argument("--forcing-n", type=int, default=None, help="Target |B| for random/patch/corners (None = auto-match ring)")
    ap.add_argument("--forcing-width", type=int, default=2, help="Ring width for forcing_shape=ring")
    ap.add_argument("--drift-hold", type=int, default=0, help="Steps between drift events (0 disables)")
    ap.add_argument("--drift-frac", type=float, default=0.1, help="Fraction of B that attempts to drift")
    ap.add_argument("--steps-per-sec", type=float, default=50.0)
    ap.add_argument("--diffusion", type=float, default=0.18)
    ap.add_argument("--damping", type=float, default=0.001)
    ap.add_argument("--forcing-strength", type=float, default=0.12)
    ap.add_argument("--motor", type=float, default=0.05)
    ap.add_argument("--motor-sample", type=int, default=256)
    ap.add_argument("--history-window", type=int, default=100)
    ap.add_argument("--history-stride", type=int, default=1, help="Store snapshot every k steps (daemon)")
    ap.add_argument("--stability-quantile", type=float, default=0.3)
    ap.add_argument("--save-interval", type=int, default=5000)
    ap.add_argument("--report-interval", type=int, default=500)
    ap.add_argument("--state-dir", type=Path, default=Path.home() / ".tbu_honest")
    ap.add_argument("--fresh", action="store_true", help="Delete existing state and start fresh (ignore saved B)")
    ap.add_argument("--ablation", action="store_true", help="Run ablation table and exit")
    ap.add_argument("--ablation-steps", type=int, default=10000)
    ap.add_argument("--susceptibility", action="store_true", help="Run susceptibility (χ) validation test and exit")
    ap.add_argument("--susceptibility-steps", type=int, default=2000, help="Steps before running susceptibility test")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")

    args = ap.parse_args()

    if args.ablation:
        run_ablation(
            size=args.size,
            n_steps=args.ablation_steps,
            forcing_width=args.forcing_width,
            seed=args.seed,
            diffusion=args.diffusion,
            damping=args.damping,
            forcing_strength=args.forcing_strength,
        )
        return

    if args.susceptibility:
        run_susceptibility_test(
            size=args.size,
            n_steps=args.susceptibility_steps,
            seed=args.seed,
            forcing_shape=args.forcing_shape,
        )
        return

    # Handle --fresh: delete existing state before creating substrate
    if args.fresh:
        state_file = args.state_dir / "state.npz"
        meta_file = args.state_dir / "meta.json"
        if state_file.exists() or meta_file.exists():
            print(f"--fresh: Deleting existing state in {args.state_dir}")
            if state_file.exists():
                state_file.unlink()
            if meta_file.exists():
                meta_file.unlink()

    sub = HonestSubstrate(
        size=args.size,
        forcing_shape=args.forcing_shape,
        forcing_n_pixels=args.forcing_n,
        forcing_ring_width=args.forcing_width,
        drift_hold=args.drift_hold,
        drift_frac=args.drift_frac,
        diffusion=args.diffusion,
        damping=args.damping,
        forcing_strength=args.forcing_strength,
        motor_strength=args.motor,
        motor_sample=args.motor_sample,
        history_window=args.history_window,
        history_stride=args.history_stride,
        stability_quantile=args.stability_quantile,
        state_dir=args.state_dir,
        motor_dir=args.state_dir / "motor",
        save_interval=args.save_interval,
        report_interval=args.report_interval,
        seed=args.seed,
    )

    Daemon(sub, steps_per_sec=args.steps_per_sec).run()


if __name__ == "__main__":
    main()
