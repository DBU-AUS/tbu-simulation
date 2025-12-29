#!/usr/bin/env python3
"""
================================================================================
LEVEL 39: PATTERN IDENTITY
================================================================================

THE GAP FROM L38:
  L38 measured activity level (S_stability = 1/median_var).
  When forcing stopped, S_stability increased (mesh got quieter).
  But we don't know if the PATTERN persists or degrades.
  
  "Quiet" ≠ "preserved identity"
  
  We measured: Is the mesh stable?
  We need:    Does it still look like training?

WHAT L39 ASKS:
  Does the mesh pattern persist when forcing stops, or does reconditioning
  smooth it away?

THE TEST:
  1. Store a training snapshot of the core region pattern
  2. After forcing stops, periodically compare current core to training snapshot
  3. Track pattern correlation over time
  
  If pattern persists → reconditioning preserves high-traffic patterns
  If pattern degrades → reconditioning smooths patterns without reinforcement

METHODOLOGY:
  - Pattern snapshot: actual pixel values in core region (not variance)
  - Pattern correlation: correlation coefficient between snapshot and current
  - Track both PATTERN identity and ACTIVITY level (S_stability)
  
  This separates:
    - "Is it quiet?" (S_stability)
    - "Does it look the same?" (pattern correlation)

REAL TBU STACK:
  - SharedSiliconSubstrate (96×48, basin-bottleneck)
  - SharedBasinConsequence (consequence loop)
  - Observer (EMA, volatility, cores, coherence)
  - Real NDBC ocean buoy data

================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque

# Try to import real ocean loader (works if multi_physics_loader.py is in path)
try:
    from multi_physics_loader import load_ocean_buoy
    HAS_LOADER = True
except ImportError:
    HAS_LOADER = False


# =============================================================================
# REAL TBU STACK (from L38)
# =============================================================================

@dataclass
class SharedSiliconSubstrate:
    """Real TBU substrate with basin-bottleneck geometry and pattern tracking."""
    
    width: int = 96
    height: int = 48
    neck_width: int = 8
    forcing_width: int = 3
    diffusion: float = 0.12
    diffusion_neck: float = 0.04
    damping: float = 0.015
    forcing_strength: float = 0.06
    seed: int = 42
    n_observers: int = 4
    history_window: int = 100
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.grid = self.rng.normal(0, 0.1, (self.height, self.width))
        
        # Neck mask (bottleneck between basins)
        mid_x = self.width // 2
        half_neck = self.neck_width // 2
        self.neck_mask = np.zeros((self.height, self.width), dtype=bool)
        self.neck_mask[:, mid_x - half_neck:mid_x + half_neck] = True
        
        # Core regions (where patterns form)
        quarter_x = self.width // 4
        mid_y = self.height // 2
        core_size = 8
        
        self.core_A_mask = (
            (np.abs(np.arange(self.height)[:, None] - mid_y) < core_size) &
            (np.abs(np.arange(self.width)[None, :] - quarter_x) < core_size)
        )
        self.core_A_coords = np.where(self.core_A_mask)
        
        self.core_B_mask = (
            (np.abs(np.arange(self.height)[:, None] - mid_y) < core_size) &
            (np.abs(np.arange(self.width)[None, :] - (3 * quarter_x)) < core_size)
        )
        self.core_B_coords = np.where(self.core_B_mask)
        
        # Basin masks
        self.basin_A_mask = np.zeros((self.height, self.width), dtype=bool)
        self.basin_A_mask[:, :mid_x - half_neck] = True
        
        self.basin_B_mask = np.zeros((self.height, self.width), dtype=bool)
        self.basin_B_mask[:, mid_x + half_neck:] = True
        
        # Observer boundary regions
        obs_height = self.height // 2 - 2
        
        self.observer_coords = []
        self.observer_basin = []
        
        for i in range(2):
            y_start = 2 if i == 0 else self.height // 2 + 2
            coords = []
            for y in range(y_start, y_start + obs_height):
                for x in range(self.forcing_width, self.forcing_width + 3):
                    coords.append((y, x))
            for y in range(y_start, y_start + 3):
                for x in range(self.forcing_width + 3, self.forcing_width + 12):
                    coords.append((y, x))
            coords = coords[:72]
            rows, cols = zip(*coords)
            self.observer_coords.append((np.array(rows), np.array(cols)))
            self.observer_basin.append("A")
        
        for i in range(2):
            y_start = 2 if i == 0 else self.height // 2 + 2
            coords = []
            for y in range(y_start, y_start + obs_height):
                for x in range(self.width - self.forcing_width - 3, self.width - self.forcing_width):
                    coords.append((y, x))
            for y in range(y_start, y_start + 3):
                for x in range(self.width - self.forcing_width - 12, self.width - self.forcing_width - 3):
                    coords.append((y, x))
            coords = coords[:72]
            rows, cols = zip(*coords)
            self.observer_coords.append((np.array(rows), np.array(cols)))
            self.observer_basin.append("B")
        
        self.observer_consumptions = [0.0] * self.n_observers
        self.history: List[np.ndarray] = []
        
        # Pattern snapshots (for L39)
        self.training_snapshot_A: Optional[np.ndarray] = None
        self.training_snapshot_B: Optional[np.ndarray] = None
        
        # Histories
        self.core_A_history: List[float] = []
        self.core_B_history: List[float] = []
    
    def step(self, consumptions: List[float] = None, 
             basin_forcing: Dict[str, float] = None):
        if consumptions is None:
            consumptions = [0.0] * self.n_observers
        self.observer_consumptions = consumptions
        
        self.history.append(self.grid.copy())
        if len(self.history) > self.history_window:
            self.history.pop(0)
        
        # Diffusion (reconditioning - entropy maximizing)
        laplacian = (
            np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
            np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) - 4 * self.grid
        )
        
        eff_diff = np.clip(self.diffusion, 0.02, 0.18)
        diff_coeff = np.where(self.neck_mask, self.diffusion_neck, eff_diff)
        self.grid += diff_coeff * laplacian
        
        # Observer forcing
        for i, (coords, cons) in enumerate(zip(self.observer_coords, consumptions)):
            local_forcing = self.forcing_strength * (1.0 + 0.5 * cons)
            noise = self.rng.normal(0, local_forcing, len(coords[0]))
            self.grid[coords] += noise
        
        # Basin forcing (external pattern)
        if basin_forcing:
            if "A" in basin_forcing and basin_forcing["A"] != 0:
                pattern_A = basin_forcing["A"] * 0.02
                self.grid[self.core_A_coords] += pattern_A
            if "B" in basin_forcing and basin_forcing["B"] != 0:
                pattern_B = basin_forcing["B"] * 0.02
                self.grid[self.core_B_coords] += pattern_B
        
        # Damping
        self.grid *= (1.0 - self.damping)
        self.grid = np.clip(self.grid, -5.0, 5.0)
        
        # Track core means
        self.core_A_history.append(self.get_core_mean("A"))
        self.core_B_history.append(self.get_core_mean("B"))
    
    def get_core_pattern(self, basin: str) -> np.ndarray:
        """Get the current pattern in a core region (actual pixel values)."""
        if basin == "A":
            return self.grid[self.core_A_mask].copy()
        return self.grid[self.core_B_mask].copy()
    
    def store_training_snapshot(self, basin: str):
        """Store the current core pattern as the training reference."""
        pattern = self.get_core_pattern(basin)
        if basin == "A":
            self.training_snapshot_A = pattern.copy()
        else:
            self.training_snapshot_B = pattern.copy()
    
    def get_pattern_correlation(self, basin: str) -> float:
        """
        Correlate current core pattern with training snapshot.
        
        This measures PATTERN IDENTITY, not activity level.
        High correlation = pattern preserved
        Low correlation = pattern degraded/changed
        """
        current = self.get_core_pattern(basin)
        
        if basin == "A":
            snapshot = self.training_snapshot_A
        else:
            snapshot = self.training_snapshot_B
        
        if snapshot is None:
            return 0.0
        
        # Mean-center both (compare shape, not absolute level)
        current_centered = current - np.mean(current)
        snapshot_centered = snapshot - np.mean(snapshot)
        
        # Correlation coefficient
        std_c = np.std(current_centered)
        std_s = np.std(snapshot_centered)
        
        if std_c < 1e-10 or std_s < 1e-10:
            return 0.0
        
        r = np.corrcoef(current_centered, snapshot_centered)[0, 1]
        
        if not np.isfinite(r):
            return 0.0
        
        return float(r)
    
    def get_pattern_cosine(self, basin: str) -> float:
        """
        Cosine similarity between current pattern and training snapshot.
        
        Alternative to correlation - may be more stable.
        """
        current = self.get_core_pattern(basin)
        
        if basin == "A":
            snapshot = self.training_snapshot_A
        else:
            snapshot = self.training_snapshot_B
        
        if snapshot is None:
            return 0.0
        
        norm_c = np.linalg.norm(current)
        norm_s = np.linalg.norm(snapshot)
        
        if norm_c < 1e-10 or norm_s < 1e-10:
            return 0.0
        
        cos = np.dot(current, snapshot) / (norm_c * norm_s)
        
        return float(cos)
    
    def get_stability_metrics(self, basin: str) -> Dict:
        """Get S_stability and related metrics for a basin."""
        if len(self.history) < 20:
            return {"S_stability": 0, "median_var": 1e-6}
        
        mask = self.basin_A_mask if basin == "A" else self.basin_B_mask
        
        H = np.array(self.history[-min(self.history_window, len(self.history)):])
        var = np.var(H, axis=0) + 1e-12
        
        region_var = var[mask]
        median_var = float(np.median(region_var))
        S_stability = 1.0 / (median_var + 1e-12)
        
        return {
            "S_stability": S_stability,
            "median_var": median_var,
        }
    
    def get_core_mean(self, basin: str) -> float:
        if basin == "A":
            return float(self.grid[self.core_A_coords].mean())
        return float(self.grid[self.core_B_coords].mean())
    
    def get_observation(self, observer_id: int) -> np.ndarray:
        return self.grid[self.observer_coords[observer_id]]


@dataclass
class SharedBasinConsequence:
    """Real consequence loop with depletion and recovery."""
    
    depletion_rate: float = 0.012
    recovery_rate: float = 0.008
    noise_base: float = 0.01
    noise_scale: float = 0.4
    lag: int = 25
    
    def __post_init__(self):
        self.sensing_health = 1.0
        self.consumption_buffer: List[float] = []
        self.health_history: List[float] = []
    
    def update(self, total_consumption: float) -> float:
        self.consumption_buffer.append(total_consumption)
        if len(self.consumption_buffer) > self.lag:
            lagged = self.consumption_buffer.pop(0)
            depletion = self.depletion_rate * lagged
            recovery = self.recovery_rate * (1.0 - self.sensing_health)
            self.sensing_health = float(np.clip(
                self.sensing_health - depletion + recovery, 0.0, 1.0
            ))
        self.health_history.append(self.sensing_health)
        return self.noise_base + self.noise_scale * (1.0 - self.sensing_health)


@dataclass
class Observer:
    """Real observer with L35 parameters."""
    
    observer_id: int
    seed: int = 42
    n_blocks: int = 6
    ema_alpha: float = 0.05
    vol_window: int = 50
    vol_gain: float = 25.0
    n_cores: int = 16
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed + self.observer_id * 1000)
        self.core_states = np.zeros(self.n_cores)
        self.core_thresholds = self.rng.uniform(0.3, 0.7, self.n_cores)
        
        self.ema = None
        self.input_history: List[np.ndarray] = []
        self.state_history: List[np.ndarray] = []
        
        self.ema_mean_history: List[float] = []
        self.volatility_history: List[float] = []
        self.consumption_history: List[float] = []
        self.coherence_history: List[float] = []
    
    def _block_sample(self, obs: np.ndarray) -> np.ndarray:
        n = len(obs)
        block_size = n // self.n_blocks
        means = []
        for i in range(self.n_blocks):
            start = i * block_size
            end = start + block_size if i < self.n_blocks - 1 else n
            means.append(np.mean(obs[start:end]))
        return np.array(means)
    
    def step(self, observation: np.ndarray, external_noise: float = 0.1) -> Dict:
        noisy_obs = observation + self.rng.normal(0, external_noise, len(observation))
        sampled = self._block_sample(noisy_obs)
        
        if self.ema is None:
            self.ema = sampled.copy()
        else:
            self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * sampled
        
        self.ema_mean_history.append(float(np.mean(self.ema)))
        
        self.input_history.append(self.ema.copy())
        if len(self.input_history) > self.vol_window + 20:
            self.input_history.pop(0)
        
        volatility = self._compute_volatility()
        self.volatility_history.append(volatility)
        
        capacity = 1.0 / (1.0 + self.vol_gain * volatility)
        consumption = 0.05 + 0.85 * capacity
        self.consumption_history.append(consumption)
        
        self._update_cores(self.ema)
        
        coherence = self._compute_coherence()
        self.coherence_history.append(coherence)
        
        return {
            "consumption": consumption,
            "coherence": coherence,
            "volatility": volatility,
            "ema_mean": self.ema_mean_history[-1],
        }
    
    def _compute_volatility(self) -> float:
        if len(self.input_history) < 2:
            return 0.0
        window = min(self.vol_window, len(self.input_history))
        recent = self.input_history[-window:]
        if len(recent) < 2:
            return 0.0
        diffs = [np.mean(np.abs(recent[i] - recent[i-1])) for i in range(1, len(recent))]
        return float(np.mean(diffs))
    
    def _update_cores(self, inputs: np.ndarray):
        n_inputs = len(inputs)
        for i in range(self.n_cores):
            idx = i % n_inputs
            drive = float(inputs[idx])
            total = np.clip(drive - self.core_thresholds[i], -10, 10)
            activation = 1.0 / (1.0 + np.exp(-4.0 * total))
            self.core_states[i] = 0.9 * self.core_states[i] + 0.1 * activation
        
        self.state_history.append(self.core_states.copy())
        if len(self.state_history) > 100:
            self.state_history.pop(0)
    
    def _compute_coherence(self) -> float:
        if len(self.state_history) < 20:
            return 0.5
        recent = np.array(self.state_history[-50:])
        mean_state = recent.mean(axis=1)
        if np.std(mean_state) < 1e-10:
            return 0.5
        r = np.corrcoef(mean_state[:-1], mean_state[1:])[0, 1]
        return float((r + 1) / 2) if np.isfinite(r) else 0.5


# =============================================================================
# REAL OCEAN DATA
# =============================================================================

def load_real_ocean_data(hours: int = 48) -> Tuple[np.ndarray, Dict]:
    """Load real NDBC ocean data."""
    
    if HAS_LOADER:
        try:
            data, meta = load_ocean_buoy(station="46025", hours=hours)
            print(f"Loaded REAL ocean data: {meta['n_samples']} samples from {meta['station']}")
            return data[:, 0], meta
        except Exception as e:
            print(f"Ocean data fetch failed: {e}")
    
    print("Using synthetic ocean data (real data unavailable)")
    rng = np.random.default_rng(42)
    n = hours * 6
    t = np.linspace(0, hours, n)
    
    data = (
        0.3 * np.sin(2*np.pi*t/12.42) +
        0.2 * np.sin(2*np.pi*t/24) +
        0.1 * np.sin(2*np.pi*t/6) +
        0.15 * rng.normal(0, 1, n)
    )
    
    return data, {"source": "synthetic", "n_samples": n}


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_pattern_identity(n_steps: int = 25000, seed: int = 42):
    """
    Test: Does the mesh pattern persist when forcing stops?
    
    Protocol:
      Phase 1 (0-5000): TRAINING - Drive Basin A, store pattern snapshot
      Phase 2 (5000-15000): REMOVAL - No forcing, track pattern identity
      Phase 3 (15000-25000): RETURN - Resume forcing, track pattern recovery
    
    Track BOTH:
      - Pattern identity: correlation with training snapshot
      - Activity level: S_stability (1/median_var)
    
    This separates:
      "Is it quiet?" from "Does it look the same?"
    """
    
    print("=" * 70)
    print("EXPERIMENT 1: PATTERN IDENTITY PERSISTENCE")
    print("=" * 70)
    print()
    print("Question: Does the mesh pattern persist when forcing stops?")
    print()
    print("Protocol:")
    print("  Phase 1 (0-5k): TRAINING - Drive Basin A, store snapshot")
    print("  Phase 2 (5k-15k): REMOVAL - No forcing, track pattern identity")
    print("  Phase 3 (15k-25k): RETURN - Resume forcing, track recovery")
    print()
    print("Tracking BOTH:")
    print("  - Pattern identity: correlation with training snapshot")
    print("  - Activity level: S_stability (1/median_var)")
    print()
    
    ocean, meta = load_real_ocean_data(hours=150)
    
    substrate = SharedSiliconSubstrate(seed=seed)
    observer = Observer(observer_id=0, seed=seed)
    consequence = SharedBasinConsequence()
    
    # Track metrics over time
    pattern_corr_history: List[Tuple[int, float]] = []
    pattern_cos_history: List[Tuple[int, float]] = []
    stability_history: List[Tuple[int, float]] = []
    
    # Key points
    key_points = {}
    
    print(f"Running {n_steps} steps...")
    
    for step in range(n_steps):
        wave = ocean[step % len(ocean)]
        lo, hi = np.quantile(ocean, 0.05), np.quantile(ocean, 0.95)
        x = (np.clip(wave, lo, hi) - lo) / (hi - lo + 1e-8)
        substrate.diffusion = 0.08 + 0.12 * x
        
        cons = observer.consumption_history[-1] if observer.consumption_history else 0.5
        noise = consequence.update(cons)
        
        # Phase-specific forcing
        basin_forcing = {}
        
        if step < 5000:
            # Phase 1: Training
            if step % 50 < 20:
                basin_forcing["A"] = wave
        elif step < 15000:
            # Phase 2: Removal (no forcing)
            pass
        else:
            # Phase 3: Return
            if step % 50 < 20:
                basin_forcing["A"] = wave
        
        substrate.step([cons, 0.5, 0.5, 0.5], basin_forcing=basin_forcing)
        observation = substrate.get_observation(0)
        observer.step(observation, external_noise=noise)
        
        # Store training snapshot at end of training
        if step == 4999:
            substrate.store_training_snapshot("A")
            print(f"  Stored training snapshot at step {step}")
            
            # Record key point
            metrics = substrate.get_stability_metrics("A")
            key_points["end_training"] = {
                "pattern_corr": 1.0,  # By definition
                "pattern_cos": 1.0,
                "S_stability": metrics["S_stability"],
                "median_var": metrics["median_var"],
            }
        
        # Track metrics every 100 steps after training
        if step > 5000 and step % 100 == 0:
            corr = substrate.get_pattern_correlation("A")
            cos = substrate.get_pattern_cosine("A")
            metrics = substrate.get_stability_metrics("A")
            
            pattern_corr_history.append((step, corr))
            pattern_cos_history.append((step, cos))
            stability_history.append((step, metrics["S_stability"]))
            
            # Key points
            if step == 5100:
                key_points["start_removal"] = {
                    "pattern_corr": corr,
                    "pattern_cos": cos,
                    "S_stability": metrics["S_stability"],
                }
            elif step == 10000:
                key_points["mid_removal"] = {
                    "pattern_corr": corr,
                    "pattern_cos": cos,
                    "S_stability": metrics["S_stability"],
                }
            elif step == 14900:
                key_points["end_removal"] = {
                    "pattern_corr": corr,
                    "pattern_cos": cos,
                    "S_stability": metrics["S_stability"],
                }
            elif step == 15100:
                key_points["start_return"] = {
                    "pattern_corr": corr,
                    "pattern_cos": cos,
                    "S_stability": metrics["S_stability"],
                }
            elif step == 20000:
                key_points["mid_return"] = {
                    "pattern_corr": corr,
                    "pattern_cos": cos,
                    "S_stability": metrics["S_stability"],
                }
            elif step == n_steps - 100:
                key_points["end_return"] = {
                    "pattern_corr": corr,
                    "pattern_cos": cos,
                    "S_stability": metrics["S_stability"],
                }
    
    print()
    print("-" * 70)
    print("RESULTS: PATTERN IDENTITY PERSISTENCE")
    print("-" * 70)
    print()
    
    # Key points table
    print("Key Points:")
    print()
    print(f"{'Phase':<15} {'Point':<12} {'Pattern r':<12} {'Pattern cos':<12} {'log10(S)':<12}")
    print("-" * 65)
    
    for name in ["end_training", "start_removal", "mid_removal", "end_removal",
                 "start_return", "mid_return", "end_return"]:
        if name in key_points:
            kp = key_points[name]
            phase = name.split("_")[0].upper()
            point = name.split("_")[1].upper() if "_" in name else ""
            log_S = np.log10(kp["S_stability"] + 1e-10)
            print(f"{phase:<15} {point:<12} {kp['pattern_corr']:+.3f}       {kp['pattern_cos']:+.3f}       {log_S:.2f}")
    
    print()
    
    # Phase averages
    removal_corr = [c for t, c in pattern_corr_history if 5000 <= t < 15000]
    return_corr = [c for t, c in pattern_corr_history if t >= 15000]
    removal_S = [s for t, s in stability_history if 5000 <= t < 15000]
    return_S = [s for t, s in stability_history if t >= 15000]
    
    print("Phase Averages:")
    print()
    print(f"  {'Phase':<15} {'Mean Pattern r':<18} {'log10(S_stability)':<20}")
    print(f"  {'-'*55}")
    if removal_corr:
        print(f"  {'Removal':<15} {np.mean(removal_corr):+.3f}             {np.log10(np.mean(removal_S) + 1e-10):.2f}")
    if return_corr:
        print(f"  {'Return':<15} {np.mean(return_corr):+.3f}             {np.log10(np.mean(return_S) + 1e-10):.2f}")
    print()
    
    # Key question: does pattern persist?
    if removal_corr:
        early_removal = removal_corr[:len(removal_corr)//3]
        late_removal = removal_corr[-len(removal_corr)//3:]
        
        early_mean = np.mean(early_removal)
        late_mean = np.mean(late_removal)
        
        print("PATTERN IDENTITY ANALYSIS:")
        print(f"  Early removal (5k-8k): pattern r = {early_mean:+.3f}")
        print(f"  Late removal (12k-15k): pattern r = {late_mean:+.3f}")
        print(f"  Change: {late_mean - early_mean:+.3f}")
        print()
        
        if late_mean > 0.5:
            print("  ✓ PATTERN PERSISTS: Correlation still positive after 10k steps")
        elif late_mean > 0.0:
            print("  ~ PATTERN DEGRADED: Correlation weakened but still positive")
        else:
            print("  ✗ PATTERN LOST: Correlation near zero or negative")
        
        # Compare to S_stability
        early_S = removal_S[:len(removal_S)//3]
        late_S = removal_S[-len(removal_S)//3:]
        
        print()
        print("ACTIVITY vs IDENTITY:")
        print(f"  S_stability went {'UP' if np.mean(late_S) > np.mean(early_S) else 'DOWN'} (mesh got {'quieter' if np.mean(late_S) > np.mean(early_S) else 'more active'})")
        print(f"  Pattern correlation went {'UP' if late_mean > early_mean else 'DOWN'}")
        
        if np.mean(late_S) > np.mean(early_S) and late_mean < early_mean:
            print("  → Mesh got quieter BUT pattern degraded")
            print("  → Activity and identity are DECOUPLED")
        elif np.mean(late_S) > np.mean(early_S) and late_mean > early_mean - 0.1:
            print("  → Mesh got quieter AND pattern preserved")
            print("  → Quiet mesh retains pattern")
    
    return {
        "pattern_corr_history": pattern_corr_history,
        "pattern_cos_history": pattern_cos_history,
        "stability_history": stability_history,
        "key_points": key_points,
    }


def experiment_reconditioning_vs_pattern(n_steps: int = 30000, seed: int = 42):
    """
    Test: Does reconditioning actively degrade the pattern, or just let it freeze?
    
    Compare two basins:
      - Basin A: Trained then removed (no forcing)
      - Basin B: Never trained (control)
    
    If reconditioning degrades patterns:
      - Basin A pattern should degrade toward Basin B pattern (both become generic)
    
    If reconditioning just freezes:
      - Basin A retains its pattern, Basin B retains its (different) pattern
    """
    
    print("=" * 70)
    print("EXPERIMENT 2: RECONDITIONING vs PATTERN")
    print("=" * 70)
    print()
    print("Question: Does reconditioning actively degrade, or just freeze?")
    print()
    print("Protocol:")
    print("  Phase 1 (0-5k): Train Basin A only")
    print("  Phase 2 (5k-30k): No forcing on either basin")
    print()
    print("Compare:")
    print("  - Basin A: trained then abandoned")
    print("  - Basin B: never trained (control)")
    print()
    
    ocean, meta = load_real_ocean_data(hours=180)
    
    substrate = SharedSiliconSubstrate(seed=seed)
    consequence = SharedBasinConsequence()
    
    # Track both basins
    pattern_A_history: List[Tuple[int, float]] = []
    pattern_B_history: List[Tuple[int, float]] = []
    
    # Also track cross-correlation (A vs B pattern similarity)
    cross_corr_history: List[Tuple[int, float]] = []
    
    print(f"Running {n_steps} steps...")
    
    for step in range(n_steps):
        wave = ocean[step % len(ocean)]
        lo, hi = np.quantile(ocean, 0.05), np.quantile(ocean, 0.95)
        x = (np.clip(wave, lo, hi) - lo) / (hi - lo + 1e-8)
        substrate.diffusion = 0.08 + 0.12 * x
        
        noise = consequence.update(0.5)
        
        # Only force Basin A during training
        basin_forcing = {}
        if step < 5000:
            if step % 50 < 20:
                basin_forcing["A"] = wave
        
        substrate.step([0.5, 0.5, 0.5, 0.5], basin_forcing=basin_forcing)
        
        # Store snapshots at end of training
        if step == 4999:
            substrate.store_training_snapshot("A")
            substrate.store_training_snapshot("B")
            print(f"  Stored snapshots at step {step}")
        
        # Track every 200 steps after training
        if step > 5000 and step % 200 == 0:
            corr_A = substrate.get_pattern_correlation("A")
            corr_B = substrate.get_pattern_correlation("B")
            
            pattern_A_history.append((step, corr_A))
            pattern_B_history.append((step, corr_B))
            
            # Cross-correlation: how similar are A and B to each other?
            pattern_A = substrate.get_core_pattern("A")
            pattern_B = substrate.get_core_pattern("B")
            
            # Mean-center
            A_centered = pattern_A - np.mean(pattern_A)
            B_centered = pattern_B - np.mean(pattern_B)
            
            if np.std(A_centered) > 1e-10 and np.std(B_centered) > 1e-10:
                cross = np.corrcoef(A_centered, B_centered)[0, 1]
                if np.isfinite(cross):
                    cross_corr_history.append((step, cross))
    
    print()
    print("-" * 70)
    print("RESULTS: RECONDITIONING vs PATTERN")
    print("-" * 70)
    print()
    
    # Analyze
    if pattern_A_history and pattern_B_history:
        early_A = [c for t, c in pattern_A_history if t < 15000]
        late_A = [c for t, c in pattern_A_history if t > 20000]
        early_B = [c for t, c in pattern_B_history if t < 15000]
        late_B = [c for t, c in pattern_B_history if t > 20000]
        
        print("Pattern Correlation with Training Snapshot:")
        print()
        print(f"  {'Basin':<10} {'Early (5k-15k)':<18} {'Late (20k-30k)':<18} {'Change':<12}")
        print(f"  {'-'*55}")
        print(f"  {'A (trained)':<10} {np.mean(early_A):+.3f}             {np.mean(late_A):+.3f}             {np.mean(late_A) - np.mean(early_A):+.3f}")
        print(f"  {'B (control)':<10} {np.mean(early_B):+.3f}             {np.mean(late_B):+.3f}             {np.mean(late_B) - np.mean(early_B):+.3f}")
        print()
        
        # Cross-correlation
        if cross_corr_history:
            early_cross = [c for t, c in cross_corr_history if t < 15000]
            late_cross = [c for t, c in cross_corr_history if t > 20000]
            
            print("Cross-Correlation (A ↔ B):")
            print(f"  Early: {np.mean(early_cross):+.3f}")
            print(f"  Late: {np.mean(late_cross):+.3f}")
            print(f"  Change: {np.mean(late_cross) - np.mean(early_cross):+.3f}")
            print()
            
            if np.mean(late_cross) > np.mean(early_cross) + 0.1:
                print("  → Basins converging (reconditioning homogenizes)")
            elif np.mean(late_cross) < np.mean(early_cross) - 0.1:
                print("  → Basins diverging")
            else:
                print("  → Basins independent (reconditioning doesn't homogenize)")
        
        # Interpretation
        print()
        print("INTERPRETATION:")
        
        A_persists = np.mean(late_A) > 0.3
        B_persists = np.mean(late_B) > 0.3
        
        if A_persists and B_persists:
            print("  Both patterns persist → reconditioning FREEZES, doesn't degrade")
        elif A_persists and not B_persists:
            print("  Trained pattern persists, control drifted → training stabilizes")
        elif not A_persists and not B_persists:
            print("  Both patterns degraded → reconditioning actively smooths")
        else:
            print("  Mixed results")
    
    return {
        "pattern_A_history": pattern_A_history,
        "pattern_B_history": pattern_B_history,
        "cross_corr_history": cross_corr_history,
    }


def experiment_repetition_strengthens_pattern(n_steps: int = 30000, seed: int = 42):
    """
    Test: Does repetition strengthen pattern persistence?
    
    Compare:
      - Basin A: Trained with repeated forcing (every 100 steps)
      - Basin B: Trained once (single burst)
    
    Then remove forcing and track which pattern persists longer.
    """
    
    print("=" * 70)
    print("EXPERIMENT 3: REPETITION STRENGTHENS PATTERN")
    print("=" * 70)
    print()
    print("Question: Does repetition strengthen pattern persistence?")
    print()
    print("Protocol:")
    print("  Phase 1 (0-10k): Train Basin A (repeated), Basin B (once at 5k)")
    print("  Phase 2 (10k-30k): No forcing, track pattern persistence")
    print()
    
    ocean, meta = load_real_ocean_data(hours=180)
    
    substrate = SharedSiliconSubstrate(seed=seed)
    consequence = SharedBasinConsequence()
    
    pattern_A_history: List[Tuple[int, float]] = []
    pattern_B_history: List[Tuple[int, float]] = []
    
    print(f"Running {n_steps} steps...")
    
    for step in range(n_steps):
        wave = ocean[step % len(ocean)]
        lo, hi = np.quantile(ocean, 0.05), np.quantile(ocean, 0.95)
        x = (np.clip(wave, lo, hi) - lo) / (hi - lo + 1e-8)
        substrate.diffusion = 0.08 + 0.12 * x
        
        noise = consequence.update(0.5)
        
        basin_forcing = {}
        
        if step < 10000:
            # Basin A: repeated forcing
            if step % 100 < 30:
                basin_forcing["A"] = wave
            
            # Basin B: single burst at step 5000
            if 5000 <= step < 5500:
                basin_forcing["B"] = wave
        
        substrate.step([0.5, 0.5, 0.5, 0.5], basin_forcing=basin_forcing)
        
        # Store snapshots at end of training
        if step == 9999:
            substrate.store_training_snapshot("A")
            substrate.store_training_snapshot("B")
            print(f"  Stored snapshots at step {step}")
        
        # Track every 200 steps after training
        if step > 10000 and step % 200 == 0:
            corr_A = substrate.get_pattern_correlation("A")
            corr_B = substrate.get_pattern_correlation("B")
            
            pattern_A_history.append((step, corr_A))
            pattern_B_history.append((step, corr_B))
    
    print()
    print("-" * 70)
    print("RESULTS: REPETITION STRENGTHENS PATTERN")
    print("-" * 70)
    print()
    
    if pattern_A_history and pattern_B_history:
        early_A = [c for t, c in pattern_A_history if t < 15000]
        late_A = [c for t, c in pattern_A_history if t > 25000]
        early_B = [c for t, c in pattern_B_history if t < 15000]
        late_B = [c for t, c in pattern_B_history if t > 25000]
        
        print("Pattern Correlation with Training Snapshot:")
        print()
        print(f"  {'Basin':<15} {'Early (10k-15k)':<18} {'Late (25k-30k)':<18} {'Change':<12}")
        print(f"  {'-'*60}")
        print(f"  {'A (repeated)':<15} {np.mean(early_A):+.3f}             {np.mean(late_A):+.3f}             {np.mean(late_A) - np.mean(early_A):+.3f}")
        print(f"  {'B (once)':<15} {np.mean(early_B):+.3f}             {np.mean(late_B):+.3f}             {np.mean(late_B) - np.mean(early_B):+.3f}")
        print()
        
        # Compare persistence
        A_better = np.mean(late_A) > np.mean(late_B)
        A_decay = np.mean(late_A) - np.mean(early_A)
        B_decay = np.mean(late_B) - np.mean(early_B)
        
        print("INTERPRETATION:")
        if A_better:
            advantage = np.mean(late_A) - np.mean(late_B)
            print(f"  ✓ REPETITION ADVANTAGE: Basin A retains {advantage:+.3f} more correlation")
        else:
            print(f"  ~ No clear repetition advantage")
        
        if abs(A_decay) < abs(B_decay):
            print(f"  ✓ SLOWER DECAY: Basin A decayed {abs(A_decay):.3f} vs Basin B {abs(B_decay):.3f}")
    
    return {
        "pattern_A_history": pattern_A_history,
        "pattern_B_history": pattern_B_history,
    }


def run_level39():
    """Run complete Level 39 analysis."""
    
    print("=" * 70)
    print("LEVEL 39: PATTERN IDENTITY")
    print("=" * 70)
    print()
    print("L38 measured activity level (S_stability).")
    print("L39 measures PATTERN IDENTITY (correlation with training snapshot).")
    print()
    print("This separates:")
    print("  - 'Is it quiet?' (S_stability)")
    print("  - 'Does it look the same?' (pattern correlation)")
    print()
    print("Using REAL TBU stack with REAL ocean data.")
    print()
    
    r1 = experiment_pattern_identity()
    print()
    
    r2 = experiment_reconditioning_vs_pattern()
    print()
    
    r3 = experiment_repetition_strengthens_pattern()
    print()
    
    print("=" * 70)
    print("LEVEL 39 SUMMARY")
    print("=" * 70)
    print()
    
    # Summarize
    if r1["key_points"]:
        kp = r1["key_points"]
        if "end_removal" in kp:
            print(f"PATTERN PERSISTENCE:")
            print(f"  End of removal: pattern r = {kp['end_removal']['pattern_corr']:+.3f}")
            if "end_return" in kp:
                print(f"  End of return:  pattern r = {kp['end_return']['pattern_corr']:+.3f}")
    
    print()
    print("=" * 70)
    print("LEVEL 39 COMPLETE")
    print("=" * 70)
    
    return {
        "identity": r1,
        "reconditioning": r2,
        "repetition": r3,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Level 39: Pattern Identity")
    p.add_argument("--identity", action="store_true", help="Pattern identity persistence")
    p.add_argument("--reconditioning", action="store_true", help="Reconditioning vs pattern")
    p.add_argument("--repetition", action="store_true", help="Repetition strengthens pattern")
    p.add_argument("--all", action="store_true", help="Run all")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    run_all = args.all or not any([args.identity, args.reconditioning, args.repetition])
    
    if run_all:
        run_level39()
    else:
        if args.identity:
            experiment_pattern_identity(seed=args.seed)
        if args.reconditioning:
            experiment_reconditioning_vs_pattern(seed=args.seed)
        if args.repetition:
            experiment_repetition_strengthens_pattern(seed=args.seed)


if __name__ == "__main__":
    main()
