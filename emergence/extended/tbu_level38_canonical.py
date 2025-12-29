#!/usr/bin/env python3
"""
================================================================================
LEVEL 38: TEMPLATE DYNAMICS (ALIGNMENT WITHOUT FORCING)
================================================================================

WHAT L37 ESTABLISHED:
  - Long-term memory access behaves as coupling/re-entry to 4D geometry
  - Repetition strengthens measured coupling
  - Absence degrades coupling, return rebuilds it
  - Two orthogonal memory systems: internal (3D, τ≈36-47) and external (4D, τ≈77)

WHAT L38 ASKS:
  Can the observer reproduce alignment dynamics learned under one boundary
  condition when that boundary is absent?

THE KEY DISTINCTION:
  - Template ≠ stored content
  - Template = learned alignment dynamics
  - Memory of Rome = re-entering alignment state created when you were there
  - Not accessing Rome, but approximating how your system behaved when Rome constrained it

THE PREDICTION (TBU-consistent):
  1. Alignment signatures should persist briefly after forcing removed
  2. Signatures should degrade without reinforcement (reconditioning)
  3. Signatures should NOT be perfect — approximation, not replay
  4. Return should be faster than original learning

IF THIS WERE STORAGE (not TBU):
  1. Signatures would persist indefinitely
  2. No degradation without external cause
  3. Perfect replay possible

PROTOCOL:
  Phase 1: TRAINING - Drive Basin A with specific pattern
  Phase 2: REMOVAL - Stop all forcing, measure alignment signatures
  Phase 3: DECAY - Continue without forcing, track signature degradation
  Phase 4: RETURN - Resume forcing, measure realignment speed

ALIGNMENT SIGNATURES:
  - Volatility level (lower = more aligned)
  - Core activation patterns (characteristic of training)
  - Phase relationships between cores
  - EMA trajectory similarity to training period

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
# REAL TBU STACK (from L37)
# =============================================================================

@dataclass
class SharedSiliconSubstrate:
    """Real TBU substrate with basin-bottleneck geometry."""
    
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
        
        # Core regions
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
        self.M_A_history: List[float] = []
        self.M_B_history: List[float] = []
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
        
        laplacian = (
            np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
            np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) - 4 * self.grid
        )
        
        eff_diff = np.clip(self.diffusion, 0.02, 0.18)
        diff_coeff = np.where(self.neck_mask, self.diffusion_neck, eff_diff)
        self.grid += diff_coeff * laplacian
        
        for i, (coords, cons) in enumerate(zip(self.observer_coords, consumptions)):
            local_forcing = self.forcing_strength * (1.0 + 0.5 * cons)
            noise = self.rng.normal(0, local_forcing, len(coords[0]))
            self.grid[coords] += noise
        
        if basin_forcing:
            if "A" in basin_forcing and basin_forcing["A"] != 0:
                pattern_A = basin_forcing["A"] * 0.02
                self.grid[self.core_A_coords] += pattern_A
            if "B" in basin_forcing and basin_forcing["B"] != 0:
                pattern_B = basin_forcing["B"] * 0.02
                self.grid[self.core_B_coords] += pattern_B
        
        self.grid *= (1.0 - self.damping)
        self.grid = np.clip(self.grid, -5.0, 5.0)
        
        self.core_A_history.append(self.get_core_mean("A"))
        self.core_B_history.append(self.get_core_mean("B"))
        
        if len(self.history) >= 20:
            self.M_A_history.append(self._compute_M(self.basin_A_mask))
            self.M_B_history.append(self._compute_M(self.basin_B_mask))
    
    def _compute_M(self, mask: np.ndarray) -> Dict:
        """
        Compute constraint strength metrics.
        
        Returns multiple metrics because M (ratio-based) can be unstable:
        - M_contrast: original ratio metric (can explode/collapse with variance regime)
        - median_var: raw median variance (lower = more stable)
        - S_stability: 1/median_var (higher = more constrained)
        
        M_contrast measures "how uneven is the variance landscape"
        S_stability measures "how stable is the region overall"
        """
        H = np.array(self.history[-min(self.history_window, len(self.history)):])
        var = np.var(H, axis=0) + 1e-12
        
        region_var = var[mask]
        median_var = float(np.median(region_var))
        mean_var = float(np.mean(region_var))
        
        # Original M (ratio-based, can be unstable)
        M_contrast = float(np.mean(median_var / (region_var + 1e-12)))
        
        # Robust stability metric (doesn't blow up)
        S_stability = 1.0 / (median_var + 1e-12)
        
        return {
            "M_contrast": M_contrast,
            "median_var": median_var,
            "mean_var": mean_var,
            "S_stability": S_stability,
        }
    
    def get_M_A(self) -> Dict:
        """Get all M metrics for Basin A."""
        return self._compute_M(self.basin_A_mask)
    
    def get_M_B(self) -> Dict:
        """Get all M metrics for Basin B."""
        return self._compute_M(self.basin_B_mask)
    
    def get_observation(self, observer_id: int) -> np.ndarray:
        return self.grid[self.observer_coords[observer_id]]
    
    def get_basin(self, observer_id: int) -> str:
        return self.observer_basin[observer_id]
    
    def get_core_mean(self, basin: str) -> float:
        if basin == "A":
            return float(self.grid[self.core_A_coords].mean())
        return float(self.grid[self.core_B_coords].mean())


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
    """Real observer with L35 parameters and template tracking."""
    
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
        
        # For template tracking
        self.core_state_history: List[np.ndarray] = []
        self.mismatch_history: List[float] = []
    
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
        current_mean = float(np.mean(sampled))
        
        if self.ema is not None:
            expectation = float(np.mean(self.ema))
            mismatch = abs(current_mean - expectation)
        else:
            expectation = current_mean
            mismatch = 0.0
        
        self.mismatch_history.append(mismatch)
        
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
        
        # Store core state for template analysis
        self.core_state_history.append(self.core_states.copy())
        
        coherence = self._compute_coherence()
        self.coherence_history.append(coherence)
        
        return {
            "consumption": consumption,
            "coherence": coherence,
            "volatility": volatility,
            "ema_mean": self.ema_mean_history[-1],
            "mismatch": mismatch,
            "core_state": self.core_states.copy(),
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
    
    def get_template_signature(self, window: int = 100) -> Dict:
        """
        Extract alignment signature from recent history.
        
        This is what we'll use to test template persistence.
        """
        if len(self.core_state_history) < window:
            return None
        
        recent_cores = np.array(self.core_state_history[-window:])
        recent_vol = np.array(self.volatility_history[-window:])
        recent_ema = np.array(self.ema_mean_history[-window:])
        
        return {
            "mean_core_state": recent_cores.mean(axis=0),
            "core_variance": recent_cores.var(axis=0),
            "mean_volatility": float(np.mean(recent_vol)),
            "volatility_trend": float(np.polyfit(range(len(recent_vol)), recent_vol, 1)[0]),
            "ema_mean": float(np.mean(recent_ema)),
            "ema_variance": float(np.var(recent_ema)),
            "coherence": self.coherence_history[-1] if self.coherence_history else 0.5,
        }


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
# TEMPLATE SIMILARITY METRICS
# =============================================================================

def template_similarity(sig1: Dict, sig2: Dict) -> Dict:
    """
    Compare two template signatures.
    
    All similarities are computed consistently and bounded [0, 1]:
    - Cosine similarity for vectors, mapped from [-1,1] to [0,1]
    - 1 - |relative_diff| for scalars (bounded [0, 1])
    """
    if sig1 is None or sig2 is None:
        return {"overall": 0.0, "core_similarity": 0.0, "volatility_similarity": 0.0,
                "ema_similarity": 0.0, "coherence_similarity": 0.0}
    
    # Core state similarity (cosine similarity, then map to [0,1])
    c1 = sig1["mean_core_state"]
    c2 = sig2["mean_core_state"]
    
    norm1 = np.linalg.norm(c1)
    norm2 = np.linalg.norm(c2)
    
    if norm1 > 1e-10 and norm2 > 1e-10:
        core_cos = float(np.dot(c1, c2) / (norm1 * norm2))  # in [-1, 1]
        core_sim = 0.5 * (core_cos + 1.0)  # map to [0, 1]
    else:
        core_sim = 0.5  # neutral if can't compute
    
    # Volatility similarity: 1 - |relative difference|
    v1 = sig1["mean_volatility"]
    v2 = sig2["mean_volatility"]
    if max(v1, v2) > 1e-10:
        vol_sim = 1.0 - abs(v1 - v2) / max(v1, v2)
    else:
        vol_sim = 1.0
    vol_sim = max(0.0, vol_sim)
    
    # EMA variance similarity (compare distribution width, not mean)
    # Mean can drift with environment; variance reflects internal dynamics
    ev1 = sig1["ema_variance"]
    ev2 = sig2["ema_variance"]
    if max(ev1, ev2) > 1e-10:
        ema_sim = 1.0 - abs(ev1 - ev2) / max(ev1, ev2)
    else:
        ema_sim = 1.0
    ema_sim = max(0.0, ema_sim)
    
    # Coherence similarity: 1 - |difference|
    coh_sim = 1.0 - abs(sig1["coherence"] - sig2["coherence"])
    coh_sim = max(0.0, coh_sim)
    
    # Overall (weighted average, all components now in [0,1])
    overall = 0.4 * core_sim + 0.3 * vol_sim + 0.2 * ema_sim + 0.1 * coh_sim
    
    return {
        "core_similarity": core_sim,
        "volatility_similarity": vol_sim,
        "ema_similarity": ema_sim,
        "coherence_similarity": coh_sim,
        "overall": overall,
    }


def compute_coupling_metrics(ema_history: List[float], core_history: List[float], 
                              window: int = 200) -> Dict:
    """
    Compute sign-invariant coupling metrics.
    
    Since coupling can oscillate between positive and negative (phase-sensitive),
    we need metrics that capture alignment strength regardless of sign.
    """
    if len(ema_history) < window or len(core_history) < window:
        return {"abs_r": 0.0, "episode_strength": 0.0, "r_signed": 0.0}
    
    ema = np.array(ema_history[-window:])
    core = np.array(core_history[-window:])
    
    if np.std(ema) < 1e-10 or np.std(core) < 1e-10:
        return {"abs_r": 0.0, "episode_strength": 0.0, "r_signed": 0.0}
    
    # Signed correlation (what we had before)
    r_signed = np.corrcoef(ema, core)[0, 1]
    if not np.isfinite(r_signed):
        r_signed = 0.0
    
    # Absolute correlation (sign-invariant)
    abs_r = abs(r_signed)
    
    # Episode strength: fraction of mini-windows where |r| > 0.3
    mini_window = 50
    n_episodes = 0
    n_strong = 0
    
    for i in range(0, window - mini_window, mini_window // 2):
        e_chunk = ema[i:i+mini_window]
        c_chunk = core[i:i+mini_window]
        
        if np.std(e_chunk) > 1e-10 and np.std(c_chunk) > 1e-10:
            r_chunk = np.corrcoef(e_chunk, c_chunk)[0, 1]
            if np.isfinite(r_chunk):
                n_episodes += 1
                if abs(r_chunk) > 0.3:
                    n_strong += 1
    
    episode_strength = n_strong / n_episodes if n_episodes > 0 else 0.0
    
    # Best lag correlation (check lags 0-50)
    best_r = abs_r
    for lag in range(1, min(51, window // 4)):
        r_lag = np.corrcoef(ema[lag:], core[:-lag])[0, 1]
        if np.isfinite(r_lag) and abs(r_lag) > best_r:
            best_r = abs(r_lag)
    
    return {
        "r_signed": float(r_signed),
        "abs_r": float(abs_r),
        "episode_strength": float(episode_strength),
        "best_lag_r": float(best_r),
    }


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_template_persistence(n_steps: int = 20000, seed: int = 42):
    """
    Test: Do alignment signatures persist after forcing is removed?
    
    Protocol:
      Phase 1 (0-5000): TRAINING - Drive Basin A with ocean pattern
      Phase 2 (5000-10000): REMOVAL - No forcing, measure signature persistence
      Phase 3 (10000-15000): DECAY - Continue no forcing, track degradation
      Phase 4 (15000-20000): RETURN - Resume forcing, measure realignment speed
    
    This tests the core L38 hypothesis:
      Template = learned alignment dynamics, not stored content
      
    TRACKING BOTH SIDES:
      - Observer side: template signature similarity
      - Geometry side: M (constraint strength proxy)
    """
    
    print("=" * 70)
    print("EXPERIMENT 1: TEMPLATE PERSISTENCE")
    print("=" * 70)
    print()
    print("Question: Do alignment signatures persist after forcing is removed?")
    print()
    print("Protocol:")
    print("  Phase 1 (0-5000): TRAINING - Drive Basin A")
    print("  Phase 2 (5000-10000): REMOVAL - No forcing")
    print("  Phase 3 (10000-15000): DECAY - Continue no forcing")
    print("  Phase 4 (15000-20000): RETURN - Resume forcing")
    print()
    print("Tracking BOTH:")
    print("  - Observer side: template signature")
    print("  - Geometry side: M (constraint strength)")
    print()
    
    ocean, meta = load_real_ocean_data(hours=120)
    
    substrate = SharedSiliconSubstrate(seed=seed)
    observer = Observer(observer_id=0, seed=seed)
    consequence = SharedBasinConsequence()
    
    # Store signatures at key points
    signatures = {
        "end_training": None,
        "start_removal": None,
        "mid_removal": None,
        "end_removal": None,
        "mid_decay": None,
        "end_decay": None,
        "mid_return": None,
        "end_return": None,
    }
    
    # Track M at same points
    M_at_points = {}
    
    # Track signature similarity and M over time
    similarity_to_training: List[Tuple[int, float]] = []
    M_over_time: List[Tuple[int, float]] = []
    
    training_signature = None
    
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
        elif step < 10000:
            # Phase 2: Removal (no forcing)
            pass
        elif step < 15000:
            # Phase 3: Decay (still no forcing)
            pass
        else:
            # Phase 4: Return
            if step % 50 < 20:
                basin_forcing["A"] = wave
        
        substrate.step([cons, 0.5, 0.5, 0.5], basin_forcing=basin_forcing)
        observation = substrate.get_observation(0)
        observer.step(observation, external_noise=noise)
        
        # Capture signatures and M metrics at key points
        # M_A_history now contains dicts with M_contrast, median_var, S_stability
        current_M = substrate.M_A_history[-1] if substrate.M_A_history else {"M_contrast": 0, "S_stability": 0, "median_var": 0}
        
        if step == 4999:
            signatures["end_training"] = observer.get_template_signature()
            training_signature = signatures["end_training"]
            M_at_points["end_training"] = current_M
        elif step == 5000:
            signatures["start_removal"] = observer.get_template_signature()
            M_at_points["start_removal"] = current_M
        elif step == 7500:
            signatures["mid_removal"] = observer.get_template_signature()
            M_at_points["mid_removal"] = current_M
        elif step == 9999:
            signatures["end_removal"] = observer.get_template_signature()
            M_at_points["end_removal"] = current_M
        elif step == 12500:
            signatures["mid_decay"] = observer.get_template_signature()
            M_at_points["mid_decay"] = current_M
        elif step == 14999:
            signatures["end_decay"] = observer.get_template_signature()
            M_at_points["end_decay"] = current_M
        elif step == 17500:
            signatures["mid_return"] = observer.get_template_signature()
            M_at_points["mid_return"] = current_M
        elif step == n_steps - 1:
            signatures["end_return"] = observer.get_template_signature()
            M_at_points["end_return"] = current_M
        
        # Track similarity and M every 100 steps after training
        if step > 5000 and step % 100 == 0 and training_signature is not None:
            current_sig = observer.get_template_signature()
            if current_sig is not None:
                sim = template_similarity(training_signature, current_sig)
                similarity_to_training.append((step, sim["overall"]))
            
            if substrate.M_A_history:
                M_over_time.append((step, substrate.M_A_history[-1]))
    
    print()
    print("-" * 70)
    print("RESULTS: TEMPLATE PERSISTENCE")
    print("-" * 70)
    print()
    
    # Combined table: similarity AND both M metrics
    print("Observer Template vs Geometry Metrics:")
    print()
    print("  M_contrast = ratio metric (can explode/collapse with variance regime)")
    print("  S_stability = 1/median_var (robust, higher = more constrained)")
    print()
    print(f"{'Phase':<12} {'Point':<10} {'Template':<10} {'M_contrast':<12} {'S_stability':<12} {'median_var':<12}")
    print("-" * 70)
    
    for name in ["end_training", "start_removal", "mid_removal", "end_removal", 
                 "mid_decay", "end_decay", "mid_return", "end_return"]:
        sig = signatures.get(name)
        M_dict = M_at_points.get(name, {"M_contrast": 0, "S_stability": 0, "median_var": 0})
        
        if sig is not None and training_signature is not None:
            sim = template_similarity(training_signature, sig)
            phase = name.split("_")[0].upper()
            point = name.split("_")[1].upper() if "_" in name else ""
            
            M_c = M_dict.get("M_contrast", 0)
            S_s = M_dict.get("S_stability", 0)
            m_var = M_dict.get("median_var", 0)
            
            # Use log scale for S_stability since it varies over orders of magnitude
            log_S = np.log10(S_s + 1e-10)
            
            print(f"{phase:<12} {point:<10} {sim['overall']:.3f}     {M_c:.1f}        log10(S)={log_S:.2f}   {m_var:.2e}")
    
    print()
    
    # Analyze phases
    if similarity_to_training:
        removal_sims = [s for t, s in similarity_to_training if 5000 <= t < 10000]
        decay_sims = [s for t, s in similarity_to_training if 10000 <= t < 15000]
        return_sims = [s for t, s in similarity_to_training if t >= 15000]
        
        # Extract S_stability (robust metric) for phase averages
        removal_S = [m["S_stability"] for t, m in M_over_time if 5000 <= t < 10000]
        decay_S = [m["S_stability"] for t, m in M_over_time if 10000 <= t < 15000]
        return_S = [m["S_stability"] for t, m in M_over_time if t >= 15000]
        
        print("Phase Averages (using S_stability, the robust metric):")
        print(f"  {'Phase':<15} {'Template Sim':<15} {'log10(S_stability)':<20}")
        print(f"  {'-'*50}")
        if removal_sims and removal_S:
            print(f"  {'Removal':<15} {np.mean(removal_sims):.3f}           {np.log10(np.mean(removal_S) + 1e-10):.2f}")
        if decay_sims and decay_S:
            print(f"  {'Decay':<15} {np.mean(decay_sims):.3f}           {np.log10(np.mean(decay_S) + 1e-10):.2f}")
        if return_sims and return_S:
            print(f"  {'Return':<15} {np.mean(return_sims):.3f}           {np.log10(np.mean(return_S) + 1e-10):.2f}")
        print()
        
        # Key result: when does the drop happen?
        if removal_sims:
            initial_drop = 1.0 - np.mean(removal_sims[:len(removal_sims)//2])
            print(f"Initial drop (first half of Removal): {initial_drop:.3f}")
        
        # Template degradation (Removal to Decay)
        if removal_sims and decay_sims:
            template_change = np.mean(decay_sims) - np.mean(removal_sims)
            print(f"Template change (Removal → Decay): {template_change:+.3f}")
            if template_change < -0.01:
                print("  ✓ Template continues degrading")
            else:
                print("  ~ Template plateaus (most loss is early)")
        
        # Stability dynamics (using robust S_stability metric)
        if removal_S and decay_S:
            S_change_log = np.log10(np.mean(decay_S) + 1e-10) - np.log10(np.mean(removal_S) + 1e-10)
            print(f"Stability change (Removal → Decay): {S_change_log:+.2f} log units")
            
            # Diagnostic: did mesh actually change, or just M_contrast regime?
            print()
            print("DIAGNOSTIC: Mesh change vs M estimator regime")
            print(f"  If S_stability is stable but M_contrast swings wildly,")
            print(f"  the mesh didn't change — only variance contrast did.")
        
        # Return recovery
        if decay_sims and return_sims:
            template_recovery = np.mean(return_sims) - np.mean(decay_sims)
            print(f"Template recovery on return: {template_recovery:+.3f}")
            if template_recovery > 0.01:
                print("  ✓ RETURN REBUILDS TEMPLATE")
    
    return {
        "signatures": signatures,
        "M_at_points": M_at_points,
        "similarity_to_training": similarity_to_training,
        "M_over_time": M_over_time,
    }


def experiment_realignment_speed(n_steps: int = 25000, seed: int = 42):
    """
    Test: Is realignment faster than initial learning?
    
    Protocol:
      Phase 1 (0-5000): INITIAL LEARNING - First exposure to Basin A pattern
      Phase 2 (5000-10000): ABSENCE - No forcing
      Phase 3 (10000-15000): REALIGNMENT - Second exposure
      Phase 4 (15000-20000): ABSENCE 2 - No forcing again
      Phase 5 (20000-25000): NAIVE LEARNING - Basin B (never seen before)
    
    Compare: How fast does coupling build in Phase 3 (realignment) vs Phase 5 (naive)?
    
    USING SIGN-INVARIANT METRICS:
      - |r| instead of r (coupling can flip sign)
      - Episode strength (fraction of windows with |r| > 0.3)
      - Best-lag correlation
    """
    
    print("=" * 70)
    print("EXPERIMENT 2: REALIGNMENT SPEED (SIGN-INVARIANT)")
    print("=" * 70)
    print()
    print("Question: Is realignment faster than initial learning?")
    print()
    print("Protocol:")
    print("  Phase 1 (0-5k): Initial learning (Basin A)")
    print("  Phase 2 (5k-10k): Absence")
    print("  Phase 3 (10k-15k): Realignment (Basin A again)")
    print("  Phase 4 (15k-20k): Absence 2")
    print("  Phase 5 (20k-25k): Naive learning (Basin B - never seen)")
    print()
    print("Using SIGN-INVARIANT metrics (|r|, episode strength, best-lag)")
    print()
    
    ocean, meta = load_real_ocean_data(hours=150)
    
    substrate = SharedSiliconSubstrate(seed=seed)
    observer = Observer(observer_id=0, seed=seed)
    consequence = SharedBasinConsequence()
    
    # Track coupling metrics over time
    coupling_history: List[Tuple[int, Dict, str]] = []
    
    for step in range(n_steps):
        wave = ocean[step % len(ocean)]
        lo, hi = np.quantile(ocean, 0.05), np.quantile(ocean, 0.95)
        x = (np.clip(wave, lo, hi) - lo) / (hi - lo + 1e-8)
        substrate.diffusion = 0.08 + 0.12 * x
        
        cons = observer.consumption_history[-1] if observer.consumption_history else 0.5
        noise = consequence.update(cons)
        
        basin_forcing = {}
        phase = ""
        
        if step < 5000:
            phase = "initial"
            if step % 50 < 20:
                basin_forcing["A"] = wave
        elif step < 10000:
            phase = "absence1"
        elif step < 15000:
            phase = "realign"
            if step % 50 < 20:
                basin_forcing["A"] = wave
        elif step < 20000:
            phase = "absence2"
        else:
            phase = "naive"
            if step % 50 < 20:
                basin_forcing["B"] = wave
        
        substrate.step([cons, 0.5, 0.5, 0.5], basin_forcing=basin_forcing)
        observation = substrate.get_observation(0)
        observer.step(observation, external_noise=noise)
        
        # Measure coupling every 200 steps using sign-invariant metrics
        if step > 500 and step % 200 == 0:
            window = 200
            if len(observer.ema_mean_history) >= window:
                if phase in ["initial", "realign", "absence1", "absence2"]:
                    core_hist = substrate.core_A_history
                else:
                    core_hist = substrate.core_B_history
                
                if len(core_hist) >= window:
                    metrics = compute_coupling_metrics(
                        observer.ema_mean_history, core_hist, window
                    )
                    coupling_history.append((step, metrics, phase))
    
    print("-" * 70)
    print("RESULTS: REALIGNMENT SPEED (SIGN-INVARIANT)")
    print("-" * 70)
    print()
    
    # Analyze by phase using sign-invariant metrics
    def analyze_phase(phase_name: str, data: List[Tuple[int, Dict, str]]) -> Dict:
        phase_data = [(t, m) for t, m, p in data if p == phase_name]
        if len(phase_data) < 3:
            return {"mean_abs_r": 0, "mean_episode": 0, "slope_abs_r": 0}
        
        times = [t for t, m in phase_data]
        abs_rs = [m["abs_r"] for t, m in phase_data]
        episodes = [m["episode_strength"] for t, m in phase_data]
        
        t_start = min(times)
        t_norm = [t - t_start for t in times]
        
        slope_abs_r = np.polyfit(t_norm, abs_rs, 1)[0] if len(t_norm) > 1 else 0
        slope_episode = np.polyfit(t_norm, episodes, 1)[0] if len(t_norm) > 1 else 0
        
        return {
            "mean_abs_r": float(np.mean(abs_rs)),
            "final_abs_r": float(abs_rs[-1]) if abs_rs else 0,
            "mean_episode": float(np.mean(episodes)),
            "slope_abs_r": float(slope_abs_r * 1000),  # per 1000 steps
            "slope_episode": float(slope_episode * 1000),
        }
    
    initial = analyze_phase("initial", coupling_history)
    realign = analyze_phase("realign", coupling_history)
    naive = analyze_phase("naive", coupling_history)
    
    print(f"{'Phase':<15} {'Mean |r|':<12} {'Final |r|':<12} {'Episode %':<12} {'Slope |r|/1k':<12}")
    print("-" * 65)
    print(f"{'Initial (A)':<15} {initial['mean_abs_r']:.3f}       {initial['final_abs_r']:.3f}        {initial['mean_episode']:.3f}        {initial['slope_abs_r']:+.4f}")
    print(f"{'Realign (A)':<15} {realign['mean_abs_r']:.3f}       {realign['final_abs_r']:.3f}        {realign['mean_episode']:.3f}        {realign['slope_abs_r']:+.4f}")
    print(f"{'Naive (B)':<15} {naive['mean_abs_r']:.3f}       {naive['final_abs_r']:.3f}        {naive['mean_episode']:.3f}        {naive['slope_abs_r']:+.4f}")
    print()
    
    # Compare realignment vs naive
    if realign['mean_abs_r'] > naive['mean_abs_r']:
        advantage = (realign['mean_abs_r'] - naive['mean_abs_r']) / naive['mean_abs_r'] * 100 if naive['mean_abs_r'] > 0 else 0
        print(f"✓ REALIGNMENT HAS STRONGER COUPLING: {advantage:.1f}% higher |r| than naive")
    
    if realign['mean_episode'] > naive['mean_episode']:
        ep_adv = (realign['mean_episode'] - naive['mean_episode']) / naive['mean_episode'] * 100 if naive['mean_episode'] > 0 else 0
        print(f"✓ REALIGNMENT HAS MORE EPISODES: {ep_adv:.1f}% more strong-coupling windows")
    
    if realign['slope_abs_r'] > naive['slope_abs_r']:
        print(f"✓ REALIGNMENT BUILDS FASTER: slope {realign['slope_abs_r']:+.4f} vs {naive['slope_abs_r']:+.4f}")
    
    if not any([realign['mean_abs_r'] > naive['mean_abs_r'],
                realign['mean_episode'] > naive['mean_episode'],
                realign['slope_abs_r'] > naive['slope_abs_r']]):
        print("~ No clear realignment advantage")
    
    return {
        "coupling_history": coupling_history,
        "initial": initial,
        "realign": realign,
        "naive": naive,
    }


def experiment_signature_degradation(n_steps: int = 30000, seed: int = 42):
    """
    Test: How do different signature components degrade?
    
    Track individual components:
      - Core state pattern
      - Volatility level
      - Coherence
      - EMA statistics
    
    This tests what aspects of the "template" persist longest.
    """
    
    print("=" * 70)
    print("EXPERIMENT 3: SIGNATURE COMPONENT DEGRADATION")
    print("=" * 70)
    print()
    print("Question: Which template components degrade fastest?")
    print()
    print("Protocol:")
    print("  Phase 1 (0-5k): Training")
    print("  Phase 2 (5k-30k): Long decay - track component degradation")
    print()
    
    ocean, meta = load_real_ocean_data(hours=180)
    
    substrate = SharedSiliconSubstrate(seed=seed)
    observer = Observer(observer_id=0, seed=seed)
    consequence = SharedBasinConsequence()
    
    training_signature = None
    
    # Track component similarities over decay
    decay_tracking = {
        "core": [],
        "volatility": [],
        "ema": [],
        "coherence": [],
        "overall": [],
    }
    
    for step in range(n_steps):
        wave = ocean[step % len(ocean)]
        lo, hi = np.quantile(ocean, 0.05), np.quantile(ocean, 0.95)
        x = (np.clip(wave, lo, hi) - lo) / (hi - lo + 1e-8)
        substrate.diffusion = 0.08 + 0.12 * x
        
        cons = observer.consumption_history[-1] if observer.consumption_history else 0.5
        noise = consequence.update(cons)
        
        basin_forcing = {}
        
        if step < 5000:
            if step % 50 < 20:
                basin_forcing["A"] = wave
        
        substrate.step([cons, 0.5, 0.5, 0.5], basin_forcing=basin_forcing)
        observation = substrate.get_observation(0)
        observer.step(observation, external_noise=noise)
        
        # Capture training signature
        if step == 4999:
            training_signature = observer.get_template_signature()
        
        # Track decay every 500 steps after training
        if step > 5000 and step % 500 == 0 and training_signature is not None:
            current_sig = observer.get_template_signature()
            if current_sig is not None:
                sim = template_similarity(training_signature, current_sig)
                
                decay_tracking["core"].append((step, sim["core_similarity"]))
                decay_tracking["volatility"].append((step, sim["volatility_similarity"]))
                decay_tracking["ema"].append((step, sim["ema_similarity"]))
                decay_tracking["coherence"].append((step, sim["coherence_similarity"]))
                decay_tracking["overall"].append((step, sim["overall"]))
    
    print("-" * 70)
    print("RESULTS: COMPONENT DEGRADATION")
    print("-" * 70)
    print()
    
    # Analyze decay rates
    print("Change = Initial - Final (positive = degraded, negative = improved)")
    print()
    print(f"{'Component':<15} {'Initial':<10} {'Final':<10} {'Change':<10} {'Interpretation':<20}")
    print("-" * 70)
    
    for component, data in decay_tracking.items():
        if not data:
            continue
        
        initial = data[0][1]
        final = data[-1][1]
        change = initial - final
        
        # Interpret the change
        if change > 0.02:
            interp = "DEGRADED"
        elif change < -0.02:
            interp = "IMPROVED (suspicious)"
        else:
            interp = "STABLE"
        
        print(f"{component:<15} {initial:.3f}     {final:.3f}     {change:+.3f}     {interp}")
    
    print()
    
    # Which actually degraded?
    decay_rates = {}
    for component, data in decay_tracking.items():
        if data:
            initial = data[0][1]
            final = data[-1][1]
            decay_rates[component] = initial - final
    
    if decay_rates:
        # Filter to only components that actually degraded (positive change)
        degraded = {k: v for k, v in decay_rates.items() if v > 0.01}
        improved = {k: v for k, v in decay_rates.items() if v < -0.01}
        stable = {k: v for k, v in decay_rates.items() if -0.01 <= v <= 0.01}
        
        if degraded:
            worst = max(degraded, key=degraded.get)
            print(f"Most degraded: {worst} ({degraded[worst]:+.3f})")
        
        if stable:
            print(f"Stable components: {list(stable.keys())}")
        
        if improved:
            print(f"Improved (check metric): {list(improved.keys())}")
    
    return {
        "decay_tracking": decay_tracking,
        "training_signature": training_signature,
    }


def run_level38():
    """Run complete Level 38 analysis."""
    
    print("=" * 70)
    print("LEVEL 38: TEMPLATE DYNAMICS")
    print("=" * 70)
    print()
    print("You remember Rome by re-entering the alignment state created when")
    print("you were there; because it was visited once and not reinforced,")
    print("reconditioning gradually degrades the precision of that re-entry.")
    print()
    print("TESTING:")
    print("  1. Do alignment signatures persist after forcing is removed?")
    print("  2. Is realignment faster than initial learning?")
    print("  3. Which template components degrade fastest?")
    print()
    print("Using REAL TBU stack with REAL ocean data.")
    print()
    
    r1 = experiment_template_persistence()
    print()
    
    r2 = experiment_realignment_speed()
    print()
    
    r3 = experiment_signature_degradation()
    print()
    
    print("=" * 70)
    print("LEVEL 38 SUMMARY")
    print("=" * 70)
    print()
    
    # Summarize findings
    if r1["similarity_to_training"]:
        removal_sims = [s for t, s in r1["similarity_to_training"] if 5000 <= t < 10000]
        decay_sims = [s for t, s in r1["similarity_to_training"] if 10000 <= t < 15000]
        return_sims = [s for t, s in r1["similarity_to_training"] if t >= 15000]
        
        print("TEMPLATE PERSISTENCE:")
        if removal_sims:
            print(f"  After removal: {np.mean(removal_sims):.3f} similarity to training")
        if decay_sims:
            print(f"  After decay:   {np.mean(decay_sims):.3f} similarity to training")
        if return_sims:
            print(f"  After return:  {np.mean(return_sims):.3f} similarity to training")
    
    print()
    print("REALIGNMENT SPEED (sign-invariant |r|):")
    print(f"  Initial learning:  mean |r|={r2['initial']['mean_abs_r']:.3f}, slope={r2['initial']['slope_abs_r']:+.4f}/1k")
    print(f"  Realignment:       mean |r|={r2['realign']['mean_abs_r']:.3f}, slope={r2['realign']['slope_abs_r']:+.4f}/1k")
    print(f"  Naive learning:    mean |r|={r2['naive']['mean_abs_r']:.3f}, slope={r2['naive']['slope_abs_r']:+.4f}/1k")
    
    print()
    print("NOTE ON GEOMETRY METRICS:")
    print("  M_contrast (ratio-based) can swing wildly with forcing regime")
    print("  S_stability (1/median_var) is more robust for 'is mesh stable?'")
    print("  Use log10(S_stability) for interpretable comparisons")
    
    print()
    print("=" * 70)
    print("LEVEL 38 COMPLETE")
    print("=" * 70)
    
    return {
        "persistence": r1,
        "realignment": r2,
        "degradation": r3,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Level 38: Template Dynamics")
    p.add_argument("--persistence", action="store_true", help="Template persistence test")
    p.add_argument("--speed", action="store_true", help="Realignment speed test")
    p.add_argument("--degradation", action="store_true", help="Component degradation test")
    p.add_argument("--all", action="store_true", help="Run all")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    run_all = args.all or not any([args.persistence, args.speed, args.degradation])
    
    if run_all:
        run_level38()
    else:
        if args.persistence:
            experiment_template_persistence(seed=args.seed)
        if args.speed:
            experiment_realignment_speed(seed=args.seed)
        if args.degradation:
            experiment_signature_degradation(seed=args.seed)


if __name__ == "__main__":
    main()
