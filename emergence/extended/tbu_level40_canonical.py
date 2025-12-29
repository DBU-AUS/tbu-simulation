#!/usr/bin/env python3
"""
================================================================================
LEVEL 40: PREDICTIVE EDGE
================================================================================

THE QUESTION:
  What is memory FOR?
  
  L38-L39 established: Memory is template-based alignment dynamics
  L40 asks: Do templates give PREDICTIVE EDGE that improves N[ω] outcomes?

APPROACH:
  NOT adding new mechanisms. Measuring what already emerges.
  
  The template lives in:
    - EMA (α=0.05) - slow integration
    - Core states (0.9 decay) - persistence
    - Core thresholds (fixed) - learned sensitivity
    - Volatility window (50 steps) - history sensitivity
  
  The EMA is an IMPLICIT PREDICTION - "geometry will be like recent history"
  Prediction error = |EMA_t - observation_{t+1}|
  
  If templates give predictive edge, trained observers should show:
    - Lower prediction error on familiar geometry (main test)
    - Higher coherence (integration stability)
    - Possibly lower volatility (but this measures motion, not prediction)

METRICS:
  - Prediction Error: |EMA_t - observation_{t+1}| - the actual prediction test
  - Coherence: internal integration stability
  - Volatility: motion amplitude (may not show advantage - see L40 analysis)

EXPERIMENTS:
  1. Template vs Naive: Compare trained vs untrained on same geometry
  2. Cross-Reference: Systematic vs isolated training on related patterns
  3. Quality-Outcomes: Does template quality correlate with N[ω] measures?

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
# REAL TBU STACK (from L38/L39)
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
        
        # Core histories for pattern tracking
        self.core_A_history: List[float] = []
        self.core_B_history: List[float] = []
    
    def step(self, consumptions: List[float] = None, 
             basin_forcing: Dict[str, float] = None,
             diffusion_mod: float = 1.0):
        """Step substrate physics."""
        if consumptions is None:
            consumptions = [0.0] * self.n_observers
        self.observer_consumptions = consumptions
        
        self.history.append(self.grid.copy())
        if len(self.history) > self.history_window:
            self.history.pop(0)
        
        # Diffusion (reconditioning)
        laplacian = (
            np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
            np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) - 4 * self.grid
        )
        
        eff_diff = np.clip(self.diffusion * diffusion_mod, 0.02, 0.18)
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
    """Real observer with L35 parameters - where templates live."""
    
    observer_id: int
    seed: int = 42
    n_blocks: int = 6
    ema_alpha: float = 0.05  # Slow integration - template persistence
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
        self.prediction_error_history: List[float] = []
        self._last_ema: Optional[np.ndarray] = None  # For prediction error
    
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
        
        # Prediction error: how wrong was our EMA prediction?
        # EMA_t is implicit prediction of "what comes next"
        # Error = |EMA_t - actual_t+1|
        if self._last_ema is not None:
            pred_error = float(np.mean(np.abs(self._last_ema - sampled)))
            self.prediction_error_history.append(pred_error)
        else:
            self.prediction_error_history.append(0.0)
        
        if self.ema is None:
            self.ema = sampled.copy()
        else:
            self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * sampled
        
        # Store current EMA as prediction for next step
        self._last_ema = self.ema.copy()
        
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
            "prediction_error": self.prediction_error_history[-1],
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
    
    def get_template_state(self) -> Dict:
        """Extract current template (what persists)."""
        return {
            "ema": self.ema.copy() if self.ema is not None else None,
            "core_states": self.core_states.copy(),
            "core_thresholds": self.core_thresholds.copy(),
        }
    
    def get_template_similarity(self, reference: Dict) -> float:
        """Compute similarity to a reference template."""
        if reference["ema"] is None or self.ema is None:
            return 0.0
        
        # Core similarity (cosine)
        norm_c = np.linalg.norm(self.core_states)
        norm_r = np.linalg.norm(reference["core_states"])
        if norm_c < 1e-10 or norm_r < 1e-10:
            core_sim = 0.5
        else:
            core_sim = (np.dot(self.core_states, reference["core_states"]) / 
                       (norm_c * norm_r) + 1) / 2
        
        # EMA similarity (correlation)
        if np.std(self.ema) < 1e-10 or np.std(reference["ema"]) < 1e-10:
            ema_sim = 0.5
        else:
            r = np.corrcoef(self.ema, reference["ema"])[0, 1]
            ema_sim = (r + 1) / 2 if np.isfinite(r) else 0.5
        
        return float(0.7 * core_sim + 0.3 * ema_sim)


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
# EXPERIMENT 1: TEMPLATE VS NAIVE
# =============================================================================

def experiment_template_vs_naive(n_train: int = 3000, n_test: int = 2000, seed: int = 42):
    """
    Test: Does prior training reduce volatility on familiar geometry?
    
    Protocol:
      1. Run two observers (A=trained, B=naive) on Basin A
      2. TRAINING phase: Only Observer A runs (builds template)
      3. GAP phase: Both idle (template persists in A)
      4. TEST phase: Both A and B run on same geometry
      
    Measure:
      - Volatility (lower = better prediction)
      - Coherence (higher = better integration)
      - Time to stabilize
    
    If template gives predictive edge:
      - A should show lower volatility than B on familiar geometry
    """
    
    print("=" * 70)
    print("EXPERIMENT 1: TEMPLATE VS NAIVE")
    print("=" * 70)
    print()
    print("Question: Does prior training reduce volatility on familiar geometry?")
    print()
    
    # Load real ocean data
    ocean_data, ocean_meta = load_real_ocean_data(hours=72)
    
    # Create substrate and consequence
    substrate = SharedSiliconSubstrate(seed=seed)
    consequence = SharedBasinConsequence()
    
    # Two observers - same seed structure but different history
    obs_trained = Observer(observer_id=0, seed=seed)
    obs_naive = Observer(observer_id=1, seed=seed + 1000)  # Different seed for fairness
    
    results = {
        "training": {"trained": [], "naive": []},
        "gap": {"trained": [], "naive": []},
        "test": {"trained": [], "naive": []},
    }
    
    print(f"Ocean data: {ocean_meta}")
    print(f"Training: {n_train} steps, Gap: 500 steps, Test: {n_test} steps")
    print()
    
    # Phase 1: TRAINING (only trained observer runs)
    print("Phase 1: TRAINING (observer A builds template)...")
    
    for step in range(n_train):
        # Ocean modulates diffusion
        ocean_idx = step % len(ocean_data)
        ocean_val = ocean_data[ocean_idx]
        diff_mod = 0.8 + 0.4 * (ocean_val - ocean_data.min()) / (ocean_data.max() - ocean_data.min() + 1e-10)
        
        # Basin A forcing (consistent pattern)
        basin_forcing = {"A": np.sin(step * 0.01) + 0.5 * np.sin(step * 0.023)}
        
        # Trained observer processes
        obs_data = substrate.get_observation(0)
        trained_result = obs_trained.step(obs_data, external_noise=consequence.update(obs_trained.consumption_history[-1] if obs_trained.consumption_history else 0.5))
        
        # Naive observer does NOT run during training
        
        # Step substrate
        substrate.step(
            consumptions=[trained_result["consumption"], 0, 0, 0],
            basin_forcing=basin_forcing,
            diffusion_mod=diff_mod
        )
        
        if step % 500 == 0 or step == n_train - 1:
            results["training"]["trained"].append({
                "step": step,
                "volatility": trained_result["volatility"],
                "coherence": trained_result["coherence"],
            })
    
    # Store trained template
    trained_template = obs_trained.get_template_state()
    print(f"  Trained template captured. Final vol={trained_result['volatility']:.4f}, coh={trained_result['coherence']:.3f}")
    
    # Phase 2: GAP (both idle - tests template persistence)
    print("Phase 2: GAP (500 steps idle)...")
    
    for step in range(500):
        substrate.step(consumptions=[0, 0, 0, 0], diffusion_mod=1.0)
    
    # Phase 3: TEST (both observers on same familiar geometry)
    print("Phase 3: TEST (both observers on familiar geometry)...")
    
    # Reset naive observer to fresh state but same structure
    obs_naive = Observer(observer_id=1, seed=seed + 1000)
    
    for step in range(n_test):
        ocean_idx = (n_train + 500 + step) % len(ocean_data)
        ocean_val = ocean_data[ocean_idx]
        diff_mod = 0.8 + 0.4 * (ocean_val - ocean_data.min()) / (ocean_data.max() - ocean_data.min() + 1e-10)
        
        # Same forcing pattern as training
        effective_step = step  # Restart pattern
        basin_forcing = {"A": np.sin(effective_step * 0.01) + 0.5 * np.sin(effective_step * 0.023)}
        
        # Both observers process same geometry
        obs_data_0 = substrate.get_observation(0)
        obs_data_1 = substrate.get_observation(1)  # Adjacent position in same basin
        
        noise_level = consequence.update(
            (obs_trained.consumption_history[-1] + obs_naive.consumption_history[-1] 
             if obs_naive.consumption_history else obs_trained.consumption_history[-1]) / 2
        )
        
        trained_result = obs_trained.step(obs_data_0, external_noise=noise_level)
        naive_result = obs_naive.step(obs_data_1, external_noise=noise_level)
        
        substrate.step(
            consumptions=[trained_result["consumption"], naive_result["consumption"], 0, 0],
            basin_forcing=basin_forcing,
            diffusion_mod=diff_mod
        )
        
        if step % 200 == 0 or step == n_test - 1:
            results["test"]["trained"].append({
                "step": step,
                "volatility": trained_result["volatility"],
                "coherence": trained_result["coherence"],
                "prediction_error": trained_result["prediction_error"],
            })
            results["test"]["naive"].append({
                "step": step,
                "volatility": naive_result["volatility"],
                "coherence": naive_result["coherence"],
                "prediction_error": naive_result["prediction_error"],
            })
    
    # Analysis
    print()
    print("RESULTS:")
    print("-" * 50)
    
    trained_vol = [d["volatility"] for d in results["test"]["trained"]]
    naive_vol = [d["volatility"] for d in results["test"]["naive"]]
    trained_coh = [d["coherence"] for d in results["test"]["trained"]]
    naive_coh = [d["coherence"] for d in results["test"]["naive"]]
    trained_pred = [d["prediction_error"] for d in results["test"]["trained"]]
    naive_pred = [d["prediction_error"] for d in results["test"]["naive"]]
    
    print(f"  Trained - mean volatility: {np.mean(trained_vol):.4f} ± {np.std(trained_vol):.4f}")
    print(f"  Naive   - mean volatility: {np.mean(naive_vol):.4f} ± {np.std(naive_vol):.4f}")
    print()
    print(f"  Trained - mean coherence: {np.mean(trained_coh):.3f} ± {np.std(trained_coh):.3f}")
    print(f"  Naive   - mean coherence: {np.mean(naive_coh):.3f} ± {np.std(naive_coh):.3f}")
    print()
    print(f"  Trained - mean prediction error: {np.mean(trained_pred):.4f} ± {np.std(trained_pred):.4f}")
    print(f"  Naive   - mean prediction error: {np.mean(naive_pred):.4f} ± {np.std(naive_pred):.4f}")
    print()
    
    vol_advantage = (np.mean(naive_vol) - np.mean(trained_vol)) / (np.mean(naive_vol) + 1e-10)
    coh_advantage = (np.mean(trained_coh) - np.mean(naive_coh)) / (np.mean(naive_coh) + 1e-10)
    pred_advantage = (np.mean(naive_pred) - np.mean(trained_pred)) / (np.mean(naive_pred) + 1e-10)
    
    print(f"  Volatility advantage: {vol_advantage*100:.1f}% (positive = trained better)")
    print(f"  Coherence advantage: {coh_advantage*100:.1f}% (positive = trained better)")
    print(f"  Prediction advantage: {pred_advantage*100:.1f}% (positive = trained better)")
    print()
    
    if vol_advantage > 0.05:
        print("  ✓ TEMPLATE REDUCES VOLATILITY")
    elif vol_advantage > 0:
        print("  ~ Slight volatility advantage")
    else:
        print("  ✗ No volatility advantage detected")
    
    if coh_advantage > 0.02:
        print("  ✓ TEMPLATE IMPROVES COHERENCE")
    else:
        print("  ✗ No coherence advantage detected")
    
    if pred_advantage > 0.05:
        print("  ✓ TEMPLATE REDUCES PREDICTION ERROR - predictive edge confirmed")
    elif pred_advantage > 0:
        print("  ~ Slight prediction advantage")
    else:
        print("  ✗ No prediction advantage detected")
    
    # Template persistence check
    final_similarity = obs_trained.get_template_similarity(trained_template)
    print(f"\n  Template similarity after gap+test: {final_similarity:.3f}")
    
    return results


# =============================================================================
# EXPERIMENT 2: CROSS-REFERENCE ADVANTAGE
# =============================================================================

def experiment_cross_reference(n_train: int = 2000, n_test: int = 1500, seed: int = 42):
    """
    Test: Does systematic training on related patterns improve performance?
    
    Protocol:
      1. Observer A: Train on pattern 1, then pattern 2 (systematic)
      2. Observer B: Train on pattern 1 only (isolated)
      3. Test both on COMBINED pattern (1+2)
      
    If cross-reference helps:
      - A should perform better on combined pattern than B
    """
    
    print("=" * 70)
    print("EXPERIMENT 2: CROSS-REFERENCE ADVANTAGE")
    print("=" * 70)
    print()
    print("Question: Does systematic training improve performance on novel combinations?")
    print()
    
    ocean_data, _ = load_real_ocean_data(hours=48)
    
    substrate = SharedSiliconSubstrate(seed=seed)
    consequence = SharedBasinConsequence()
    
    obs_systematic = Observer(observer_id=0, seed=seed)
    obs_isolated = Observer(observer_id=1, seed=seed + 1000)
    
    # Pattern definitions
    pattern_1 = lambda s: np.sin(s * 0.01)  # Slow oscillation
    pattern_2 = lambda s: 0.5 * np.sin(s * 0.037)  # Different frequency
    pattern_combined = lambda s: pattern_1(s) + pattern_2(s)  # Both together
    
    print("Phase 1: Train systematic observer on pattern 1...")
    
    for step in range(n_train):
        ocean_idx = step % len(ocean_data)
        diff_mod = 0.8 + 0.4 * (ocean_data[ocean_idx] - ocean_data.min()) / (ocean_data.max() - ocean_data.min() + 1e-10)
        
        basin_forcing = {"A": pattern_1(step)}
        
        obs_data = substrate.get_observation(0)
        result = obs_systematic.step(obs_data, external_noise=consequence.update(result["consumption"] if step > 0 else 0.5))
        
        substrate.step(
            consumptions=[result["consumption"], 0, 0, 0],
            basin_forcing=basin_forcing,
            diffusion_mod=diff_mod
        )
    
    print(f"  Pattern 1 complete. vol={result['volatility']:.4f}")
    
    print("Phase 2: Train systematic observer on pattern 2...")
    
    for step in range(n_train):
        ocean_idx = (n_train + step) % len(ocean_data)
        diff_mod = 0.8 + 0.4 * (ocean_data[ocean_idx] - ocean_data.min()) / (ocean_data.max() - ocean_data.min() + 1e-10)
        
        basin_forcing = {"A": pattern_2(step)}
        
        obs_data = substrate.get_observation(0)
        result = obs_systematic.step(obs_data, external_noise=consequence.update(result["consumption"]))
        
        substrate.step(
            consumptions=[result["consumption"], 0, 0, 0],
            basin_forcing=basin_forcing,
            diffusion_mod=diff_mod
        )
    
    print(f"  Pattern 2 complete. vol={result['volatility']:.4f}")
    
    # Reset substrate, train isolated observer on pattern 1 only (same duration as systematic)
    substrate = SharedSiliconSubstrate(seed=seed)
    consequence = SharedBasinConsequence()
    
    print("Phase 3: Train isolated observer on pattern 1 only (2x duration)...")
    
    for step in range(2 * n_train):
        ocean_idx = step % len(ocean_data)
        diff_mod = 0.8 + 0.4 * (ocean_data[ocean_idx] - ocean_data.min()) / (ocean_data.max() - ocean_data.min() + 1e-10)
        
        basin_forcing = {"A": pattern_1(step)}
        
        obs_data = substrate.get_observation(1)
        result = obs_isolated.step(obs_data, external_noise=consequence.update(result["consumption"] if step > 0 else 0.5))
        
        substrate.step(
            consumptions=[0, result["consumption"], 0, 0],
            basin_forcing=basin_forcing,
            diffusion_mod=diff_mod
        )
    
    print(f"  Pattern 1 (2x) complete. vol={result['volatility']:.4f}")
    
    # Test both on combined pattern
    print("Phase 4: Test both on COMBINED pattern...")
    
    substrate = SharedSiliconSubstrate(seed=seed + 999)  # Fresh substrate
    consequence = SharedBasinConsequence()
    
    results_sys = []
    results_iso = []
    
    for step in range(n_test):
        ocean_idx = step % len(ocean_data)
        diff_mod = 0.8 + 0.4 * (ocean_data[ocean_idx] - ocean_data.min()) / (ocean_data.max() - ocean_data.min() + 1e-10)
        
        basin_forcing = {"A": pattern_combined(step)}
        
        obs_data_0 = substrate.get_observation(0)
        obs_data_1 = substrate.get_observation(1)
        
        noise = consequence.update(0.5)
        
        result_sys = obs_systematic.step(obs_data_0, external_noise=noise)
        result_iso = obs_isolated.step(obs_data_1, external_noise=noise)
        
        substrate.step(
            consumptions=[result_sys["consumption"], result_iso["consumption"], 0, 0],
            basin_forcing=basin_forcing,
            diffusion_mod=diff_mod
        )
        
        if step % 150 == 0 or step == n_test - 1:
            results_sys.append({"step": step, "vol": result_sys["volatility"], "coh": result_sys["coherence"]})
            results_iso.append({"step": step, "vol": result_iso["volatility"], "coh": result_iso["coherence"]})
    
    # Analysis
    print()
    print("RESULTS (on combined pattern):")
    print("-" * 50)
    
    sys_vol = np.mean([r["vol"] for r in results_sys])
    iso_vol = np.mean([r["vol"] for r in results_iso])
    sys_coh = np.mean([r["coh"] for r in results_sys])
    iso_coh = np.mean([r["coh"] for r in results_iso])
    
    print(f"  Systematic - mean volatility: {sys_vol:.4f}")
    print(f"  Isolated   - mean volatility: {iso_vol:.4f}")
    print()
    print(f"  Systematic - mean coherence: {sys_coh:.3f}")
    print(f"  Isolated   - mean coherence: {iso_coh:.3f}")
    print()
    
    vol_advantage = (iso_vol - sys_vol) / (iso_vol + 1e-10)
    
    if vol_advantage > 0.05:
        print(f"  ✓ CROSS-REFERENCE ADVANTAGE: {vol_advantage*100:.1f}% lower volatility")
    elif vol_advantage > 0:
        print(f"  ~ Slight cross-reference advantage: {vol_advantage*100:.1f}%")
    else:
        print(f"  ✗ No cross-reference advantage detected")
    
    return {"systematic": results_sys, "isolated": results_iso}


# =============================================================================
# EXPERIMENT 3: TEMPLATE QUALITY vs N[ω] OUTCOMES
# =============================================================================

def experiment_quality_outcomes(n_steps: int = 4000, seed: int = 42):
    """
    Test: Does template quality correlate with N[ω]-relevant outcomes?
    
    Protocol:
      1. Train observers for varying durations (weak to strong templates)
      2. Measure template quality (similarity to full-training reference)
      3. Test each on familiar geometry
      4. Correlate template quality with outcomes (volatility, coherence)
      
    If template quality matters:
      - Better templates should produce lower volatility, higher coherence
    """
    
    print("=" * 70)
    print("EXPERIMENT 3: TEMPLATE QUALITY vs OUTCOMES")
    print("=" * 70)
    print()
    print("Question: Does template quality correlate with N[ω] outcomes?")
    print()
    
    ocean_data, _ = load_real_ocean_data(hours=48)
    
    training_durations = [0, 500, 1000, 2000, 4000]  # Increasing template strength
    
    results = {}
    
    # First, create reference template (full training)
    print("Creating reference template (4000 steps)...")
    
    ref_substrate = SharedSiliconSubstrate(seed=seed)
    ref_consequence = SharedBasinConsequence()
    ref_observer = Observer(observer_id=0, seed=seed)
    
    for step in range(4000):
        ocean_idx = step % len(ocean_data)
        diff_mod = 0.8 + 0.4 * (ocean_data[ocean_idx] - ocean_data.min()) / (ocean_data.max() - ocean_data.min() + 1e-10)
        
        basin_forcing = {"A": np.sin(step * 0.01) + 0.3 * np.sin(step * 0.029)}
        
        obs_data = ref_substrate.get_observation(0)
        result = ref_observer.step(obs_data, external_noise=ref_consequence.update(result["consumption"] if step > 0 else 0.5))
        
        ref_substrate.step(
            consumptions=[result["consumption"], 0, 0, 0],
            basin_forcing=basin_forcing,
            diffusion_mod=diff_mod
        )
    
    reference_template = ref_observer.get_template_state()
    print(f"  Reference template captured. vol={result['volatility']:.4f}, coh={result['coherence']:.3f}")
    print()
    
    # Test each training duration
    for train_dur in training_durations:
        print(f"Testing {train_dur} steps training...")
        
        substrate = SharedSiliconSubstrate(seed=seed + train_dur)
        consequence = SharedBasinConsequence()
        observer = Observer(observer_id=0, seed=seed)
        
        # Training phase
        for step in range(train_dur):
            ocean_idx = step % len(ocean_data)
            diff_mod = 0.8 + 0.4 * (ocean_data[ocean_idx] - ocean_data.min()) / (ocean_data.max() - ocean_data.min() + 1e-10)
            
            basin_forcing = {"A": np.sin(step * 0.01) + 0.3 * np.sin(step * 0.029)}
            
            obs_data = substrate.get_observation(0)
            result = observer.step(obs_data, external_noise=consequence.update(result["consumption"] if step > 0 else 0.5))
            
            substrate.step(
                consumptions=[result["consumption"], 0, 0, 0],
                basin_forcing=basin_forcing,
                diffusion_mod=diff_mod
            )
        
        # Get template quality
        template_quality = observer.get_template_similarity(reference_template) if train_dur > 0 else 0.0
        
        # Test phase (fresh substrate, same pattern)
        test_substrate = SharedSiliconSubstrate(seed=seed + 9999)
        test_consequence = SharedBasinConsequence()
        
        test_results = []
        
        for step in range(2000):
            ocean_idx = step % len(ocean_data)
            diff_mod = 0.8 + 0.4 * (ocean_data[ocean_idx] - ocean_data.min()) / (ocean_data.max() - ocean_data.min() + 1e-10)
            
            basin_forcing = {"A": np.sin(step * 0.01) + 0.3 * np.sin(step * 0.029)}
            
            obs_data = test_substrate.get_observation(0)
            result = observer.step(obs_data, external_noise=test_consequence.update(result["consumption"] if step > 0 else 0.5))
            
            test_substrate.step(
                consumptions=[result["consumption"], 0, 0, 0],
                basin_forcing=basin_forcing,
                diffusion_mod=diff_mod
            )
            
            if step >= 500:  # Skip warmup
                test_results.append({
                    "volatility": result["volatility"],
                    "coherence": result["coherence"],
                    "prediction_error": result["prediction_error"],
                })
        
        mean_vol = np.mean([r["volatility"] for r in test_results])
        mean_coh = np.mean([r["coherence"] for r in test_results])
        mean_pred = np.mean([r["prediction_error"] for r in test_results])
        
        results[f"{train_dur} steps"] = {
            "training_duration": train_dur,
            "template_quality": template_quality,
            "volatility": mean_vol,
            "coherence": mean_coh,
            "prediction_error": mean_pred,
        }
        
        print(f"  quality={template_quality:.3f}, vol={mean_vol:.4f}, coh={mean_coh:.3f}, pred_err={mean_pred:.4f}")
    
    # Correlation analysis
    print()
    print("CORRELATION ANALYSIS:")
    print("-" * 50)
    
    qualities = [results[k]["template_quality"] for k in results]
    vols = [results[k]["volatility"] for k in results]
    cohs = [results[k]["coherence"] for k in results]
    preds = [results[k]["prediction_error"] for k in results]
    
    # Quality vs Volatility (expect negative - better quality = lower vol)
    if np.std(qualities) > 1e-10 and np.std(vols) > 1e-10:
        r_qual_vol = np.corrcoef(qualities, vols)[0, 1]
    else:
        r_qual_vol = 0.0
    
    # Quality vs Coherence (expect positive - better quality = higher coh)
    if np.std(qualities) > 1e-10 and np.std(cohs) > 1e-10:
        r_qual_coh = np.corrcoef(qualities, cohs)[0, 1]
    else:
        r_qual_coh = 0.0
    
    # Quality vs Prediction Error (expect negative - better quality = lower error)
    if np.std(qualities) > 1e-10 and np.std(preds) > 1e-10:
        r_qual_pred = np.corrcoef(qualities, preds)[0, 1]
    else:
        r_qual_pred = 0.0
    
    print(f"  Template Quality vs Volatility:       r = {r_qual_vol:.3f} (expect negative)")
    print(f"  Template Quality vs Coherence:        r = {r_qual_coh:.3f} (expect positive)")
    print(f"  Template Quality vs Prediction Error: r = {r_qual_pred:.3f} (expect negative)")
    print()
    
    if r_qual_vol < -0.3:
        print("  ✓ BETTER TEMPLATES → LOWER VOLATILITY")
    else:
        print("  ✗ No clear quality-volatility relationship")
    
    if r_qual_coh > 0.3:
        print("  ✓ BETTER TEMPLATES → HIGHER COHERENCE")
    else:
        print("  ✗ No clear quality-coherence relationship")
    
    if r_qual_pred < -0.3:
        print("  ✓ BETTER TEMPLATES → LOWER PREDICTION ERROR")
    else:
        print("  ✗ No clear quality-prediction relationship")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_level40():
    """Run all L40 experiments."""
    
    print()
    print("=" * 70)
    print("LEVEL 40: PREDICTIVE EDGE")
    print("=" * 70)
    print()
    print("THE QUESTION: What is memory FOR?")
    print()
    print("L38-L39 established memory is template-based alignment dynamics.")
    print("L40 tests whether templates give PREDICTIVE EDGE.")
    print()
    print("NOT adding new mechanisms - measuring what already emerges.")
    print()
    
    r1 = experiment_template_vs_naive(seed=42)
    print()
    
    r2 = experiment_cross_reference(seed=42)
    print()
    
    r3 = experiment_quality_outcomes(seed=42)
    
    # Summary
    print()
    print("=" * 70)
    print("LEVEL 40 SUMMARY")
    print("=" * 70)
    print()
    
    # Extract key findings
    if r1["test"]["trained"] and r1["test"]["naive"]:
        t_vol = np.mean([d["volatility"] for d in r1["test"]["trained"]])
        n_vol = np.mean([d["volatility"] for d in r1["test"]["naive"]])
        vol_adv = (n_vol - t_vol) / (n_vol + 1e-10) * 100
        
        t_coh = np.mean([d["coherence"] for d in r1["test"]["trained"]])
        n_coh = np.mean([d["coherence"] for d in r1["test"]["naive"]])
        coh_adv = (t_coh - n_coh) / (n_coh + 1e-10) * 100
        
        t_pred = np.mean([d["prediction_error"] for d in r1["test"]["trained"]])
        n_pred = np.mean([d["prediction_error"] for d in r1["test"]["naive"]])
        pred_adv = (n_pred - t_pred) / (n_pred + 1e-10) * 100
        
        print(f"TEMPLATE VS NAIVE:")
        print(f"  Volatility advantage:  {vol_adv:+.1f}%")
        print(f"  Coherence advantage:   {coh_adv:+.1f}%")
        print(f"  Prediction advantage:  {pred_adv:+.1f}%")
    
    if r2["systematic"] and r2["isolated"]:
        s_vol = np.mean([d["vol"] for d in r2["systematic"]])
        i_vol = np.mean([d["vol"] for d in r2["isolated"]])
        cr_adv = (i_vol - s_vol) / (i_vol + 1e-10) * 100
        print(f"\nCROSS-REFERENCE: {cr_adv:+.1f}% volatility advantage on novel combination")
    
    # Quality correlations
    qualities = [r3[k]["template_quality"] for k in r3]
    vols = [r3[k]["volatility"] for k in r3]
    cohs = [r3[k]["coherence"] for k in r3]
    preds = [r3[k]["prediction_error"] for k in r3]
    
    print(f"\nQUALITY CORRELATIONS:")
    if np.std(qualities) > 1e-10:
        r_qv = np.corrcoef(qualities, vols)[0, 1] if np.std(vols) > 1e-10 else 0
        r_qc = np.corrcoef(qualities, cohs)[0, 1] if np.std(cohs) > 1e-10 else 0
        r_qp = np.corrcoef(qualities, preds)[0, 1] if np.std(preds) > 1e-10 else 0
        print(f"  Quality vs Volatility:       r = {r_qv:+.3f}")
        print(f"  Quality vs Coherence:        r = {r_qc:+.3f}")
        print(f"  Quality vs Prediction Error: r = {r_qp:+.3f}")
    
    print()
    print("=" * 70)
    
    return {"exp1": r1, "exp2": r2, "exp3": r3}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Level 40: Predictive Edge")
    p.add_argument("--exp1", action="store_true", help="Template vs naive")
    p.add_argument("--exp2", action="store_true", help="Cross-reference advantage")
    p.add_argument("--exp3", action="store_true", help="Quality vs outcomes")
    p.add_argument("--all", action="store_true", help="Run all")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    run_all = args.all or not any([args.exp1, args.exp2, args.exp3])
    
    if run_all:
        run_level40()
    else:
        if args.exp1:
            experiment_template_vs_naive(seed=args.seed)
        if args.exp2:
            experiment_cross_reference(seed=args.seed)
        if args.exp3:
            experiment_quality_outcomes(seed=args.seed)


if __name__ == "__main__":
    main()
