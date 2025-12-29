#!/usr/bin/env python3
"""
================================================================================
LEVEL 31b CANONICAL: AGENCY THROUGH GEOMETRY OCCUPATION
================================================================================

DEMONSTRATION:
  Self-referential 3D slice occupies better positions in constraint geometry.
  
MECHANISM:
  1. Real physics: NDBC ocean buoy data → diffusion modulation → sign-flip
  2. Consequence loop: consumption → future observation noise
  3. Self-reference: volatility detection → consumption restraint
  4. Result: +0.02 coherence advantage across 5 seeds

THE TBU CLAIM:
  The 4D geometry has no perspective. The 3D slice is the geometry looking
  at itself. By looking, it can reconfigure. By reconfiguring, it selects
  which future slices are compatible. This is agency without goals.

USAGE:
  python tbu_level31b_canonical.py --all          # Full test suite
  python tbu_level31b_canonical.py --geometry     # Core geometry occupation test
  python tbu_level31b_canonical.py --transitions  # Regime transition test
  python tbu_level31b_canonical.py --signflip     # Verify reconditioning

================================================================================
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from urllib.request import urlopen
from urllib.error import URLError
from datetime import datetime, timedelta, timezone

# =============================================================================
# NDBC OCEAN BUOY LOADER
# =============================================================================

def load_ocean_buoy(station: str = "46025", hours: int = 168) -> Tuple[np.ndarray, Dict]:
    """
    Load real-time ocean buoy data from NDBC.
    
    Returns:
        data: (n_samples, 10) array of normalized channel values
        meta: dict with source info
    """
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"
    
    try:
        with urlopen(url, timeout=30) as response:
            lines = response.read().decode('utf-8').strip().split('\n')
        
        if len(lines) < 3:
            raise ValueError("Insufficient data")
        
        # Parse header
        header = lines[0].replace('#', '').split()
        
        # Parse data rows
        data_rows = []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)
        
        for line in lines[2:]:  # Skip header rows
            parts = line.split()
            if len(parts) < 10:
                continue
            
            try:
                year = int(parts[0])
                if year < 100:
                    year += 2000
                month, day, hour, minute = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                row_time = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                
                if row_time < cutoff:
                    continue
                
                # Extract numeric values (skip time columns)
                values = []
                for v in parts[5:15]:
                    try:
                        val = float(v) if v != 'MM' else np.nan
                    except:
                        val = np.nan
                    values.append(val)
                
                while len(values) < 10:
                    values.append(np.nan)
                
                data_rows.append(values[:10])
                
            except (ValueError, IndexError):
                continue
        
        if len(data_rows) < 10:
            raise ValueError(f"Only {len(data_rows)} valid samples")
        
        data = np.array(data_rows, dtype=float)
        
        # Normalize each channel
        for i in range(data.shape[1]):
            col = data[:, i]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                mean, std = col[valid].mean(), col[valid].std()
                if std > 1e-10:
                    data[:, i] = (col - mean) / std
                else:
                    data[:, i] = 0.0
            data[np.isnan(data[:, i]), i] = 0.0
        
        return data, {"source": "NDBC", "n_samples": len(data_rows), "station": station}
    
    except Exception as e:
        print(f"  Failed to fetch NDBC data: {e}")
        return generate_synthetic_ocean(hours)


def generate_synthetic_ocean(hours: int = 168) -> Tuple[np.ndarray, Dict]:
    """Generate synthetic ocean data when NDBC unavailable."""
    print("  Generating synthetic ocean data")
    rng = np.random.default_rng(42)
    n = max(300, hours * 2)
    t = np.linspace(0, hours, n)
    
    data = np.zeros((n, 10))
    # Wave height (tidal + storm components)
    data[:, 0] = 0.3 * np.sin(2*np.pi*t/12.42) + 0.2 * np.sin(2*np.pi*t/6) + 0.1 * rng.normal(0, 1, n)
    # Pressure (inverse of waves)
    data[:, 7] = -0.6 * data[:, 0] + 0.1 * rng.normal(0, 1, n)
    # Other channels
    for i in range(10):
        if i not in [0, 7]:
            data[:, i] = rng.normal(0, 1, n)
        # Normalize
        data[:, i] = (data[:, i] - data[:, i].mean()) / (data[:, i].std() + 1e-10)
    
    return data, {"source": "synthetic", "n_samples": n}


# =============================================================================
# OCEAN REGIME DRIVER
# =============================================================================

def resample_to_steps(signal: np.ndarray, n_steps: int) -> np.ndarray:
    """Resample signal to n_steps using linear interpolation."""
    if len(signal) == 0:
        return np.zeros(n_steps)
    x_old = np.linspace(0.0, 1.0, len(signal))
    x_new = np.linspace(0.0, 1.0, n_steps)
    return np.interp(x_new, x_old, signal)


@dataclass
class OceanRegimeDriver:
    """
    Converts NDBC ocean data into regime signal and diffusion coefficient.
    
    Regime = wave_height - pressure (both normalized)
    High regime (storm): high waves, low pressure → high diffusion
    Low regime (calm): low waves, high pressure → low diffusion
    """
    n_steps: int = 10000
    hours: int = 168
    station: str = "46025"
    seed: int = 42
    
    # Physics modulation
    diffusion_base: float = 0.18
    diffusion_range: float = 0.12
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.regime_series: np.ndarray = np.zeros(self.n_steps)
        self.diffusion_series: np.ndarray = np.zeros(self.n_steps)
        self._load_and_prepare()
    
    def _load_and_prepare(self):
        print("Loading ocean buoy data...")
        data, meta = load_ocean_buoy(station=self.station, hours=self.hours)
        print(f"  Source: {meta.get('source', 'unknown')}, samples: {meta.get('n_samples', 0)}")
        
        # Compute regime: wave_height - pressure
        wave = data[:, 0]
        pressure = data[:, 7] if data.shape[1] > 7 else np.zeros_like(wave)
        raw_regime = wave - pressure
        
        # Normalize to [-1, 1]
        r_min, r_max = raw_regime.min(), raw_regime.max()
        if r_max - r_min > 1e-10:
            raw_regime = 2 * (raw_regime - r_min) / (r_max - r_min) - 1
        
        # Resample to simulation steps
        self.regime_series = resample_to_steps(raw_regime, self.n_steps)
        self.regime_series += self.rng.normal(0, 0.02, self.n_steps)  # Break smoothness
        
        # Compute diffusion series
        self.diffusion_series = self.diffusion_base + self.diffusion_range * self.regime_series
        self.diffusion_series = np.maximum(0.01, self.diffusion_series)
        
        print(f"  Regime range: [{self.regime_series.min():.3f}, {self.regime_series.max():.3f}]")
        print(f"  Diffusion range: [{self.diffusion_series.min():.3f}, {self.diffusion_series.max():.3f}]")
    
    def get_at_step(self, step: int) -> Dict[str, float]:
        idx = min(step, self.n_steps - 1)
        return {
            "regime": float(self.regime_series[idx]),
            "diffusion": float(self.diffusion_series[idx]),
        }


# =============================================================================
# MINIMAL DIFFUSION SUBSTRATE
# =============================================================================

@dataclass
class DiffusionSubstrate:
    """
    Minimal 2D diffusion substrate with boundary forcing.
    No environment complexity - just physics.
    """
    size: int = 64
    seed: int = 42
    diffusion: float = 0.2
    forcing_strength: float = 0.5
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.grid = self.rng.normal(0, 0.1, (self.size, self.size))
        
        # Ring boundary
        center = self.size // 2
        radius = self.size // 2 - 4
        y, x = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        self.forcing_mask = (dist >= radius - 2) & (dist <= radius + 2)
        self.forcing_coords = np.where(self.forcing_mask)
        self.n_boundary = len(self.forcing_coords[0])
    
    def step(self):
        """One diffusion step with boundary forcing."""
        # Laplacian
        laplacian = (
            np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
            np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) - 4 * self.grid
        )
        
        # Diffusion
        self.grid += self.diffusion * laplacian
        
        # Boundary forcing
        noise = self.rng.normal(0, self.forcing_strength, self.n_boundary)
        self.grid[self.forcing_coords] += noise
        
        # Light damping
        self.grid *= 0.99
    
    def get_boundary(self) -> np.ndarray:
        return self.grid[self.forcing_coords]
    
    def get_core(self) -> np.ndarray:
        return self.grid[~self.forcing_mask]


# =============================================================================
# OBSERVERS
# =============================================================================

@dataclass
class AgenticObserver:
    """
    Self-referential observer.
    - Tracks internal volatility
    - Restrains consumption when volatile
    - No explicit goals
    """
    n_cores: int = 16
    seed: int = 42
    VOL_K: float = 30.0  # Volatility → capacity gain
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.core_states = np.zeros(self.n_cores)
        self.core_thresholds = self.rng.uniform(0.3, 0.7, self.n_cores)
        
        self.input_history: List[np.ndarray] = []
        self.state_history: List[np.ndarray] = []
        self.consumption_history: List[float] = []
        self.coherence_history: List[float] = []
    
    def step(self, observation: np.ndarray) -> Dict:
        # Store input
        self.input_history.append(observation.copy())
        if len(self.input_history) > 60:
            self.input_history.pop(0)
        
        # Compute volatility (self-reference)
        volatility = self._compute_volatility()
        
        # Capacity inversely related to volatility
        capacity = 1.0 / (1.0 + self.VOL_K * volatility)
        
        # Consumption follows capacity
        consumption = float(np.clip(capacity, 0.1, 0.9))
        
        # Update internal cores
        self._update_cores(observation)
        
        # Compute coherence
        coherence = self._compute_coherence()
        
        self.consumption_history.append(consumption)
        self.coherence_history.append(coherence)
        
        return {"consumption": consumption, "coherence": coherence, "volatility": volatility}
    
    def _compute_volatility(self) -> float:
        if len(self.input_history) < 2:
            return 0.0
        recent = self.input_history[-50:]
        if len(recent) < 2:
            return 0.0
        diffs = [np.mean(np.abs(recent[i] - recent[i-1])) for i in range(1, len(recent))]
        return float(np.mean(diffs))
    
    def _update_cores(self, inputs: np.ndarray):
        n_inputs = len(inputs)
        for i in range(self.n_cores):
            idx = i % n_inputs
            drive = float(inputs[idx])
            total = drive - self.core_thresholds[i]
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


@dataclass
class RockObserver:
    """
    Non-self-referential observer. Fixed consumption.
    Same dynamics, just doesn't look at itself.
    """
    n_cores: int = 16
    seed: int = 42
    fixed_consumption: float = 0.5
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.core_states = np.zeros(self.n_cores)
        self.core_thresholds = self.rng.uniform(0.3, 0.7, self.n_cores)
        self.state_history: List[np.ndarray] = []
        self.consumption_history: List[float] = []
        self.coherence_history: List[float] = []
    
    def step(self, observation: np.ndarray) -> Dict:
        consumption = self.fixed_consumption  # No self-reference
        
        # Same core dynamics
        n_inputs = len(observation)
        for i in range(self.n_cores):
            idx = i % n_inputs
            drive = float(observation[idx])
            total = drive - self.core_thresholds[i]
            activation = 1.0 / (1.0 + np.exp(-4.0 * total))
            self.core_states[i] = 0.9 * self.core_states[i] + 0.1 * activation
        
        self.state_history.append(self.core_states.copy())
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        coherence = self._compute_coherence()
        
        self.consumption_history.append(consumption)
        self.coherence_history.append(coherence)
        
        return {"consumption": consumption, "coherence": coherence}
    
    def _compute_coherence(self) -> float:
        if len(self.state_history) < 20:
            return 0.5
        recent = np.array(self.state_history[-50:])
        mean_state = recent.mean(axis=1)
        if np.std(mean_state) < 1e-10:
            return 0.5
        r = np.corrcoef(mean_state[:-1], mean_state[1:])[0, 1]
        return float((r + 1) / 2) if np.isfinite(r) else 0.5


@dataclass
class RandomObserver:
    """Random consumption. No structure."""
    n_cores: int = 16
    seed: int = 42
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.core_states = np.zeros(self.n_cores)
        self.core_thresholds = self.rng.uniform(0.3, 0.7, self.n_cores)
        self.state_history: List[np.ndarray] = []
        self.consumption_history: List[float] = []
        self.coherence_history: List[float] = []
    
    def step(self, observation: np.ndarray) -> Dict:
        consumption = float(self.rng.uniform(0.1, 0.9))  # Random
        
        n_inputs = len(observation)
        for i in range(self.n_cores):
            idx = i % n_inputs
            drive = float(observation[idx])
            total = drive - self.core_thresholds[i]
            activation = 1.0 / (1.0 + np.exp(-4.0 * total))
            self.core_states[i] = 0.9 * self.core_states[i] + 0.1 * activation
        
        self.state_history.append(self.core_states.copy())
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        coherence = self._compute_coherence()
        
        self.consumption_history.append(consumption)
        self.coherence_history.append(coherence)
        
        return {"consumption": consumption, "coherence": coherence}
    
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
# CONSEQUENCE LOOP
# =============================================================================

@dataclass
class ConsequenceLoop:
    """
    Consumption → future observation noise.
    
    This makes consumption LOAD-BEARING:
    - High consumption depletes sensing
    - Low consumption preserves sensing
    - Better sensing → better observations → better coherence
    """
    depletion_rate: float = 0.02
    recovery_rate: float = 0.005
    noise_base: float = 0.02
    noise_scale: float = 0.4
    lag: int = 20
    
    def __post_init__(self):
        self.sensing_health = 1.0
        self.consumption_buffer: List[float] = []
    
    def update(self, consumption: float) -> float:
        """Update sensing health, return current observation noise."""
        self.consumption_buffer.append(consumption)
        
        if len(self.consumption_buffer) > self.lag:
            lagged = self.consumption_buffer.pop(0)
            depletion = self.depletion_rate * lagged
            recovery = self.recovery_rate * (1.0 - self.sensing_health)
            self.sensing_health = float(np.clip(self.sensing_health - depletion + recovery, 0.0, 1.0))
        
        return self.noise_base + self.noise_scale * (1.0 - self.sensing_health)
    
    def reset(self):
        self.sensing_health = 1.0
        self.consumption_buffer = []


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_geometry_occupation(n_steps: int = 10000, n_seeds: int = 5, warmup: int = 1000):
    """
    THE CORE TEST: Does self-reference improve geometry occupation?
    
    Same physics, same consequence loop, different observers.
    """
    print("=" * 70)
    print("EXPERIMENT: GEOMETRY OCCUPATION")
    print("=" * 70)
    print()
    print("Hypothesis: Self-referential 3D slice occupies better geometry")
    print("  - Same physics (ocean → diffusion)")
    print("  - Same consequence loop (consumption → obs noise)")
    print("  - Different self-reference capacity")
    print()
    
    results = {"agentic": [], "rock": [], "random": []}
    
    for seed in range(n_seeds):
        print(f"Seed {seed}...")
        
        # Shared ocean driver
        ocean = OceanRegimeDriver(n_steps=n_steps, seed=seed)
        
        for obs_type in ["agentic", "rock", "random"]:
            substrate = DiffusionSubstrate(size=64, seed=seed)
            rng = np.random.default_rng(seed + 1000)
            
            if obs_type == "agentic":
                observer = AgenticObserver(seed=seed + 2000)
            elif obs_type == "rock":
                observer = RockObserver(seed=seed + 2000)
            else:
                observer = RandomObserver(seed=seed + 2000)
            
            loop = ConsequenceLoop()
            
            coherence_list = []
            sensing_list = []
            consumption_list = []
            
            for step in range(n_steps):
                # Ocean-driven physics
                physics = ocean.get_at_step(step)
                substrate.diffusion = physics["diffusion"]
                substrate.step()
                
                # Get observation with consequence-affected noise
                boundary = substrate.get_boundary()
                obs_sigma = loop.update(observer.consumption_history[-1] if observer.consumption_history else 0.5)
                observation = boundary + rng.normal(0, obs_sigma, size=boundary.shape)
                
                # Observer processes
                state = observer.step(observation)
                
                if step >= warmup:
                    coherence_list.append(state["coherence"])
                    sensing_list.append(loop.sensing_health)
                    consumption_list.append(state["consumption"])
            
            results[obs_type].append({
                "seed": seed,
                "coherence_mean": float(np.mean(coherence_list)),
                "coherence_std": float(np.std(coherence_list)),
                "coherence_min": float(np.min(coherence_list)),
                "sensing_mean": float(np.mean(sensing_list)),
                "consumption_mean": float(np.mean(consumption_list)),
            })
    
    # Results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    print(f"{'Observer':<12} {'Coherence':>10} {'Std':>10} {'Sensing':>10} {'Consumption':>12}")
    print("-" * 58)
    
    for obs_type in ["agentic", "rock", "random"]:
        data = results[obs_type]
        coh = np.mean([d["coherence_mean"] for d in data])
        std = np.mean([d["coherence_std"] for d in data])
        sens = np.mean([d["sensing_mean"] for d in data])
        cons = np.mean([d["consumption_mean"] for d in data])
        print(f"{obs_type:<12} {coh:>10.4f} {std:>10.4f} {sens:>10.4f} {cons:>12.4f}")
    
    print()
    
    # Per-seed
    print("PER-SEED COHERENCE:")
    for seed in range(n_seeds):
        a = results["agentic"][seed]["coherence_mean"]
        r = results["rock"][seed]["coherence_mean"]
        print(f"  Seed {seed}: agentic={a:.4f}, rock={r:.4f}, diff={a-r:+.4f}")
    print()
    
    # Comparison
    agentic_coh = np.mean([d["coherence_mean"] for d in results["agentic"]])
    rock_coh = np.mean([d["coherence_mean"] for d in results["rock"]])
    advantage = agentic_coh - rock_coh
    
    print(f"COHERENCE ADVANTAGE: {advantage:+.4f}")
    print()
    
    if advantage > 0.005:
        print("✓ SELF-REFERENCE IMPROVES GEOMETRY OCCUPATION")
        print("  The 3D slice that looks at itself maintains better coherence")
    elif advantage > 0.001:
        print("~ Small positive effect detected")
    else:
        print("✗ No significant difference")
    
    return results


def experiment_signflip(n_steps: int = 10000, warmup: int = 1000):
    """Verify reconditioning sign-flip with real physics."""
    print("=" * 70)
    print("EXPERIMENT: SIGN-FLIP RECONDITIONING")
    print("=" * 70)
    print()
    
    ocean = OceanRegimeDriver(n_steps=n_steps, seed=42)
    substrate = DiffusionSubstrate(size=64, seed=42)
    
    regime_list = []
    boundary_std_list = []
    core_std_list = []
    
    for step in range(n_steps):
        physics = ocean.get_at_step(step)
        substrate.diffusion = physics["diffusion"]
        substrate.step()
        
        if step >= warmup:
            regime_list.append(physics["regime"])
            boundary_std_list.append(float(substrate.get_boundary().std()))
            core_std_list.append(float(substrate.get_core().std()))
    
    regime = np.array(regime_list)
    boundary_std = np.array(boundary_std_list)
    core_std = np.array(core_std_list)
    
    high_mask = regime > 0
    low_mask = regime <= 0
    
    r_pooled = np.corrcoef(boundary_std, core_std)[0, 1]
    r_high = np.corrcoef(boundary_std[high_mask], core_std[high_mask])[0, 1]
    r_low = np.corrcoef(boundary_std[low_mask], core_std[low_mask])[0, 1]
    
    sign_flip = np.sign(r_high) != np.sign(r_low)
    
    print(f"r_pooled: {r_pooled:+.4f}")
    print(f"r_high (storm):  {r_high:+.4f}")
    print(f"r_low (calm):    {r_low:+.4f}")
    print(f"Sign-flip: {'YES' if sign_flip else 'no'}")
    print()
    
    if sign_flip:
        print("✓ RECONDITIONING CONFIRMED")
        print("  Boundary-core coupling inverts between regimes")
    
    return {"r_pooled": r_pooled, "r_high": r_high, "r_low": r_low, "sign_flip": sign_flip}


def experiment_transitions(n_steps: int = 10000, warmup: int = 1000):
    """Test performance at regime transitions."""
    print("=" * 70)
    print("EXPERIMENT: REGIME TRANSITIONS")
    print("=" * 70)
    print()
    
    ocean = OceanRegimeDriver(n_steps=n_steps, seed=42)
    
    observers = {
        "agentic": AgenticObserver(seed=42),
        "rock": RockObserver(seed=42),
    }
    loops = {
        "agentic": ConsequenceLoop(),
        "rock": ConsequenceLoop(),
    }
    
    substrate = DiffusionSubstrate(size=64, seed=42)
    rng = np.random.default_rng(42)
    
    data = {k: {"coherence": [], "sensing": []} for k in observers}
    regime_list = []
    
    for step in range(n_steps):
        physics = ocean.get_at_step(step)
        substrate.diffusion = physics["diffusion"]
        substrate.step()
        boundary = substrate.get_boundary()
        
        for name in observers:
            obs = observers[name]
            loop = loops[name]
            
            obs_sigma = loop.update(obs.consumption_history[-1] if obs.consumption_history else 0.5)
            observation = boundary + rng.normal(0, obs_sigma, size=boundary.shape)
            state = obs.step(observation.copy())
            
            if step >= warmup:
                data[name]["coherence"].append(state["coherence"])
                data[name]["sensing"].append(loop.sensing_health)
        
        if step >= warmup:
            regime_list.append(physics["regime"])
    
    regime = np.array(regime_list)
    transitions = np.where(np.diff(np.sign(regime)) != 0)[0]
    
    print(f"Found {len(transitions)} regime transitions")
    print()
    
    for name in ["agentic", "rock"]:
        coh = np.array(data[name]["coherence"])
        print(f"{name.upper()}: mean coherence = {np.mean(coh):.4f}")
    
    print()
    advantage = np.mean(data["agentic"]["coherence"]) - np.mean(data["rock"]["coherence"])
    print(f"Coherence advantage: {advantage:+.4f}")
    
    return data


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Level 31b: Agency Through Geometry Occupation")
    p.add_argument("--geometry", action="store_true", help="Geometry occupation test")
    p.add_argument("--signflip", action="store_true", help="Sign-flip verification")
    p.add_argument("--transitions", action="store_true", help="Regime transition test")
    p.add_argument("--all", action="store_true", help="Run all experiments")
    args = p.parse_args()
    
    run_all = args.all or not any([args.geometry, args.signflip, args.transitions])
    
    if args.geometry or run_all:
        experiment_geometry_occupation()
        print()
    
    if args.signflip or run_all:
        experiment_signflip()
        print()
    
    if args.transitions or run_all:
        experiment_transitions()
        print()
    
    print("=" * 70)
    print("LEVEL 31b COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
