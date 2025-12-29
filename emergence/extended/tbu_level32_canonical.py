#!/usr/bin/env python3
"""
================================================================================
LEVEL 32: OBSERVING EMERGENCE IN THE TBU STACK
================================================================================

APPROACH:
  Level 31b established the proper TBU architecture. Now we OBSERVE what
  the 3D slice actually does when embedded in this substrate.
  
  No hypothesis to prove. No gains to tune. Just observation.

QUESTIONS:
  1. TEMPORAL: What timescales does the agentic observer naturally operate on?
     - Does coherence cluster at certain temporal scales?
     - Does consumption correlate with regime at certain lags?
  
  2. GEOMETRIC: Where does coherence accumulate in the substrate?
     - Boundary vs core?
     - Basin A vs Basin B?
     - Near bottleneck vs far from it?
  
  3. RECONDITIONING: How does the observer respond to regime changes?
     - Before, during, after transitions?
     - Symmetric or asymmetric (storm→calm vs calm→storm)?

THE TBU STACK (from 31b):
  1. Real physics: NDBC ocean data
  2. Basin/bottleneck substrate with boundary/core topology
  3. Consequence loop (consumption → observation noise)
  4. Agentic observer (volatility → consumption restraint)
  5. Rock observer (fixed consumption, control)

OUTPUT:
  Time series and statistics for analysis. Let the data speak.

================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from urllib.request import urlopen
from datetime import datetime, timedelta, timezone
import json


# =============================================================================
# NDBC OCEAN DATA LOADER (Real Physics)
# =============================================================================

def load_ocean_buoy(station: str = "46025", hours: int = 168) -> Tuple[np.ndarray, Dict]:
    """Load real-time ocean buoy data from NDBC."""
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"
    
    try:
        with urlopen(url, timeout=30) as response:
            lines = response.read().decode('utf-8').strip().split('\n')
        
        if len(lines) < 3:
            raise ValueError("Insufficient data")
        
        data_rows = []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)
        
        for line in lines[2:]:
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
        
        print(f"  Loaded {len(data_rows)} ocean samples from NDBC {station}")
        return data, {"source": "NDBC", "station": station, "n_samples": len(data_rows)}
    
    except Exception as e:
        print(f"  NDBC failed: {e}, generating synthetic")
        return generate_synthetic_ocean(hours)


def generate_synthetic_ocean(hours: int = 168) -> Tuple[np.ndarray, Dict]:
    """Synthetic ocean with regime structure."""
    rng = np.random.default_rng(42)
    n = max(500, hours * 3)
    t = np.linspace(0, hours, n)
    
    # Multi-scale structure
    regime = np.sin(2*np.pi*t/36)  # 36-hour weather cycles
    regime += 0.3 * np.sin(2*np.pi*t/12.42)  # Tidal
    regime += 0.2 * np.sin(2*np.pi*t/168)  # Weekly
    regime += rng.normal(0, 0.15, n)
    
    data = np.zeros((n, 10))
    for i in range(10):
        phase = rng.uniform(0, 2*np.pi)
        scale = rng.uniform(0.3, 0.7)
        data[:, i] = scale * np.sin(2*np.pi*t/(6+i*3) + phase)
        data[:, i] += 0.3 * regime  # Couple to regime
        data[:, i] += 0.1 * rng.normal(0, 1, n)
        data[:, i] = (data[:, i] - data[:, i].mean()) / (data[:, i].std() + 1e-10)
    
    print(f"  Generated {n} synthetic ocean samples")
    return data, {"source": "synthetic", "n_samples": n}


# =============================================================================
# OCEAN REGIME DRIVER
# =============================================================================

@dataclass
class OceanRegimeDriver:
    """Maps ocean conditions to diffusion physics."""
    n_steps: int = 10000
    seed: int = 42
    diffusion_base: float = 0.12
    diffusion_range: float = 0.04
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        
        ocean_data, self.meta = load_ocean_buoy(hours=168)
        self.ocean_data = ocean_data
        
        # Use wave height as regime driver
        wave_height = ocean_data[:, 0]
        
        # Resample
        x_old = np.linspace(0, 1, len(wave_height))
        x_new = np.linspace(0, 1, self.n_steps)
        self.regime_series = np.interp(x_new, x_old, wave_height)
        
        # Normalize to [-1, 1]
        self.regime_series = np.clip(self.regime_series / 2, -1, 1)
        
        # Diffusion
        self.diffusion_series = self.diffusion_base + self.diffusion_range * self.regime_series
        self.diffusion_series = np.clip(self.diffusion_series, 0.06, 0.18)
        
        print(f"  Regime: [{self.regime_series.min():.3f}, {self.regime_series.max():.3f}]")
        print(f"  Diffusion: [{self.diffusion_series.min():.3f}, {self.diffusion_series.max():.3f}]")
    
    def get_at_step(self, step: int) -> Dict[str, float]:
        idx = min(step, self.n_steps - 1)
        return {
            "regime": float(self.regime_series[idx]),
            "diffusion": float(self.diffusion_series[idx]),
        }


# =============================================================================
# BASIN-BOTTLENECK SUBSTRATE
# =============================================================================

@dataclass
class BasinBottleneckSubstrate:
    """Two basins connected by bottleneck."""
    width: int = 96
    height: int = 48
    neck_width: int = 8
    forcing_width: int = 3
    diffusion: float = 0.12
    diffusion_neck: float = 0.04
    damping: float = 0.015
    forcing_strength: float = 0.06
    seed: int = 42
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.grid = self.rng.uniform(-0.02, 0.02, (self.height, self.width))
        self._create_geometry()
    
    def _create_geometry(self):
        neck_center = self.width // 2
        neck_left = neck_center - self.neck_width // 2
        neck_right = neck_center + self.neck_width // 2
        
        self.neck_mask = np.zeros((self.height, self.width), dtype=bool)
        self.neck_mask[:, neck_left:neck_right] = True
        
        self.basin_A_mask = np.zeros((self.height, self.width), dtype=bool)
        self.basin_A_mask[:, :neck_left] = True
        
        self.basin_B_mask = np.zeros((self.height, self.width), dtype=bool)
        self.basin_B_mask[:, neck_right:] = True
        
        self.forcing_A_mask = np.zeros((self.height, self.width), dtype=bool)
        self.forcing_A_mask[:, :self.forcing_width] = True
        
        self.forcing_B_mask = np.zeros((self.height, self.width), dtype=bool)
        self.forcing_B_mask[:, -self.forcing_width:] = True
        
        self.boundary_mask = self.forcing_A_mask | self.forcing_B_mask
        
        core_margin = 8
        basin_A_center = neck_left // 2
        basin_B_center = neck_right + (self.width - neck_right) // 2
        
        self.core_A_mask = np.zeros((self.height, self.width), dtype=bool)
        self.core_A_mask[core_margin:-core_margin, basin_A_center-4:basin_A_center+4] = True
        
        self.core_B_mask = np.zeros((self.height, self.width), dtype=bool)
        self.core_B_mask[core_margin:-core_margin, basin_B_center-4:basin_B_center+4] = True
        
        self.core_mask = self.core_A_mask | self.core_B_mask
        
        self.boundary_coords = np.where(self.boundary_mask)
        self.core_coords = np.where(self.core_mask)
        self.n_boundary = len(self.boundary_coords[0])
        self.n_core = len(self.core_coords[0])
    
    def step(self):
        laplacian = (
            np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
            np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) - 4 * self.grid
        )
        
        eff_diff = np.clip(self.diffusion, 0.02, 0.18)
        diff_coeff = np.where(self.neck_mask, self.diffusion_neck, eff_diff)
        
        self.grid += diff_coeff * laplacian
        
        noise = self.rng.normal(0, self.forcing_strength, self.n_boundary)
        self.grid[self.boundary_coords] += noise
        
        self.grid *= (1.0 - self.damping)
        self.grid = np.clip(self.grid, -5.0, 5.0)
    
    def get_boundary(self) -> np.ndarray:
        return self.grid[self.boundary_coords]
    
    def get_core(self) -> np.ndarray:
        return self.grid[self.core_coords]
    
    def get_core_A(self) -> np.ndarray:
        return self.grid[self.core_A_mask]
    
    def get_core_B(self) -> np.ndarray:
        return self.grid[self.core_B_mask]
    
    def get_neck(self) -> np.ndarray:
        return self.grid[self.neck_mask]


# =============================================================================
# CONSEQUENCE LOOP
# =============================================================================

@dataclass
class ConsequenceLoop:
    """Consumption → future observation noise."""
    depletion_rate: float = 0.015
    recovery_rate: float = 0.008
    noise_base: float = 0.01
    noise_scale: float = 0.3
    lag: int = 25
    
    def __post_init__(self):
        self.sensing_health = 1.0
        self.consumption_buffer: List[float] = []
    
    def update(self, consumption: float) -> float:
        self.consumption_buffer.append(consumption)
        
        if len(self.consumption_buffer) > self.lag:
            lagged = self.consumption_buffer.pop(0)
            depletion = self.depletion_rate * lagged
            recovery = self.recovery_rate * (1.0 - self.sensing_health)
            self.sensing_health = float(np.clip(
                self.sensing_health - depletion + recovery, 0.0, 1.0
            ))
        
        return self.noise_base + self.noise_scale * (1.0 - self.sensing_health)
    
    def reset(self):
        self.sensing_health = 1.0
        self.consumption_buffer = []


# =============================================================================
# OBSERVERS (from 31b)
# =============================================================================

@dataclass
class AgenticObserver:
    """Self-referential observer. Tracks volatility, restrains consumption."""
    n_cores: int = 16
    seed: int = 42
    VOL_K: float = 25.0
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.core_states = np.zeros(self.n_cores)
        self.core_thresholds = self.rng.uniform(0.3, 0.7, self.n_cores)
        
        self.input_history: List[np.ndarray] = []
        self.state_history: List[np.ndarray] = []
        self.consumption_history: List[float] = []
        self.coherence_history: List[float] = []
        self.volatility_history: List[float] = []
    
    def step(self, observation: np.ndarray) -> Dict:
        self.input_history.append(observation.copy())
        if len(self.input_history) > 60:
            self.input_history.pop(0)
        
        volatility = self._compute_volatility()
        
        capacity = 1.0 / (1.0 + self.VOL_K * volatility)
        consumption = float(np.clip(capacity, 0.1, 0.9))
        
        self._update_cores(observation)
        coherence = self._compute_coherence()
        
        self.consumption_history.append(consumption)
        self.coherence_history.append(coherence)
        self.volatility_history.append(volatility)
        
        return {"consumption": consumption, "coherence": coherence, "volatility": volatility}
    
    def _compute_volatility(self) -> float:
        if len(self.input_history) < 2:
            return 0.0
        recent = self.input_history[-30:]
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


@dataclass
class RockObserver:
    """Fixed consumption. Control."""
    n_cores: int = 16
    seed: int = 42
    fixed_consumption: float = 0.4
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.core_states = np.zeros(self.n_cores)
        self.core_thresholds = self.rng.uniform(0.3, 0.7, self.n_cores)
        
        self.state_history: List[np.ndarray] = []
        self.consumption_history: List[float] = []
        self.coherence_history: List[float] = []
    
    def step(self, observation: np.ndarray) -> Dict:
        consumption = self.fixed_consumption
        
        n_inputs = len(observation)
        for i in range(self.n_cores):
            idx = i % n_inputs
            drive = float(observation[idx])
            total = np.clip(drive - self.core_thresholds[i], -10, 10)
            activation = 1.0 / (1.0 + np.exp(-4.0 * total))
            self.core_states[i] = 0.9 * self.core_states[i] + 0.1 * activation
        
        self.state_history.append(self.core_states.copy())
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        coherence = self._compute_coherence()
        
        self.consumption_history.append(consumption)
        self.coherence_history.append(coherence)
        
        return {"consumption": consumption, "coherence": coherence, "volatility": 0.0}
    
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
# OBSERVATION EXPERIMENT
# =============================================================================

def observe_emergence(n_steps: int = 8000, warmup: int = 500):
    """
    Run the TBU stack and OBSERVE what emerges.
    No hypothesis. Just data.
    """
    print("=" * 70)
    print("LEVEL 32: OBSERVING EMERGENCE")
    print("=" * 70)
    print()
    print("Running proper TBU stack. Observing what the 3D slice does.")
    print()
    
    # Initialize
    ocean = OceanRegimeDriver(n_steps=n_steps, seed=42)
    
    substrate_ag = BasinBottleneckSubstrate(seed=42)
    substrate_rk = BasinBottleneckSubstrate(seed=42)
    
    agentic = AgenticObserver(seed=42)
    rock = RockObserver(seed=42)
    
    loop_ag = ConsequenceLoop()
    loop_rk = ConsequenceLoop()
    
    rng = np.random.default_rng(42)
    
    # Data collection
    data = {
        "regime": [],
        "diffusion": [],
        # Agentic
        "ag_coherence": [],
        "ag_consumption": [],
        "ag_volatility": [],
        "ag_sensing": [],
        "ag_boundary_std": [],
        "ag_core_std": [],
        "ag_core_A_mean": [],
        "ag_core_B_mean": [],
        "ag_neck_std": [],
        # Rock
        "rk_coherence": [],
        "rk_consumption": [],
        "rk_sensing": [],
        "rk_boundary_std": [],
        "rk_core_std": [],
    }
    
    print("Running simulation...")
    for step in range(n_steps):
        if step % 2000 == 0:
            print(f"  Step {step}/{n_steps}")
        
        # Physics
        physics = ocean.get_at_step(step)
        
        substrate_ag.diffusion = physics["diffusion"]
        substrate_rk.diffusion = physics["diffusion"]
        
        substrate_ag.step()
        substrate_rk.step()
        
        # Observations with consequence noise
        last_ag = agentic.consumption_history[-1] if agentic.consumption_history else 0.5
        last_rk = rock.consumption_history[-1] if rock.consumption_history else 0.4
        
        noise_ag = loop_ag.update(last_ag)
        noise_rk = loop_rk.update(last_rk)
        
        obs_ag = substrate_ag.get_boundary() + rng.normal(0, noise_ag, substrate_ag.n_boundary)
        obs_rk = substrate_rk.get_boundary() + rng.normal(0, noise_rk, substrate_rk.n_boundary)
        
        # Observer steps
        state_ag = agentic.step(obs_ag)
        state_rk = rock.step(obs_rk)
        
        # Collect data after warmup
        if step >= warmup:
            data["regime"].append(physics["regime"])
            data["diffusion"].append(physics["diffusion"])
            
            data["ag_coherence"].append(state_ag["coherence"])
            data["ag_consumption"].append(state_ag["consumption"])
            data["ag_volatility"].append(state_ag["volatility"])
            data["ag_sensing"].append(loop_ag.sensing_health)
            data["ag_boundary_std"].append(float(substrate_ag.get_boundary().std()))
            data["ag_core_std"].append(float(substrate_ag.get_core().std()))
            data["ag_core_A_mean"].append(float(substrate_ag.get_core_A().mean()))
            data["ag_core_B_mean"].append(float(substrate_ag.get_core_B().mean()))
            data["ag_neck_std"].append(float(substrate_ag.get_neck().std()))
            
            data["rk_coherence"].append(state_rk["coherence"])
            data["rk_consumption"].append(state_rk["consumption"])
            data["rk_sensing"].append(loop_rk.sensing_health)
            data["rk_boundary_std"].append(float(substrate_rk.get_boundary().std()))
            data["rk_core_std"].append(float(substrate_rk.get_core().std()))
    
    # Convert to arrays
    for k in data:
        data[k] = np.array(data[k])
    
    print()
    print("=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)
    
    # 1. Basic comparison
    print()
    print("1. COHERENCE ADVANTAGE")
    print("-" * 40)
    ag_coh = data["ag_coherence"].mean()
    rk_coh = data["rk_coherence"].mean()
    advantage = ag_coh - rk_coh
    print(f"   Agentic coherence:  {ag_coh:.4f} ± {data['ag_coherence'].std():.4f}")
    print(f"   Rock coherence:     {rk_coh:.4f} ± {data['rk_coherence'].std():.4f}")
    print(f"   Advantage:          {advantage:+.4f}")
    
    # 2. Consumption patterns
    print()
    print("2. CONSUMPTION PATTERNS")
    print("-" * 40)
    print(f"   Agentic: mean={data['ag_consumption'].mean():.3f}, "
          f"std={data['ag_consumption'].std():.3f}, "
          f"range=[{data['ag_consumption'].min():.3f}, {data['ag_consumption'].max():.3f}]")
    print(f"   Rock:    fixed={data['rk_consumption'].mean():.3f}")
    
    # 3. Sensing health
    print()
    print("3. SENSING HEALTH")
    print("-" * 40)
    print(f"   Agentic: mean={data['ag_sensing'].mean():.3f}, min={data['ag_sensing'].min():.3f}")
    print(f"   Rock:    mean={data['rk_sensing'].mean():.3f}, min={data['rk_sensing'].min():.3f}")
    
    # 4. Temporal correlations
    print()
    print("4. TEMPORAL STRUCTURE")
    print("-" * 40)
    
    # Regime vs volatility
    r_regime_vol = np.corrcoef(data["regime"], data["ag_volatility"])[0, 1]
    print(f"   Regime ↔ Volatility:     r = {r_regime_vol:+.3f}")
    
    # Regime vs coherence
    r_regime_coh = np.corrcoef(data["regime"], data["ag_coherence"])[0, 1]
    print(f"   Regime ↔ Ag Coherence:   r = {r_regime_coh:+.3f}")
    
    # Volatility vs coherence
    r_vol_coh = np.corrcoef(data["ag_volatility"], data["ag_coherence"])[0, 1]
    print(f"   Volatility ↔ Coherence:  r = {r_vol_coh:+.3f}")
    
    # Consumption vs coherence (lagged)
    for lag in [0, 10, 25, 50]:
        if lag > 0:
            r = np.corrcoef(data["ag_consumption"][:-lag], data["ag_coherence"][lag:])[0, 1]
        else:
            r = np.corrcoef(data["ag_consumption"], data["ag_coherence"])[0, 1]
        print(f"   Consumption → Coherence (lag {lag:2d}): r = {r:+.3f}")
    
    # 5. Geometric structure
    print()
    print("5. GEOMETRIC STRUCTURE")
    print("-" * 40)
    
    # Boundary vs Core
    r_bc = np.corrcoef(data["ag_boundary_std"], data["ag_core_std"])[0, 1]
    print(f"   Boundary ↔ Core std:     r = {r_bc:+.3f}")
    
    # Basin A vs Basin B
    r_ab = np.corrcoef(data["ag_core_A_mean"], data["ag_core_B_mean"])[0, 1]
    print(f"   Core A ↔ Core B:         r = {r_ab:+.3f}")
    
    # Neck activity
    print(f"   Neck std: mean={data['ag_neck_std'].mean():.4f}")
    
    # 6. Regime-dependent behavior
    print()
    print("6. REGIME-DEPENDENT BEHAVIOR")
    print("-" * 40)
    
    high_regime = data["regime"] > np.percentile(data["regime"], 75)
    low_regime = data["regime"] < np.percentile(data["regime"], 25)
    
    print(f"   High regime (storm):")
    print(f"      Coherence: {data['ag_coherence'][high_regime].mean():.4f}")
    print(f"      Consumption: {data['ag_consumption'][high_regime].mean():.3f}")
    print(f"      Volatility: {data['ag_volatility'][high_regime].mean():.4f}")
    
    print(f"   Low regime (calm):")
    print(f"      Coherence: {data['ag_coherence'][low_regime].mean():.4f}")
    print(f"      Consumption: {data['ag_consumption'][low_regime].mean():.3f}")
    print(f"      Volatility: {data['ag_volatility'][low_regime].mean():.4f}")
    
    # 7. Autocorrelation (temporal scales)
    print()
    print("7. TEMPORAL SCALES (autocorrelation)")
    print("-" * 40)
    
    for signal_name, signal in [("Coherence", data["ag_coherence"]), 
                                 ("Volatility", data["ag_volatility"]),
                                 ("Consumption", data["ag_consumption"])]:
        print(f"   {signal_name}:")
        for lag in [1, 5, 10, 25, 50, 100]:
            if lag < len(signal):
                ac = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
                print(f"      lag {lag:3d}: {ac:.3f}")
    
    # 8. CONDITIONED RECONDITIONING (referee check #1)
    print()
    print("8. CONDITIONED RECONDITIONING")
    print("-" * 40)
    
    # Split by diffusion regime (using actual diffusion values)
    high_diff = data["diffusion"] > np.percentile(data["diffusion"], 75)
    low_diff = data["diffusion"] < np.percentile(data["diffusion"], 25)
    
    print(f"   High diffusion samples: {high_diff.sum()}")
    print(f"   Low diffusion samples:  {low_diff.sum()}")
    
    if high_diff.sum() > 50 and low_diff.sum() > 50:
        r_bc_high = np.corrcoef(
            data["ag_boundary_std"][high_diff], 
            data["ag_core_std"][high_diff]
        )[0, 1]
        r_bc_low = np.corrcoef(
            data["ag_boundary_std"][low_diff], 
            data["ag_core_std"][low_diff]
        )[0, 1]
        
        print(f"   Boundary↔Core (high diffusion): r = {r_bc_high:+.3f}")
        print(f"   Boundary↔Core (low diffusion):  r = {r_bc_low:+.3f}")
        print(f"   Difference:                     Δr = {r_bc_high - r_bc_low:+.3f}")
        
        if abs(r_bc_high - r_bc_low) > 0.1:
            print("   ✓ Reconditioning is regime-dependent")
        else:
            print("   ~ Reconditioning present but regime-invariant")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY: WHAT EMERGED")
    print("=" * 70)
    print()
    
    if advantage > 0.01:
        print(f"✓ Coherence advantage: +{advantage:.4f}")
    elif advantage > 0:
        print(f"~ Small coherence advantage: +{advantage:.4f}")
    else:
        print(f"✗ No coherence advantage: {advantage:.4f}")
    
    if abs(r_vol_coh) > 0.1:
        if r_vol_coh < 0:
            print(f"✓ Volatility anti-correlates with coherence: {r_vol_coh:.3f}")
            print("  → Observer detects instability and adapts")
        else:
            print(f"? Volatility correlates with coherence: {r_vol_coh:.3f}")
    
    if data["ag_coherence"][high_regime].mean() != data["ag_coherence"][low_regime].mean():
        diff = data["ag_coherence"][high_regime].mean() - data["ag_coherence"][low_regime].mean()
        print(f"  Regime affects coherence: storm - calm = {diff:+.4f}")
    
    return data


def soft_control_check(n_steps: int = 8000, warmup: int = 500):
    """
    REFEREE CHECK #2: Soft Control
    
    Run rock at 0.30 consumption (closer to agentic's ~0.21) to prove
    the advantage isn't just "consume less = win".
    
    If agentic still outperforms, it's the TIMING of restraint that matters,
    not just the average consumption level.
    """
    print()
    print("=" * 70)
    print("REFEREE CHECK: SOFT CONTROL")
    print("=" * 70)
    print()
    print("Testing: Is the advantage just 'consume less = win'?")
    print("Running rock at 0.30 consumption (vs agentic's adaptive ~0.21)")
    print()
    
    ocean = OceanRegimeDriver(n_steps=n_steps, seed=42)
    
    substrate_ag = BasinBottleneckSubstrate(seed=42)
    substrate_rk = BasinBottleneckSubstrate(seed=42)
    
    agentic = AgenticObserver(seed=42)
    rock_soft = RockObserver(seed=42, fixed_consumption=0.30)  # Softer rock
    
    loop_ag = ConsequenceLoop()
    loop_rk = ConsequenceLoop()
    
    rng = np.random.default_rng(42)
    
    ag_coherence = []
    rk_coherence = []
    ag_consumption = []
    ag_sensing = []
    rk_sensing = []
    
    print("Running simulation...")
    for step in range(n_steps):
        if step % 2000 == 0:
            print(f"  Step {step}/{n_steps}")
        
        physics = ocean.get_at_step(step)
        
        substrate_ag.diffusion = physics["diffusion"]
        substrate_rk.diffusion = physics["diffusion"]
        
        substrate_ag.step()
        substrate_rk.step()
        
        last_ag = agentic.consumption_history[-1] if agentic.consumption_history else 0.5
        last_rk = rock_soft.consumption_history[-1] if rock_soft.consumption_history else 0.3
        
        noise_ag = loop_ag.update(last_ag)
        noise_rk = loop_rk.update(last_rk)
        
        obs_ag = substrate_ag.get_boundary() + rng.normal(0, noise_ag, substrate_ag.n_boundary)
        obs_rk = substrate_rk.get_boundary() + rng.normal(0, noise_rk, substrate_rk.n_boundary)
        
        state_ag = agentic.step(obs_ag)
        state_rk = rock_soft.step(obs_rk)
        
        if step >= warmup:
            ag_coherence.append(state_ag["coherence"])
            rk_coherence.append(state_rk["coherence"])
            ag_consumption.append(state_ag["consumption"])
            ag_sensing.append(loop_ag.sensing_health)
            rk_sensing.append(loop_rk.sensing_health)
    
    ag_coherence = np.array(ag_coherence)
    rk_coherence = np.array(rk_coherence)
    ag_consumption = np.array(ag_consumption)
    ag_sensing = np.array(ag_sensing)
    rk_sensing = np.array(rk_sensing)
    
    print()
    print("RESULTS: SOFT CONTROL")
    print("-" * 50)
    print()
    print(f"{'Metric':<20} {'Agentic':>12} {'Rock (0.30)':>12}")
    print("-" * 50)
    print(f"{'Coherence':<20} {ag_coherence.mean():>12.4f} {rk_coherence.mean():>12.4f}")
    print(f"{'Coherence std':<20} {ag_coherence.std():>12.4f} {rk_coherence.std():>12.4f}")
    print(f"{'Consumption':<20} {ag_consumption.mean():>12.3f} {0.30:>12.3f}")
    print(f"{'Sensing health':<20} {ag_sensing.mean():>12.3f} {rk_sensing.mean():>12.3f}")
    
    advantage = ag_coherence.mean() - rk_coherence.mean()
    consumption_diff = ag_consumption.mean() - 0.30
    
    print()
    print(f"Coherence advantage: {advantage:+.4f}")
    print(f"Consumption difference: {consumption_diff:+.3f}")
    print()
    
    if advantage > 0.005:
        if consumption_diff < 0:
            print("✓ AGENTIC WINS despite consuming LESS than soft rock")
            print("  → The advantage is TIMING, not just average consumption")
        else:
            print("✓ AGENTIC WINS even with similar consumption levels")
            print("  → Adaptive restraint outperforms fixed restraint")
    elif advantage > 0:
        print("~ Small advantage remains")
        print("  → Effect is partially explained by consumption level")
    else:
        print("✗ No advantage over soft rock")
        print("  → Effect may be purely consumption-level dependent")
    
    return {
        "ag_coherence": ag_coherence.mean(),
        "rk_coherence": rk_coherence.mean(),
        "advantage": advantage,
        "ag_consumption": ag_consumption.mean(),
        "ag_sensing": ag_sensing.mean(),
        "rk_sensing": rk_sensing.mean(),
    }


def ablation_no_reconditioning(n_steps: int = 8000, warmup: int = 500):
    """
    ABLATION: Remove reconditioning by fixing diffusion constant.
    
    If reconditioning is necessary for the advantage, fixing diffusion
    should eliminate or reduce the coherence advantage.
    
    What we're testing:
    - With reconditioning: diffusion varies with regime → boundary-core coupling changes
    - Without reconditioning: diffusion fixed → boundary-core coupling constant
    
    If advantage disappears → 3D slice was surfing the reconditioning
    If advantage persists → advantage is just sensing preservation (no reconditioning needed)
    """
    print()
    print("=" * 70)
    print("ABLATION: REMOVE RECONDITIONING")
    print("=" * 70)
    print()
    print("Testing: Is reconditioning necessary for the advantage?")
    print("Method: Fix diffusion constant (no regime modulation)")
    print()
    
    ocean = OceanRegimeDriver(n_steps=n_steps, seed=42)
    
    # Two conditions: WITH and WITHOUT reconditioning
    results = {}
    
    for condition in ["WITH reconditioning", "WITHOUT reconditioning"]:
        print(f"\n--- {condition} ---")
        
        substrate_ag = BasinBottleneckSubstrate(seed=42)
        substrate_rk = BasinBottleneckSubstrate(seed=42)
        
        agentic = AgenticObserver(seed=42)
        rock = RockObserver(seed=42, fixed_consumption=0.40)
        
        loop_ag = ConsequenceLoop()
        loop_rk = ConsequenceLoop()
        
        rng = np.random.default_rng(42)
        
        ag_coherence = []
        rk_coherence = []
        ag_volatility = []
        boundary_std = []
        core_std = []
        diffusion_used = []
        
        # Fixed diffusion for ablation (mean of normal range)
        fixed_diffusion = 0.12
        
        for step in range(n_steps):
            physics = ocean.get_at_step(step)
            
            if condition == "WITH reconditioning":
                # Normal: diffusion varies with regime
                substrate_ag.diffusion = physics["diffusion"]
                substrate_rk.diffusion = physics["diffusion"]
                diffusion_used.append(physics["diffusion"])
            else:
                # Ablation: diffusion fixed
                substrate_ag.diffusion = fixed_diffusion
                substrate_rk.diffusion = fixed_diffusion
                diffusion_used.append(fixed_diffusion)
            
            substrate_ag.step()
            substrate_rk.step()
            
            last_ag = agentic.consumption_history[-1] if agentic.consumption_history else 0.5
            last_rk = rock.consumption_history[-1] if rock.consumption_history else 0.4
            
            noise_ag = loop_ag.update(last_ag)
            noise_rk = loop_rk.update(last_rk)
            
            obs_ag = substrate_ag.get_boundary() + rng.normal(0, noise_ag, substrate_ag.n_boundary)
            obs_rk = substrate_rk.get_boundary() + rng.normal(0, noise_rk, substrate_rk.n_boundary)
            
            state_ag = agentic.step(obs_ag)
            state_rk = rock.step(obs_rk)
            
            if step >= warmup:
                ag_coherence.append(state_ag["coherence"])
                rk_coherence.append(state_rk["coherence"])
                ag_volatility.append(state_ag["volatility"])
                boundary_std.append(float(substrate_ag.get_boundary().std()))
                core_std.append(float(substrate_ag.get_core().std()))
        
        ag_coherence = np.array(ag_coherence)
        rk_coherence = np.array(rk_coherence)
        boundary_std = np.array(boundary_std)
        core_std = np.array(core_std)
        diffusion_used = np.array(diffusion_used[warmup:])
        
        # Check if reconditioning is present
        r_bc = np.corrcoef(boundary_std, core_std)[0, 1]
        
        # Check regime-dependent coupling
        high_diff = diffusion_used > np.percentile(diffusion_used, 75)
        low_diff = diffusion_used < np.percentile(diffusion_used, 25)
        
        if high_diff.sum() > 50 and low_diff.sum() > 50 and condition == "WITH reconditioning":
            r_bc_high = np.corrcoef(boundary_std[high_diff], core_std[high_diff])[0, 1]
            r_bc_low = np.corrcoef(boundary_std[low_diff], core_std[low_diff])[0, 1]
            delta_r = r_bc_high - r_bc_low
        else:
            r_bc_high = r_bc_low = r_bc
            delta_r = 0.0
        
        advantage = ag_coherence.mean() - rk_coherence.mean()
        
        results[condition] = {
            "ag_coherence": ag_coherence.mean(),
            "rk_coherence": rk_coherence.mean(),
            "advantage": advantage,
            "r_bc": r_bc,
            "r_bc_high": r_bc_high,
            "r_bc_low": r_bc_low,
            "delta_r": delta_r,
            "volatility": np.mean(ag_volatility),
        }
        
        print(f"   Coherence advantage: {advantage:+.4f}")
        print(f"   Boundary↔Core: r = {r_bc:+.3f}")
        if condition == "WITH reconditioning":
            print(f"   Reconditioning Δr: {delta_r:+.3f}")
    
    # Compare
    print()
    print("=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print()
    
    with_adv = results["WITH reconditioning"]["advantage"]
    without_adv = results["WITHOUT reconditioning"]["advantage"]
    
    print(f"{'Condition':<25} {'Advantage':>12} {'B↔C corr':>12} {'Δr':>10}")
    print("-" * 62)
    print(f"{'WITH reconditioning':<25} {with_adv:>+12.4f} {results['WITH reconditioning']['r_bc']:>+12.3f} {results['WITH reconditioning']['delta_r']:>+10.3f}")
    print(f"{'WITHOUT reconditioning':<25} {without_adv:>+12.4f} {results['WITHOUT reconditioning']['r_bc']:>+12.3f} {results['WITHOUT reconditioning']['delta_r']:>+10.3f}")
    
    print()
    advantage_reduction = with_adv - without_adv
    pct_reduction = (1 - without_adv/with_adv)*100 if with_adv > 0 else 0
    print(f"Advantage reduction: {advantage_reduction:+.4f} ({pct_reduction:.0f}%)")
    
    print()
    if without_adv < 0.005:
        print("✓ ABLATION CONFIRMS: Reconditioning is necessary")
        print("  Without reconditioning, advantage disappears")
        print("  → 3D slice was surfing the reconditioning structure")
    elif without_adv < with_adv * 0.5:
        print("✓ ABLATION SUPPORTS: Reconditioning contributes significantly")
        print(f"  Advantage reduced by {pct_reduction:.0f}%")
        print("  → 3D slice benefits from reconditioning structure")
    elif without_adv < with_adv:
        print("~ PARTIAL: Reconditioning contributes but not essential")
        print(f"  Advantage reduced by {pct_reduction:.0f}%")
    else:
        print("✗ ABLATION FAILS: Reconditioning not necessary")
        print("  Advantage persists without reconditioning")
        print("  → Advantage is just sensing preservation")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    p = argparse.ArgumentParser(description="Level 32: Observe Emergence")
    p.add_argument("--steps", type=int, default=8000, help="Simulation steps")
    p.add_argument("--save", type=str, default=None, help="Save data to JSON")
    p.add_argument("--soft-control", action="store_true", help="Run soft control check only")
    p.add_argument("--ablation", action="store_true", help="Run ablation test only")
    p.add_argument("--all", action="store_true", help="Run all checks")
    args = p.parse_args()
    
    if args.soft_control:
        soft_control_check(n_steps=args.steps)
    elif args.ablation:
        ablation_no_reconditioning(n_steps=args.steps)
    else:
        data = observe_emergence(n_steps=args.steps)
        
        if args.save:
            save_data = {k: v.tolist() for k, v in data.items()}
            with open(args.save, 'w') as f:
                json.dump(save_data, f)
            print(f"\nData saved to {args.save}")
        
        if args.all:
            soft_control_check(n_steps=args.steps)
            ablation_no_reconditioning(n_steps=args.steps)
    
    print()
    print("=" * 70)
    print("LEVEL 32 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
