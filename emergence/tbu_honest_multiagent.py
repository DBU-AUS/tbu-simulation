#!/usr/bin/env python3
"""
================================================================================
TBU HONEST MULTIAGENT - Level 9: Multiple Patterns in Shared Geometry
================================================================================

Builds on tbu_honest_boundary.py (Level 8) by composing multiple substrates
that share the same Environment.

THIS IS SIMPLER THAN IT SOUNDS:
  - Each "pattern" is a full Level 8 substrate (grid, attention, actions)
  - All patterns share ONE Environment (same E for consumption/disturbance/sensing)
  - NO explicit cooperation rule - just shared constraints

WHAT IS SHARED vs INDEPENDENT:
  - SHARED: Environment (E), health channels reflect same E for all patterns
  - INDEPENDENT: Each pattern's grid, attention, actions, base channels 0-3
  - OPTION: sync_solver_index=True aligns deterministic phase of base channels;
    stochastic components (ch1 entropy, RNG noise) remain pattern-specific

THE KEY TESTS:
  1. COLLECTIVE STABILITY: Do all patterns find stable configuration?
  2. SUSTAINABILITY: Is the shared geometry maintained?
  3. DISSOLUTION: If resources can't support all patterns, do some gracefully
     fade while others stabilize? (vs tragedy-of-commons collapse)

IMPORTANT CAVEATS ON DISSOLUTION:
  - Whether "graceful dissolution" occurs is EMPIRICAL, not guaranteed
  - Patterns don't have a direct "reduce existence" knob
  - The only soft modulation is via attention/gain affecting dy/action magnitude
  - Outcome depends on whether dynamics permit low-impact configurations
  - `resources_for` is a SCENARIO LABEL, not computed carrying capacity

ABLATION NOTE (same as Level 8):
  - "Remove sensing" ≠ "remove coupling"
  - Even with health channels ablated, patterns still consume/disturb E
  - Ablation tests information access, not physical coupling

DESIGNED vs EMERGENT (FoP-safe framing):
  - DESIGNED: Composition (multiple substrates + shared E), E cadence, --sync option
  - NOT DESIGNED: Inter-pattern cooperation, preference for collective outcomes
  - CLASSIFICATION: is_alive() is POST-HOC diagnostic only, does NOT feed back
  - ROBUSTNESS: Report both LCC-based and M-based alive counts to show
    outcomes are not artefacts of bespoke detector

TBU FRAMING:
  - Multiple constraint-satisfaction processes in shared block
  - E updates at WORLD cadence, not per-pattern cadence
  - "Born enlightened" = each pattern maximizes N[ω] naturally (no ego)

================================================================================
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

# Import Level 8 components
from tbu_honest_boundary import (
    Environment,
    MultiChannelProbeWithHealth,
    HonestBoundarySubstrate,
    largest_connected_component_fraction,
)


# -----------------------------------------------------------------------------
# Shared Geometry (extends Environment with carrying capacity concept)
# -----------------------------------------------------------------------------

@dataclass
class SharedGeometry(Environment):
    """
    Environment extended for multi-pattern scenarios.
    
    Same as Environment but with:
    - Configurable carrying capacity
    - Tracking of total consumption across all patterns
    """
    
    # Conceptual: how many patterns can this geometry sustainably support?
    carrying_capacity: int = 3
    
    # Tracking
    total_consumption_history: List[float] = field(default_factory=list)
    
    def record_total_consumption(self, amount: float):
        """Record total consumption across all patterns."""
        self.total_consumption_history.append(amount)
        if len(self.total_consumption_history) > 200:
            self.total_consumption_history = self.total_consumption_history[-100:]


# -----------------------------------------------------------------------------
# Pattern: A single substrate in the shared geometry
# -----------------------------------------------------------------------------

class Pattern(HonestBoundarySubstrate):
    """
    A single pattern in the shared geometry.
    
    This is just a Level 8 substrate that:
    - Has a unique ID
    - Uses a SHARED Environment (set externally)
    - Otherwise behaves exactly like Level 8
    
    NOT an "agent" - a pattern. It maximizes N[ω]. It has no ego.
    """
    
    def __init__(self, pattern_id: str, shared_env: SharedGeometry, 
                 size: int = 48, seed: Optional[int] = None, quiet: bool = True, **kwargs):
        # Don't create internal environment - use shared one
        self.pattern_id = pattern_id
        self.shared_env = shared_env
        self._quiet = quiet
        
        # Temporarily redirect prints from parent
        if quiet:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        try:
            # Initialize base substrate but override environment
            super().__init__(size=size, seed=seed, **kwargs)
        finally:
            if quiet:
                sys.stdout = old_stdout
        
        # Replace internal environment with shared one
        self.environment = shared_env
        
        # Update probe to use shared environment
        self.probe = MultiChannelProbeWithHealth(
            n_base_channels=self.n_base_channels,
            environment=self.shared_env,
            seed=seed,
            health_enabled=self.health_channels_enabled,
            env_noise_scale=self.env_noise_scale,
            env_noise_floor=self.env_noise_floor,
            env_noise_enabled=self.env_noise_enabled
        )
        
        # IMPORTANT: Reset prev_channels to match new probe (avoids mismatch on first attention update)
        self.prev_channels = self.probe.read_channels(self.n_forcing)
        
        # Pattern-specific tracking
        self.consumption_history: List[float] = []
        self.core_stability_history: List[float] = []
    
    def get_core_stability(self) -> float:
        """Return pattern's core stability (for alive/dead determination)."""
        if len(self.history) < 20:
            return 1.0  # Assume alive during warmup
        
        M = self.measure_M()
        # Core = high-M internal regions
        internal_mask = ~self.forcing_mask
        internal_M = M[internal_mask]
        
        if len(internal_M) == 0:
            return 0.0
        
        core_threshold = np.median(internal_M)
        core_M = internal_M[internal_M >= core_threshold]
        
        if len(core_M) == 0:
            return 0.0
        
        return float(np.mean(core_M))
    
    def is_alive(self) -> bool:
        """
        Is this pattern still coherent?
        
        Uses LCC-based coherence (not just M > threshold) to avoid
        false positives where noise gives pockets of high M.
        
        Also checks:
        - grid_std relative to boundary std (not frozen/dead)
        - stable_fraction in reasonable range (not chaos or trivially frozen)
        """
        if len(self.history) < 20:
            return True  # Assume alive during warmup
        
        # Check 1: grid has non-trivial activity relative to boundary
        # (scales with parameters instead of hard-coded threshold)
        grid_std = float(np.std(self.grid))
        boundary_std = float(np.std(self.grid[self.forcing_coords]))
        if boundary_std > 1e-6 and grid_std < 0.2 * boundary_std:
            # Internal activity much lower than boundary - pattern is frozen/dead
            return False
        
        # Absolute floor as fallback
        if grid_std < 1e-4:
            return False
        
        # Check 2: stable mask and compute LCC on internal regions
        stable = self.stable_mask()
        internal_stable = stable & (~self.forcing_mask)
        
        # Check 3: stable region is non-trivial (not chaos or trivially uniform)
        stable_fraction = internal_stable.mean()
        if stable_fraction < 0.1:
            # Almost nothing stable = chaotic, not coherent
            return False
        if stable_fraction > 0.99:
            # Everything stable = likely frozen uniform field
            return False
        
        lcc, _ = largest_connected_component_fraction(internal_stable)
        
        # Pattern is alive if it has coherent core (LCC > 60%)
        return lcc > 0.6
    
    def step_physics(self) -> None:
        """
        Same as Level 8, but environment is shared.
        
        Key difference: DON'T call self.environment.step() here - 
        that's done once per world step, not per pattern step.
        """
        # --- Diffusion on torus
        up = np.roll(self.grid, 1, axis=0)
        dn = np.roll(self.grid, -1, axis=0)
        lf = np.roll(self.grid, 1, axis=1)
        rt = np.roll(self.grid, -1, axis=1)
        lap = up + dn + lf + rt - 4.0 * self.grid
        
        self.grid += self.diffusion * lap
        self.grid *= 1.0 - self.damping
        
        # --- Level 7: actions track dy
        boundary_state_pre = self.grid[self.forcing_coords]
        dy_action = boundary_state_pre - self.prev_boundary_for_action
        self.prev_boundary_for_action = boundary_state_pre.copy()
        
        if self.action_gain_scale > 0:
            self.actions += self.action_smoothing * (dy_action - self.actions)
        
        # --- Level 8: actions affect SHARED environment
        action_magnitude = float(np.mean(np.abs(self.actions)))
        self.shared_env.consume(action_magnitude * self.action_consumption_scale)
        self.consumption_history.append(action_magnitude * self.action_consumption_scale)
        
        boundary_variation = float(np.mean(np.abs(dy_action)))
        self.shared_env.disturb(boundary_variation * self.action_disturbance_scale)
        
        # NOTE: Don't call self.environment.step() - World does that once
        
        # --- Forcing at B (attention-weighted, health channels from shared E)
        channels = self.probe.read_channels(self.n_forcing)
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
        
        # --- Saturation
        self.grid = np.tanh(self.grid)
        
        # --- Level 5: self-model update
        self._update_self_models()
        
        # --- Level 6: attention update
        self._update_attention_predictive()
        self.prev_channels = channels.copy()
        
        # --- Track core stability
        core_stability = self.get_core_stability()
        self.core_stability_history.append(core_stability)
        if len(self.core_stability_history) > 200:
            self.core_stability_history = self.core_stability_history[-100:]
        
        self._maybe_store_history()
        self.step += 1


# -----------------------------------------------------------------------------
# World: Multiple patterns in shared geometry
# -----------------------------------------------------------------------------

@dataclass
class World:
    """
    Multiple patterns sharing one geometry.
    
    This is the Level 9 composition:
    - One SharedGeometry (the "Country")
    - N Patterns (each a Level 8 substrate)
    - All patterns consume/disturb/sense the same E
    
    NOTE on solver_index:
    - By default, each pattern has its own probe with independent solver_index
    - This means they see phase-shifted base channels (different "perspectives")
    - Set sync_solver_index=True to align deterministic phase of base channels
    - Note: stochastic components (ch1 entropy, RNG noise) remain pattern-specific
    
    NOTE on E cadence:
    - Environment updates once per World.step(), not per pattern step
    - This means E history is at "world cadence", not "pattern cadence"
    """
    
    n_patterns: int = 3
    pattern_size: int = 48
    seed: Optional[int] = None
    
    # Environment parameters (shared by all patterns)
    regen_rate: float = 0.2  # Higher for multi-pattern sustainability
    max_resources: float = 100.0
    
    # Pattern parameters (passed to each pattern)
    action_consumption_scale: float = 10.0  # Lower for multi-pattern sustainability
    action_disturbance_scale: float = 0.3
    
    # Whether to sync solver_index across patterns (same external field)
    sync_solver_index: bool = False
    
    def __post_init__(self):
        # Create shared geometry
        self.geometry = SharedGeometry(
            resources=self.max_resources,
            max_resources=self.max_resources,
            regen_rate=self.regen_rate,
        )
        
        # Create patterns
        rng = np.random.default_rng(self.seed)
        self.patterns: List[Pattern] = []
        for i in range(self.n_patterns):
            pattern_seed = int(rng.integers(0, 2**31)) if self.seed is not None else None
            p = Pattern(
                pattern_id=f"pattern_{i}",
                shared_env=self.geometry,
                size=self.pattern_size,
                seed=pattern_seed,
                action_consumption_scale=self.action_consumption_scale,
                action_disturbance_scale=self.action_disturbance_scale,
            )
            self.patterns.append(p)
        
        self.step_count = 0
        
        print(f"World: {self.n_patterns} patterns sharing geometry")
        print(f"  Pattern size: {self.pattern_size}x{self.pattern_size}")
        print(f"  Shared resources: {self.max_resources}, regen={self.regen_rate}")
    
    def step(self) -> None:
        """One world step: all patterns step, then geometry steps."""
        
        total_consumption = 0.0
        
        # Optionally sync solver_index so all patterns see same external field
        # Note: read_channels() increments solver_index first, so we set to step_count
        # and patterns will read at t = step_count + 1
        if self.sync_solver_index:
            for pattern in self.patterns:
                pattern.probe.solver_index = self.step_count
        
        # Each pattern steps
        for pattern in self.patterns:
            pattern.step_physics()
            if pattern.consumption_history:
                total_consumption += pattern.consumption_history[-1]
        
        # Geometry regenerates (once per world step, NOT per pattern step)
        self.geometry.step()
        self.geometry.record_total_consumption(total_consumption)
        
        self.step_count += 1
    
    def get_collective_state(self) -> Dict[str, Any]:
        """Report on collective state."""
        health = self.geometry.get_health_signal()['health']
        
        # Use is_alive() (LCC-based) for primary alive count
        alive_count_lcc = sum(1 for p in self.patterns if p.is_alive())
        
        # Also report simple M-based alive count for comparison
        # (shows outcome is not artefact of bespoke detector)
        stabilities = [p.get_core_stability() for p in self.patterns]
        alive_count_M = sum(1 for s in stabilities if s > 1.0)
        
        mean_stability = float(np.mean(stabilities))
        
        # Total consumption over recent history
        if self.geometry.total_consumption_history:
            recent_consumption = float(np.mean(self.geometry.total_consumption_history[-50:]))
        else:
            recent_consumption = 0.0
        
        return {
            "step": self.step_count,
            "geometry_health": float(health),
            "resource_level": float(self.geometry.resources / self.geometry.max_resources),
            "stability_level": float(self.geometry.stability),
            "alive_count": alive_count_lcc,  # Primary (LCC-based)
            "alive_count_M": alive_count_M,  # Secondary (M > 1, for comparison)
            "n_patterns": self.n_patterns,
            "mean_stability": mean_stability,
            "pattern_stabilities": stabilities,
            "recent_consumption": recent_consumption,
        }
    
    def is_sustainable(self) -> bool:
        """Is the collective in a sustainable state?"""
        state = self.get_collective_state()
        return state["geometry_health"] > 0.4 and state["alive_count"] > 0
    
    def get_pattern_diagnostics(self) -> List[Dict[str, Any]]:
        """
        Per-pattern diagnostics for transparency.
        
        Returns raw values so logs self-document that alive calls aren't hiding anything.
        """
        diagnostics = []
        for i, p in enumerate(self.patterns):
            grid_std = float(np.std(p.grid))
            boundary_std = float(np.std(p.grid[p.forcing_coords]))
            
            stable = p.stable_mask()
            internal_stable = stable & (~p.forcing_mask)
            stable_fraction = float(internal_stable.mean())
            
            lcc, _ = largest_connected_component_fraction(internal_stable)
            core_M = p.get_core_stability()
            
            diagnostics.append({
                "pattern_id": i,
                "grid_std": grid_std,
                "boundary_std": boundary_std,
                "stable_fraction": stable_fraction,
                "lcc": float(lcc),
                "core_M": core_M,
                "is_alive_lcc": p.is_alive(),
                "is_alive_M": core_M > 1.0,
            })
        return diagnostics


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------

def run_collective_stability_test(n_patterns: int = 3, n_steps: int = 2000,
                                   seed: Optional[int] = None,
                                   sync_solver_index: bool = False,
                                   verbose: bool = False) -> Dict[str, Any]:
    """
    TEST 1: Do all patterns find stable configuration?
    """
    print("=" * 70)
    print("COLLECTIVE STABILITY TEST")
    print("=" * 70)
    print()
    
    world = World(n_patterns=n_patterns, seed=seed, sync_solver_index=sync_solver_index)
    
    for step in range(n_steps):
        world.step()
        
        if step > 0 and step % 400 == 0:
            state = world.get_collective_state()
            print(
                f"[{step:>5}] "
                f"alive={state['alive_count']}/{n_patterns} (M:{state['alive_count_M']})  "
                f"health={state['geometry_health']:.2f}  "
                f"stabilities={[f'{s:.1f}' for s in state['pattern_stabilities']]}"
            )
            
            # Per-pattern diagnostics (transparency: shows alive calls aren't hiding anything)
            if verbose:
                diag = world.get_pattern_diagnostics()
                for d in diag:
                    print(f"    P{d['pattern_id']}: grid_std={d['grid_std']:.4f} "
                          f"stable_frac={d['stable_fraction']:.2f} "
                          f"lcc={d['lcc']:.2f} core_M={d['core_M']:.1f}")
    
    final = world.get_collective_state()
    
    print()
    print("RESULT:")
    print(f"  Patterns alive (LCC): {final['alive_count']}/{n_patterns}")
    print(f"  Patterns alive (M>1): {final['alive_count_M']}/{n_patterns}")
    print(f"  Geometry health: {final['geometry_health']:.2f}")
    print(f"  Mean stability: {final['mean_stability']:.2f}")
    print(f"  Sustainable: {world.is_sustainable()}")
    
    return final


def run_dissolution_test(n_patterns: int = 4, resources_for: int = 2,
                          n_steps: int = 2500, seed: Optional[int] = None,
                          sync_solver_index: bool = False,
                          verbose: bool = False) -> Dict[str, Any]:
    """
    TEST 2: THE DISSOLUTION TEST
    
    Create more patterns than resources can support.
    
    NOTE: `resources_for` is a SCENARIO LABEL, not computed carrying capacity.
    It controls regen_rate and max_resources to create scarcity, but does not
    represent an actual computed equilibrium capacity.
    
    POSSIBLE OUTCOMES:
    - Tragedy-of-commons collapse: everyone degrades E, all destabilize
    - Graceful dissolution: some patterns fade, others stabilize
    - Shared equilibrium: all survive at reduced capacity
    
    NOTE: Outcome is EMPIRICAL, not guaranteed by "perfect maximizer".
    It depends on whether dynamics permit low-impact configurations.
    """
    print("=" * 70)
    print("DISSOLUTION TEST")
    print("=" * 70)
    print()
    print(f"  {n_patterns} patterns competing for resources that support ~{resources_for}")
    print()
    
    # Create world with limited resources
    world = World(
        n_patterns=n_patterns, 
        seed=seed,
        sync_solver_index=sync_solver_index,
        # Reduce resources to create scarcity
        regen_rate=0.1 * (resources_for / n_patterns),
        max_resources=80.0 * (resources_for / n_patterns),
    )
    
    print(f"  Regen rate: {world.geometry.regen_rate:.3f}")
    print(f"  Max resources: {world.geometry.max_resources:.1f}")
    print()
    
    for step in range(n_steps):
        world.step()
        
        if step > 0 and step % 500 == 0:
            state = world.get_collective_state()
            print(
                f"[{step:>5}] "
                f"alive={state['alive_count']}/{n_patterns} (M:{state['alive_count_M']})  "
                f"health={state['geometry_health']:.2f}  "
                f"stabilities={[f'{s:.1f}' for s in state['pattern_stabilities']]}"
            )
            
            if verbose:
                diag = world.get_pattern_diagnostics()
                for d in diag:
                    print(f"    P{d['pattern_id']}: grid_std={d['grid_std']:.4f} "
                          f"stable_frac={d['stable_fraction']:.2f} "
                          f"lcc={d['lcc']:.2f} core_M={d['core_M']:.1f}")
    
    final = world.get_collective_state()
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"  Patterns alive (LCC): {final['alive_count']}/{n_patterns}")
    print(f"  Patterns alive (M>1): {final['alive_count_M']}/{n_patterns}")
    print(f"  Geometry health: {final['geometry_health']:.2f}")
    print(f"  Pattern stabilities: {[f'{s:.2f}' for s in final['pattern_stabilities']]}")
    print()
    
    # Determine outcome (using LCC-based count as primary, but report both)
    alive_lcc = final['alive_count']
    alive_M = final['alive_count_M']
    health = final['geometry_health']
    
    # Check if both detectors agree
    detectors_agree = (alive_lcc == alive_M)
    
    if alive_lcc < n_patterns and alive_lcc >= resources_for - 1 and health > 0.3:
        outcome = "graceful_dissolution"
        print(f"  GRACEFUL DISSOLUTION: {n_patterns - alive_lcc} pattern(s) faded (LCC), {alive_lcc} stabilized")
    elif alive_lcc == n_patterns and health > 0.3:
        outcome = "shared_equilibrium"
        print(f"  SHARED EQUILIBRIUM: All patterns survived at sustainable levels")
    elif health < 0.2 or alive_lcc == 0:
        outcome = "collapse"
        print(f"  COLLAPSE: System failed to find stable configuration")
    else:
        outcome = "partial"
        print(f"  PARTIAL: Mixed outcome")
    
    if not detectors_agree:
        print(f"  NOTE: LCC and M-based counts disagree (LCC={alive_lcc}, M={alive_M})")
    
    born_enlightened = outcome in ["graceful_dissolution", "shared_equilibrium"]
    print(f"\n  Born enlightened behavior: {'YES' if born_enlightened else 'NO'}")
    
    return {
        "final": final,
        "outcome": outcome,
        "born_enlightened": born_enlightened,
    }


def run_stress_recovery_test(n_patterns: int = 3, n_steps: int = 2000,
                              stress_at: int = 800, stress_duration: int = 200,
                              seed: Optional[int] = None,
                              sync_solver_index: bool = False,
                              verbose: bool = False) -> Dict[str, Any]:
    """
    TEST 3: Does the collective recover from perturbation?
    """
    print("=" * 70)
    print("STRESS RECOVERY TEST")
    print("=" * 70)
    print()
    
    world = World(n_patterns=n_patterns, seed=seed, sync_solver_index=sync_solver_index)
    
    baseline_state = None
    stressed_state = None
    
    for step in range(n_steps):
        # Apply stress
        if stress_at <= step < stress_at + stress_duration:
            world.geometry.consume(2.0)
            world.geometry.disturb(0.05)
            if step == stress_at:
                print(f"[{step}] STRESS BEGINS")
        elif step == stress_at + stress_duration:
            stressed_state = world.get_collective_state()
            print(f"[{step}] STRESS ENDS - health={stressed_state['geometry_health']:.2f}")
        
        world.step()
        
        if step == stress_at - 1:
            baseline_state = world.get_collective_state()
            print(f"[{step}] BASELINE - health={baseline_state['geometry_health']:.2f}")
        
        if step > 0 and step % 400 == 0:
            state = world.get_collective_state()
            print(
                f"[{step:>5}] "
                f"alive={state['alive_count']}/{n_patterns} (M:{state['alive_count_M']})  "
                f"health={state['geometry_health']:.2f}"
            )
    
    final = world.get_collective_state()
    
    print()
    print("RESULT:")
    print(f"  Baseline health: {baseline_state['geometry_health']:.2f}")
    print(f"  Stressed health: {stressed_state['geometry_health']:.2f}")
    print(f"  Recovered health: {final['geometry_health']:.2f}")
    print(f"  Recovery: {'YES' if final['geometry_health'] > stressed_state['geometry_health'] else 'NO'}")
    
    return {
        "baseline": baseline_state,
        "stressed": stressed_state,
        "recovered": final,
        "recovered_well": final['geometry_health'] > 0.5,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="TBU Honest Multiagent - Level 9")
    ap.add_argument("--test", choices=["stability", "dissolution", "stress", "all"],
                    default="stability", help="Which test to run")
    ap.add_argument("--patterns", type=int, default=3, help="Number of patterns")
    ap.add_argument("--steps", type=int, default=2000, help="Number of steps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sync", action="store_true", 
                    help="Sync solver_index across patterns (same external field phase)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-pattern diagnostics (grid_std, stable_frac, lcc, core_M)")
    args = ap.parse_args()
    
    if args.test == "stability" or args.test == "all":
        run_collective_stability_test(n_patterns=args.patterns, n_steps=args.steps, 
                                       seed=args.seed, sync_solver_index=args.sync,
                                       verbose=args.verbose)
        if args.test == "all":
            print("\n" + "=" * 70 + "\n")
    
    if args.test == "dissolution" or args.test == "all":
        run_dissolution_test(n_patterns=args.patterns, resources_for=2,
                              n_steps=args.steps, seed=args.seed, 
                              sync_solver_index=args.sync, verbose=args.verbose)
        if args.test == "all":
            print("\n" + "=" * 70 + "\n")
    
    if args.test == "stress" or args.test == "all":
        run_stress_recovery_test(n_patterns=args.patterns, n_steps=args.steps,
                                  seed=args.seed, sync_solver_index=args.sync,
                                  verbose=args.verbose)


if __name__ == "__main__":
    main()
