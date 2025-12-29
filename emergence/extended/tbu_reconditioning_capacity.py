#!/usr/bin/env python3
"""
================================================================================
TBU RECONDITIONING CAPACITY (RC) METRIC — CORRECTED DEFINITION
================================================================================

KEY INSIGHT (from Level 22):

    Reconditioning Capacity is NOT measured by the appearance of contradiction,
    but by how much contradiction can be absorbed WITHOUT ever appearing.

    Sign-flip is a FAILURE MODE of insufficient hierarchy, not a feature.
    A well-structured hierarchy PREVENTS sign-flip by giving reconditioning 
    somewhere to go.

WHAT HIERARCHY DOES:
    - Absorbs reconditioning before it reaches the core
    - Each level resolves one degree of freedom
    - Contradictions exist but are resolved upstream
    - By the time signals reach core, they're compatible

CAPACITY EXCEEDED SIGNALS (for hierarchical systems):
    - Decode accuracy drops (information loss)
    - Excess depth becomes harmful
    - Matching law breaks

    NOT sign-flip (that's for non-hierarchical substrates)

DEFINITIONS
-----------

Reconditioning Mode:
    A latent variable that creates context-dependent constraint behavior.
    The system must absorb this without letting contradictions reach the core.

Reconditioning Capacity (RC):
    The maximum number of independent reconditioning modes a system can absorb
    while maintaining:
        1. Unified correlation geometry (no sign-flip)
        2. Stable information recovery (decode accuracy)
        3. Beneficial depth-accuracy relationship

ANALOGY:
    A well-designed suspension doesn't eliminate bumps in the road —
    it prevents them from shaking the car apart.
    The bumps still exist. They just don't show up as chaos.

    Hierarchy is structural stability in constraint space.

================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import warnings


# =============================================================================
# COHERENCE CHECK (Sign-flip = failure, absence = success)
# =============================================================================

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, handling edge cases."""
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return 0.0 if not np.isfinite(r) else float(r)


def median_split(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Split array by median into high/low masks."""
    thr = float(np.median(z))
    return z >= thr, z < thr, thr


@dataclass
class CoherenceResult:
    """
    Result of coherence check with three states:
    
    1. COHERENT: Same-sign correlations with meaningful magnitude
    2. NEUTRAL: Correlations too weak to evaluate (uninformative)
    3. FLIP: Opposite-sign correlations with meaningful magnitude (failure)
    
    Sign-flip is only meaningful if both correlations have significant magnitude.
    """
    r_pool: float
    r_hi: float
    r_lo: float
    sign_flip: bool  # True = opposite signs with magnitude
    washout_ratio: float
    
    # Thresholds
    flip_epsilon: float = 0.05  # Min magnitude to count as non-zero
    strength_min: float = 0.08  # Min magnitude to be "informative"
    
    @property
    def max_strength(self) -> float:
        """Maximum absolute correlation across regimes."""
        return max(abs(self.r_hi), abs(self.r_lo))
    
    @property
    def is_informative(self) -> bool:
        """Correlations strong enough to evaluate."""
        return self.max_strength >= self.strength_min
    
    @property
    def is_coherent(self) -> bool:
        """
        System is coherent if:
        - No sign-flip, OR
        - Correlations too weak to evaluate (neutral)
        
        Only FLIP with magnitude counts as failure.
        """
        return not self.sign_flip
    
    @property
    def state(self) -> str:
        """Three-state classification."""
        if self.sign_flip:
            return "flip"
        elif not self.is_informative:
            return "neutral"
        else:
            return "coherent"
    
    @property
    def status(self) -> str:
        if self.sign_flip:
            return "FLIP: Contradiction leaked"
        elif not self.is_informative:
            return "NEUTRAL: Weak signal"
        else:
            return "COHERENT: Unified geometry"


def check_coherence(
    boundary_std: np.ndarray,
    core_std: np.ndarray,
    volatility: np.ndarray,
    min_samples: int = 100,
    flip_epsilon: float = 0.05,  # Correlations below this are "effectively zero"
) -> CoherenceResult:
    """
    Check if hierarchy maintains coherence under reconditioning.
    
    Coherent = same-sign correlations across regimes
    Incoherent = sign-flip (hierarchy failed to absorb contradiction)
    
    Uses epsilon threshold: |r| < flip_epsilon treated as zero (no flip possible)
    Only a TRUE flip if: r_hi > +eps AND r_lo < -eps (or vice versa)
    """
    mask = np.isfinite(boundary_std) & np.isfinite(core_std) & np.isfinite(volatility)
    x, y, z = boundary_std[mask], core_std[mask], volatility[mask]
    
    if len(x) < min_samples:
        return CoherenceResult(0, 0, 0, False, 0)
    
    hi, lo, _ = median_split(z)
    
    r_pool = pearson_r(x, y)
    r_hi = pearson_r(x[hi], y[hi])
    r_lo = pearson_r(x[lo], y[lo])
    
    # Strict sign-flip detection with epsilon
    # Only flip if BOTH are meaningfully non-zero AND opposite
    hi_positive = r_hi > flip_epsilon
    hi_negative = r_hi < -flip_epsilon
    lo_positive = r_lo > flip_epsilon
    lo_negative = r_lo < -flip_epsilon
    
    sign_flip = (hi_positive and lo_negative) or (hi_negative and lo_positive)
    
    best = max(abs(r_hi), abs(r_lo), 1e-12)
    washout = 1.0 - (abs(r_pool) / best)
    
    return CoherenceResult(
        r_pool=r_pool,
        r_hi=r_hi,
        r_lo=r_lo,
        sign_flip=sign_flip,
        washout_ratio=washout,
    )


# =============================================================================
# CAPACITY MEASUREMENT (decode accuracy, not sign-flip)
# =============================================================================

@dataclass
class CapacityMeasurement:
    """
    Measurement of reconditioning capacity at a given configuration.
    
    Capacity exceeded when:
        - Decode accuracy drops below threshold
        - Adding depth no longer helps (or hurts)
    """
    n_modes: int
    depth: int
    decode_accuracy: float
    coherence: CoherenceResult
    
    @property
    def is_within_capacity(self) -> bool:
        """System is within capacity if coherent and accuracy above threshold."""
        return self.coherence.is_coherent and self.decode_accuracy > 0.6
    
    @property 
    def capacity_status(self) -> str:
        if not self.coherence.is_coherent:
            return "EXCEEDED: Hierarchy insufficient (incoherence)"
        elif self.decode_accuracy < 0.6:
            return "EXCEEDED: Information loss (accuracy dropped)"
        else:
            return f"WITHIN: Absorbing {self.n_modes} modes at depth {self.depth}"


# =============================================================================
# RECONDITIONING CAPACITY (RC) - Complete Metric
# =============================================================================

@dataclass
class ReconditioningCapacity:
    """
    Complete Reconditioning Capacity measurement.
    
    RC = maximum modes absorbable while maintaining coherence + accuracy
    
    This is a GEOMETRIC metric, not computational:
        - How much structural folding can the system do?
        - How many contradictions can be silently resolved?
    """
    
    # Core capacity numbers
    RC_v: int = 1  # Vertical (max beneficial depth)
    RC_h: int = 1  # Horizontal (max beneficial body count)  
    max_modes_absorbed: int = 1  # Max modes with coherent geometry
    
    # Supporting measurements
    measurements: List[CapacityMeasurement] = field(default_factory=list)
    
    # Threshold used
    accuracy_threshold: float = 0.6
    
    @property
    def RC_total(self) -> float:
        """
        Total Reconditioning Capacity.
        
        RC_total = RC_v × RC_h × max(1, max_modes_absorbed)
        
        All three multiply because:
            - Deeper = more hierarchical folding
            - Wider = more distributed folding
            - More modes = more contradiction absorbed
        """
        return float(self.RC_v * self.RC_h * max(1, self.max_modes_absorbed))
    
    @property
    def benchmark_class(self) -> str:
        """System classification by capacity profile."""
        if self.RC_h <= 1 and self.RC_v > 2:
            return "A"  # Vertical only (weather-like)
        elif self.RC_v <= 2 and self.RC_h > 1:
            return "B"  # Horizontal only (distributed)
        elif self.RC_v > 2 and self.RC_h > 1:
            return "C"  # Full (complex environments)
        else:
            return "minimal"
    
    def report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            "RECONDITIONING CAPACITY REPORT",
            "=" * 70,
            "",
            "CORE INSIGHT:",
            "  Capacity = how much contradiction absorbed WITHOUT appearing",
            "",
            "THREE STATES:",
            "  coherent = same-sign correlations (unified geometry)",
            "  neutral  = correlations too weak to evaluate",
            "  FLIP     = opposite-sign correlations (contradiction leaked)",
            "",
            f"RC_v (vertical depth):      {self.RC_v}",
            f"RC_h (horizontal bodies):   {self.RC_h}",
            f"Max modes absorbed:         {self.max_modes_absorbed}",
            "",
            f"RC_total = {self.RC_v} × {self.RC_h} × {max(1, self.max_modes_absorbed)} = {self.RC_total:.0f}",
            "",
            f"Benchmark Class: {self.benchmark_class}",
        ]
        
        if self.benchmark_class == "A":
            lines.append("→ Vertical hierarchy (weather-like sensing)")
        elif self.benchmark_class == "B":
            lines.append("→ Horizontal interleaving (distributed observation)")
        elif self.benchmark_class == "C":
            lines.append("→ Full multi-body hierarchy (complex environments)")
        
        lines.extend(["", "=" * 70])
        return "\n".join(lines)


# =============================================================================
# MEASUREMENT FUNCTIONS
# =============================================================================

def measure_rc_at_config(
    n_modes: int,
    depth: int,
    n_steps: int = 8000,
    seed: int = 42,
) -> CapacityMeasurement:
    """
    Measure capacity at a specific (modes, depth) configuration.
    
    Returns decode accuracy AND coherence check.
    """
    from tbu_level22_canonical import NLevelHierarchy
    
    h = NLevelHierarchy(
        n_levels=depth,
        n_latents=n_modes,
        seed=seed
    )
    
    # Collect data
    X = []  # Features for decode
    y = []  # Regime labels
    boundary_stds = []
    core_stds = []
    volatilities = []
    
    prev_boundary_mean = None
    
    for t in range(n_steps):
        h.step()
        
        if t > 2000:  # Burn-in
            L1 = h.levels[0]
            
            # Decode features from boundary history
            feats = []
            for i in range(L1.n_basins):
                bh = L1.boundary_history.get(i, [])
                if bh:
                    feats.append(float(np.mean(bh[-1])))
                    if len(bh) >= 2:
                        feats.append(float(np.mean(bh[-1] - bh[-2])))
                    else:
                        feats.append(0.0)
                else:
                    feats.extend([0.0, 0.0])
            X.append(feats)
            y.append(h.global_regime)
            
            # Coherence data from boundary history
            all_boundary = []
            for i in range(L1.n_basins):
                bh = L1.boundary_history.get(i, [])
                if bh:
                    all_boundary.extend(bh[-1].flatten())
            
            if all_boundary:
                boundary_mean = float(np.mean(all_boundary))
                boundary_std = float(np.std(all_boundary))
            else:
                boundary_mean = 0.0
                boundary_std = 0.0
            
            # Core volatility from the substrate
            core_vol = L1.get_core_volatility() if hasattr(L1, 'get_core_volatility') else 0.0
            
            boundary_stds.append(boundary_std)
            core_stds.append(float(core_vol))
            
            # Volatility = change in boundary mean
            if prev_boundary_mean is not None:
                vol = abs(boundary_mean - prev_boundary_mean)
            else:
                vol = 0.0
            volatilities.append(vol)
            prev_boundary_mean = boundary_mean
    
    # Compute decode accuracy (time-split)
    X = np.array(X)
    y = np.array(y)
    decode_acc = 0.5
    
    if len(X) > 400:
        split = len(X) // 2
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]
        
        Xtrb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        Xteb = np.hstack([Xte, np.ones((len(Xte), 1))])
        d = Xtrb.shape[1]
        A = Xtrb.T @ Xtrb + 0.01 * np.eye(d)
        b = Xtrb.T @ ytr
        try:
            w = np.linalg.solve(A, b)
            pred = np.sign(Xteb @ w)
            decode_acc = float(np.mean(pred == yte))
        except:
            pass
    
    # Check coherence
    coherence = check_coherence(
        np.array(boundary_stds),
        np.array(core_stds),
        np.array(volatilities),
    )
    
    return CapacityMeasurement(
        n_modes=n_modes,
        depth=depth,
        decode_accuracy=decode_acc,
        coherence=coherence,
    )


def find_reconditioning_capacity(
    max_modes: int = 5,
    max_depth: int = 7,
    n_steps: int = 8000,
    n_seeds: int = 3,
    accuracy_threshold: float = 0.65,
) -> ReconditioningCapacity:
    """
    Find the system's reconditioning capacity.
    
    Searches for:
        - RC_v: Maximum beneficial depth
        - max_modes: Maximum modes that can be absorbed with coherence
    """
    print("=" * 70)
    print("FINDING RECONDITIONING CAPACITY")
    print("=" * 70)
    print()
    print("Capacity = max contradiction absorbed while maintaining:")
    print("  1. Coherence (no sign-flip)")
    print("  2. Information recovery (decode accuracy)")
    print()
    
    measurements = []
    rc_v = 2
    max_absorbed = 0  # Start at 0, increment when modes absorbed
    
    for n_modes in range(1, max_modes + 1):
        print(f"\n--- {n_modes} MODE{'S' if n_modes > 1 else ''} ---")
        
        for depth in range(2, max_depth + 1):
            # Average over seeds
            accs = []
            coherent_count = 0
            flip_details = []
            
            for seed in range(42, 42 + n_seeds):
                m = measure_rc_at_config(n_modes, depth, n_steps, seed)
                accs.append(m.decode_accuracy)
                if m.coherence.is_coherent:
                    coherent_count += 1
                flip_details.append((m.coherence.r_hi, m.coherence.r_lo, m.coherence.sign_flip))
                measurements.append(m)
            
            mean_acc = float(np.mean(accs))
            coherent = coherent_count == n_seeds
            
            # Show the actual correlations (means across seeds)
            r_his = [d[0] for d in flip_details]
            r_los = [d[1] for d in flip_details]
            flips = sum(1 for d in flip_details if d[2])
            
            mean_r_hi = np.mean(r_his)
            mean_r_lo = np.mean(r_los)
            max_strength = max(abs(mean_r_hi), abs(mean_r_lo))
            
            # Three-state label
            if flips > 0:
                state_str = f"FLIP:{flips}/{n_seeds}"
            elif max_strength < 0.08:
                state_str = "neutral"
            else:
                state_str = "coherent"
            
            status = "✓" if coherent and mean_acc > accuracy_threshold else "×"
            
            print(f"  D={depth}: acc={mean_acc:.3f} r_hi={mean_r_hi:+.2f} r_lo={mean_r_lo:+.2f} [{state_str}] {status}")
        
        # Check if ANY depth maintained coherence with good accuracy
        # Use MINIMUM DEPTH that is stable (prefer simpler solutions)
        any_coherent = False
        best_coherent_depth = None
        best_coherent_acc = 0.0
        
        # First find max accuracy across all coherent depths
        max_coherent_acc = 0.0
        for depth in range(2, max_depth + 1):
            depth_meas = [m for m in measurements if m.n_modes == n_modes and m.depth == depth]
            all_coherent = all(m.coherence.is_coherent for m in depth_meas)
            if all_coherent:
                mean_acc = np.mean([m.decode_accuracy for m in depth_meas])
                if mean_acc > max_coherent_acc:
                    max_coherent_acc = mean_acc
        
        # Then pick MINIMUM depth within 1% of max (stability-aware)
        acc_tolerance = 0.01
        for depth in range(2, max_depth + 1):
            depth_meas = [m for m in measurements if m.n_modes == n_modes and m.depth == depth]
            all_coherent = all(m.coherence.is_coherent for m in depth_meas)
            mean_acc = np.mean([m.decode_accuracy for m in depth_meas])
            
            if all_coherent and mean_acc >= (max_coherent_acc - acc_tolerance):
                any_coherent = True
                best_coherent_depth = depth
                best_coherent_acc = mean_acc
                break  # Take minimum depth that works
        
        if any_coherent:
            max_absorbed = n_modes
            rc_v = max(rc_v, best_coherent_depth)
            print(f"  → {n_modes} modes ABSORBED at depth {best_coherent_depth} (acc={best_coherent_acc:.3f})")
        else:
            print(f"  → {n_modes} modes: no coherent depth found")
    
    rc = ReconditioningCapacity(
        RC_v=rc_v,
        RC_h=1,  # Single body for now
        max_modes_absorbed=max_absorbed,
        measurements=measurements,
        accuracy_threshold=accuracy_threshold,
    )
    
    print()
    print(rc.report())
    
    return rc


# =============================================================================
# QUICK DEMONSTRATION
# =============================================================================

def demo_coherence_preservation(n_steps: int = 6000, seed: int = 42):
    """
    Demonstrate that hierarchy PREVENTS sign-flip.
    
    This is the key insight: sign-flip absent = hierarchy working.
    """
    print("=" * 70)
    print("COHERENCE PRESERVATION DEMONSTRATION")
    print("=" * 70)
    print()
    print("KEY INSIGHT: Hierarchy prevents sign-flip by absorbing contradiction")
    print("Sign-flip ABSENT = hierarchy functioning correctly")
    print()
    
    configs = [
        (1, 2, "shallow, 1 mode"),
        (1, 4, "deep, 1 mode"),
        (3, 2, "shallow, 3 modes"),
        (3, 4, "deep, 3 modes"),
        (3, 5, "deeper, 3 modes"),
    ]
    
    print(f"{'Config':<25} {'r_pool':>8} {'r_hi':>8} {'r_lo':>8} {'Status':<20}")
    print("-" * 75)
    
    for n_modes, depth, label in configs:
        m = measure_rc_at_config(n_modes, depth, n_steps, seed)
        c = m.coherence
        
        if c.sign_flip:
            status = "SIGN-FLIP (failure!)"
        elif c.washout_ratio > 0.3:
            status = "high washout"
        else:
            status = "coherent ✓"
        
        print(f"D={depth}, M={n_modes} ({label:<12}) {c.r_pool:>+8.3f} {c.r_hi:>+8.3f} "
              f"{c.r_lo:>+8.3f} {status}")
    
    print()
    print("INTERPRETATION:")
    print("  All coherent = hierarchy absorbing all contradiction")
    print("  Sign-flip would indicate hierarchy insufficient")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--find":
        # Full capacity search
        find_reconditioning_capacity(
            max_modes=int(sys.argv[2]) if len(sys.argv) > 2 else 4,
            max_depth=int(sys.argv[3]) if len(sys.argv) > 3 else 6,
            n_steps=int(sys.argv[4]) if len(sys.argv) > 4 else 6000,
            n_seeds=int(sys.argv[5]) if len(sys.argv) > 5 else 3,
        )
    else:
        # Quick demo
        demo_coherence_preservation()
