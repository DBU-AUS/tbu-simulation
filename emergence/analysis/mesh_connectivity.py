#!/usr/bin/env python3
"""
================================================================================
TBU CONSTRAINT MESH CONNECTIVITY ANALYSIS
================================================================================

Question: When we perceive a constraint variable, what's the extent of access?
- Local only (nearest neighbors)?
- Complete (all conditioned relationships equally)?
- Gradient (decay with distance)?

Methods:
1. Build graph of variable relationships
2. Measure path lengths and clustering
3. Test if perceiving one variable propagates through chains
4. Check for isolated clusters

================================================================================
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, '/mnt/project')
from tbu_honest_boundary import HonestBoundarySubstrate


def run_substrate_and_log(n_steps: int = 6000, warmup: int = 500, seed: int = 42) -> List[Dict]:
    """Run substrate and log comprehensive state."""
    print(f"Running substrate for {n_steps} steps...")
    
    substrate = HonestBoundarySubstrate(seed=seed)
    
    log = []
    prev_boundary = substrate.grid[substrate.forcing_coords].copy()
    
    for _ in range(warmup):
        substrate.step_physics()
    
    prev_boundary = substrate.grid[substrate.forcing_coords].copy()
    
    for step in range(n_steps - warmup):
        substrate.step_physics()
        
        grid = substrate.grid
        boundary = grid[substrate.forcing_coords]
        core = grid[~substrate.forcing_mask]
        
        dy = boundary - prev_boundary
        boundary_volatility = float(np.mean(np.abs(dy)))
        prev_boundary = boundary.copy()
        
        report = substrate.report()
        mean_attn = substrate.attention.mean(axis=0)
        
        log.append({
            'grid_mean': float(grid.mean()),
            'grid_std': float(grid.std()),
            'core_mean': float(core.mean()),
            'core_std': float(core.std()),
            'boundary_mean': float(boundary.mean()),
            'boundary_std': float(boundary.std()),
            'boundary_volatility': boundary_volatility,
            'boundary_dy_mean': float(np.mean(dy)),
            'boundary_dy_std': float(np.std(dy)),
            'M_ratio': float(report['M_ratio']),
            'core_coherence': float(report['core_coherence']),
            'env_health': float(report['env_health']),
            'env_resources': float(report['env_resources']),
            'env_stability': float(report['env_stability']),
            'obs_sigma': float(report['obs_sigma']),
            'attention_predictable': float(mean_attn[0] + mean_attn[3]),
            'attention_entropy': float(mean_attn[1]),
            'attention_health': float(mean_attn[4] + mean_attn[5]),
            'action_mean': float(report['action_mean']),
            'action_std': float(report['action_std']),
            'action_stability': float(report['action_stability']),
        })
        
        if (step + 1) % 1000 == 0:
            print(f"  Step {step + warmup + 1}/{n_steps}")
    
    return log


def build_relationship_graph(log: List[Dict], min_washout: float = 0.3) -> Dict:
    """
    Build graph where:
    - Nodes = observable variables
    - Edges = hidden relationships (conditioned correlations that wash out)
    - Edge attributes = which condition reveals it, strength
    """
    
    observables = [
        'grid_mean', 'grid_std', 'core_mean', 'core_std',
        'boundary_mean', 'boundary_std', 'boundary_volatility',
        'boundary_dy_mean', 'boundary_dy_std',
        'attention_predictable', 'attention_entropy', 'attention_health',
        'action_mean', 'action_std',
    ]
    
    conditions = [
        'M_ratio', 'core_coherence', 'boundary_volatility',
        'env_health', 'obs_sigma', 'action_stability',
    ]
    
    edges = defaultdict(list)  # (var1, var2) -> list of {condition, strength, sign_flip}
    
    for var_x, var_y in combinations(observables, 2):
        for var_z in conditions:
            if var_z in [var_x, var_y]:
                continue
            
            x = np.array([d[var_x] for d in log])
            y = np.array([d[var_y] for d in log])
            z = np.array([d[var_z] for d in log])
            
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[mask], y[mask], z[mask]
            
            if len(x) < 200:
                continue
            
            z_thresh = np.median(z)
            high, low = z >= z_thresh, z < z_thresh
            
            if high.sum() < 100 or low.sum() < 100:
                continue
            
            r_pooled = np.corrcoef(x, y)[0, 1]
            r_high = np.corrcoef(x[high], y[high])[0, 1]
            r_low = np.corrcoef(x[low], y[low])[0, 1]
            
            if not all(np.isfinite([r_pooled, r_high, r_low])):
                continue
            
            best = max(abs(r_high), abs(r_low))
            washout = 1.0 - abs(r_pooled) / best if best > 0.01 else 0
            sign_flip = (r_high * r_low < 0) and abs(r_high) > 0.05 and abs(r_low) > 0.05
            
            if washout > min_washout or sign_flip:
                pair = tuple(sorted([var_x, var_y]))
                edges[pair].append({
                    'condition': var_z,
                    'washout': washout,
                    'sign_flip': sign_flip,
                    'r_high': r_high,
                    'r_low': r_low,
                    'r_pooled': r_pooled,
                    'interaction': abs(r_high - r_low),
                })
    
    return dict(edges)


def analyze_graph_structure(edges: Dict) -> Dict:
    """Analyze basic graph properties."""
    
    # Extract nodes
    nodes = set()
    for (v1, v2) in edges.keys():
        nodes.add(v1)
        nodes.add(v2)
    
    # Adjacency list
    adj = defaultdict(set)
    for (v1, v2) in edges.keys():
        adj[v1].add(v2)
        adj[v2].add(v1)
    
    # Degree distribution
    degrees = {n: len(adj[n]) for n in nodes}
    
    # Connected components (BFS)
    visited = set()
    components = []
    
    for start in nodes:
        if start in visited:
            continue
        component = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)
    
    # Path lengths (BFS from each node)
    all_distances = {}
    for start in nodes:
        distances = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        all_distances[start] = distances
    
    # Clustering coefficient
    clustering = {}
    for node in nodes:
        neighbors = list(adj[node])
        if len(neighbors) < 2:
            clustering[node] = 0
            continue
        # Count edges between neighbors
        neighbor_edges = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if n2 in adj[n1]:
                    neighbor_edges += 1
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        clustering[node] = neighbor_edges / possible if possible > 0 else 0
    
    return {
        'nodes': nodes,
        'n_nodes': len(nodes),
        'n_edges': len(edges),
        'adj': dict(adj),
        'degrees': degrees,
        'components': components,
        'n_components': len(components),
        'distances': all_distances,
        'clustering': clustering,
    }


def analyze_reach_from_perception(log: List[Dict], edges: Dict, graph: Dict) -> Dict:
    """
    Key question: If we perceive condition Z, how much of the mesh becomes accessible?
    
    Test: For each condition Z, measure:
    1. Direct reach: relationships Z reveals directly
    2. Indirect reach: relationships that become "accessible" through chains
    3. Strength decay: does access strength diminish with graph distance?
    """
    
    conditions = ['M_ratio', 'core_coherence', 'boundary_volatility', 
                  'env_health', 'obs_sigma', 'action_stability']
    
    # What does each condition reveal directly?
    direct_reach = defaultdict(set)
    relationship_strength = {}  # (pair, condition) -> strength
    
    for pair, rels in edges.items():
        for rel in rels:
            z = rel['condition']
            direct_reach[z].add(pair)
            relationship_strength[(pair, z)] = rel['interaction']
    
    # Analyze strength vs graph distance
    results = {}
    
    for condition in conditions:
        revealed = direct_reach[condition]
        
        if not revealed:
            continue
        
        # Get all nodes touched by this condition
        touched_nodes = set()
        for (v1, v2) in revealed:
            touched_nodes.add(v1)
            touched_nodes.add(v2)
        
        # For each revealed relationship, what's its "centrality" in the mesh?
        # (How connected are its endpoints to other revealed relationships?)
        
        strengths = []
        connectivities = []
        
        for pair in revealed:
            v1, v2 = pair
            # How connected are these nodes in the full graph?
            connectivity = (graph['degrees'].get(v1, 0) + graph['degrees'].get(v2, 0)) / 2
            strength = relationship_strength.get((pair, condition), 0)
            strengths.append(strength)
            connectivities.append(connectivity)
        
        results[condition] = {
            'n_direct': len(revealed),
            'touched_nodes': touched_nodes,
            'n_touched': len(touched_nodes),
            'mean_strength': np.mean(strengths) if strengths else 0,
            'std_strength': np.std(strengths) if strengths else 0,
            'strength_connectivity_corr': np.corrcoef(strengths, connectivities)[0, 1] if len(strengths) > 2 else 0,
        }
    
    return results


def analyze_transitive_access(log: List[Dict], edges: Dict, graph: Dict) -> Dict:
    """
    Test transitive closure:
    If A↔B|Z is revealed and B↔C|Z is revealed,
    can we detect A↔C correlation in Z-conditioned analysis even if not directly present?
    
    This tests whether perception propagates through chains.
    """
    
    print("\n" + "-" * 70)
    print("TRANSITIVE ACCESS ANALYSIS")
    print("-" * 70)
    print("\nIf A↔B|Z and B↔C|Z are revealed, is A↔C accessible?")
    
    conditions = ['core_coherence', 'boundary_volatility', 'env_health']
    
    results = {}
    
    for condition in conditions:
        # Get pairs revealed by this condition
        revealed_pairs = set()
        for pair, rels in edges.items():
            for rel in rels:
                if rel['condition'] == condition:
                    revealed_pairs.add(pair)
        
        if len(revealed_pairs) < 3:
            continue
        
        # Build adjacency from revealed pairs
        adj = defaultdict(set)
        for (v1, v2) in revealed_pairs:
            adj[v1].add(v2)
            adj[v2].add(v1)
        
        # Find triangles: A-B-C where A↔B and B↔C but maybe not A↔C
        triangles_tested = 0
        triangles_transitive = 0
        transitive_strength = []
        
        nodes = list(adj.keys())
        for i, A in enumerate(nodes):
            for B in adj[A]:
                for C in adj[B]:
                    if C == A or C in adj[A]:  # Skip if A↔C already revealed
                        continue
                    
                    # A↔B|Z and B↔C|Z exist, but A↔C|Z not in revealed set
                    # Test if A↔C shows conditioning effect anyway
                    
                    x = np.array([d[A] for d in log])
                    y = np.array([d[C] for d in log])
                    z = np.array([d[condition] for d in log])
                    
                    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                    x, y, z = x[mask], y[mask], z[mask]
                    
                    if len(x) < 200:
                        continue
                    
                    z_thresh = np.median(z)
                    high, low = z >= z_thresh, z < z_thresh
                    
                    r_pooled = np.corrcoef(x, y)[0, 1]
                    r_high = np.corrcoef(x[high], y[high])[0, 1]
                    r_low = np.corrcoef(x[low], y[low])[0, 1]
                    
                    if not all(np.isfinite([r_pooled, r_high, r_low])):
                        continue
                    
                    triangles_tested += 1
                    
                    # Does A↔C show transitive conditioning effect?
                    interaction = abs(r_high - r_low)
                    if interaction > 0.2:  # Threshold for "transitive access"
                        triangles_transitive += 1
                        transitive_strength.append(interaction)
        
        if triangles_tested > 0:
            results[condition] = {
                'triangles_tested': triangles_tested,
                'triangles_transitive': triangles_transitive,
                'transitive_rate': triangles_transitive / triangles_tested,
                'mean_transitive_strength': np.mean(transitive_strength) if transitive_strength else 0,
            }
            
            print(f"\n{condition}:")
            print(f"  Triangles tested: {triangles_tested}")
            print(f"  Transitive access: {triangles_transitive} ({100*triangles_transitive/triangles_tested:.1f}%)")
            if transitive_strength:
                print(f"  Mean transitive strength: {np.mean(transitive_strength):.3f}")
    
    return results


def analyze_strength_gradient(log: List[Dict], edges: Dict, graph: Dict) -> Dict:
    """
    Does access strength decay with distance in the mesh?
    
    For each condition, look at:
    - Directly revealed relationships (distance 0)
    - Relationships one hop away in the graph
    - Relationships two hops away
    
    Measure: average interaction strength at each distance
    """
    
    print("\n" + "-" * 70)
    print("STRENGTH GRADIENT ANALYSIS")
    print("-" * 70)
    print("\nDoes access strength decay with graph distance?")
    
    conditions = ['core_coherence', 'boundary_volatility', 'env_health']
    observables = list(graph['nodes'])
    
    results = {}
    
    for condition in conditions:
        # Get directly revealed pairs and their nodes
        direct_pairs = set()
        for pair, rels in edges.items():
            for rel in rels:
                if rel['condition'] == condition:
                    direct_pairs.add(pair)
        
        if not direct_pairs:
            continue
        
        # Measure strength at different distances
        strength_by_distance = defaultdict(list)
        
        for var_x, var_y in combinations(observables, 2):
            if condition in [var_x, var_y]:
                continue
            
            pair = tuple(sorted([var_x, var_y]))
            
            # Compute conditioned correlation
            x = np.array([d[var_x] for d in log])
            y = np.array([d[var_y] for d in log])
            z = np.array([d[condition] for d in log])
            
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[mask], y[mask], z[mask]
            
            if len(x) < 200:
                continue
            
            z_thresh = np.median(z)
            high, low = z >= z_thresh, z < z_thresh
            
            r_high = np.corrcoef(x[high], y[high])[0, 1]
            r_low = np.corrcoef(x[low], y[low])[0, 1]
            
            if not all(np.isfinite([r_high, r_low])):
                continue
            
            interaction = abs(r_high - r_low)
            
            # Determine distance from directly revealed pairs
            if pair in direct_pairs:
                distance = 0
            else:
                # Check if adjacent to a direct pair (share a node)
                pair_nodes = set(pair)
                min_dist = float('inf')
                for dp in direct_pairs:
                    dp_nodes = set(dp)
                    if pair_nodes & dp_nodes:  # Share a node
                        min_dist = min(min_dist, 1)
                    else:
                        # Check 2-hop (share neighbor)
                        for pn in pair_nodes:
                            for dpn in dp_nodes:
                                if dpn in graph['adj'].get(pn, set()):
                                    min_dist = min(min_dist, 2)
                distance = min_dist if min_dist < float('inf') else 3
            
            strength_by_distance[distance].append(interaction)
        
        # Summarize
        gradient = {}
        for dist in sorted(strength_by_distance.keys()):
            strengths = strength_by_distance[dist]
            gradient[dist] = {
                'n': len(strengths),
                'mean': np.mean(strengths),
                'std': np.std(strengths),
            }
        
        results[condition] = gradient
        
        print(f"\n{condition}:")
        print(f"  {'Distance':<10} {'N':>6} {'Mean Strength':>15} {'Std':>10}")
        print(f"  {'-'*45}")
        for dist in sorted(gradient.keys()):
            g = gradient[dist]
            print(f"  {dist:<10} {g['n']:>6} {g['mean']:>15.3f} {g['std']:>10.3f}")
    
    return results


def analyze_cluster_isolation(edges: Dict, graph: Dict) -> Dict:
    """
    Are there isolated clusters in the mesh?
    I.e., groups of variables that are internally connected but not connected to others?
    """
    
    print("\n" + "-" * 70)
    print("CLUSTER ISOLATION ANALYSIS")
    print("-" * 70)
    
    components = graph['components']
    
    print(f"\nNumber of connected components: {len(components)}")
    
    for i, comp in enumerate(components):
        print(f"\nComponent {i+1} ({len(comp)} nodes):")
        for node in sorted(comp):
            deg = graph['degrees'].get(node, 0)
            clust = graph['clustering'].get(node, 0)
            print(f"  {node}: degree={deg}, clustering={clust:.2f}")
    
    # Check if any component is isolated
    if len(components) == 1:
        print("\n→ The mesh is FULLY CONNECTED")
        print("  All variables can reach all others through the constraint graph")
    else:
        print(f"\n→ The mesh has {len(components)} ISOLATED CLUSTERS")
        print("  Perceiving one cluster does not give access to others")
    
    return {'n_components': len(components), 'components': components}


def main():
    print("=" * 70)
    print("TBU CONSTRAINT MESH CONNECTIVITY ANALYSIS")
    print("=" * 70)
    print()
    print("Question: Local access, complete access, or gradient?")
    print()
    
    # Run substrate
    log = run_substrate_and_log(n_steps=6000, warmup=500, seed=42)
    print(f"\nLogged {len(log)} steps")
    
    # Build graph
    print("\nBuilding relationship graph...")
    edges = build_relationship_graph(log, min_washout=0.3)
    print(f"Found {len(edges)} variable pairs with hidden structure")
    
    # Analyze structure
    print("\nAnalyzing graph structure...")
    graph = analyze_graph_structure(edges)
    
    print(f"\n{'='*70}")
    print("GRAPH STRUCTURE")
    print("="*70)
    print(f"\nNodes: {graph['n_nodes']}")
    print(f"Edges: {graph['n_edges']}")
    print(f"Components: {graph['n_components']}")
    print(f"Average degree: {np.mean(list(graph['degrees'].values())):.2f}")
    print(f"Average clustering: {np.mean(list(graph['clustering'].values())):.2f}")
    
    # Degree distribution
    print("\nDegree distribution:")
    for node in sorted(graph['degrees'].keys(), key=lambda n: -graph['degrees'][n]):
        print(f"  {node}: {graph['degrees'][node]} connections")
    
    # Path length distribution
    all_path_lengths = []
    for start, distances in graph['distances'].items():
        for end, dist in distances.items():
            if start < end:  # Avoid double counting
                all_path_lengths.append(dist)
    
    if all_path_lengths:
        print(f"\nPath lengths:")
        print(f"  Min: {min(all_path_lengths)}")
        print(f"  Max: {max(all_path_lengths)}")
        print(f"  Mean: {np.mean(all_path_lengths):.2f}")
        print(f"  Diameter: {max(all_path_lengths)}")
    
    # Cluster isolation
    cluster_results = analyze_cluster_isolation(edges, graph)
    
    # Reach analysis
    print(f"\n{'='*70}")
    print("REACH FROM PERCEPTION")
    print("="*70)
    reach_results = analyze_reach_from_perception(log, edges, graph)
    
    print(f"\n{'Condition':<20} {'Direct':>8} {'Nodes':>8} {'Mean Str':>10}")
    print("-" * 50)
    for cond, stats in sorted(reach_results.items(), key=lambda x: -x[1]['n_direct']):
        print(f"{cond:<20} {stats['n_direct']:>8} {stats['n_touched']:>8} {stats['mean_strength']:>10.3f}")
    
    # Transitive access
    transitive_results = analyze_transitive_access(log, edges, graph)
    
    # Strength gradient
    gradient_results = analyze_strength_gradient(log, edges, graph)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: LOCAL VS GLOBAL ACCESS")
    print("="*70)
    
    if graph['n_components'] == 1:
        print("\n✓ Mesh is FULLY CONNECTED - no isolated clusters")
    else:
        print(f"\n✗ Mesh has {graph['n_components']} isolated clusters")
    
    # Check gradient
    has_gradient = False
    for cond, gradient in gradient_results.items():
        if 0 in gradient and 2 in gradient:
            d0 = gradient[0]['mean']
            d2 = gradient[2]['mean'] if 2 in gradient else gradient.get(3, {}).get('mean', 0)
            if d0 > d2 * 1.3:  # 30% stronger at distance 0
                has_gradient = True
    
    if has_gradient:
        print("✓ GRADIENT EXISTS - strength decays with distance")
    else:
        print("✗ No clear gradient - access is relatively uniform")
    
    # Check transitive
    has_transitive = any(r.get('transitive_rate', 0) > 0.3 for r in transitive_results.values())
    
    if has_transitive:
        print("✓ TRANSITIVE ACCESS - chains propagate information")
    else:
        print("✗ Limited transitive access - mostly direct relationships")
    
    print()


if __name__ == "__main__":
    main()
