from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np

EPS = 1e-9

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def pairwise_min_distance(positions: np.ndarray) -> float:
    n = positions.shape[0]
    if n <= 1:
        return float("inf")
    if cKDTree is not None:
        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=2)
        return float(np.min(dists[:, 1]))
    dmin = float("inf")
    for i in range(n):
        d = np.linalg.norm(positions[i + 1 :] - positions[i], axis=1)
        if d.size:
            dmin = min(dmin, float(np.min(d)))
    return dmin


def count_pairs_within(positions: np.ndarray, radius: float) -> int:
    n = positions.shape[0]
    if n <= 1:
        return 0
    if cKDTree is not None:
        tree = cKDTree(positions)
        return len(tree.query_pairs(radius))
    count = 0
    r2 = radius * radius
    for i in range(n):
        diff = positions[i + 1 :] - positions[i]
        count += int(np.sum(np.einsum("ij,ij->i", diff, diff) < r2))
    return count


def connected_components_from_edges(n: int, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    seen = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(comp)
    return comps


def connectivity_metrics(positions: np.ndarray, comm_radius: float) -> Dict[str, float]:
    n = positions.shape[0]
    if n <= 1:
        return {"largest_component_ratio": 1.0, "n_components": 1.0}
    if cKDTree is not None:
        tree = cKDTree(positions)
        edges = list(tree.query_pairs(comm_radius))
    else:
        edges = []
        r2 = comm_radius * comm_radius
        for i in range(n):
            diff = positions[i + 1 :] - positions[i]
            d2 = np.einsum("ij,ij->i", diff, diff)
            for off in np.where(d2 <= r2)[0]:
                edges.append((i, i + 1 + int(off)))
    comps = connected_components_from_edges(n, edges)
    largest = max(len(c) for c in comps)
    return {
        "largest_component_ratio": float(largest / n),
        "n_components": float(len(comps)),
    }


def order_parameter(velocities: np.ndarray) -> float:
    speeds = np.linalg.norm(velocities, axis=1)
    moving = speeds > 1e-4
    if not np.any(moving):
        return 0.0
    unit = velocities[moving] / speeds[moving, None]
    return float(np.linalg.norm(np.mean(unit, axis=0)))


def mission_progress(positions: np.ndarray, goals: np.ndarray, goal_tolerance: float) -> Tuple[float, float]:
    dist = np.linalg.norm(positions - goals, axis=1)
    reached = dist <= goal_tolerance
    return float(np.mean(reached)), float(np.mean(dist))
