"""Derive per-observation reference labels from the arc-level .route file.

The .route file is a chronological arc sequence with no timestamps, so
there is no direct row-to-row mapping between GPS fixes and route arcs.
We align each fix t to a route position j(t) with the constraint that
j(t) is non-decreasing (the vehicle never goes back along its own route),
minimising the total point-to-arc distance. Solved exactly by dynamic
programming with a prefix-minimum recurrence:

    D[t, j] = c[t, j] + min_{j' <= j} D[t-1, j']

The reference arc for fix t is then route[j(t)].
"""
import numpy as np
from .geo import project_points_to_segments


def align_track_to_route(xy: np.ndarray, net, route: np.ndarray):
    T, J = len(xy), len(route)
    A, B = net.A[route], net.B[route]
    D = np.empty((T, J), dtype=np.float32)
    prev = None
    for t in range(T):
        P = np.repeat(xy[t:t + 1], J, axis=0)
        c, _, _ = project_points_to_segments(P, A, B)
        c = c.astype(np.float32)
        if prev is None:
            D[t] = c
        else:
            D[t] = c + np.minimum.accumulate(prev)
        prev = D[t]
    j = np.empty(T, dtype=np.int64)
    j[-1] = int(np.argmin(D[-1]))
    for t in range(T - 2, -1, -1):
        j[t] = int(np.argmin(D[t, : j[t + 1] + 1]))
    labels = route[j]
    d, _ = net.dist_to_arc(xy, labels)
    return labels, j, d
