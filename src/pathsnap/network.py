"""Road network: arc geometry, spatial index and bounded shortest paths."""
import heapq
import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, bearing_deg, project_points_to_segments
from . import config as C


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                 densify_step: float = C.DENSIFY_STEP_M):
        self.proj = proj
        self.nodes_ll = nodes_lonlat
        self.nodes = proj.to_xy(nodes_lonlat)
        self.arcs = arcs.astype(np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        self.A = self.nodes[self.arcs[:, 0]]
        self.B = self.nodes[self.arcs[:, 1]]
        d = self.B - self.A
        self.length = np.linalg.norm(d, axis=1)
        self.bearing = bearing_deg(d)
        self.out_deg = np.bincount(self.arcs[:, 0], minlength=self.n_nodes)
        self.in_deg = np.bincount(self.arcs[:, 1], minlength=self.n_nodes)
        # reverse-arc existence (two-way street)
        key = self.arcs[:, 0] * self.n_nodes + self.arcs[:, 1]
        rkey = self.arcs[:, 1] * self.n_nodes + self.arcs[:, 0]
        self.has_reverse = np.isin(rkey, key)
        # CSR adjacency node -> (to_node, arc_id, length)
        order = np.argsort(self.arcs[:, 0], kind="stable")
        self._adj_arc = order
        self._adj_ptr = np.r_[0, np.cumsum(np.bincount(self.arcs[:, 0], minlength=self.n_nodes))]
        self._build_index(densify_step)
        self._sp_cache: dict = {}

    # ---------------- spatial index -----------------
    def _build_index(self, step):
        n = np.maximum(1, np.ceil(self.length / step).astype(int)) + 1
        arc_ids = np.repeat(np.arange(self.n_arcs), n)
        start = np.r_[0, np.cumsum(n)[:-1]]
        k = np.arange(n.sum()) - np.repeat(start, n)
        frac = k / np.repeat(n - 1, n).clip(min=1)
        pts = self.A[arc_ids] + (self.B - self.A)[arc_ids] * frac[:, None]
        self._sample_arc = arc_ids
        self._tree = cKDTree(pts)
        self._step = step

    def candidates(self, P: np.ndarray, radius: float = C.CAND_RADIUS_M, k: int = C.CAND_MAX_K):
        """For each point return list of (arc_ids, dist, offset) sorted by distance."""
        hits = self._tree.query_ball_point(P, r=radius + self._step / 2.0)
        out = []
        for i, h in enumerate(hits):
            if not h:
                out.append((np.empty(0, np.int64), np.empty(0), np.empty(0)))
                continue
            arcs = np.unique(self._sample_arc[h])
            Pi = np.repeat(P[i:i + 1], len(arcs), axis=0)
            d, off, _ = project_points_to_segments(Pi, self.A[arcs], self.B[arcs])
            m = d <= radius
            arcs, d, off = arcs[m], d[m], off[m]
            o = np.lexsort((arcs, d))[:k]
            out.append((arcs[o], d[o], off[o]))
        return out

    def dist_to_arc(self, P: np.ndarray, arc_ids: np.ndarray):
        d, off, _ = project_points_to_segments(P, self.A[arc_ids], self.B[arc_ids])
        return d, off

    # ---------------- connectivity -----------------
    def connected(self, a: int, b: int) -> bool:
        return a == b or self.arcs[a, 1] == self.arcs[b, 0]

    def bounded_sp(self, src: int, cutoff: float) -> dict:
        """Dijkstra from node src, exploring up to `cutoff` metres. Cached."""
        key = src
        c = self._sp_cache.get(key)
        if c is not None and c[0] >= cutoff:
            return c[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        ptr, adj, L, to = self._adj_ptr, self._adj_arc, self.length, self.arcs[:, 1]
        while pq:
            du, u = heapq.heappop(pq)
            if du > dist.get(u, np.inf) or du > cutoff:
                continue
            for e in adj[ptr[u]:ptr[u + 1]]:
                v = to[e]
                nd = du + L[e]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        if len(self._sp_cache) > 200_000:
            self._sp_cache.clear()
        self._sp_cache[key] = (cutoff, dist)
        return dist

    def arcs_lonlat(self, arc_ids):
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        a = self.nodes_ll[self.arcs[arc_ids, 0]]
        b = self.nodes_ll[self.arcs[arc_ids, 1]]
        return a, b

    def components(self):
        """Weakly / strongly connected component counts (scipy)."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        g = csr_matrix((np.ones(self.n_arcs), (self.arcs[:, 0], self.arcs[:, 1])),
                       shape=(self.n_nodes, self.n_nodes))
        nw, lw = connected_components(g, directed=True, connection="weak")
        ns, ls = connected_components(g, directed=True, connection="strong")
        return nw, lw, ns, ls
