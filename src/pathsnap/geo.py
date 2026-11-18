"""Geometry helpers.

Distances: each trip covers at most a few tens of km, so we use a local
equirectangular (tangent-plane) projection centred on the trip. Over
<50 km the distortion vs. true geodesic distance is well below GPS noise
(<0.1 %), and it lets us use fast Euclidean / vector maths in metres.
"""
import numpy as np

R_EARTH = 6_371_008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = np.cos(np.radians(lat0)) * R_EARTH * np.pi / 180.0
        self.ky = R_EARTH * np.pi / 180.0

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=np.float64)
        return np.column_stack(((lonlat[:, 0] - self.lon0) * self.kx,
                                (lonlat[:, 1] - self.lat0) * self.ky))

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return np.column_stack((xy[:, 0] / self.kx + self.lon0,
                                xy[:, 1] / self.ky + self.lat0))


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def project_points_to_segments(P, A, B):
    """Project points P (n,2) onto segments A->B (n,2) element-wise.
    Returns (distance, offset_along_segment_m, segment_length)."""
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    L = np.sqrt(L2)
    t = np.einsum("ij,ij->i", P - A, AB) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    proj = A + AB * t[:, None]
    d = np.linalg.norm(P - proj, axis=1)
    return d, t * L, L


def bearing_deg(dxdy: np.ndarray) -> np.ndarray:
    """Compass-style bearing (0 = north, clockwise) of vectors in local xy."""
    return (np.degrees(np.arctan2(dxdy[:, 0], dxdy[:, 1])) + 360.0) % 360.0


def angle_diff_deg(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
