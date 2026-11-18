"""Trip processing, candidate generation and feature engineering.

The learning problem is framed as *candidate ranking*: for every GPS fix
we generate nearby road arcs (candidates) and a classifier predicts
P(candidate is the reference arc | features). Per fix, the arg-max
candidate is the classifier-only prediction; the probabilities also
serve as HMM emission scores.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from . import config as C
from .geo import LocalProjection, bearing_deg, angle_diff_deg
from .io import RawTrip, clean_track
from .network import RoadNetwork
from .labels import align_track_to_route

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_cands",
    "dist_ctx_mean", "dist_prev", "dist_next",
    "heading_diff", "heading_diff_rank", "heading_diff_ctx", "speed", "heading_valid",
    "arc_len", "offset_frac", "from_out_deg", "to_in_deg", "two_way",
    "dt_prev", "dist_softmax", "heading_softmax",
]


@dataclass
class TripData:
    trip_id: str
    proj: LocalProjection
    net: RoadNetwork
    lonlat: np.ndarray
    xy: np.ndarray
    t: np.ndarray
    cands: list
    clean_report: dict
    label: Optional[np.ndarray] = None       # reference arc per fix
    label_dist: Optional[np.ndarray] = None  # fix-to-reference-arc distance
    route: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)


def prepare_trip(raw: RawTrip, with_labels: bool = True,
                 radius: float = C.CAND_RADIUS_M, k: int = C.CAND_MAX_K) -> TripData:
    track, rep = clean_track(raw.track)
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    proj = LocalProjection(lon0, lat0)
    net = RoadNetwork(raw.nodes, raw.arcs, proj)
    xy = proj.to_xy(track[:, :2])
    cands = net.candidates(xy, radius=radius, k=k)
    td = TripData(raw.trip_id, proj, net, track[:, :2], xy, track[:, 2], cands, rep, route=raw.route)
    if with_labels and raw.route is not None:
        td.label, _, td.label_dist = align_track_to_route(xy, net, raw.route)
    return td


def _heading(xy, t, k):
    n = len(xy)
    i0 = np.clip(np.arange(n) - k, 0, n - 1)
    i1 = np.clip(np.arange(n) + k, 0, n - 1)
    v = xy[i1] - xy[i0]
    dist = np.linalg.norm(v, axis=1)
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    return bearing_deg(v), dist / dt, dist


def build_features(td: TripData) -> pd.DataFrame:
    xy, t, net = td.xy, td.t, td.net
    T = len(xy)
    hdg, speed, hdist = _heading(xy, t, C.HEADING_HALF_WINDOW)
    heading_valid = (hdist > 3.0).astype(np.float32)   # heading meaningless when stationary
    dt_prev = np.r_[1.0, np.diff(t)]

    obs, arc, dist, off = [], [], [], []
    for i, (a, d, o) in enumerate(td.cands):
        obs.append(np.full(len(a), i)); arc.append(a); dist.append(d); off.append(o)
    if not obs:
        return pd.DataFrame(columns=["obs", "arc"] + FEATURES + ["y"])
    obs = np.concatenate(obs); arc = np.concatenate(arc)
    dist = np.concatenate(dist); off = np.concatenate(off)
    df = pd.DataFrame({"obs": obs, "arc": arc, "dist": dist, "offset": off})
    if len(df) == 0:
        return pd.DataFrame(columns=["obs", "arc"] + FEATURES + ["y"])

    g = df.groupby("obs")["dist"]
    df["dist_rank"] = g.rank(method="first") - 1
    dmin = g.transform("min")
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio_min"] = df["dist"] / (dmin + 1.0)
    df["n_cands"] = g.transform("size")

    # temporal context: distance of neighbouring fixes to the SAME arc
    W = C.CONTEXT_HALF_WINDOW
    ctx = np.zeros(len(df)); cnt = np.zeros(len(df))
    dprev = np.full(len(df), np.nan); dnext = np.full(len(df), np.nan)
    for s in range(-W, W + 1):
        j = obs + s
        valid = (j >= 0) & (j < T)
        dd = np.full(len(df), np.nan)
        dd[valid], _ = net.dist_to_arc(xy[j[valid]], arc[valid])
        ctx[valid] += dd[valid]; cnt[valid] += 1
        if s == -1: dprev = dd
        if s == 1: dnext = dd
    df["dist_ctx_mean"] = ctx / np.maximum(cnt, 1)
    df["dist_prev"] = np.where(np.isnan(dprev), df["dist"], dprev)
    df["dist_next"] = np.where(np.isnan(dnext), df["dist"], dnext)

    hd = angle_diff_deg(hdg[obs], net.bearing[arc])
    df["heading_diff"] = np.where(heading_valid[obs] > 0, hd, 90.0)
    df["heading_diff_rank"] = df.groupby("obs")["heading_diff"].rank(method="first") - 1
    # heading agreement over a wider window (robust to single noisy fix)
    hdg2, _, hdist2 = _heading(xy, t, 2 * C.HEADING_HALF_WINDOW + 1)
    hd2 = angle_diff_deg(hdg2[obs], net.bearing[arc])
    df["heading_diff_ctx"] = np.where(hdist2[obs] > 3.0, hd2, 90.0)
    df["speed"] = speed[obs]
    df["heading_valid"] = heading_valid[obs]

    df["arc_len"] = net.length[arc]
    df["offset_frac"] = df["offset"] / np.maximum(df["arc_len"], 1e-6)
    df["from_out_deg"] = net.out_deg[net.arcs[arc, 0]]
    df["to_in_deg"] = net.in_deg[net.arcs[arc, 1]]
    df["two_way"] = net.has_reverse[arc].astype(np.float32)
    df["dt_prev"] = dt_prev[obs]

    # within-fix normalised scores (Gaussian distance / von-Mises-like heading)
    sd = np.exp(-0.5 * (df["dist_ctx_mean"] / 10.0) ** 2)
    df["dist_softmax"] = sd / sd.groupby(df["obs"]).transform("sum").clip(lower=1e-12)
    sh = np.exp(np.cos(np.radians(df["heading_diff"])) * 2.0)
    df["heading_softmax"] = sh / sh.groupby(df["obs"]).transform("sum")

    if td.label is not None:
        df["y"] = (arc == td.label[obs]).astype(np.int8)
    else:
        df["y"] = -1
    df["trip"] = td.trip_id
    for c in FEATURES:
        df[c] = df[c].astype(np.float32)
    return df
