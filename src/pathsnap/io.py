"""Raw-file readers for a PathSnap trip folder.

Each trip folder <id>/ contains:
  <id>.track : lon lat t          (GPS fixes, ~1 Hz)
  <id>.nodes : lon lat            (row index = node id)
  <id>.arcs  : from_node to_node  (row index = arc id, directed)
  <id>.route : arc_id             (reference route, chronological; optional)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class RawTrip:
    trip_id: str
    track: np.ndarray            # (T,3) lon, lat, t
    nodes: np.ndarray            # (N,2) lon, lat
    arcs: np.ndarray             # (M,2) from, to
    route: Optional[np.ndarray]  # (R,) arc ids or None


def _read(path: Path, cols: int, dtype) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=dtype)
    arr = df.to_numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, cols)
    return arr


def list_trips(part_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(part_dir).iterdir() if p.is_dir())


def load_trip(part_dir: Path, trip_id: str, with_route: bool = True) -> RawTrip:
    d = Path(part_dir) / trip_id
    track = _read(d / f"{trip_id}.track", 3, np.float64)
    nodes = _read(d / f"{trip_id}.nodes", 2, np.float64)
    arcs = _read(d / f"{trip_id}.arcs", 2, np.int64)
    route = None
    rp = d / f"{trip_id}.route"
    if with_route and rp.exists():
        route = _read(rp, 1, np.int64).ravel()
    return RawTrip(trip_id, track, nodes, arcs, route)


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Basic cleaning: drop NaN / out-of-range coords, sort by time,
    drop duplicate timestamps. Returns cleaned track + a report dict."""
    rep = {"raw_points": len(track)}
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    rep["invalid_dropped"] = int((~ok).sum())
    tr = track[ok]
    order = np.argsort(tr[:, 2], kind="stable")
    rep["unsorted_fixed"] = int((order != np.arange(len(order))).sum())
    tr = tr[order]
    keep = np.r_[True, np.diff(tr[:, 2]) > 0]
    rep["dup_time_dropped"] = int((~keep).sum())
    tr = tr[keep]
    rep["clean_points"] = len(tr)
    return tr, rep
