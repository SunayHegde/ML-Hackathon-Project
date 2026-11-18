"""Stage 1: dataset inspection & exploratory data analysis.

Produces (partition is stated in every file name):
  results/eda_inventory_<part>.csv      per-trip inventory: files, sizes, counts,
                                        missing values, time gaps, extents,
                                        network size/connectivity, route checks
  results/eda_summary.csv               partition-level summary
  results/data_quality_log.md           data-quality issues & decisions
  results/figures/eda_*.png             plots
  maps/eda_trip_<id>.html               folium map: GPS + network + reference route

Usage: python scripts/01_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium

from pathsnap import config as C
from pathsnap.io import list_trips, load_trip
from pathsnap.geo import LocalProjection, haversine_m
from pathsnap.network import RoadNetwork

PARTS = {"train": C.TRAIN_DIR, "student_test": C.STUDENT_TEST_DIR}


def inspect_trip(part_dir, tid):
    d = Path(part_dir) / tid
    files = {ext: (d / f"{tid}.{ext}") for ext in ("track", "nodes", "arcs", "route")}
    raw = load_trip(part_dir, tid)
    tr = raw.track
    row = dict(trip=tid)
    for ext, p in files.items():
        row[f"has_{ext}"] = p.exists()
        row[f"size_kb_{ext}"] = round(p.stat().st_size / 1024, 1) if p.exists() else 0
    row["n_obs"] = len(tr)
    row["nan_values"] = int(np.isnan(tr).sum())
    row["invalid_coords"] = int(((np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum())
    dt = np.diff(tr[:, 2])
    row["non_increasing_ts"] = int((dt <= 0).sum())
    row["dt_frac_1s"] = float(np.mean(np.isclose(dt, 1.0)))
    row["dt_max"] = float(dt.max()) if len(dt) else 0
    row["gaps_gt_10s"] = int((dt > 10).sum())
    step = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    speed = step / np.maximum(dt, 1e-6)
    row["duration_s"] = float(tr[-1, 2] - tr[0, 2])
    row["length_km"] = float(step.sum() / 1000)
    row["speed_p95_kmh"] = float(np.percentile(speed, 95) * 3.6)
    row["jumps_gt_60ms"] = int((speed > 60).sum())        # > 216 km/h -> implausible
    row["stationary_frac"] = float(np.mean(step < 0.5))
    row["lon_min"], row["lon_max"] = tr[:, 0].min(), tr[:, 0].max()
    row["lat_min"], row["lat_max"] = tr[:, 1].min(), tr[:, 1].max()
    row["extent_km"] = float(haversine_m(row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"]) / 1000)
    proj = LocalProjection(tr[:, 0].mean(), tr[:, 1].mean())
    net = RoadNetwork(raw.nodes, raw.arcs, proj)
    row["n_nodes"], row["n_arcs"] = net.n_nodes, net.n_arcs
    used = np.zeros(net.n_nodes, bool); used[raw.arcs.ravel()] = True
    row["isolated_nodes"] = int((~used).sum())
    nw, lw, ns, ls = net.components()
    row["weak_components"] = nw
    row["largest_weak_comp_frac"] = float(np.bincount(lw).max() / net.n_nodes)
    row["strong_components"] = ns
    row["largest_strong_comp_frac"] = float(np.bincount(ls).max() / net.n_nodes)
    row["two_way_arc_frac"] = float(net.has_reverse.mean())
    row["arc_len_median_m"] = float(np.median(net.length))
    row["zero_len_arcs"] = int((net.length < 0.01).sum())
    if raw.route is not None:
        r = raw.route
        row["route_len_arcs"] = len(r)
        row["route_len_km"] = float(net.length[r].sum() / 1000)
        row["route_connected_frac"] = float(np.mean(raw.arcs[r[:-1], 1] == raw.arcs[r[1:], 0])) if len(r) > 1 else 1.0
        row["route_repeated_arcs"] = int(len(r) - len(np.unique(r)))
        row["route_ids_valid"] = bool((r >= 0).all() and (r < net.n_arcs).all())
    return row, raw, net, proj


def folium_trip_map(raw, net, proj, out, title):
    tr = raw.track
    center = [tr[:, 1].mean(), tr[:, 0].mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    # road network near the track
    xy = proj.to_xy(tr[:, :2])
    near = np.unique(np.concatenate([c[0] for c in net.candidates(xy[::3], radius=150, k=60)]))
    fg_net = folium.FeatureGroup(name=f"Road network near track ({len(near)} arcs)")
    a, b = net.arcs_lonlat(near)
    for p, q in zip(a, b):
        folium.PolyLine([[p[1], p[0]], [q[1], q[0]]], color="#888888", weight=2, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)
    if raw.route is not None:
        fg_r = folium.FeatureGroup(name="Reference route (.route)")
        a, b = net.arcs_lonlat(raw.route)
        for p, q in zip(a, b):
            folium.PolyLine([[p[1], p[0]], [q[1], q[0]]], color="green", weight=5, opacity=0.8).add_to(fg_r)
        fg_r.add_to(m)
    fg_g = folium.FeatureGroup(name="Raw GPS fixes (.track)")
    for lon, lat, t in tr[:: max(1, len(tr) // 1500)]:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, fill_opacity=0.8,
                            tooltip=f"t={t:.0f}s").add_to(fg_g)
    fg_g.add_to(m)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:6px 10px;'
        f'border:1px solid #999;font:13px sans-serif"><b>{title}</b><br>'
        f'<span style="color:blue">&#9679;</span> GPS fixes &nbsp; <span style="color:green">&#9644;</span> reference route '
        f'&nbsp; <span style="color:#888">&#9644;</span> road network</div>'))
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(str(out))


def main():
    rows = []
    samples = {}
    all_dt, all_speed = {}, {}
    for part, pdir in PARTS.items():
        prow = []
        for tid in list_trips(pdir):
            row, raw, net, proj = inspect_trip(pdir, tid)
            row["partition"] = part
            prow.append(row)
            tr = raw.track
            dt = np.diff(tr[:, 2])
            all_dt.setdefault(part, []).append(dt)
            st = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
            all_speed.setdefault(part, []).append(st / np.maximum(dt, 1e-6) * 3.6)
            if tid in ("00000001", "00000000", "00000003"):
                samples[(part, tid)] = (raw, net, proj)
            print(part, tid, row["n_obs"], row["n_arcs"])
        df = pd.DataFrame(prow)
        df.to_csv(C.RESULTS_DIR / f"eda_inventory_{part}.csv", index=False)
        rows.append(df)
    inv = pd.concat(rows)
    summ = inv.groupby("partition").agg(
        trips=("trip", "count"), obs_total=("n_obs", "sum"), obs_median=("n_obs", "median"),
        obs_min=("n_obs", "min"), obs_max=("n_obs", "max"), nan_values=("nan_values", "sum"),
        invalid_coords=("invalid_coords", "sum"), non_increasing_ts=("non_increasing_ts", "sum"),
        dt_frac_1s_mean=("dt_frac_1s", "mean"), gaps_gt_10s=("gaps_gt_10s", "sum"),
        jumps_gt_60ms=("jumps_gt_60ms", "sum"), nodes_median=("n_nodes", "median"),
        arcs_median=("n_arcs", "median"), isolated_nodes=("isolated_nodes", "sum"),
        route_connected_frac_mean=("route_connected_frac", "mean"),
        all_have_route=("has_route", "all"))
    summ.to_csv(C.RESULTS_DIR / "eda_summary.csv")
    print(summ.T)

    # ---------------- plots -----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part in PARTS:
        sub = inv[inv.partition == part]
        ax[0].hist(sub.n_obs, bins=30, alpha=0.6, label=part)
        ax[1].hist(sub.length_km, bins=30, alpha=0.6, label=part)
        ax[2].hist(sub.n_arcs, bins=30, alpha=0.6, label=part)
    ax[0].set(title="GPS fixes per trip", xlabel="# fixes"); ax[1].set(title="Track length per trip", xlabel="km")
    ax[2].set(title="Road arcs per trip network", xlabel="# arcs")
    for a in ax: a.legend()
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "eda_trip_size_distributions.png", dpi=120); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    dt = np.concatenate(all_dt["train"])
    ax[0].hist(np.clip(dt, 0, 20), bins=40, log=True)
    ax[0].set(title=f"Sampling interval (train, n={len(dt):,}); {np.mean(np.isclose(dt,1)):.1%} exactly 1 s",
              xlabel="dt (s, clipped at 20)")
    sp = np.concatenate(all_speed["train"])
    ax[1].hist(np.clip(sp, 0, 200), bins=80, log=True)
    ax[1].set(title="Implied speed between fixes (train)", xlabel="km/h (clipped 200)")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "eda_sampling_and_speed_train.png", dpi=120); plt.close()

    # missing-value analysis
    mv = inv.groupby("partition")[["nan_values", "invalid_coords", "non_increasing_ts", "gaps_gt_10s", "jumps_gt_60ms"]].sum()
    mv.plot(kind="bar", figsize=(9, 4), logy=False, title="Missing / invalid values & anomalies per partition")
    plt.ylabel("count"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(C.FIG_DIR / "eda_missing_values.png", dpi=120); plt.close()

    # geographic distribution of trips
    plt.figure(figsize=(10, 5))
    for part, mk in (("train", "o"), ("student_test", "s")):
        sub = inv[inv.partition == part]
        plt.scatter((sub.lon_min + sub.lon_max) / 2, (sub.lat_min + sub.lat_max) / 2, marker=mk, label=part, alpha=.7)
    plt.xlabel("longitude"); plt.ylabel("latitude"); plt.title("Trip centroids (all partitions given)"); plt.legend()
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "eda_trip_centroids.png", dpi=120); plt.close()

    # trajectory + road network plot for sample trips
    for (part, tid), (raw, net, proj) in samples.items():
        xy = proj.to_xy(raw.track[:, :2])
        fig, ax = plt.subplots(figsize=(8, 8))
        pad = 300
        m = ((net.A[:, 0] > xy[:, 0].min() - pad) & (net.A[:, 0] < xy[:, 0].max() + pad) &
             (net.A[:, 1] > xy[:, 1].min() - pad) & (net.A[:, 1] < xy[:, 1].max() + pad))
        for a, b in zip(net.A[m], net.B[m]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbbbbb", lw=0.6)
        if raw.route is not None:
            for a, b in zip(net.A[raw.route], net.B[raw.route]):
                ax.plot([a[0], b[0]], [a[1], b[1]], color="green", lw=2.5)
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c=raw.track[:, 2], cmap="viridis", zorder=3)
        ax.set_aspect("equal"); ax.set_title(f"[{part}] trip {tid}: GPS (colour=time), route (green), network (grey)")
        ax.set_xlabel("x (m, local projection)"); ax.set_ylabel("y (m)")
        plt.tight_layout(); plt.savefig(C.FIG_DIR / f"eda_trajectory_{part}_{tid}.png", dpi=120); plt.close()
        folium_trip_map(raw, net, proj, C.MAPS_DIR / f"eda_trip_{part}_{tid}.html",
                        f"[{part}] trip {tid} - raw data")

    # network connectivity plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(inv.largest_strong_comp_frac, bins=20); ax[0].set(title="Largest strongly-connected comp. (frac of nodes)")
    ax[1].hist(inv.two_way_arc_frac, bins=20); ax[1].set(title="Fraction of two-way arcs per network")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "eda_network_connectivity.png", dpi=120); plt.close()
    print("EDA done")


if __name__ == "__main__":
    main()
