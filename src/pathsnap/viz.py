"""Folium map generation (used by scripts/06_make_maps.py and the API)."""
import numpy as np
import folium


def _merge_arcs(net, arc_seq):
    """Arc sequence -> list of polylines ([lat, lon] lists), joining connected arcs."""
    lines, cur = [], []
    last = None
    for a in arc_seq:
        if a is None or a < 0 or a == last:
            continue
        p = net.nodes_ll[net.arcs[a, 0]]; q = net.nodes_ll[net.arcs[a, 1]]
        if cur and last is not None and net.arcs[last, 1] == net.arcs[a, 0]:
            cur.append([q[1], q[0]])
        else:
            if cur:
                lines.append(cur)
            cur = [[p[1], p[0]], [q[1], q[0]]]
        last = a
    if cur:
        lines.append(cur)
    return lines


def route_map(td, layers: dict, title: str, show_network=True, gps_step=None):
    """layers: name -> (arc sequence, colour, weight, dash)."""
    ll = td.lonlat
    m = folium.Map(location=[ll[:, 1].mean(), ll[:, 0].mean()], zoom_start=15, control_scale=True)
    if show_network:
        near = np.unique(np.concatenate([c[0] for c in td.cands] + [np.empty(0, np.int64)])).astype(int)
        fg = folium.FeatureGroup(name=f"Road network (candidate arcs, {len(near)})", show=True)
        for a in near:
            p = td.net.nodes_ll[td.net.arcs[a, 0]]; q = td.net.nodes_ll[td.net.arcs[a, 1]]
            folium.PolyLine([[p[1], p[0]], [q[1], q[0]]], color="#9e9e9e", weight=2, opacity=0.5).add_to(fg)
        fg.add_to(m)
    for name, (seq, colour, weight, dash) in layers.items():
        fg = folium.FeatureGroup(name=name)
        for line in _merge_arcs(td.net, seq):
            folium.PolyLine(line, color=colour, weight=weight, opacity=0.85, dash_array=dash, tooltip=name).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup(name="Raw GPS fixes")
    step = gps_step or max(1, len(ll) // 2000)
    for i in range(0, len(ll), step):
        folium.CircleMarker([ll[i, 1], ll[i, 0]], radius=2, color="#1565c0", fill=True, fill_opacity=0.9,
                            weight=1, tooltip=f"fix {i}  t={td.t[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker([ll[0, 1], ll[0, 0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([ll[-1, 1], ll[-1, 0]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = "".join(f'<span style="color:{c}">&#9644;&#9644;</span> {n}<br>' for n, (_, c, _, _) in layers.items())
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:20px;left:10px;z-index:9999;background:white;padding:8px 12px;'
        f'border:1px solid #888;border-radius:4px;font:12px sans-serif"><b>{title}</b><br>'
        f'<span style="color:#1565c0">&#9679;</span> Raw GPS fixes<br>'
        f'<span style="color:#9e9e9e">&#9644;&#9644;</span> Road network<br>{legend}</div>'))
    m.fit_bounds([[ll[:, 1].min(), ll[:, 0].min()], [ll[:, 1].max(), ll[:, 0].max()]])
    return m
