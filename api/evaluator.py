"""Server-side scoring. The only place the API reads the reference route.

Called AFTER prediction is complete; returns aggregate metrics only (the
reference route / labels are never returned to the client).
"""
import numpy as np

from pathsnap.io import load_trip
from pathsnap.labels import align_track_to_route
from pathsnap.metrics import route_continuity, route_set_metrics
from pathsnap import config as C


def score(eval_dir, trip_id, td, clf_pred, hmm_pred):
    raw = load_trip(eval_dir, trip_id, with_route=True)
    out = dict(n_fixes=len(td.xy),
               fixes_without_candidates=int(sum(len(c[0]) == 0 for c in td.cands)))
    if raw.route is None:
        out["note"] = "no reference route on server; accuracy unavailable"
        return out
    labels, _, ldist = align_track_to_route(td.xy, td.net, raw.route)
    matchable = ldist <= C.CAND_RADIUS_M
    for tag, pred in (("classifier", clf_pred), ("hmm", hmm_pred)):
        if pred is None:
            continue
        ok = pred == labels
        out[f"top1_accuracy_{tag}"] = round(float(ok.mean()), 4)
        out[f"top1_accuracy_matchable_{tag}"] = round(float(ok[matchable].mean()), 4) if matchable.any() else None
        out[f"route_continuity_{tag}"] = round(route_continuity(pred, raw.arcs), 4)
        for k, v in route_set_metrics(pred, raw.route, td.net.length).items():
            out[f"{k}_{tag}"] = round(float(v), 4)
    return out
