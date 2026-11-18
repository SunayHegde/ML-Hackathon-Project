"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health              liveness + loaded model / decoder config
  GET  /trips               evaluation trip ids available on the server (no labels)
  POST /predict {trip_id}   full pipeline: raw files -> candidates -> classifier
                            -> HMM/Viterbi; then server-side scoring against the
                            reference route. Returns predicted route + metrics.
  GET  /map/{trip_id}       folium HTML of the latest prediction for a trip

Ground truth (.route) is read ONLY inside evaluator.score() after prediction;
it is never passed to the predictor and never returned to the client.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR  folder with the evaluation trips (default data/student_test)
"""
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pathsnap import config as C
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import PathSnapPredictor
from pathsnap.viz import route_map
from api.evaluator import score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", C.STUDENT_TEST_DIR))
app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = PathSnapPredictor()
_last = {}   # trip_id -> (TripData, prediction) for /map


class PredictRequest(BaseModel):
    trip_id: str
    use_hmm: bool = True


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info(f"{request.client.host if request.client else '-'} {request.method} {request.url.path} "
             f"-> {resp.status_code} ({(time.perf_counter()-t0)*1000:.0f} ms)")
    return resp


def _check_trip(trip_id: str):
    trips = list_trips(EVAL_DIR)
    if trip_id not in trips:
        raise HTTPException(404, f"unknown trip_id {trip_id!r}")


@app.get("/health")
def health():
    return dict(status="ok", eval_dir=str(EVAL_DIR), model=type(predictor.model).__name__,
                hmm=predictor.hmm.to_dict())


@app.get("/trips")
def trips():
    out = []
    for t in list_trips(EVAL_DIR):
        n = sum(1 for _ in open(EVAL_DIR / t / f"{t}.track"))
        out.append(dict(trip_id=t, n_fixes=n))
    return dict(eval_dir=EVAL_DIR.name, trips=out)


def _polylines(net, seq):
    lines, cur, last = [], [], None
    for a in seq:
        a = int(a)
        if a < 0 or a == last:
            continue
        p = net.nodes_ll[net.arcs[a, 0]]; q = net.nodes_ll[net.arcs[a, 1]]
        if cur and last is not None and net.arcs[last, 1] == net.arcs[a, 0]:
            cur.append([round(q[1], 6), round(q[0], 6)])
        else:
            if cur:
                lines.append(cur)
            cur = [[round(p[1], 6), round(p[0], 6)], [round(q[1], 6), round(q[0], 6)]]
        last = a
    if cur:
        lines.append(cur)
    return lines


@app.post("/predict")
def predict(req: PredictRequest):
    _check_trip(req.trip_id)
    log.info(f"PREDICT request trip={req.trip_id} use_hmm={req.use_hmm}")
    t0 = time.perf_counter()
    raw = load_trip(EVAL_DIR, req.trip_id, with_route=False)        # inference input: no labels
    out = predictor.run(raw, use_hmm=req.use_hmm)
    td = out["td"]
    final = out["hmm_pred"] if req.use_hmm else out["clf_pred"]
    t_pred = time.perf_counter() - t0
    metrics = score(EVAL_DIR, req.trip_id, td, out["clf_pred"], out["hmm_pred"])  # server-side only
    _last[req.trip_id] = (td, final)
    log.info(f"PREDICT done trip={req.trip_id} fixes={len(td.xy)} "
             f"top1_hmm={metrics.get('top1_accuracy_hmm')} top1_clf={metrics.get('top1_accuracy_classifier')} "
             f"route_f1={metrics.get('route_f1_hmm')} time={t_pred:.2f}s")
    step = max(1, len(td.lonlat) // 3000)
    return dict(
        trip_id=req.trip_id,
        n_fixes=len(td.xy),
        gps=[[round(lat, 6), round(lon, 6)] for lon, lat in td.lonlat[::step]],
        predicted_route=_polylines(td.net, final),
        classifier_route=_polylines(td.net, out["clf_pred"]),
        predicted_arc_per_fix=[int(x) for x in final],
        metrics=metrics,
        timing=dict(classifier_s=round(out["t_classifier"], 3), hmm_s=round(out["t_hmm"], 3),
                    total_s=round(t_pred, 3)),
        hmm_breaks=out["diag"].get("breaks"),
    )


@app.get("/map/{trip_id}", response_class=HTMLResponse)
def trip_map(trip_id: str):
    _check_trip(trip_id)
    if trip_id not in _last:
        raise HTTPException(404, "run /predict for this trip first")
    td, final = _last[trip_id]
    m = route_map(td, {"Predicted route (classifier + HMM)": (list(final), "#d81b60", 5, None)},
                  f"PathSnap prediction - trip {trip_id}")
    return m.get_root().render()
