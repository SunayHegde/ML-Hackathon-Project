"""End-to-end inference pipeline: raw trip -> candidates -> classifier -> HMM."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURES, prepare_trip, build_features, TripData
from .hmm import HMMParams, viterbi
from .io import RawTrip


def candidate_probs(td: TripData, df: pd.DataFrame, p: np.ndarray):
    """Re-attach classifier probabilities to td.cands order (per fix)."""
    probs = [np.zeros(len(c[0])) for c in td.cands]
    if len(df):
        key = dict(zip(zip(df["obs"].to_numpy(), df["arc"].to_numpy()), p))
        for i, (a, _, _) in enumerate(td.cands):
            probs[i] = np.array([key.get((i, int(x)), 0.0) for x in a])
    return probs


def classifier_only(td: TripData, probs):
    return np.array([int(a[np.argmax(p)]) if len(a) else -1 for (a, _, _), p in zip(td.cands, probs)])


class PathSnapPredictor:
    def __init__(self, model_path=C.MODELS_DIR / "final_model.joblib",
                 hmm_path=C.CONFIGS_DIR / "hmm_params.json"):
        self.model = joblib.load(model_path)
        params = json.loads(Path(hmm_path).read_text()) if Path(hmm_path).exists() else {}
        self.hmm = HMMParams(**params)

    def predict_proba(self, df):
        if len(df) == 0:
            return np.zeros(0)
        return self.model.predict_proba(df[FEATURES])[:, 1]

    def run(self, raw: RawTrip, use_hmm: bool = True):
        t0 = time.perf_counter()
        td = prepare_trip(raw, with_labels=False)   # labels never enter inference
        df = build_features(td)
        p = self.predict_proba(df)
        probs = candidate_probs(td, df, p)
        t1 = time.perf_counter()
        clf_pred = classifier_only(td, probs)
        diag = {}
        hmm_pred = None
        if use_hmm:
            hmm_pred, diag = viterbi(td.net, td.xy, td.t, td.cands, probs, self.hmm)
        t2 = time.perf_counter()
        return dict(td=td, clf_pred=clf_pred, hmm_pred=hmm_pred, diag=diag,
                    t_classifier=t1 - t0, t_hmm=t2 - t1)
