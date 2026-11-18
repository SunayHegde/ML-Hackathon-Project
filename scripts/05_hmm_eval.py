"""Stage 5: HMM / Viterbi route reconstruction - tuning and evaluation.

1. Decoder parameters (beta, w_emission) are tuned by grid search on TRAINING
   trips using OUT-OF-FOLD classifier probabilities (cache/oof_train.parquet from
   04_bayes_opt.py), so emissions are as (un)confident as on unseen data.
2. The tuned decoder is evaluated on the 10 student-testing trips with the final
   tuned model, comparing classifier-only vs classifier + HMM:
   observation Top-1 accuracy, route continuity, length-weighted route
   precision/recall/F1 (sequence level), chain breaks and decode time.

Outputs: results/hmm_grid.csv, configs/hmm_params.json,
         results/hmm_vs_classifier_student_test.csv, results/figures/hmm_*.png

Usage: python scripts/05_hmm_eval.py [--tune-trips 25]
"""
import argparse, sys, time, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathsnap import config as C
from pathsnap.features import prepare_trip, build_features, FEATURES
from pathsnap.hmm import HMMParams, prune, transition_tables, decode
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import route_continuity, route_set_metrics
from pathsnap.pipeline import candidate_probs, classifier_only

BETAS = [2, 5, 10, 20, 40, 80]
WEIGHTS = [0.25, 0.5, 1.0, 2.0]


def probs_from_table(td, tab):
    key = dict(zip(zip(tab["obs"].to_numpy(), tab["arc"].to_numpy()), tab["p"].to_numpy()))
    return [np.array([key.get((i, int(x)), 0.0) for x in a]) for i, (a, _, _) in enumerate(td.cands)]


def evaluate_trip(td, raw, probs, params, cache=None):
    clf = classifier_only(td, probs)
    t0 = time.perf_counter()
    if cache is None:
        cands, pr = prune(td.cands, probs, params.max_states)
        valid, tables = transition_tables(td.net, td.xy, td.t, cands, params)
        cache = (cands, pr, valid, tables)
    cands, pr, valid, tables = cache
    hmm, diag = decode(cands, pr, valid, tables, params)
    tdec = time.perf_counter() - t0
    out = dict(trip=td.trip_id, n_obs=len(td.xy),
               clf_top1=float(np.mean(clf == td.label)), hmm_top1=float(np.mean(hmm == td.label)),
               clf_continuity=route_continuity(clf, raw.arcs), hmm_continuity=route_continuity(hmm, raw.arcs),
               hmm_breaks=diag["breaks"], decode_s=tdec)
    for tag, pred in (("clf", clf), ("hmm", hmm)):
        for k, v in route_set_metrics(pred, raw.route, td.net.length).items():
            out[f"{tag}_{k}"] = v
    return out, cache, clf, hmm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-trips", type=int, default=25)
    a = ap.parse_args()
    rng = np.random.default_rng(C.SEED)

    # ---------------- 1. tune on training trips (OOF probabilities) ----------------
    oof = pd.read_parquet(C.CACHE_DIR / "oof_train.parquet")
    st = pd.read_csv(C.RESULTS_DIR / "candidate_stats_train.csv", dtype={"trip": str})
    st["trip"] = st.trip.str.zfill(8)
    pool = st[(st.n_obs <= 4000)].trip.tolist()          # keep tuning time reasonable
    tune = sorted(rng.choice(pool, size=min(a.tune_trips, len(pool)), replace=False))
    print("tuning trips:", tune)
    base = HMMParams()
    prepared = []
    for tid in tune:
        raw = load_trip(C.TRAIN_DIR, tid)
        td = prepare_trip(raw)
        probs = probs_from_table(td, oof[oof.trip == tid])
        _, cache, _, _ = evaluate_trip(td, raw, probs, base)
        prepared.append((td, raw, probs, cache))
    grid = []
    for beta, w in itertools.product(BETAS, WEIGHTS):
        p = HMMParams(beta=beta, w_emission=w)
        rows = [evaluate_trip(td, raw, probs, p, cache)[0] for td, raw, probs, cache in prepared]
        r = pd.DataFrame(rows)
        n = r.n_obs.sum()
        grid.append(dict(beta=beta, w_emission=w,
                         clf_top1=(r.clf_top1 * r.n_obs).sum() / n, hmm_top1=(r.hmm_top1 * r.n_obs).sum() / n,
                         hmm_continuity=r.hmm_continuity.mean(), hmm_route_f1=r.hmm_route_f1.mean()))
        print(grid[-1])
    grid = pd.DataFrame(grid)
    grid.to_csv(C.RESULTS_DIR / "hmm_grid.csv", index=False)
    best = grid.sort_values(["hmm_top1", "hmm_route_f1"], ascending=False).iloc[0]
    params = HMMParams(beta=float(best.beta), w_emission=float(best.w_emission))
    json.dump(params.to_dict(), open(C.CONFIGS_DIR / "hmm_params.json", "w"), indent=2)
    print("BEST decoder params:", params)

    piv = grid.pivot(index="beta", columns="w_emission", values="hmm_top1")
    plt.figure(figsize=(6, 4)); plt.imshow(piv.values, cmap="viridis", aspect="auto")
    plt.xticks(range(len(piv.columns)), piv.columns); plt.yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            plt.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    plt.xlabel("emission weight w_e"); plt.ylabel("beta (m)")
    plt.title(f"HMM Top-1 on {len(tune)} training trips (OOF probs)"); plt.colorbar()
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "hmm_grid.png", dpi=120); plt.close()

    # ---------------- 2. evaluate on student-testing trips ----------------
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    rows = []
    for tid in list_trips(C.STUDENT_TEST_DIR):
        raw = load_trip(C.STUDENT_TEST_DIR, tid)
        td = prepare_trip(raw)
        df = build_features(td)
        t0 = time.perf_counter()
        p = model.predict_proba(df[FEATURES])[:, 1]
        tclf = time.perf_counter() - t0
        probs = candidate_probs(td, df, p)
        r, _, clf, hmm = evaluate_trip(td, raw, probs, params)
        r["classifier_s"] = tclf
        rows.append(r)
        np.save(C.CACHE_DIR / f"pred_{tid}.npy", np.vstack([clf, hmm]))
        print(r)
    res = pd.DataFrame(rows)
    n = res.n_obs.sum()
    tot = dict(trip="ALL (fix-weighted)", n_obs=n,
               clf_top1=(res.clf_top1 * res.n_obs).sum() / n, hmm_top1=(res.hmm_top1 * res.n_obs).sum() / n)
    for c in res.columns:
        if c not in tot and c != "trip":
            tot[c] = res[c].mean() if c not in ("hmm_breaks", "decode_s", "classifier_s") else res[c].sum()
    res = pd.concat([res, pd.DataFrame([tot])], ignore_index=True)
    res.to_csv(C.RESULTS_DIR / "hmm_vs_classifier_student_test.csv", index=False)
    print(res.tail(1).T)

    r = res.iloc[:-1].set_index("trip")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    r[["clf_top1", "hmm_top1"]].plot(kind="bar", ax=ax[0], ylim=(0.7, 1.0), title="Top-1 accuracy per student-test trip")
    r[["clf_continuity", "hmm_continuity"]].plot(kind="bar", ax=ax[1], ylim=(0.8, 1.01), title="Route continuity")
    r[["clf_route_f1", "hmm_route_f1"]].plot(kind="bar", ax=ax[2], ylim=(0.5, 1.01), title="Route F1 (length-weighted)")
    for x in ax: x.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "hmm_vs_classifier.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
