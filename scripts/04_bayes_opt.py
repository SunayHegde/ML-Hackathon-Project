"""Stage 4: Bayesian hyper-parameter optimisation of the selected model (LightGBM).

Search  : Optuna TPE sampler (sequential model-based / Bayesian optimisation),
          median pruner across folds.
Objective: maximise mean candidate-level macro F1 over a 3-fold GroupKFold
          (groups = trips) on the 80 training trips only.
After search: refit tuned model on all training trips, compare untuned vs tuned
with the SAME 5-fold GroupKFold CV + student-test evaluation, save the final model
and out-of-fold probabilities (used to tune the HMM without leakage).

Outputs: results/optuna_trials.csv, configs/best_params.json,
         results/tuned_vs_untuned.csv, results/figures/optuna_*.png,
         models/final_model.joblib, cache/oof_train.parquet

Usage: python scripts/04_bayes_opt.py [--trials 30]
"""
import argparse, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from pathsnap import config as C
from pathsnap.features import FEATURES
from pathsnap.io import load_trip
from pathsnap.metrics import route_continuity
from pathsnap.models import MODEL_ZOO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
tm = import_module("03_train_models")

SEARCH_SPACE = {
    "n_estimators": ("int", 100, 800),
    "learning_rate": ("float_log", 0.01, 0.2),
    "num_leaves": ("int", 15, 255),
    "min_child_samples": ("int", 10, 300),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.4, 1.0),
    "reg_lambda": ("float_log", 1e-3, 10.0),
    "reg_alpha": ("float_log", 1e-3, 10.0),
}


def suggest(trial):
    p = {}
    for k, (kind, lo, hi) in SEARCH_SPACE.items():
        if kind == "int":
            p[k] = trial.suggest_int(k, lo, hi)
        elif kind == "float":
            p[k] = trial.suggest_float(k, lo, hi)
        else:
            p[k] = trial.suggest_float(k, lo, hi, log=True)
    return p


def make(params):
    return LGBMClassifier(**params, subsample_freq=1, random_state=C.SEED, n_jobs=-1, verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--search-frac", type=float, default=1.0,
                    help="fraction of training trips' fixes used during the search (speed)")
    ap.add_argument("--cmp-folds", type=int, default=5, help="GroupKFold folds for tuned-vs-untuned")
    a = ap.parse_args()

    tr, st_tr = tm.load("train")
    te, st_te = tm.load("student_test")
    reliable = (tr.label_dist <= C.CAND_RADIUS_M).to_numpy()
    groups = tr["trip"].to_numpy()
    # optional speed-up: search on a random subset of fixes (whole candidate sets kept)
    if a.search_frac < 1.0:
        keys = tr[["trip", "obs"]].drop_duplicates().sample(frac=a.search_frac, random_state=C.SEED)
        srch = tr.reset_index().merge(keys, on=["trip", "obs"]).set_index("index")
    else:
        srch = tr
    s_rel = (srch.label_dist <= C.CAND_RADIUS_M).to_numpy()
    folds3 = list(GroupKFold(n_splits=3).split(srch, groups=srch["trip"].to_numpy()))

    def objective(trial):
        params = suggest(trial)
        scores = []
        for k, (itr, iva) in enumerate(folds3):
            itr = itr[s_rel[itr]]
            m = make(params)
            m.fit(srch.iloc[itr][FEATURES], srch.iloc[itr]["y"])
            p = m.predict_proba(srch.iloc[iva][FEATURES])[:, 1]
            scores.append(f1_score(srch.iloc[iva]["y"], (p >= .5).astype(int), average="macro"))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=8)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
                                study_name="pathsnap_lightgbm")
    # seed the search with the untuned configuration as trial 0 (fair baseline)
    study.enqueue_trial(dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3, reg_alpha=1e-3))
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number:2d} {t.state.name:8s} value={t.value} "
                                                 f"best={s.best_value:.5f} ({time.time()-t0:.0f}s)", flush=True)])
    search_secs = time.time() - t0
    df_trials = study.trials_dataframe()
    df_trials.to_csv(C.RESULTS_DIR / "optuna_trials.csv", index=False)
    best = study.best_params
    json.dump(dict(model="LightGBM", best_params=best, best_cv3_macro_f1=study.best_value,
                   n_trials=len(study.trials), search_seconds=search_secs,
                   search_space={k: list(v) for k, v in SEARCH_SPACE.items()},
                   objective=f"mean macro F1, 3-fold GroupKFold on 80 train trips "
                             f"({a.search_frac:.0%} of fixes sampled)", sampler="TPE (seed 42)"),
              open(C.CONFIGS_DIR / "best_params.json", "w"), indent=2)
    print("BEST", best, study.best_value)

    # plots
    done = df_trials[df_trials.state == "COMPLETE"]
    plt.figure(figsize=(7, 4))
    plt.scatter(done.number, done.value, label="trial")
    plt.plot(done.number, done.value.cummax(), color="red", label="best so far")
    plt.xlabel("trial"); plt.ylabel("3-fold CV macro F1"); plt.title("Optuna TPE optimisation history"); plt.legend()
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "optuna_history.png", dpi=120); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(7, 4)); plt.barh(list(imp)[::-1], list(imp.values())[::-1])
        plt.title("Hyper-parameter importance (fANOVA)"); plt.tight_layout()
        plt.savefig(C.FIG_DIR / "optuna_param_importance.png", dpi=120); plt.close()
    except Exception as e:
        print("importance failed", e)

    # ---------- untuned vs tuned: same 5-fold CV + student test ----------
    folds5 = list(GroupKFold(n_splits=a.cmp_folds).split(tr, groups=groups))
    n_obs_tr = st_tr.set_index("trip")["n_obs"]
    nm = tr.loc[tr.label_dist <= C.CAND_RADIUS_M, ["trip", "obs"]].drop_duplicates().groupby("trip").size()
    nm_te = te.loc[te.label_dist <= C.CAND_RADIUS_M, ["trip", "obs"]].drop_duplicates().shape[0]
    arcs_te = {t: load_trip(C.STUDENT_TEST_DIR, t, with_route=False).arcs for t in st_te.trip}
    untuned_factory = MODEL_ZOO["LightGBM"][1]
    rows = []
    oof = np.zeros(len(tr))
    for label, factory in (("untuned", untuned_factory), ("tuned", lambda: make(best))):
        f1s, top1s, lls = [], [], []
        for itr, iva in folds5:
            itr_r = itr[reliable[itr]]
            m = factory(); m.fit(tr.iloc[itr_r][FEATURES], tr.iloc[itr_r]["y"])
            p = m.predict_proba(tr.iloc[iva][FEATURES])[:, 1]
            if label == "tuned":
                oof[iva] = p
            vt = tr.iloc[iva]["trip"].unique()
            mm, _ = tm.eval_split(tr.iloc[iva], p, n_obs_tr.loc[vt].sum(), nm.reindex(vt).fillna(0).sum())
            f1s.append(mm["macro_f1"]); top1s.append(mm["top1_acc"]); lls.append(mm["log_loss"])
        m = factory()
        t0 = time.perf_counter(); m.fit(tr[reliable][FEATURES], tr[reliable]["y"]); ttrain = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict_proba(te[FEATURES])[:, 1]; tinf = time.perf_counter() - t0
        mm, t1 = tm.eval_split(te, p, st_te.n_obs.sum(), nm_te)
        cont = np.nanmean([route_continuity(g.sort_values("obs")["arc"].to_numpy(), arcs_te[t]) for t, g in t1.groupby("trip")])
        rows.append(dict(model=f"LightGBM-{label}", cv_macro_f1=np.mean(f1s), cv_macro_f1_std=np.std(f1s),
                         cv_top1=np.mean(top1s), cv_log_loss=np.mean(lls),
                         test_top1=mm["top1_acc"], test_macro_f1=mm["macro_f1"], test_log_loss=mm["log_loss"],
                         test_precision=mm["macro_precision"], test_recall=mm["macro_recall"],
                         route_continuity=cont, train_time_s=ttrain,
                         infer_ms_per_fix=1000 * tinf / st_te.n_obs.sum()))
        print(rows[-1])
        if label == "tuned":
            joblib.dump(m, C.MODELS_DIR / "final_model.joblib", compress=3)
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / "tuned_vs_untuned.csv", index=False)
    tr[["trip", "obs", "arc", "y", "label_dist"]].assign(p=oof).to_parquet(C.CACHE_DIR / "oof_train.parquet", index=False)

    r = res.set_index("model")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    r[["cv_macro_f1", "test_macro_f1", "test_top1"]].T.plot(kind="bar", ax=ax[0], ylim=(0.8, 1.0)); ax[0].set_title("Higher is better")
    r[["cv_log_loss", "test_log_loss"]].T.plot(kind="bar", ax=ax[1]); ax[1].set_title("Log loss (lower is better)")
    r[["infer_ms_per_fix"]].T.plot(kind="bar", ax=ax[2]); ax[2].set_title("Inference ms / fix")
    for x in ax: x.tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "tuned_vs_untuned.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
