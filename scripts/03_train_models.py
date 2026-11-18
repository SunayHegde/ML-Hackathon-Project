"""Stage 3: train + compare >= 5 models.

Protocol
  * 80 training trips, 5-fold GroupKFold (groups = trip) -> CV metrics.
    Training rows exclude fixes whose reference arc is > CAND_RADIUS away
    (off-network / parked segments -> unreliable labels). Validation uses ALL
    rows of the held-out trips.
  * Refit on all 80 trips -> evaluate on the 10 student-testing trips.
  * Metrics: candidate-level macro F1 / precision / recall / log loss,
    observation-level Top-1 accuracy (fixes without candidates count wrong),
    route continuity of classifier-only predictions, train / inference time.

Outputs: results/model_cv_folds.csv, results/model_comparison.csv,
         results/figures/*.png, models/<name>.joblib

Usage: python scripts/03_train_models.py [--models LightGBM MLP ...]
"""
import argparse, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from pathsnap import config as C
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics, top1_predictions, route_continuity
from pathsnap.models import MODEL_ZOO
from pathsnap.io import load_trip


def load(part):
    df = pd.read_parquet(C.CACHE_DIR / f"features_{part}.parquet")
    st = pd.read_csv(C.RESULTS_DIR / f"candidate_stats_{part}.csv", dtype={"trip": str})
    st["trip"] = st["trip"].str.zfill(8)
    return df, st


def subsample(df, n, rng):
    if n is None or len(df) <= n:
        return df
    # sample whole fixes so every fix keeps its full candidate set
    keys = df[["trip", "obs"]].drop_duplicates()
    frac = n / len(df)
    keys = keys.sample(frac=frac, random_state=int(rng.integers(1e9)))
    return df.merge(keys, on=["trip", "obs"])


def eval_split(df, p, n_obs, n_obs_matchable):
    m = candidate_metrics(df["y"].to_numpy(), p)
    t1 = top1_predictions(df, p)
    m["top1_acc"] = t1["y"].sum() / n_obs
    matchable = df.loc[df.label_dist <= C.CAND_RADIUS_M, ["trip", "obs"]].drop_duplicates()
    t1m = t1.merge(matchable, on=["trip", "obs"])
    m["top1_acc_matchable"] = t1m["y"].sum() / max(n_obs_matchable, 1)
    return m, t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_ZOO))
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(C.SEED)

    tr, st_tr = load("train")
    te, st_te = load("student_test")
    n_obs_tr = st_tr.set_index("trip")["n_obs"]
    reliable = tr.label_dist <= C.CAND_RADIUS_M
    print(f"train rows {len(tr):,} (reliable {reliable.sum():,}); student-test rows {len(te):,}")

    # matchable fix counts per trip (reference arc within radius)
    def n_match(df):
        return df.loc[df.label_dist <= C.CAND_RADIUS_M, ["trip", "obs"]].drop_duplicates().groupby("trip").size()
    nm_tr = n_match(tr)

    gkf = GroupKFold(n_splits=a.folds)
    trips = tr["trip"].to_numpy()
    folds = list(gkf.split(tr, groups=trips))

    arcs_te = {t: load_trip(C.STUDENT_TEST_DIR, t, with_route=False).arcs for t in st_te.trip}

    fold_rows, summary = [], []
    for name in a.models:
        family, factory, max_rows, cfg = MODEL_ZOO[name]
        print(f"\n===== {name} ({family}) : {cfg}")
        for k, (itr, iva) in enumerate(folds):
            dtr = subsample(tr.iloc[itr][reliable.iloc[itr].to_numpy()], max_rows, rng)
            dva = tr.iloc[iva]
            model = factory()
            t0 = time.perf_counter()
            model.fit(dtr[FEATURES], dtr["y"])
            ttrain = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = model.predict_proba(dva[FEATURES])[:, 1]
            tinf = time.perf_counter() - t0
            va_trips = dva["trip"].unique()
            m, _ = eval_split(dva, p, n_obs_tr.loc[va_trips].sum(), nm_tr.reindex(va_trips).fillna(0).sum())
            # training-fold fit (overfitting check) on a sample of the training rows
            dfit = subsample(dtr, 200_000, rng)
            m["train_macro_f1"] = f1_score(dfit["y"], (model.predict_proba(dfit[FEATURES])[:, 1] >= .5).astype(int), average="macro")
            m.update(model=name, fold=k, train_time_s=ttrain, infer_time_s=tinf, n_train_rows=len(dtr))
            fold_rows.append(m)
            print(f"  fold {k}: macroF1 {m['macro_f1']:.4f} (train {m['train_macro_f1']:.4f}) top1 {m['top1_acc']:.4f} "
                  f"logloss {m['log_loss']:.4f}  train {ttrain:.1f}s")

        # ---- refit on all training trips, evaluate on student test
        dtr = subsample(tr[reliable], max_rows, rng)
        model = factory()
        t0 = time.perf_counter()
        if name == "LightGBM":
            # hold out 10% of training trips purely to record a real train/valid loss curve
            ut = np.array(sorted(dtr.trip.unique())); hold = set(rng.choice(ut, size=8, replace=False))
            mh = dtr.trip.isin(hold)
            model.fit(dtr.loc[~mh, FEATURES], dtr.loc[~mh, "y"],
                      eval_set=[(dtr.loc[~mh, FEATURES], dtr.loc[~mh, "y"]), (dtr.loc[mh, FEATURES], dtr.loc[mh, "y"])],
                      eval_names=["train", "valid"], eval_metric="binary_logloss")
            hist = model.evals_result_
            plt.figure(figsize=(6, 4))
            plt.plot(hist["train"]["binary_logloss"], label="train (72 trips)")
            plt.plot(hist["valid"]["binary_logloss"], label="valid (8 held-out trips)")
            plt.xlabel("boosting iteration"); plt.ylabel("binary log loss"); plt.title("LightGBM training history"); plt.legend()
            plt.tight_layout(); plt.savefig(C.FIG_DIR / "curve_LightGBM.png", dpi=120); plt.close()
            pd.DataFrame({k: v["binary_logloss"] for k, v in hist.items()}).to_csv(C.LOG_DIR / "lightgbm_history.csv", index_label="iter")
            model = factory(); model.fit(dtr[FEATURES], dtr["y"])  # final model on all rows
        else:
            model.fit(dtr[FEATURES], dtr["y"])
        ttrain = time.perf_counter() - t0
        if name == "MLP":
            mlp = model[-1]
            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(mlp.loss_curve_, label="training loss (log loss + L2)", color="C0")
            ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss")
            ax2 = ax1.twinx(); ax2.plot(mlp.validation_scores_, color="C1", label="validation accuracy (10% split)")
            ax2.set_ylabel("validation accuracy")
            fig.legend(loc="upper right"); plt.title("MLP training history"); plt.tight_layout()
            plt.savefig(C.FIG_DIR / "curve_MLP.png", dpi=120); plt.close()
            pd.DataFrame({"train_loss": mlp.loss_curve_, "val_acc": mlp.validation_scores_}).to_csv(C.LOG_DIR / "mlp_history.csv", index_label="epoch")
        t0 = time.perf_counter()
        p = model.predict_proba(te[FEATURES])[:, 1]
        tinf = time.perf_counter() - t0
        m, t1 = eval_split(te, p, st_te.n_obs.sum(), n_match(te).sum())
        cont = []
        for trip, g in t1.groupby("trip"):
            cont.append(route_continuity(g.sort_values("obs")["arc"].to_numpy(), arcs_te[trip]))
        m["route_continuity"] = float(np.nanmean(cont))
        cm = confusion_matrix(te["y"], (p >= .5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["not ref arc", "ref arc"]).plot(values_format="d")
        plt.title(f"{name} - student-test candidate confusion matrix"); plt.tight_layout()
        plt.savefig(C.FIG_DIR / f"cm_{name}.png", dpi=120); plt.close()
        cvf = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        row = dict(model=name, family=family, config=cfg,
                   cv_macro_f1=cvf.macro_f1.mean(), cv_macro_f1_std=cvf.macro_f1.std(),
                   cv_top1=cvf.top1_acc.mean(), cv_log_loss=cvf.log_loss.mean(),
                   cv_train_macro_f1=cvf.train_macro_f1.mean(),
                   test_top1=m["top1_acc"], test_top1_matchable=m["top1_acc_matchable"],
                   test_macro_f1=m["macro_f1"], test_precision=m["macro_precision"], test_recall=m["macro_recall"],
                   test_log_loss=m["log_loss"], route_continuity=m["route_continuity"],
                   train_time_s=ttrain, infer_ms_per_fix=1000 * tinf / st_te.n_obs.sum(), n_train_rows=len(dtr))
        summary.append(row)
        print("  STUDENT-TEST:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        if name != "KNN":  # KNN stores its training data; refit is cheap, so not persisted
            joblib.dump(model, C.MODELS_DIR / f"{name}.joblib", compress=3)

    out_f = C.RESULTS_DIR / "model_cv_folds.csv"
    out_s = C.RESULTS_DIR / "model_comparison.csv"
    new_f, new_s = pd.DataFrame(fold_rows), pd.DataFrame(summary)
    if out_s.exists() and set(a.models) != set(MODEL_ZOO):  # partial rerun: merge
        new_f = pd.concat([pd.read_csv(out_f).query("model not in @a.models"), new_f])
        new_s = pd.concat([pd.read_csv(out_s).query("model not in @a.models"), new_s])
    new_f.to_csv(out_f, index=False); new_s.to_csv(out_s, index=False)
    plot_comparison(new_f, new_s)


def plot_comparison(folds, s):
    s = s.set_index("model")
    fig, ax = plt.subplots(figsize=(8, 4))
    folds.boxplot(column="macro_f1", by="model", ax=ax)
    ax.set_title("5-fold GroupKFold CV macro F1 (train partition)"); plt.suptitle(""); ax.set_ylabel("macro F1")
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(C.FIG_DIR / "cmp_cv_macro_f1.png", dpi=120); plt.close()

    ax = s[["test_top1", "test_macro_f1"]].plot(kind="bar", figsize=(9, 4), ylim=(0.5, 1.0))
    ax.set_title("Student-test: observation Top-1 accuracy & candidate macro F1"); plt.xticks(rotation=20)
    for c in ax.containers: ax.bar_label(c, fmt="%.3f", fontsize=7)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "cmp_student_test.png", dpi=120); plt.close()

    ax = s[["cv_log_loss", "test_log_loss"]].plot(kind="bar", figsize=(9, 4), logy=True)
    ax.set_title("Log loss (lower is better; log scale)"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "cmp_log_loss.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(s.infer_ms_per_fix, s.test_top1)
    for n, r in s.iterrows(): ax.annotate(n, (r.infer_ms_per_fix, r.test_top1))
    ax.set_xscale("log"); ax.set_xlabel("inference time (ms per GPS fix, log)"); ax.set_ylabel("student-test Top-1")
    ax.set_title("Accuracy vs. inference cost"); plt.tight_layout()
    plt.savefig(C.FIG_DIR / "cmp_accuracy_vs_time.png", dpi=120); plt.close()

    ax = s[["cv_train_macro_f1", "cv_macro_f1", "test_macro_f1"]].plot(kind="bar", figsize=(9, 4), ylim=(0.5, 1.0))
    ax.set_title("Over/under-fitting check: train-fold vs CV vs student-test macro F1"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "cmp_overfitting.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
