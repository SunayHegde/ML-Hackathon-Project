"""Stage 2: preprocessing + candidate generation + feature engineering.

For every trip in a partition: clean the track, project to local metres,
generate candidate arcs, align the reference route to fixes (labels) and
compute candidate features. Output: cache/features_<partition>.parquet
and results/candidate_stats_<partition>.csv.

Usage:  python scripts/02_build_dataset.py [--parts train student_test]
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config as C
from pathsnap.io import list_trips, load_trip
from pathsnap.features import prepare_trip, build_features

PARTS = {"train": C.TRAIN_DIR, "student_test": C.STUDENT_TEST_DIR}


def process(part_dir, tid):
    t0 = time.time()
    td = prepare_trip(load_trip(part_dir, tid))
    df = build_features(td)
    T = len(td.xy)
    has_pos = np.zeros(T, bool)
    if len(df):
        pos_obs = df.loc[df.y == 1, "obs"].to_numpy()
        has_pos[pos_obs] = True
    ld = td.label_dist
    df["label_dist"] = ld[df["obs"].to_numpy()].astype(np.float32)
    stats = dict(trip=tid, n_obs=T, n_rows=len(df),
                 mean_cands=len(df) / max(T, 1),
                 no_candidates=int(sum(len(a) == 0 for a, _, _ in td.cands)),
                 candidate_miss_rate=float(1 - has_pos.mean()),
                 label_dist_median=float(np.median(ld)),
                 label_dist_p95=float(np.percentile(ld, 95)),
                 frac_label_dist_gt_radius=float((ld > C.CAND_RADIUS_M).mean()),
                 secs=time.time() - t0, **td.clean_report)
    return df, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=list(PARTS))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    for part in a.parts:
        trips = list_trips(PARTS[part])
        print(f"[{part}] {len(trips)} trips")
        out = Parallel(n_jobs=a.jobs, verbose=5)(delayed(process)(PARTS[part], t) for t in trips)
        df = pd.concat([o[0] for o in out], ignore_index=True)
        st = pd.DataFrame([o[1] for o in out])
        df.to_parquet(C.CACHE_DIR / f"features_{part}.parquet", index=False)
        st.to_csv(C.RESULTS_DIR / f"candidate_stats_{part}.csv", index=False)
        tot_obs = st.n_obs.sum()
        miss = (st.candidate_miss_rate * st.n_obs).sum() / tot_obs
        print(f"[{part}] rows={len(df):,} obs={tot_obs:,} positives={int(df.y.sum()):,} "
              f"candidate_miss_rate={miss:.4f}")


if __name__ == "__main__":
    main()
