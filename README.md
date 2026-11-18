# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap snaps noisy ~1 Hz GPS fixes onto a directed road network and rebuilds the route the vehicle drove. It works in two stages:

1. **Candidate classifier.** For every GPS fix, nearby road arcs become candidates. A trained classifier scores each one with *P(this arc is the reference arc)*.
2. **HMM + Viterbi decoder.** The classifier probabilities act as emissions. Transitions come from network driving distances, which turns per-fix guesses into one connected route.

The system runs as a **FastAPI backend** (Laptop 1) and a **browser UI** (Laptop 2).

## Results (all produced by the scripts in this repo)

### Dataset (EDA: `results/eda_summary.csv`)
| Partition | Trips | GPS fixes | NaN / invalid coords | Gaps > 10 s | Speed jumps > 216 km/h | Candidate miss rate* |
|---|---|---|---|---|---|---|
| train | 80 | 208,417 | 0 / 0 | 590 | 546 | 11.7 % |
| student_test | 10 | 19,160 | 0 / 0 | 27 | 2 | 0.2 % |

\*A miss means the fix's reference arc is not among its candidates (50 m radius, 12 nearest). The training miss rate mostly comes from two trips, 00000054 and 00000033. Both have hours of parked or off-network GPS fixes where the reference arc is more than 100 m away. These fixes (reference arc more than 50 m away) are dropped from **training** but always counted as errors in **evaluation**.

### Model comparison (`results/model_comparison.csv`)
CV = 5-fold GroupKFold on the 80 training trips. Test = the 10 student-test trips.

| Model | Family | CV macro F1 (±sd) | CV train-fold F1 | Test Top-1 | Test macro F1 | Test log loss | Route continuity | Train s | Inference ms/fix |
|---|---|---|---|---|---|---|---|---|---|
| Gaussian NB | Probabilistic | 0.842 ± 0.030 | 0.849 | 0.842 | 0.842 | 0.608 | 0.927 | 0.6 | 0.003 |
| Logistic Regression | Linear | 0.918 ± 0.013 | 0.922 | 0.872 | 0.919 | 0.102 | 0.933 | 4.5 | 0.003 |
| KNN (k=31) | Instance-based | 0.919 ± 0.013 | **1.000** (memorises) | 0.869 | 0.922 | 0.120 | 0.938 | 0.1 | 2.061 |
| MLP (64,32) | Neural network | 0.932 ± 0.009 | 0.947 | 0.886 | 0.932 | 0.076 | 0.945 | 54.9 | 0.008 |
| Random Forest | Bagging | 0.936 ± 0.010 | 0.962 | 0.897 | 0.937 | 0.069 | 0.944 | 100.6 | 0.038 |
| **LightGBM** | Boosting | **0.937 ± 0.012** | 0.969 | **0.905** | **0.943** | **0.067** | **0.951** | 47.5 | 0.049 |

**LightGBM was selected.** It scored best on every quality metric: CV F1, test Top-1, test F1, log loss and continuity. Its log loss was also the lowest, which means its probabilities are the best calibrated. That matters because those probabilities become the HMM emissions. It also trains about twice as fast as Random Forest.

### Bayesian hyperparameter optimisation (`configs/best_params.json`, `results/optuna_trials.csv`)
* **Search:** Optuna TPE sampler (seed 42) with a median pruner. The 12 trials consisted of 8 completed and 4 pruned. Trial 0 was seeded with the untuned configuration.
* **Search space:** `n_estimators` 100–800, `learning_rate` 0.01–0.2 (log), `num_leaves` 15–255, `min_child_samples` 10–300, `subsample` 0.5–1, `colsample_bytree` 0.4–1, `reg_lambda` and `reg_alpha` 1e-3–10 (log).
* **Objective:** mean macro F1 over a 3-fold GroupKFold on the training trips. For time reasons, the search used a 20 % random sample of fixes, keeping each fix's full candidate set.
* **Best:** `n_estimators=142, learning_rate=0.065, num_leaves=147, min_child_samples=207, subsample=0.81, colsample_bytree=0.41, reg_lambda=0.005, reg_alpha=0.004`.

| | CV macro F1 (3-fold) | CV log loss | Test Top-1 | Test macro F1 | Test log loss |
|---|---|---|---|---|---|
| LightGBM untuned | 0.9388 | 0.0770 | 0.9046 | 0.9426 | 0.0667 |
| LightGBM tuned | **0.9389** | **0.0752** | 0.9021 | **0.9431** | **0.0657** |

Tuning gave only a marginal gain: slightly better F1 and log loss, and a 0.3-point drop in test Top-1. The default configuration was already near the optimum for these features. The tuned model is about 3× smaller (142 trees instead of 400) and faster at inference, so it is used as `models/final_model.joblib`.

### HMM / Viterbi vs classifier-only (`results/hmm_vs_classifier_student_test.csv`)
The decoder was grid-tuned on 8 training trips using out-of-fold probabilities. The best values were **β = 5 m** and **emission weight 0.25**.

| Student-test (10 trips, 19,160 fixes) | Classifier only | Classifier + HMM |
|---|---|---|
| Observation Top-1 accuracy | 0.902 | **0.926** |
| Route continuity | 0.953 | **0.979** |
| Route F1 (length-weighted) | 0.956 | **0.983** |
| Chain breaks (total) | – | 7 |

The HMM improves 9 of the 10 trips. **Trip 00000007 is the exception**: Top-1 drops from 0.969 to 0.890. The vehicle is parked at the end of that trip for about 150 fixes on a two-way street. With no motion there is no usable heading, and the decoder picks the reverse-direction arc. Its route F1 still rises, from 0.989 to 0.991.

Interactive maps: `maps/student_test_<trip>.html`. Each shows the GPS fixes, road network, reference route, classifier-only route and HMM route, with a legend and layer control.

---

## Repository layout

```
pathsnap/
├── src/pathsnap/            core library
│   ├── config.py            paths + pipeline constants
│   ├── io.py                raw-file readers, track cleaning
│   ├── geo.py               local metric projection, point→segment projection
│   ├── network.py           RoadNetwork: arc geometry, KD-tree index, bounded Dijkstra
│   ├── labels.py            route → per-fix reference labels (monotone DP alignment)
│   ├── features.py          candidate generation + feature engineering
│   ├── models.py            the 6 compared models
│   ├── metrics.py           macro-F1, Top-1, log loss, route continuity, route P/R/F1
│   ├── hmm.py               HMM + Viterbi decoder
│   ├── pipeline.py          end-to-end inference (used by the API)
│   └── viz.py               Folium map builder
├── scripts/
│   ├── 01_eda.py            data inspection / EDA / Folium EDA maps
│   ├── 02_build_dataset.py  preprocessing + candidates + features → cache/
│   ├── 03_train_models.py   6 models, 5-fold GroupKFold CV + student-test evaluation
│   ├── 04_bayes_opt.py      Optuna (TPE) Bayesian optimisation of LightGBM, final model
│   ├── 05_hmm_eval.py       decoder tuning + classifier-only vs classifier+HMM
│   └── 06_make_maps.py      Folium maps of predictions for student-test trips
├── api/                     FastAPI backend (main.py) + server-side scorer (evaluator.py)
├── ui/                      evaluator UI (index.html + serve.py)
├── models/                  saved models (final_model.joblib = tuned LightGBM)
├── configs/                 best_params.json (Optuna), hmm_params.json (decoder)
├── results/                 CSVs, logs/, figures/
├── maps/                    interactive Folium HTML maps
├── requirements.txt
└── run_pipeline.sh / run_pipeline.ps1
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13, scikit-learn 1.9, LightGBM 4, Optuna 4.

### Data

The dataset is not committed because of its size. Put it here:

```
data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          # 80 trips
data/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   # 10 trips
```

To point at a different location, set `PATHSNAP_DATA`. Evaluator-testing trips are never used while developing. The API only reads them through `PATHSNAP_EVAL_DIR` at demo time.

## Reproducing the experiments

```bash
bash run_pipeline.sh          # or: powershell -File run_pipeline.ps1
```

| Step | Script | Main outputs |
|---|---|---|
| EDA | `01_eda.py` | `results/eda_inventory_*.csv`, `eda_summary.csv`, `figures/eda_*.png`, `maps/eda_trip_*.html` |
| Features | `02_build_dataset.py` | `cache/features_*.parquet`, `results/candidate_stats_*.csv` |
| Model comparison | `03_train_models.py` | `results/model_comparison.csv`, `model_cv_folds.csv`, `figures/cmp_*.png`, `cm_*.png`, `curve_*.png` |
| Bayesian optimisation | `04_bayes_opt.py` | `results/optuna_trials.csv`, `configs/best_params.json`, `results/tuned_vs_untuned.csv`, `models/final_model.joblib` |
| HMM | `05_hmm_eval.py` | `results/hmm_grid.csv`, `configs/hmm_params.json`, `results/hmm_vs_classifier_student_test.csv` |
| Maps | `06_make_maps.py` | `maps/student_test_*.html` |

All random seeds are fixed (`SEED = 42`).

## Deployment (two laptops)

**Laptop 1 runs the model and API.** It needs the repo, `models/`, `configs/` and the evaluation trip folders:

```bash
# PowerShell: $env:PATHSNAP_EVAL_DIR="D:\path\to\eval_trips"
export PATHSNAP_EVAL_DIR=/path/to/evaluator_test     # default: data/student_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

For the final evaluation, extract the evaluator-test trips to `data/evaluator_test/` and run `.\start_api_eval.ps1` (port 9000 by default). It sets `PATHSNAP_EVAL_DIR` for you.

Every request is logged to that console, for example `PREDICT request trip=…` and `PREDICT done trip=… top1_hmm=…`. The firewall must allow port 8000.

**Laptop 2 runs the UI.** Only the `ui/` folder is needed:

```bash
python ui/serve.py --port 8501
# open http://localhost:8501/?api=http://<LAPTOP1_IP>:8000
```

The UI lists the trips that the API exposes. When you click a trip, the UI sends its `trip_id` to `POST /predict` and then shows:
- the GPS fixes, the predicted route and the classifier-only route (as layers),
- the accuracy and route metrics that the server computed,
- the timings.

### API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, model type and decoder parameters |
| GET | `/trips` | Trip IDs in `PATHSNAP_EVAL_DIR` (no labels) |
| POST | `/predict` | Body `{"trip_id": "00000052", "use_hmm": true}`. Returns the predicted route polylines, the arc for each fix, metrics and timings |
| GET | `/map/{trip_id}` | Folium HTML of the last prediction for that trip |

**Leakage safeguards**
- The predictor loads trips with `with_route=False`.
- The `.route` file is read only in `api/evaluator.py`, after prediction has finished. Only aggregate metrics come back to the client; the reference route is never returned.
- No prediction is hardcoded.
- `trip_id` is checked against the directory listing.

## Method summary

* **Projection.** Each trip uses a local equirectangular tangent plane centred on the trip. Trips span less than 50 km, so the distance error stays far below GPS noise, and distances can be computed in metres.
* **Labels.** `.route` lists arcs with no timestamps. Each fix is aligned to a position in the route with a monotone dynamic-programming alignment that minimises the total point-to-arc distance. The reference arc for a fix is the route arc at its aligned position.
* **Candidates.** Arcs are densified every 10 m into a KD-tree. Each fix keeps every arc within 50 m, up to the nearest 12.
* **Features (21).**
  - Distance: distance, rank, gap to the best candidate, ratio, and mean distance over a ±3-fix window.
  - Neighbours: distance from the previous and next fixes.
  - Heading: heading difference over ±2 and ±5 fixes and its rank, plus speed and a heading-validity flag.
  - Arc: length, offset fraction, node degrees, whether the arc is two-way.
  - Timing: `dt`.
  - Scores normalised within each fix (softmax-style), for distance and for heading.
* **Models.** Logistic Regression, Gaussian NB, KNN, Random Forest, LightGBM and MLP. Each is evaluated with 5-fold GroupKFold, grouped by trip, then evaluated on the 10 student-test trips.
* **Bayesian optimisation.** Optuna's TPE sampler searches 8 LightGBM hyperparameters, maximising macro F1 under 3-fold GroupKFold.
* **HMM.** The hidden states are candidate arcs together with their offsets.
  - Emission: the classifier probability, normalised within the fix.
  - Transition: `exp(-|driving distance − straight-line distance| / β)` on the directed graph, using a bounded Dijkstra search.
  - The chain breaks when no transition is feasible, and fixes with no candidates are skipped.
  - β and the emission weight are grid-tuned on out-of-fold probabilities from the training trips.
