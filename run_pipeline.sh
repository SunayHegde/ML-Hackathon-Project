#!/usr/bin/env bash
# Reproduce every result in results/, models/, configs/ and maps/
set -euo pipefail
export PYTHONUNBUFFERED=1
python scripts/01_eda.py                    | tee results/logs/01_eda.log
python scripts/02_build_dataset.py          | tee results/logs/02_build_dataset.log
python -W ignore scripts/03_train_models.py | tee results/logs/03_train_models.log
python -W ignore scripts/04_bayes_opt.py    | tee results/logs/04_bayes_opt.log
python -W ignore scripts/05_hmm_eval.py     | tee results/logs/05_hmm_eval.log
python scripts/06_make_maps.py              | tee results/logs/06_make_maps.log
