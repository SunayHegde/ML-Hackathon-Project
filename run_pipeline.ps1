# Reproduce every result in results/, models/, configs/ and maps/ (Windows PowerShell)
$ErrorActionPreference = "Stop"
python scripts/01_eda.py            | Tee-Object results/logs/01_eda.log
python scripts/02_build_dataset.py  | Tee-Object results/logs/02_build_dataset.log
python -W ignore scripts/03_train_models.py | Tee-Object results/logs/03_train_models.log
python -W ignore scripts/04_bayes_opt.py    | Tee-Object results/logs/04_bayes_opt.log
python -W ignore scripts/05_hmm_eval.py     | Tee-Object results/logs/05_hmm_eval.log
python scripts/06_make_maps.py      | Tee-Object results/logs/06_make_maps.log
