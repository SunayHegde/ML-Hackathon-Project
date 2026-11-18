# Start the PathSnap API (Laptop 1) serving the evaluator-test trips.
# Usage (PowerShell, from the pathsnap folder):  .\start_api_eval.ps1  [-Port 9000]
param([int]$Port = 9000)
Set-Location $PSScriptRoot
$env:PATHSNAP_EVAL_DIR = Join-Path $PSScriptRoot "data\evaluator_test"
Write-Host "Serving trips from $env:PATHSNAP_EVAL_DIR on port $Port"
python -m uvicorn api.main:app --host 0.0.0.0 --port $Port
