$ErrorActionPreference = "Stop"

$env:WANDB_MODE = "offline"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = 1

Write-Host "Starting Experiment 0: Single-Stain Baseline (HE Only)"
& "C:\Users\301-4\anaconda3\envs\chym_proj\python.exe" mil/train.py --encoder ctranspath --mags 10 --stains HE --tag Exp0_HE_only --epochs 40 > results/Exp0_HE.log 2>&1

Write-Host "Starting Experiment 0: Single-Stain Baseline (PAS Only)"
& "C:\Users\301-4\anaconda3\envs\chym_proj\python.exe" mil/train.py --encoder ctranspath --mags 10 --stains PAS --tag Exp0_PAS_only --epochs 40 > results/Exp0_PAS.log 2>&1

Write-Host "Starting Experiment 0: Single-Stain Baseline (MT Only)"
& "C:\Users\301-4\anaconda3\envs\chym_proj\python.exe" mil/train.py --encoder ctranspath --mags 10 --stains MT --tag Exp0_MT_only --epochs 40 > results/Exp0_MT.log 2>&1

Write-Host "All Single-Stain Experiments Completed."
