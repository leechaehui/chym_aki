# 1. W&B 오프라인 모드 강제 설정
$env:WANDB_MODE = "offline"

# 2. Windows 인코딩 이슈 우회 설정
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = 1

# 3. Conda chym_proj 환경의 python 실행파일을 직접 구동해 conda run 버그 우회
& "C:\Users\301-4\anaconda3\envs\chym_proj\python.exe" Pathology_model/mil/train.py --encoder ctranspath --mags 40 --tag exp5
