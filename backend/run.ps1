# CHYM-AKI 백엔드 실행 스크립트
# 사용법: .\run.ps1 [port]
#   예시: .\run.ps1        → 8000번 포트
#         .\run.ps1 8080   → 8080번 포트

param([int]$Port = 8010)

$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "[*] venv 활성화 중..." -ForegroundColor Cyan
    & $venvActivate
} else {
    Write-Host "[!] .venv를 찾을 수 없습니다. 전역 환경으로 실행합니다." -ForegroundColor Yellow
}

Write-Host "[*] uvicorn 시작 (port=$Port, reload=on)" -ForegroundColor Green
uvicorn main:app --reload --port $Port
