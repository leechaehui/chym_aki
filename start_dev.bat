@echo off
cd /d "%~dp0"

echo ==========================================================
echo   CHYM-AKI Auto-Sync ^& Backup ^& Start Development Server
echo ==========================================================
echo.

:: 1. Create a unique backup directory name using python
for /f "usebackq tokens=*" %%a in (`python -c "from datetime import datetime; print(datetime.now().strftime('%%Y%%m%%d_%%H%%M%%S'))"`) do set TIMESTAMP=%%a
set BACKUP_DIR=backup\backup_%TIMESTAMP%

echo [1] Backing up source code, DB ^& ML models to %BACKUP_DIR%...
:: robocopy . %BACKUP_DIR% /MIR /XD node_modules .venv .git backup .idea __pycache__ uploads data keys /XF *.pyc *.db-journal rollback_audit.log pip_error.log npm_error.log db_error.log /R:0 /W:0 /NDL /NFL /NJH /NJS >nul 2>&1
echo Backup completed!

echo.
echo [2] Checking Backend Dependencies (requirements.txt)...
cd backend
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
if exist pip_error.log del pip_error.log
python -m pip install -r requirements.txt 2> pip_error.log
if %ERRORLEVEL% neq 0 goto ROLLBACK

echo.
echo [3] Checking Frontend Dependencies (package.json)...
cd ..\frontend
if exist npm_error.log del npm_error.log
call npm install 2> npm_error.log
if %ERRORLEVEL% neq 0 goto ROLLBACK

echo.
echo [3.5] Ensuring PostgreSQL service is running...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\ensure_postgres.ps1"
if %ERRORLEVEL% neq 0 goto NODB

echo.
echo [4] Applying Database Schema ^& Seeding...
cd ..\backend
call .venv\Scripts\activate.bat
if exist db_error.log del db_error.log
python -c "from core.database import init_db; init_db(); print('DB Models created.')" 2> db_error.log
if %ERRORLEVEL% neq 0 goto ROLLBACK

:: 모델에 새로 추가된 컬럼을 기존 테이블에 반영(드리프트 방지). create_all 은 없는 테이블만 만들고 컬럼은 못 채우므로 필요.
python scripts\sync_schema.py 2>> db_error.log
if %ERRORLEVEL% neq 0 goto ROLLBACK

python -c "from db.seed import seed; seed(); print('DB Seed completed.')" 2>> db_error.log
if %ERRORLEVEL% neq 0 goto ROLLBACK

python -c "from core.database import SessionLocal; from sqlalchemy import text; db=SessionLocal(); sql=open('telemetry/db_schema.sql', encoding='utf-8').read(); db.execute(text(sql)); db.commit(); db.close(); print('Telemetry schema applied.')" 2>> db_error.log
if %ERRORLEVEL% neq 0 goto ROLLBACK
if exist db_error.log del db_error.log

echo.
echo [5] Starting Servers...
cd ..
:: [경고] ?�떤 ?�이 ?�더?�도 ?�버??8010?�로 ?��??�다. (?�행 ????주석??반드???�인??�?
start "CHYM Backend" cmd /k "cd backend && call .venv\Scripts\activate.bat && uvicorn main:app --host 0.0.0.0 --port 8010 --reload"
:: --- WSI inference server (port 8001) - separate process (failure isolation) ---
:: Needs conda env with torch + openslide (chym_proj 또는 chym_aki). Auto-detect its python; override with CHYM_WSI_PYTHON.
set "WSI_PY="
if defined CHYM_WSI_PYTHON if exist "%CHYM_WSI_PYTHON%" set "WSI_PY=%CHYM_WSI_PYTHON%"
if not defined WSI_PY for /f "delims=" %%i in ('conda info --base 2^>nul') do for %%E in (chym_proj chym_aki) do if not defined WSI_PY if exist "%%i\envs\%%E\python.exe" set "WSI_PY=%%i\envs\%%E\python.exe"
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do for %%E in (chym_proj chym_aki) do if not defined WSI_PY if exist "%%~P\envs\%%E\python.exe" set "WSI_PY=%%~P\envs\%%E\python.exe"
if defined WSI_PY (
    echo [WSI] using %WSI_PY%
    start "CHYM WSI 8001" cmd /k "cd backend && "%WSI_PY%" -m uvicorn wsi_main:app --host 0.0.0.0 --port 8001"
) else (
    echo [WARN] conda env 'chym_proj' python not found - WSI server ^(8001^) NOT started.
    echo        Create env 'chym_proj' or set CHYM_WSI_PYTHON. See backend\wsi\README.md
)
start "CHYM Frontend" cmd /k "cd frontend && npm run dev"

echo Done! Servers are starting in separate windows.
exit /b 0

:NODB
echo.
echo ==========================================================
echo   [ERROR] PostgreSQL is not available (port 5432).
echo   Source code was NOT rolled back - only the DB server is down.
echo ==========================================================
echo   조치: 관리자 PowerShell에서 아래 실행 후 다시 시도하세요.
echo       Start-Service postgresql-x64-16
echo   서비스가 안 뜨면 데이터 폴더의 stale postmaster.pid 삭제 후 재시도.
echo ==========================================================
pause
exit /b 1

:ROLLBACK
echo.
echo ==========================================================
echo   [ERROR] Setup Failed!
echo   Rolling back source code, models, and DB to previous state...
echo ==========================================================
cd /d "%~dp0"
python scripts\rollback_notifier.py
robocopy %BACKUP_DIR% . /MIR /XD node_modules .venv .git backup .idea __pycache__ uploads data keys /XF *.pyc *.db-journal rollback_audit.log pip_error.log npm_error.log db_error.log /R:0 /W:0 /NDL /NFL /NJH /NJS >nul 2>&1
echo Rollback completed from %BACKUP_DIR%.
echo.
echo ==========================================================
echo   [�˸�] ����ȭ �� �浹 �Ǵ� ������ �߻��Ͽ� ���� �� ���·� �����Ǿ����ϴ�.
echo   ������ ��Ű�� ����(requirements.txt, package.json)�̳� DB ��Ű��
echo   ���� ������ �´��� Ȯ�����ּ���!
echo ==========================================================
pause
exit /b 1
