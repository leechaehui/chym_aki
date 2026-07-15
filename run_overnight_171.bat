@echo off
set PYTHON_EXE=C:\Users\301-4\anaconda3\envs\chym_proj\python.exe
set MIL_DIR=C:\team\chym_aki\Pathology_model\mil

echo ====================================================
echo 171 Patient Full Pipeline (Overnight Execution)
echo ====================================================

echo [1] Running Embedding on 117,792 patches (This will take hours)...
%PYTHON_EXE% %MIL_DIR%\embed_patches.py --encoder ctranspath --mags 10,20,30,40 --topk 100 --workers 1
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2] Training CLAM-lite on 171 Patients...
%PYTHON_EXE% %MIL_DIR%\train_clam_lite.py --encoder ctranspath --mags 10,20,30,40 --tag cdss_v5_full_171p --topk 16
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3] Building FAISS Index...
%PYTHON_EXE% %MIL_DIR%\build_faiss_index.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo PIPELINE DONE!
