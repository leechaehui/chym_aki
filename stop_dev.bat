@echo off
echo ==========================================================
echo   CHYM-AKI Stop Development Server
echo ==========================================================
echo.
echo 백엔드와 프론트엔드 서버를 종료하는 중입니다...

:: CHYM Backend 및 Frontend 이름으로 열린 콘솔 창과 자식 프로세스를 모두 강제 종료합니다.
taskkill /FI "WINDOWTITLE eq CHYM Backend*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq CHYM Frontend*" /T /F >nul 2>&1

echo.
echo 성공적으로 모든 서버가 종료되었습니다!
ping 127.0.0.1 -n 4 >nul
 