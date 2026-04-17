@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"

echo ========================================
echo  VLA MuJoCo — Web workspace
echo ========================================
echo.

set CREATED_VENV=0
if exist "venv\Scripts\python.exe" goto have_venv
echo Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv. Install Python 3.10+ on PATH.
    pause
    exit /b 1
)
set CREATED_VENV=1

:have_venv
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv.
    pause
    exit /b 1
)

if "%CREATED_VENV%"=="1" (
    echo Upgrading pip...
    python -m pip install --upgrade pip -q
)

echo Installing Python dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo Verifying packages...
python -c "import importlib; [importlib.import_module(m) for m in ['mujoco','cv2','numpy','openai','fastapi']]; print('OK.')"
if errorlevel 1 (
    echo ERROR: Import check failed.
    pause
    exit /b 1
)

if not exist "web\index.html" (
    echo ERROR: web\index.html not found.
    pause
    exit /b 1
)

echo.
echo Starting server in a new window ^(API + MuJoCo + web UI on port 8000^)...
start "VLA Server" cmd /k "call ""%ROOT%\launcher_api.bat"""

echo Waiting for http://127.0.0.1:8000 ...
set /a RETRIES=0
:wait_api
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto api_ready
set /a RETRIES+=1
if %RETRIES% GEQ 60 (
    echo WARNING: Server did not respond. Check the VLA Server window for errors.
    goto open_browser
)
timeout /t 1 /nobreak >nul
goto wait_api

:api_ready
echo Server is up.

:open_browser
echo Opening browser...
start "" "http://127.0.0.1:8000/"

echo.
echo Workspace: http://127.0.0.1:8000/
echo Close the ^"VLA Server^" window to stop the simulation and API.
echo.
pause
endlocal
