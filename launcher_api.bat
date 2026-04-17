@echo off
REM Started by run.bat — keeps the API + web UI server running.
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
    echo ERROR: venv not found. Run run.bat from the repository root.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
pause
