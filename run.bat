@echo off
setlocal

REM Change to the repository root where this script lives.
cd /d "%~dp0"

set CREATED_VENV=0
if exist venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    echo Creating virtual environment in "%cd%\venv"...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python is installed and available on PATH.
        pause
        exit /b 1
    )
    set CREATED_VENV=1
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

if "%CREATED_VENV%" == "1" (
    echo Upgrading pip...
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to upgrade pip.
        pause
        exit /b 1
    )

    echo Installing required Python packages from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Package installation failed.
        echo If "mujoco" failed, you may need to install MuJoCo separately from https://mujoco.org/ and ensure the license is configured.
        pause
        exit /b 1
    )
) else (
    echo Existing virtual environment detected. Installing/updating packages from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Package installation failed.
        pause
        exit /b 1
    )
)

echo Verifying installed packages...
python -c "import importlib; [importlib.import_module(m) for m in ['mujoco','cv2','numpy','openai','fastapi','glfw','imgui','OpenGL']]; print('All dependencies are installed successfully.')"
if errorlevel 1 (
    echo.
    echo ERROR: Dependency verification failed.
    echo Please check the earlier output for missing packages or installation errors.
    pause
    exit /b 1
)

echo.
echo Setup completed successfully.
echo.
echo Starting VLA Planning API (FastAPI / Swagger) in a separate window...
echo   In your browser use http://127.0.0.1:8000/ or http://localhost:8000/ ^(not 0.0.0.0^).
echo   The root URL redirects to Swagger UI ^(/docs^).
echo   Ensure GROQ_API_KEY is set in your environment or .env for API and controller.
start "VLA Planning API" /D "%~dp0" cmd /k "call venv\Scripts\activate.bat && uvicorn franka_emika_panda.api:app --reload --host 0.0.0.0 --port 8000"
echo.
echo Starting MuJoCo integrated viewer in a NEW window (3D + bottom-right chat).
echo   Title: "MuJoCo VLA" — if you do not see it, check the taskbar and that console for errors.
echo   Terminal CLI instead: python franka_emika_panda\vla_controller.py
start "MuJoCo VLA" /D "%~dp0" cmd /k "call venv\Scripts\activate.bat && python -m franka_emika_panda.integrated_viewer || (echo. & echo Integrated viewer exited with an error. & pause)"

echo.
echo Setup window: you can close this prompt after the MuJoCo window opens. The API window stays separate.
pause
