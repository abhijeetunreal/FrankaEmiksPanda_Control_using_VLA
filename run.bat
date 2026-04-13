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
    echo Existing virtual environment detected. Skipping package installation.
)

echo Verifying installed packages...
python -c "import importlib; [importlib.import_module(m) for m in ['mujoco','cv2','numpy','openai']]; print('All dependencies are installed successfully.')"
if errorlevel 1 (
    echo.
    echo ERROR: Dependency verification failed.
    echo Please check the earlier output for missing packages or installation errors.
    pause
    exit /b 1
)

echo.
echo Setup completed successfully.
echo Running the VLA controller now...
python franka_emika_panda\vla_controller.py
if errorlevel 1 (
    echo.
    echo ERROR: The controller script failed to run.
    pause
    exit /b 1
)

echo.
echo Controller finished successfully.
pause
