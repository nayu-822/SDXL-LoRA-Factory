@echo off
setlocal
set PYTHONUTF8=1
cd /d %~dp0

:: Keep the original Windows workflow. Set AUTO_SETUP=1 when dependencies need repair.
if not defined APP_PORT set APP_PORT=8001
if not defined APP_HOST set APP_HOST=0.0.0.0
if not defined WORKSPACE_ROOT set WORKSPACE_ROOT=%~dp0.runtime
if not defined LOCAL_DATA_ROOT set LOCAL_DATA_ROOT=%WORKSPACE_ROOT%\data
if not defined LOCAL_OUTPUT_ROOT set LOCAL_OUTPUT_ROOT=%WORKSPACE_ROOT%\output
if not defined LOCAL_JOB_ROOT set LOCAL_JOB_ROOT=%WORKSPACE_ROOT%\jobs
if not defined RCLONE_CONFIG set RCLONE_CONFIG=%WORKSPACE_ROOT%\rclone\rclone.conf
if not defined GDRIVE_REMOTE set GDRIVE_REMOTE=gdrive
if not defined AUTO_SETUP set AUTO_SETUP=1

echo [SDXL LoRA Factory] Initializing on %APP_HOST%:%APP_PORT%...

if not exist venv\ (
    echo [INFO] Creating virtual environment venv...
    python -m venv venv
)
set VENV_PYTHON=%~dp0venv\Scripts\python.exe

if "%AUTO_SETUP%"=="1" (
    "%VENV_PYTHON%" backend\setup_check.py
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Setup check failed. Please check the messages above.
        pause
        exit /b %ERRORLEVEL%
    )
)

echo.
echo [SDXL LoRA Factory] Starting backend server...
echo Access the GUI at http://localhost:%APP_PORT%
echo.

start http://localhost:%APP_PORT%
"%VENV_PYTHON%" -m backend.main

pause
