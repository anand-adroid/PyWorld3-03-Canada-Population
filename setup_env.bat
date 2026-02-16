@echo off
REM ============================================================
REM PyWorld3-03 Environment Setup for Windows
REM ============================================================
REM Creates a clean virtual environment with only the required
REM dependencies for the Canadian Population & Agriculture model.
REM
REM Usage:
REM   Double-click this file (setup_env.bat)
REM   OR run from Command Prompt:
REM   setup_env.bat
REM
REM After setup, activate with:
REM   venv\Scripts\activate
REM
REM To run the Streamlit app:
REM   streamlit run app_population.py
REM ============================================================

setlocal enabledelayedexpansion

cls
echo ============================================
echo   PyWorld3-03 Environment Setup
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Using: %PYTHON_VERSION%
echo.

REM Remove old venv if it exists
if exist venv (
    echo Removing existing venv...
    rmdir /s /q venv >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Could not remove old venv completely.
        echo Please manually delete the "venv" folder if you see errors.
    )
)

REM Create fresh virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure Python is installed with venv support.
    pause
    exit /b 1
)

echo Virtual environment created.
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Virtual environment activated.
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: pip upgrade had issues, but continuing...
)

REM Install requirements
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo To activate the environment in the future:
echo   venv\Scripts\activate
echo.
echo To run the Streamlit app:
echo   streamlit run app_population.py
echo.
echo To run Jupyter notebooks:
echo   jupyter notebook
echo.
pause
