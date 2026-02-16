@echo off
REM ============================================================
REM PyWorld3-03 Project Cleanup for Windows
REM ============================================================
REM Cleans up stale files, caches, and IDE artifacts.
REM
REM Usage:
REM   Double-click this file (cleanup.bat)
REM   OR run from Command Prompt:
REM   cleanup.bat
REM ============================================================

setlocal enabledelayedexpansion

cls
echo ============================================
echo   PyWorld3-03 Project Cleanup
echo ============================================
echo.

echo === FILES THAT WILL BE DELETED ===
echo.

echo [Cache ^& Build Artifacts]
echo   myworld3\__pycache__\     - Stale .pyc files from Python
echo   .DS_Store                  - macOS artifact (if present)
echo   .claude\worktrees\         - Stale Claude session artifacts
echo.

echo [Redundant/Old Module Files]
echo   myworld3\pollution_old.py  - Old version
echo   myworld3\resource_old.py   - Old version
echo   myworld3\classes.dot       - Generated diagram source
echo   myworld3\classes_Population.png - Generated diagram
echo   myworld3\pyworld3_population.svg - Generated diagram (large)
echo   myworld3\test_delay3.py    - Internal test
echo   myworld3\test_sectors.py   - Internal test
echo.

echo [Generated Output Images]
echo   Population_plot.png        - Generated output (43 KB)
echo   canadian_agriculture_simulation.png - Generated output (48 KB)
echo.

echo [Optional]
echo   .idea\                     - JetBrains IDE config
echo.

set /p CONFIRM="Delete these files? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo Deleting files...

REM Delete cache
if exist myworld3\__pycache__ (
    echo Removing myworld3\__pycache__\
    rmdir /s /q myworld3\__pycache__ >nul 2>&1
)

REM Delete Claude worktrees
if exist .claude\worktrees (
    echo Removing .claude\worktrees\
    rmdir /s /q .claude\worktrees >nul 2>&1
)

REM Delete old/redundant files
set files_to_delete=^
    myworld3\pollution_old.py ^
    myworld3\resource_old.py ^
    myworld3\classes.dot ^
    myworld3\classes_Population.png ^
    myworld3\pyworld3_population.svg ^
    myworld3\test_delay3.py ^
    myworld3\test_sectors.py ^
    Population_plot.png ^
    canadian_agriculture_simulation.png ^
    .DS_Store

for %%F in (%files_to_delete%) do (
    if exist %%F (
        echo Removing %%F
        del /q %%F >nul 2>&1
    )
)

REM Optional: Delete IDE config
set /p DELETE_IDE="Also delete .idea\ folder? (Y/N): "
if /i "%DELETE_IDE%"=="Y" (
    if exist .idea (
        echo Removing .idea\
        rmdir /s /q .idea >nul 2>&1
    )
)

echo.
echo ============================================
echo   Cleanup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Run setup_env.bat to create virtual environment
echo   2. Activate it: venv\Scripts\activate
echo   3. Run app: streamlit run app_population.py
echo.
pause
