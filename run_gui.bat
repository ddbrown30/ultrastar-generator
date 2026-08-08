@echo off
REM Double-click this file to launch the UltraStar Generator GUI --
REM no manual venv activation or command-line invocation needed.

REM Get the folder of this batch file (same convention as launch_env.bat/setup.bat)
set "BATCH_DIR=%~dp0"
set "VENV_PATH=%BATCH_DIR%venv"

if not exist "%VENV_PATH%\Scripts\pythonw.exe" (
    echo [ERROR] Could not find the virtual environment at "%VENV_PATH%".
    echo         Run setup.bat first to create it.
    pause
    exit /b 1
)

REM pythonw.exe (not python.exe) so no console window appears alongside
REM the GUI window itself.
start "" "%VENV_PATH%\Scripts\pythonw.exe" -m ultrastar_generator.gui
