@echo off
REM Get the folder of this batch file
set "BATCH_DIR=%~dp0"

REM Set venv path relative to batch file folder
set "VENV_PATH=%BATCH_DIR%venv"

REM Open new cmd with venv activated, quotes handle spaces
cmd /k ""%VENV_PATH%\Scripts\activate.bat""
