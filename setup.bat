@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  UltraStar Generator - Environment Setup
echo ============================================
echo.

REM --- 1. Check Python ---------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         and check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found:
python --version
echo.

REM --- 2. Check ffmpeg -----------------------------------------------------
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg was not found on PATH.
    echo           Download a build from https://www.gyan.dev/ffmpeg/builds/
    echo           ^(the "essentials" zip^), unzip it, and add its bin folder
    echo           to your PATH. The generator will not run without it.
    echo.
    pause
) else (
    echo [OK] ffmpeg found.
    echo.
)

REM --- 3. Create a CLEAN virtual environment ---------------------------------
REM Always start fresh. Reusing an existing venv folder risks inheriting
REM packages from some other project (e.g. a different torch version, or
REM something like whisperx that pulls in its own incompatible torch pin),
REM which causes exactly the kind of dependency-resolver conflicts this
REM script is trying to avoid.
if exist venv (
    echo Removing existing .\venv to guarantee a clean environment...
    rmdir /s /q venv
    if exist venv (
        echo [ERROR] Could not remove the existing venv folder. Close any
        echo         terminals/editors using it and re-run this script.
        pause
        exit /b 1
    )
)
echo Creating virtual environment in .\venv ...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment created.
echo.

REM --- 4. Activate venv -----------------------------------------------------
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.
echo.

REM --- 5. Upgrade pip ---------------------------------------------------------
python -m pip install --upgrade pip

REM --- 6. Install PyTorch (pinned to what whisperx requires, CUDA 12.6) ------
REM requirements.txt's whisperx>=3.1 pins torch~=2.8.0 and torchvision~=0.23.0.
REM Installing "latest" torch (e.g. 2.13.0) satisfies CUDA support but breaks
REM that pin and makes pip's resolver complain / leave a broken combo. So we
REM install the exact versions whisperx expects, from the cu126 index, and
REM let requirements.txt pull the matching torchaudio alongside it.
echo.
echo Removing any existing torch/torchvision/torchaudio install (avoids a
echo stale or mismatched wheel getting reused from pip's cache)...
pip uninstall torch torchvision torchaudio -y >nul 2>&1

echo Installing PyTorch 2.8.0 / torchvision 0.23.0 (CUDA 12.6 build,
echo versions chosen to match whisperx's pinned requirement)...
pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
if errorlevel 1 (
    echo [ERROR] PyTorch install failed. Check your internet connection
    echo         and that your NVIDIA driver supports CUDA 12.6.
    pause
    exit /b 1
)
echo [OK] PyTorch installed.
echo.

REM --- 6b. Hard-verify CUDA is actually usable before continuing -------------
echo Verifying the CUDA build actually works on this machine...
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo [ERROR] torch.cuda.is_available^(^) returned False after installing
    echo         the cu126 build. This usually means your NVIDIA driver
    echo         doesn't support CUDA 12.6 yet, or an old CPU wheel is
    echo         still cached elsewhere. Run "nvidia-smi" and check the
    echo         "CUDA Version" field in the top-right, then re-run this
    echo         script once the driver is updated.
    pause
    exit /b 1
)
echo [OK] CUDA build confirmed working:
python -c "import torch; print(' ', torch.__version__, '-', torch.cuda.get_device_name(0))"
echo.

REM --- 7. Install remaining requirements -------------------------------------
echo Installing remaining dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] requirements.txt install hit an error. This is most often
    echo           whisperx's own dependency pins conflicting despite the
    echo           torch/torchvision pin above ^(e.g. if whisperx bumps its
    echo           requirement in a new release^). Per requirements.txt,
    echo           whisperx is optional -- the tool falls back to
    echo           faster-whisper's own word timestamps if it's missing.
    echo.
    echo           Retrying install with whisperx skipped...
    findstr /v /i "whisperx" requirements.txt > requirements.tmp.txt
    pip install -r requirements.tmp.txt
    if errorlevel 1 (
        echo [ERROR] Install still failed even without whisperx. See the
        echo         pip output above for details.
        del requirements.tmp.txt
        pause
        exit /b 1
    )
    del requirements.tmp.txt
    echo [OK] Dependencies installed ^(without whisperx -- word-level
    echo      alignment will use faster-whisper's own timestamps^).
) else (
    echo [OK] Dependencies installed.
)
echo.

REM --- 8. Verify the CLI ------------------------------------------------------
echo Verifying ultrastar_generator CLI...
python -m ultrastar_generator -h
echo.

echo ============================================
echo  Setup complete!
echo  Run "venv\Scripts\activate.bat" in new
echo  terminals before using the tool.
echo  Remember to pass --device cuda to use the GPU.
echo ============================================
pause
