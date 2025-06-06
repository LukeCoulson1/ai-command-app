@echo off
REM Set up virtual environment, install dependencies, and run the AI Command App Streamlit UI

REM Set paths
set APP_DIR=%~dp0
set VENV_DIR=%APP_DIR%venv
set REQUIREMENTS=%APP_DIR%requirements.txt
set PYTHON_SCRIPT=%APP_DIR%src\webui.py

REM Check if venv exists, if not, create it
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate the virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip, setuptools, and wheel to avoid build issues
pip install --upgrade pip setuptools wheel

REM Install required packages except llama-cpp-python
findstr /V /I "llama-cpp-python" "%REQUIREMENTS%" > "%APP_DIR%requirements_no_llama.txt"
pip install -r "%APP_DIR%requirements_no_llama.txt"
del "%APP_DIR%requirements_no_llama.txt"

REM Default to CUDA 12.1
set CUDA_VER=cu121

REM Try to detect CUDA version using nvidia-smi
for /f "tokens=1,2 delims=." %%a in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>nul') do (
    if "%%a"=="12" set CUDA_VER=cu121
    if "%%a"=="11" set CUDA_VER=cu118
)

REM Check if llama-cpp-python is installed
pip show llama-cpp-python >nul 2>&1
if errorlevel 1 goto install_llama

goto run

:install_llama
REM Check if nvidia-smi worked
nvidia-smi >nul 2>&1
if errorlevel 1 goto cpu

goto build

:cpu
echo No CUDA GPU detected, installing CPU-only llama-cpp-python from source
pip install --upgrade cython numpy
pip install llama-cpp-python --no-binary :all:
goto run

:build
echo CUDA GPU detected, building llama-cpp-python from source with CUDA support
pip install --upgrade cython numpy
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --no-binary :all:
goto run

:run
REM Run the Streamlit app
streamlit run "%PYTHON_SCRIPT%"

pause