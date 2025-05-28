@echo off
REM Batch file to run the AI Command App Streamlit UI from any directory

REM Get the directory of this batch file
set SCRIPT_DIR=%~dp0

REM Set the Python script path relative to the batch file
set PYTHON_SCRIPT=%SCRIPT_DIR%src\webui.py

REM Activate virtual environment if needed
REM call "%SCRIPT_DIR%venv\Scripts\activate"

REM Run the Streamlit app
streamlit run "%PYTHON_SCRIPT%"