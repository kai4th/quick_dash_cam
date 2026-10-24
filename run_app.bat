@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "main.py"
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "main.py"
) else (
    python "main.py"
)
