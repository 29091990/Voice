@echo off
chcp 65001 >nul
set PYTHONUTF8=1
title VoiceStudio AI Pro - Xu Ly & Doi Giong Long Tieng Video
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "app.py"
) else (
    echo Dang khoi dong bang Python...
    ".venv\Scripts\python.exe" "app.py"
    pause
)
