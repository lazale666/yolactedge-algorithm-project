@echo off
setlocal
set SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%..\yolact_edge\.venv38\Scripts\python.exe" "%SCRIPT_DIR%gui_app.py"
