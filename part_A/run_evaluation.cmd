@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_evaluation.ps1"
exit /b %ERRORLEVEL%
