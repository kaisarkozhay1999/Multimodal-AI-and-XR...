@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_paper_annotation.ps1"
exit /b %ERRORLEVEL%
