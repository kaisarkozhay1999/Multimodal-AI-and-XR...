@echo off
"%~dp0..\.venv-samurai\Scripts\python.exe" "%~dp0scripts\compute_human_centric_indicator.py"
exit /b %ERRORLEVEL%
