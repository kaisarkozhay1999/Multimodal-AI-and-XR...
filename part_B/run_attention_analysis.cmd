@echo off
"%~dp0..\.venv-samurai\Scripts\python.exe" "%~dp0scripts\analyze_attention.py"
exit /b %ERRORLEVEL%
