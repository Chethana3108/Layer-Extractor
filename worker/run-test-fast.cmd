@echo off
REM Fast mode (lighter models) - drag an image onto this file
if "%~1"=="" (echo Drag an image onto run-test-fast.cmd & pause & exit /b 1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-test.ps1" -Image "%~1" -Fast
pause
