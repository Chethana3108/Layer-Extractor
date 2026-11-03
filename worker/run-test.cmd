@echo off
REM Drag an image file onto this file, or: run-test.cmd C:\shots\pack.jpg [-Fast]
if "%~1"=="" (echo Drag an image onto run-test.cmd & pause & exit /b 1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-test.ps1" -Image "%~1" %2 %3 %4
pause
