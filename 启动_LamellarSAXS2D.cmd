@echo off
chcp 65001 >nul 2>&1
call "%~dp0启动_WingSAXS.cmd" %*
exit /b %ERRORLEVEL%
