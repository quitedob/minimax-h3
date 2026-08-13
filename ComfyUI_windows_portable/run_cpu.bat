@echo off
chcp 65001 >nul
set PYTHONUTF8=1
.\python_embeded\python.exe -s ComfyUI\main.py --cpu --windows-standalone-build
pause
