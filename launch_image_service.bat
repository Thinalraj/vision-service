@echo off
setlocal
cd /d "%~dp0"
echo Starting Vision Image FastAPI backend on port 8002...
python -m uvicorn api:app --host 0.0.0.0 --port 8002
pause
