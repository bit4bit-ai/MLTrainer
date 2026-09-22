@echo off
setlocal
title MLTrainer - Machine Vision Model Training Studio

cd /d "%~dp0"

echo =====================================================================
echo                     Starting MLTrainer Studio
echo           Machine Vision Model Training Studio in Python
echo =====================================================================
echo.

:: 1. Check if Node.js is installed
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js is not found in PATH!
    echo Please install Node.js from https://nodejs.org/ to run this app.
    pause
    exit /b 1
)

:: 2. Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ to run this app.
    pause
    exit /b 1
)

:: 3. Setup Python Backend Environment
echo [1/4] Checking Python Virtual Environment (.venv)...
if not exist "backend\.venv\" (
    echo [INFO] Creating Python virtual environment...
    python -m venv "backend\.venv"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b %ERRORLEVEL%
    )
)

echo [2/4] Verifying backend dependencies...
call "backend\.venv\Scripts\activate.bat"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r backend\requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some backend packages had notices, proceeding...
)

:: 4. Setup Frontend Dependencies
echo [3/4] Checking Frontend dependencies...
cd frontend
if not exist "node_modules\" (
    echo [INFO] node_modules folder not found. Running npm install...
    call npm install
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] npm install failed!
        cd ..
        pause
        exit /b %ERRORLEVEL%
    )
)
cd ..

:: 5. Launch Backend Server in separate window
echo [4/4] Starting FastAPI backend on http://localhost:8000 ...
start "MLTrainer Backend (FastAPI)" cmd /k "cd /d "%~dp0" && call backend\.venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

:: 6. Launch Frontend Dev Server and open browser
echo [INFO] Starting Vite Frontend on http://localhost:3000 ...
cd frontend
call npm run dev -- --open

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with an error code.
    pause
)
