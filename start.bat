@echo off
echo ===============================================================================
echo Starting Smart Traffic Video Analytics Platform (VisionFlow)
echo ===============================================================================

:: Activate virtual environment if available
if exist .venv\Scripts\activate.bat (
    echo Activating Python virtual environment...
    call .venv\Scripts\activate.bat
)

:: Start Backend API Server
echo Starting FastAPI Backend on port 8000...
start "VisionFlow Backend" cmd /k "cd backend && python main.py"

:: Wait 2 seconds for backend initialization
timeout /t 2 /nobreak >nul

:: Start Frontend HTTP Server
echo Starting Frontend HTTP Server on port 3000...
start "VisionFlow Frontend" cmd /k "cd frontend && python -m http.server 3000"

:: Open default web browser
echo Launching VisionFlow in default web browser...
start http://localhost:3000

echo.
echo ===============================================================================
echo VisionFlow is running!
echo - Frontend: http://localhost:3000
echo - Backend:  http://localhost:8000
echo - Swagger:  http://localhost:8000/api/docs
echo ===============================================================================
