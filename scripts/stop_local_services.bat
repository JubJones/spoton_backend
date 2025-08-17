@echo off
REM SpotOn Backend - Stop Local Services Script (Windows)
REM This script stops all local development services

echo 🛑 Stopping SpotOn Backend Local Services
echo =========================================

REM Color codes (limited in batch)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Function equivalents for status messages
:print_status
echo %GREEN%✅ %~1%NC%
goto :eof

:print_error
echo %RED%❌ %~1%NC%
goto :eof

:print_warning
echo %YELLOW%⚠️  %~1%NC%
goto :eof

:print_info
echo %BLUE%ℹ️  %~1%NC%
goto :eof

REM Stop FastAPI backend
call :print_info "Stopping FastAPI backend processes..."
taskkill /f /im python.exe /fi "WINDOWTITLE eq *uvicorn*" >nul 2>&1
if !errorLevel! == 0 (
    call :print_status "FastAPI backend stopped"
) else (
    call :print_info "No FastAPI backend process found"
)

REM Stop any Python processes that might be running our app
call :print_info "Cleaning up Python processes..."
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv ^| findstr "python.exe"') do (
    set pid=%%i
    set pid=!pid:"=!
    tasklist /fi "pid eq !pid!" /fo csv | findstr "spoton\|uvicorn\|app.main" >nul 2>&1
    if !errorLevel! == 0 (
        taskkill /f /pid !pid! >nul 2>&1
        call :print_status "Stopped Python process !pid!"
    )
)

REM Stop Redis (if running)
call :print_info "Checking Redis server..."
redis-cli ping >nul 2>&1
if !errorLevel! == 0 (
    redis-cli shutdown >nul 2>&1
    if !errorLevel! == 0 (
        call :print_status "Redis stopped"
    ) else (
        call :print_warning "Could not stop Redis"
    )
) else (
    call :print_info "Redis is not running"
)

REM Note about PostgreSQL (don't stop system service)
sc query postgresql-x64-14 | findstr "RUNNING" >nul 2>&1
if !errorLevel! == 0 (
    call :print_info "PostgreSQL is still running (system service - not stopping)"
) else (
    call :print_info "PostgreSQL service is not running"
)

call :print_status "Local services cleanup complete"

echo.
echo To restart services:
echo 1. Start services: scripts\start_backend_local.bat
echo 2. Or manually: .venv\Scripts\activate.bat ^&^& python -m uvicorn app.main:app --reload --env-file .env.local

pause