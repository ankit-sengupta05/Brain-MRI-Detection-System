@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
    echo Node.js is required. Install Node.js and run this file again.
    pause
    exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
    echo npm is required. Install Node.js with npm and run this file again.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
    if errorlevel 1 (
        echo Frontend dependency installation failed.
        pause
        exit /b 1
    )
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    set "PYTHON_ARGS="
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo Python 3 is required. Install Python and run this file again.
            pause
            exit /b 1
        )
        set "PYTHON_EXE=python"
        set "PYTHON_ARGS="
    )
)

call "%PYTHON_EXE%" %PYTHON_ARGS% -c "import fastapi, uvicorn, multipart, PIL, torch, torchvision; from backend.app import app" >nul 2>nul
if errorlevel 1 (
    echo Installing missing backend requirements. PyTorch may take several minutes to download...
    call "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Backend dependency installation failed. See the error above.
        pause
        exit /b 1
    )
)

call "%PYTHON_EXE%" %PYTHON_ARGS% -c "import fastapi, uvicorn, multipart, PIL, torch, torchvision; from backend.app import app"
if errorlevel 1 (
    echo Backend dependencies could not be imported. See the error above.
    pause
    exit /b 1
)

start "Brain MRI Backend" cmd /k ""%PYTHON_EXE%" %PYTHON_ARGS% -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

start "Brain MRI Frontend" cmd /k "cd /d ""%~dp0"" && npm run dev"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5173"
echo Dashboard starting at http://127.0.0.1:5173
echo Keep both terminal windows open while using the dashboard.
endlocal
