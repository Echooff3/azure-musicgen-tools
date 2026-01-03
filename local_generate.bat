@echo off
REM Quick start script for music generation

echo ============================================================
echo MusicGen Music Generation (DirectML)
echo ============================================================
echo.

REM Check if model exists
if not exist "trained_models\final_model" (
    echo ERROR: No trained model found!
    echo Please run local_train.bat first.
    pause
    exit /b 1
)

REM Create output folder
if not exist "generated_music" mkdir generated_music

set /p PROMPT="Enter prompt (default: upbeat electronic drums): "
if "%PROMPT%"=="" set PROMPT=upbeat electronic drums

echo.
echo Generating music with prompt: "%PROMPT%"
echo.

python local_generate_directml.py ^
    --model-folder ./trained_models/final_model ^
    --base-model facebook/musicgen-small ^
    --prompt "%PROMPT%" ^
    --output ./generated_music/output.wav ^
    --max-new-tokens 256 ^
    --temperature 1.0 ^
    --guidance-scale 3.0

echo.
echo ============================================================
echo Music generated! Saved to: generated_music/output.wav
echo ============================================================
pause
