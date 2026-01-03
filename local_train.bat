@echo off
REM Quick start script for local DirectML training

echo ============================================================
echo MusicGen Local Training (DirectML)
echo ============================================================
echo.

REM Check if input folder exists
if not exist "audio_loops" (
    echo ERROR: audio_loops folder not found!
    echo Please create an 'audio_loops' folder and add your audio files.
    pause
    exit /b 1
)

REM Create output folder
if not exist "trained_models" mkdir trained_models

echo Starting training...
echo Input: audio_loops/
echo Output: trained_models/
echo.

python local_train_directml.py ^
    --input-folder ./audio_loops ^
    --output-folder ./trained_models ^
    --model-name facebook/musicgen-small ^
    --lora-rank 8 ^
    --lora-alpha 16 ^
    --learning-rate 1e-4 ^
    --num-epochs 10 ^
    --batch-size 2 ^
    --drum-mode ^
    --enhance-percussion

echo.
echo ============================================================
echo Training complete! Model saved to: trained_models/
echo ============================================================
pause
