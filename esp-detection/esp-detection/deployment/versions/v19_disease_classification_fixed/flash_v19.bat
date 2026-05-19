@echo off
REM ESP32-S3 Grape Leaf Detection v19 - Flash Script (Windows)
REM Enhanced debug version with comprehensive model loading diagnostics

echo ╔════════════════════════════════════════════════════════════╗
echo ║  ESP32-S3 Grape Leaf Detection + Disease Classification   ║
echo ║                    Version 19 (Debug)                      ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check for COM port argument
if "%1"=="" (
    echo ❌ Error: Please specify COM port
    echo    Usage: flash_v19.bat COM6
    echo.
    echo 💡 To find your COM port:
    echo    - Open Device Manager
    echo    - Look under "Ports (COM ^& LPT)"
    echo    - Find "Silicon Labs CP210x" or "USB Serial Port"
    exit /b 1
)

set PORT=%1

echo 📋 Flash Configuration:
echo    Port: %PORT%
echo    Chip: ESP32-S3
echo    Baud: 921600
echo    Flash: 16MB
echo.

REM Check if esptool.py exists
where esptool.py >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Error: esptool.py not found!
    echo    Install: pip install esptool
    exit /b 1
)

REM Erase flash
echo 🗑️  Erasing flash...
esptool.py --chip esp32s3 --port %PORT% erase_flash
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Flash erase failed!
    exit /b 1
)

echo.
echo 📤 Flashing firmware v19...
esptool.py --chip esp32s3 --port %PORT% --baud 921600 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m 0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 grape_leaf_detect.bin

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Flash complete!
    echo.
    echo 📊 To monitor serial output:
    echo    python -m serial.tools.miniterm %PORT% 115200
    echo.
    echo 🔍 Debug logging enabled - Check serial output for:
    echo    - Packed binary start/end pointers
    echo    - Binary size and header hex dump
    echo    - Model loading status
    echo    - Graceful fallback if disease model fails
) else (
    echo.
    echo ❌ Flash failed! Check connections and try again.
    exit /b 1
)
