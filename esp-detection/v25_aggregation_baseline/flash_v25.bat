@echo off
REM ESP32-S3 Grape Leaf Disease Detection v25 Flash Script
REM Usage: flash_v25.bat [PORT] [BAUD]
REM Example: flash_v25.bat COM3 460800

setlocal enabledelayedexpansion

echo ╔════════════════════════════════════════════════════════════╗
echo ║   ESP32-S3 Grape Leaf Disease Detection - v25 Flasher    ║
echo ║            Professional Aggregation Module                ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Default values
set PORT=%1
set BAUD=%2

if "%PORT%"=="" set PORT=COM3
if "%BAUD%"=="" set BAUD=460800

REM Check if esptool is available
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: Python not found!
    echo Install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

python -m esptool version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: esptool not found!
    echo Install with: pip install esptool
    pause
    exit /b 1
)

REM Check if binaries exist
if not exist "bootloader.bin" (
    echo ❌ Error: bootloader.bin not found!
    pause
    exit /b 1
)
if not exist "partition-table.bin" (
    echo ❌ Error: partition-table.bin not found!
    pause
    exit /b 1
)
if not exist "grape_leaf_detect.bin" (
    echo ❌ Error: grape_leaf_detect.bin not found!
    pause
    exit /b 1
)

echo ✓ Port: %PORT%
echo ✓ Baud Rate: %BAUD%
echo ✓ Firmware Binaries:
for %%F in (bootloader.bin partition-table.bin grape_leaf_detect.bin) do (
    echo   - %%F ^(%%~zF bytes^)
)
echo.

echo 📝 Flashing in 3 seconds... (Ctrl+C to cancel)
timeout /t 3 /nobreak >nul

echo.
echo 🚀 Flashing ESP32-S3...
echo.

python -m esptool ^
    --chip esp32s3 ^
    --port %PORT% ^
    --baud %BAUD% ^
    --before default_reset ^
    --after hard_reset ^
    write_flash ^
    --flash_mode dio ^
    --flash_size 8MB ^
    --flash_freq 80m ^
    0x0 bootloader.bin ^
    0x8000 partition-table.bin ^
    0x10000 grape_leaf_detect.bin

if %ERRORLEVEL% equ 0 (
    echo.
    echo ╔════════════════════════════════════════════════════════════╗
    echo ║           ✓ Flashing Completed Successfully!              ║
    echo ╚════════════════════════════════════════════════════════════╝
    echo.
    echo 📺 Next Steps:
    echo 1. Open serial monitor ^(115200 baud^):
    echo    - Arduino IDE: Tools ^> Serial Monitor
    echo    - PuTTY: Connection type Serial, Speed 115200
    echo    - Tera Term: Setup ^> Serial port ^> Speed 115200
    echo.
    echo 2. Look for this output:
    echo    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    echo    ┃   WEIGHTED DISEASE AGGREGATION RESULTS ^(Baseline Mode^)   ┃
    echo    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    echo.
    echo 3. Verify weighted class scores are displayed
    echo.
    echo 💡 Tip: To enable Hybrid mode, see README.md
) else (
    echo.
    echo ╔════════════════════════════════════════════════════════════╗
    echo ║              ❌ Flashing Failed!                           ║
    echo ╚════════════════════════════════════════════════════════════╝
    echo.
    echo Troubleshooting:
    echo 1. Check if device is in bootloader mode
    echo 2. Try holding BOOT button while connecting
    echo 3. Verify correct COM port in Device Manager
    echo 4. Install CH340/CP2102 USB driver if needed
    echo 5. Check USB cable ^(must be data cable, not charge-only^)
    pause
    exit /b 1
)

pause
