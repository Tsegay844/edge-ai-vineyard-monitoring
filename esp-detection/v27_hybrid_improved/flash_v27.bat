@echo off
REM Flash script for ESP32-S3 Grape Leaf Detection v27

echo ===============================================
echo   ESP32-S3 Grape Leaf Detection Firmware v27  
echo ===============================================
echo.

set PORT=%1
if "%PORT%"=="" set PORT=COM3

python -m esptool version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: esptool not found
    echo Install with: pip install esptool
    pause
    exit /b 1
)

if not exist "bootloader.bin" (
    echo Error: Binary files not found
    pause
    exit /b 1
)

echo Flashing to: %PORT%
echo.
pause

echo Starting flash...
python -m esptool --chip esp32s3 -p %PORT% -b 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 8MB --flash_freq 80m 0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 grape_leaf_detect.bin

if %errorlevel% equ 0 (
    echo.
    echo Flash completed successfully!
    echo.
) else (
    echo Flash failed!
)

pause
