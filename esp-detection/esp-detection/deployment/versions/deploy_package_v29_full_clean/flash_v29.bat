@echo off
REM Flash script for ESP32-S3 Grape Leaf Detection v29 (Full Clean Build)
REM Usage: flash_v29.bat [PORT]

set PORT=%1
if "%PORT%"=="" set PORT=COM6

set BAUD=921600

echo ======================================
echo ESP32-S3 Firmware Flash Tool v29
echo FULL CLEAN BUILD - No cache artifacts
echo ======================================
echo.
echo Port: %PORT%
echo Baud: %BAUD%
echo.

echo Flashing firmware...
python -m esptool --chip esp32s3 -p %PORT% -b %BAUD% --before=default_reset --after=hard_reset write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB 0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 grape_leaf_detect.bin

if %errorlevel% equ 0 (
    echo.
    echo Flash successful!
    echo.
    echo Monitor output with:
    echo   python -m serial.tools.miniterm %PORT% 115200
) else (
    echo.
    echo Flash failed!
    pause
    exit /b 1
)

pause
