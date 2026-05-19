@echo off
REM Flash script for ESP32-S3 Grape Leaf Detection v27 (Windows)
REM Usage: flash_v27.bat [PORT]
REM Example: flash_v27.bat COM6

SET PORT=%1
IF "%PORT%"=="" SET PORT=COM6

echo ==================================================
echo  ESP32-S3 Grape Leaf Detection Firmware v27
echo  Critical Fix: Disease class order corrected
echo ==================================================
echo.
echo Flashing to port: %PORT%
echo Baud rate: 921600
echo.

python -m esptool --chip esp32s3 --port %PORT% --baud 921600 ^
  --before default_reset --after hard_reset write_flash ^
  --flash_mode dio --flash_size 16MB --flash_freq 80m ^
  0x0 bootloader.bin ^
  0x8000 partition-table.bin ^
  0x10000 grape_leaf_detect.bin

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo Flash successful!
    echo To monitor output, run:
    echo   python -m serial.tools.miniterm %PORT% 115200
) ELSE (
    echo.
    echo Flash failed! Check:
    echo   1. ESP32 is connected
    echo   2. Correct port specified
    echo   3. esptool installed: pip install esptool
)

pause
