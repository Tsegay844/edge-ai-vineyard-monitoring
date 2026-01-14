@echo off
echo =========================================================
echo  GRAPE LEAF DISEASE DETECTION - v23 416x320 MODEL
echo  ESP32-S3 Firmware Flash Tool (Windows)
echo =========================================================
echo.
echo Build Time: Jan 13 2026
echo NEW: 416x320 detection model (30%% more pixels, 2.5%% padding)
echo.

set PORT=COM6
set BAUD=921600

echo Erasing flash...
esptool.py --chip esp32s3 --port %PORT% --baud %BAUD% erase_flash
if %errorlevel% neq 0 (
    echo ERROR: Flash erase failed!
    pause
    exit /b 1
)

echo.
echo Flashing v23 firmware with 416x320 model...
esptool.py --chip esp32s3 --port %PORT% --baud %BAUD% --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m 0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 grape_leaf_detect.bin
if %errorlevel% neq 0 (
    echo ERROR: Flashing failed!
    pause
    exit /b 1
)

echo.
echo =========================================================
echo  Flashing Complete!
echo =========================================================
echo.
echo Detection model: 416x320 (33%% more pixels than v22)
echo Expected detection time: ~320-340ms (vs 285ms in v22)
echo Letterbox padding: Only 8px (2.5%% waste)
echo.
echo To monitor serial output, run:
echo python -m serial.tools.miniterm %PORT% 115200
echo.
pause
