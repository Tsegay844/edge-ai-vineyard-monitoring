@echo off
REM ESP32-S3 Grape Leaf Detection v20 - WORKING VERSION
echo ╔═══════════════════════════════════════════════════╗
echo ║  ESP32-S3 Grape Leaf Detection v20 (WORKING!)    ║
echo ╚═══════════════════════════════════════════════════╝

if "%1"=="" (
    echo Usage: flash_v20.bat COM_PORT
    echo Example: flash_v20.bat COM6
    exit /b 1
)

set PORT=%1
echo Port: %PORT%
echo Erasing flash...
esptool.py --chip esp32s3 --port %PORT% erase_flash

echo.
echo Flashing v20...
esptool.py --chip esp32s3 --port %PORT% --baud 921600 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m 0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 grape_leaf_detect.bin

echo.
echo ✅ Done! Monitor with:
echo    python -m serial.tools.miniterm %PORT% 115200
