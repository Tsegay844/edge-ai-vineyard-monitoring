@echo off
REM ============================================================================
REM  ESP32-S3 Grape Leaf Detection - Windows Flash Script
REM  Flashes bootloader, partition table, app, and MODEL to espdet_det partition
REM ============================================================================

set DEFAULT_PORT=COM3
set DEFAULT_BAUD=460800

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║   ESP32-S3 Grape Leaf Detection - Flash Script (Windows)    ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

REM Check if port provided as argument
if "%1"=="" (
    set /p USER_PORT="Enter COM port [default: %DEFAULT_PORT%]: "
    if "!USER_PORT!"=="" set USER_PORT=%DEFAULT_PORT%
) else (
    set USER_PORT=%1
)

echo.
echo 📋 Flash Configuration:
echo    • Target Chip:  ESP32-S3
echo    • COM Port:     %USER_PORT%
echo    • Baud Rate:    %DEFAULT_BAUD%
echo    • Partition:    espdet_det (model @ 0x310000)
echo.

REM Check if esptool is installed
where esptool.py >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ ERROR: esptool.py not found!
    echo.
    echo Please install esptool:
    echo    pip install esptool
    echo.
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "espdet_pico_320_320_grape_leaf.espdl" (
    echo ❌ ERROR: Model file not found!
    echo    Missing: espdet_pico_320_320_grape_leaf.espdl
    echo.
    pause
    exit /b 1
)

echo 🔄 Starting flash process...
echo.

REM Flash command with all components including model partition
esptool.py --chip esp32s3 --port %USER_PORT% --baud %DEFAULT_BAUD% ^
    --before default_reset --after hard_reset ^
    write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB ^
    0x0 bootloader.bin ^
    0xf000 partition-table.bin ^
    0x10000 grape_leaf_detect.bin ^
    0x310000 espdet_pico_320_320_grape_leaf.espdl

if %errorlevel% equ 0 (
    echo.
    echo ╔═══════════════════════════════════════════════════════════════╗
    echo ║                  ✓ FLASH SUCCESSFUL!                         ║
    echo ╚═══════════════════════════════════════════════════════════════╝
    echo.
    echo The device will reboot automatically.
    echo Monitor output with: python -m serial.tools.miniterm %USER_PORT% 115200
    echo.
) else (
    echo.
    echo ╔═══════════════════════════════════════════════════════════════╗
    echo ║                    ❌ FLASH FAILED!                           ║
    echo ╚═══════════════════════════════════════════════════════════════╝
    echo.
    echo Troubleshooting:
    echo  • Check COM port is correct
    echo  • Ensure device is in bootloader mode (hold BOOT, press RST)
    echo  • Try lower baud rate: 115200
    echo  • Check USB cable supports data transfer
    echo.
)

pause
