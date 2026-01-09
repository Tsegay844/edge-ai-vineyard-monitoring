@echo off
REM ============================================================================
REM  ESP32-S3 Grape Leaf Detection v7 - CORRECT BINARIES
REM  Compiled: Jan 8 2026 23:15:22 with PARTITION-TO-PSRAM fix
REM ============================================================================

set DEFAULT_PORT=COM3
set DEFAULT_BAUD=460800

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║   ESP32-S3 Grape Leaf Detection v7 - Flash Script           ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

if "%1"=="" (
    set /p USER_PORT="Enter COM port [default: %DEFAULT_PORT%]: "
    if "!USER_PORT!"=="" set USER_PORT=%DEFAULT_PORT%
) else (
    set USER_PORT=%1
)

echo.
echo 📋 Flash Configuration:
echo    • Target:       ESP32-S3
echo    • Port:         %USER_PORT%
echo    • Baud:         %DEFAULT_BAUD%
echo    • Model:        espdet_pico_320_320_grape_leaf.espdl (479 KB)
echo    • Compile Time: Jan 8 2026 23:15:22
echo.

where esptool.py >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ ERROR: esptool.py not found! Install: pip install esptool
    pause
    exit /b 1
)

if not exist "espdet_pico_320_320_grape_leaf.espdl" (
    echo ❌ ERROR: Model file missing!
    pause
    exit /b 1
)

echo 🔄 Flashing 4 files: bootloader + partition-table + app + MODEL...
echo.

esptool.py --chip esp32s3 --port %USER_PORT% --baud %DEFAULT_BAUD% --before default_reset --after hard_reset write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB 0x0 bootloader.bin 0xf000 partition-table.bin 0x10000 grape_leaf_detect.bin 0x310000 espdet_pico_320_320_grape_leaf.espdl

if %errorlevel% equ 0 (
    echo.
    echo ╔═══════════════════════════════════════════════════════════════╗
    echo ║                  ✓ FLASH SUCCESSFUL!                         ║
    echo ╚═══════════════════════════════════════════════════════════════╝
    echo.
    echo Expected output:
    echo   I ^(xxxx^) app_init: Compile time:     Jan  8 2026 23:15:22
    echo   I ^(xxxx^) grape_leaf_detect: Loading model from partition...
    echo   I ^(xxxx^) grape_leaf_detect: Found espdet_det partition...
    echo   I ^(xxxx^) grape_leaf_detect: ✓ Model copied to PSRAM
    echo.
    echo Monitor: python -m serial.tools.miniterm %USER_PORT% 115200
) else (
    echo ❌ FLASH FAILED! Check port and try bootloader mode.
)
pause
