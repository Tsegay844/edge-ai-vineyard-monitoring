@echo off
REM Flash v17 Disease Classification firmware to ESP32-S3
REM Usage: flash_v17.bat [PORT]
REM Example: flash_v17.bat COM6

SET PORT=%1
IF "%PORT%"=="" SET PORT=COM6
SET BAUD=460800

echo ========================================================
echo   ESP32-S3 Grape Leaf Detection v17 - Disease Classifier
echo ========================================================
echo.
echo Target Device: ESP32-S3 (QFN56)
echo Port: %PORT%
echo Baud Rate: %BAUD%
echo.
echo Binary Sizes:
echo   - Bootloader:        23 KB
echo   - Partition Table:    3 KB
echo   - Application:      4.2 MB (dual-model system)
echo.
echo Press any key to start flashing...
pause >nul

esptool.py --chip esp32s3 --port %PORT% --baud %BAUD% ^
  --before default_reset --after hard_reset ^
  write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB ^
  0x0 bootloader.bin ^
  0x8000 partition-table.bin ^
  0x10000 grape_leaf_detect.bin

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Flashing completed successfully!
    echo.
    echo To monitor serial output, use one of:
    echo   - Arduino Serial Monitor
    echo   - PuTTY: Port=%PORT%, Baud=115200
    echo   - idf.py -p %PORT% monitor
    echo.
) ELSE (
    echo.
    echo ❌ Flashing failed!
    echo.
    echo Troubleshooting:
    echo   1. Check USB connection
    echo   2. Verify port in Device Manager
    echo   3. Hold BOOT button while connecting USB
    echo   4. Try different USB cable or port
    echo.
    pause
    exit /b 1
)

pause
