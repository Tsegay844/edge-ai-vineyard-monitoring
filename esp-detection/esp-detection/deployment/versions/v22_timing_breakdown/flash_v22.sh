#!/bin/bash
echo "========================================================="
echo " GRAPE LEAF DISEASE DETECTION - v22 TIMING BREAKDOWN"
echo " ESP32-S3 Firmware Flash Tool (Linux/Mac)"
echo "========================================================="
echo ""
echo "Build Time: Jan 10 2026 16:11:xx"
echo "NEW: Detailed micro-timing for performance profiling"
echo ""

PORT="/dev/ttyUSB0"
BAUD=921600

echo "Erasing flash..."
esptool.py --chip esp32s3 --port $PORT --baud $BAUD erase_flash || {
    echo "ERROR: Flash erase failed!"
    exit 1
}

echo ""
echo "Flashing v22 firmware..."
esptool.py --chip esp32s3 --port $PORT --baud $BAUD --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m 0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 grape_leaf_detect.bin || {
    echo "ERROR: Flashing failed!"
    exit 1
}

echo ""
echo "========================================================="
echo " Flashing Complete!"
echo "========================================================="
echo ""
echo "To monitor serial output, run:"
echo "python -m serial.tools.miniterm $PORT 115200"
echo ""
