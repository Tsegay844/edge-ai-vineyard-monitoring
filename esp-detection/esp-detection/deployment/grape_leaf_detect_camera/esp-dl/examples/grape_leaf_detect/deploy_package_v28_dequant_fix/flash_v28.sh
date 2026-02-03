#!/bin/bash
# Flash script for ESP32-S3 Grape Leaf Detection v28 (Dequantization Fix)
# Usage: sudo ./flash_v28.sh [PORT]

PORT=${1:-/dev/ttyUSB0}
BAUD=921600

echo "======================================"
echo "ESP32-S3 Firmware Flash Tool v28"
echo "CRITICAL FIX: Proper INT8 dequantization"
echo "======================================"
echo ""
echo "Port: $PORT"
echo "Baud: $BAUD"
echo ""

if [ ! -e "$PORT" ]; then
    echo "ERROR: Port $PORT not found!"
    echo "Available ports:"
    ls /dev/tty* 2>/dev/null | grep -E "(USB|ACM)" || echo "  No USB ports found"
    exit 1
fi

echo "Flashing firmware..."
python3 -m esptool --chip esp32s3 -p $PORT -b $BAUD \
    --before=default_reset --after=hard_reset write_flash \
    --flash_mode dio --flash_freq 80m --flash_size 16MB \
    0x0 bootloader.bin \
    0x8000 partition-table.bin \
    0x10000 grape_leaf_detect.bin

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Flash successful!"
    echo ""
    echo "Monitor output with:"
    echo "  python3 -m serial.tools.miniterm $PORT 115200"
else
    echo ""
    echo "✗ Flash failed!"
    exit 1
fi
