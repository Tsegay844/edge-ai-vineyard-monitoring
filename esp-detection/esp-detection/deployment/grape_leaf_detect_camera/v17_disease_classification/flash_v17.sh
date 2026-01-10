#!/bin/bash
# Flash v17 Disease Classification firmware to ESP32-S3
# Usage: ./flash_v17.sh [PORT]
# Example: ./flash_v17.sh /dev/ttyUSB0

PORT="${1:-/dev/ttyUSB0}"
BAUD=460800

echo "════════════════════════════════════════════════════════"
echo "  ESP32-S3 Grape Leaf Detection v17 - Disease Classifier"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Target Device: ESP32-S3 (QFN56)"
echo "Port: $PORT"
echo "Baud Rate: $BAUD"
echo ""
echo "Binary Sizes:"
echo "  - Bootloader:        23 KB"
echo "  - Partition Table:    3 KB"
echo "  - Application:      4.2 MB (dual-model system)"
echo ""
echo "Press ENTER to start flashing..."
read

esptool.py --chip esp32s3 --port "$PORT" --baud "$BAUD" \
  --before default_reset --after hard_reset \
  write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB \
  0x0 bootloader.bin \
  0x8000 partition-table.bin \
  0x10000 grape_leaf_detect.bin

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Flashing completed successfully!"
    echo ""
    echo "To monitor serial output:"
    echo "  screen $PORT 115200"
    echo "  OR"
    echo "  minicom -D $PORT -b 115200"
    echo "  OR"
    echo "  idf.py -p $PORT monitor"
    echo ""
else
    echo ""
    echo "❌ Flashing failed!"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check USB connection"
    echo "  2. Verify port: ls /dev/ttyUSB* or ls /dev/ttyACM*"
    echo "  3. Hold BOOT button while connecting USB"
    echo "  4. Check user permissions: sudo usermod -a -G dialout \$USER"
    exit 1
fi
