#!/bin/bash
# ============================================================================
#  ESP32-S3 Grape Leaf Detection - Linux/Mac Flash Script
#  Flashes bootloader, partition table, app, and MODEL to espdet_det partition
# ============================================================================

DEFAULT_PORT="/dev/ttyUSB0"
DEFAULT_BAUD=460800

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   ESP32-S3 Grape Leaf Detection - Flash Script (Linux/Mac)  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Use provided port or ask user
if [ -z "$1" ]; then
    read -p "Enter port [default: $DEFAULT_PORT]: " USER_PORT
    USER_PORT=${USER_PORT:-$DEFAULT_PORT}
else
    USER_PORT=$1
fi

echo ""
echo "📋 Flash Configuration:"
echo "   • Target Chip:  ESP32-S3"
echo "   • Port:         $USER_PORT"
echo "   • Baud Rate:    $DEFAULT_BAUD"
echo "   • Partition:    espdet_det (model @ 0x310000)"
echo ""

# Check if esptool is installed
if ! command -v esptool.py &> /dev/null; then
    echo "❌ ERROR: esptool.py not found!"
    echo ""
    echo "Please install esptool:"
    echo "   pip install esptool"
    echo ""
    exit 1
fi

# Check if model file exists
if [ ! -f "espdet_pico_320_320_grape_leaf.espdl" ]; then
    echo "❌ ERROR: Model file not found!"
    echo "   Missing: espdet_pico_320_320_grape_leaf.espdl"
    echo ""
    exit 1
fi

echo "🔄 Starting flash process..."
echo ""

# Flash command with all components including model partition
esptool.py --chip esp32s3 --port $USER_PORT --baud $DEFAULT_BAUD \
    --before default_reset --after hard_reset \
    write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB \
    0x0 bootloader.bin \
    0xf000 partition-table.bin \
    0x10000 grape_leaf_detect.bin \
    0x310000 espdet_pico_320_320_grape_leaf.espdl

if [ $? -eq 0 ]; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                  ✓ FLASH SUCCESSFUL!                         ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "The device will reboot automatically."
    echo "Monitor output with: python -m serial.tools.miniterm $USER_PORT 115200"
    echo ""
else
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    ❌ FLASH FAILED!                           ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Troubleshooting:"
    echo " • Check port is correct (try 'ls /dev/tty*' or 'ls /dev/cu.*')"
    echo " • Ensure device is in bootloader mode (hold BOOT, press RST)"
    echo " • Try lower baud rate: 115200"
    echo " • Check USB cable supports data transfer"
    echo " • Check port permissions: 'sudo usermod -a -G dialout $USER'"
    echo ""
    exit 1
fi
