#!/bin/bash
# ============================================================================
#  ESP32-S3 Grape Leaf Detection v7 - CORRECT BINARIES  
#  Compiled: Jan 8 2026 23:15:22 with PARTITION-TO-PSRAM fix
# ============================================================================

DEFAULT_PORT="/dev/ttyUSB0"
DEFAULT_BAUD=460800

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   ESP32-S3 Grape Leaf Detection v7 - Flash Script           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

if [ -z "$1" ]; then
    read -p "Enter port [default: $DEFAULT_PORT]: " USER_PORT
    USER_PORT=${USER_PORT:-$DEFAULT_PORT}
else
    USER_PORT=$1
fi

echo ""
echo "📋 Flash Configuration:"
echo "   • Target:       ESP32-S3"
echo "   • Port:         $USER_PORT"
echo "   • Baud:         $DEFAULT_BAUD"
echo "   • Model:        espdet_pico_320_320_grape_leaf.espdl (479 KB)"
echo "   • Compile Time: Jan 8 2026 23:15:22"
echo ""

if ! command -v esptool.py &> /dev/null; then
    echo "❌ ERROR: esptool.py not found! Install: pip install esptool"
    exit 1
fi

if [ ! -f "espdet_pico_320_320_grape_leaf.espdl" ]; then
    echo "❌ ERROR: Model file missing!"
    exit 1
fi

echo "🔄 Flashing 4 files: bootloader + partition-table + app + MODEL..."
echo ""

esptool.py --chip esp32s3 --port $USER_PORT --baud $DEFAULT_BAUD --before default_reset --after hard_reset write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB 0x0 bootloader.bin 0xf000 partition-table.bin 0x10000 grape_leaf_detect.bin 0x310000 espdet_pico_320_320_grape_leaf.espdl

if [ $? -eq 0 ]; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                  ✓ FLASH SUCCESSFUL!                         ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Expected output:"
    echo "  I (xxxx) app_init: Compile time:     Jan  8 2026 23:15:22"
    echo "  I (xxxx) grape_leaf_detect: Loading model from partition..."
    echo "  I (xxxx) grape_leaf_detect: Found espdet_det partition..."
    echo "  I (xxxx) grape_leaf_detect: ✓ Model copied to PSRAM"
    echo ""
    echo "Monitor: python -m serial.tools.miniterm $USER_PORT 115200"
else
    echo "❌ FLASH FAILED! Check port and try bootloader mode."
    exit 1
fi
