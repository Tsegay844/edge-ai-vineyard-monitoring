#!/bin/bash

# ESP32-S3 Grape Leaf Detection v19 - Flash Script (Linux/Mac)
# Enhanced debug version with comprehensive model loading diagnostics

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ESP32-S3 Grape Leaf Detection + Disease Classification   ║"
echo "║                    Version 19 (Debug)                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if esptool.py is available
if ! command -v esptool.py &> /dev/null; then
    echo "❌ Error: esptool.py not found!"
    echo "   Install: pip install esptool"
    exit 1
fi

# Detect port (Linux/Mac)
if [ -z "$1" ]; then
    echo "🔍 Detecting serial port..."
    if [ "$(uname)" == "Darwin" ]; then
        PORT=$(ls /dev/cu.usbserial-* 2>/dev/null | head -1)
        [ -z "$PORT" ] && PORT=$(ls /dev/cu.SLAB_USBtoUART 2>/dev/null | head -1)
    else
        PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -1)
        [ -z "$PORT" ] && PORT=$(ls /dev/ttyACM* 2>/dev/null | head -1)
    fi
    
    if [ -z "$PORT" ]; then
        echo "❌ No serial port detected. Please specify manually:"
        echo "   ./flash_v19.sh /dev/ttyUSB0"
        exit 1
    fi
    echo "✓ Found: $PORT"
else
    PORT=$1
fi

echo ""
echo "📋 Flash Configuration:"
echo "   Port: $PORT"
echo "   Chip: ESP32-S3"
echo "   Baud: 921600"
echo "   Flash: 16MB"
echo ""

# Erase flash
echo "🗑️  Erasing flash..."
esptool.py --chip esp32s3 --port $PORT erase_flash
if [ $? -ne 0 ]; then
    echo "❌ Flash erase failed!"
    exit 1
fi

echo ""
echo "📤 Flashing firmware v19..."
esptool.py --chip esp32s3 --port $PORT --baud 921600 \
    --before default_reset --after hard_reset \
    write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m \
    0x0 bootloader.bin \
    0x8000 partition-table.bin \
    0x10000 grape_leaf_detect.bin

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Flash complete!"
    echo ""
    echo "📊 To monitor serial output:"
    echo "   python -m serial.tools.miniterm $PORT 115200"
    echo ""
    echo "🔍 Debug logging enabled - Check serial output for:"
    echo "   - Packed binary start/end pointers"
    echo "   - Binary size and header hex dump"
    echo "   - Model loading status"
    echo "   - Graceful fallback if disease model fails"
else
    echo ""
    echo "❌ Flash failed! Check connections and try again."
    exit 1
fi
