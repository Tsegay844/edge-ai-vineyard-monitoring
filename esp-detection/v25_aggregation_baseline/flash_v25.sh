#!/bin/bash

# ESP32-S3 Grape Leaf Disease Detection v25 Flash Script
# Usage: ./flash_v25.sh [PORT] [BAUD]
# Example: ./flash_v25.sh /dev/ttyUSB0 460800

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   ESP32-S3 Grape Leaf Disease Detection - v25 Flasher    ║${NC}"
echo -e "${BLUE}║            Professional Aggregation Module                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Default values
PORT=${1:-/dev/ttyUSB0}
BAUD=${2:-460800}

# Check if esptool is available
if ! command -v esptool.py &> /dev/null; then
    echo -e "${RED}❌ Error: esptool.py not found!${NC}"
    echo "Install with: pip install esptool"
    exit 1
fi

# Check if port exists
if [ ! -e "$PORT" ]; then
    echo -e "${YELLOW}⚠️  Warning: Port $PORT not found!${NC}"
    echo ""
    echo "Available serial ports:"
    ls -1 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  No USB serial devices found"
    echo ""
    echo "Detected ports:"
    python3 -m serial.tools.list_ports 2>/dev/null || echo "  Install pyserial: pip install pyserial"
    echo ""
    read -p "Enter port (e.g., /dev/ttyUSB0): " PORT
fi

# Check if binaries exist
if [ ! -f "bootloader.bin" ] || [ ! -f "partition-table.bin" ] || [ ! -f "grape_leaf_detect.bin" ]; then
    echo -e "${RED}❌ Error: Missing firmware binaries!${NC}"
    echo "Required files:"
    echo "  - bootloader.bin"
    echo "  - partition-table.bin"
    echo "  - grape_leaf_detect.bin"
    exit 1
fi

echo -e "${GREEN}✓ Port:${NC} $PORT"
echo -e "${GREEN}✓ Baud Rate:${NC} $BAUD"
echo -e "${GREEN}✓ Firmware Binaries:${NC}"
echo "  - bootloader.bin       ($(stat -c%s bootloader.bin 2>/dev/null || stat -f%z bootloader.bin) bytes)"
echo "  - partition-table.bin  ($(stat -c%s partition-table.bin 2>/dev/null || stat -f%z partition-table.bin) bytes)"
echo "  - grape_leaf_detect.bin ($(stat -c%s grape_leaf_detect.bin 2>/dev/null || stat -f%z grape_leaf_detect.bin) bytes)"
echo ""

echo -e "${YELLOW}📝 Flashing in 3 seconds... (Ctrl+C to cancel)${NC}"
sleep 3

echo ""
echo -e "${BLUE}🚀 Flashing ESP32-S3...${NC}"
echo ""

python3 -m esptool \
    --chip esp32s3 \
    --port "$PORT" \
    --baud "$BAUD" \
    --before default_reset \
    --after hard_reset \
    write_flash \
    --flash_mode dio \
    --flash_size 8MB \
    --flash_freq 80m \
    0x0 bootloader.bin \
    0x8000 partition-table.bin \
    0x10000 grape_leaf_detect.bin

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           ✓ Flashing Completed Successfully!              ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}📺 Next Steps:${NC}"
    echo "1. Open serial monitor: screen $PORT 115200"
    echo "   (or use Arduino IDE Serial Monitor / PuTTY / minicom)"
    echo ""
    echo "2. Look for this output:"
    echo -e "   ${GREEN}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
    echo -e "   ${GREEN}┃   WEIGHTED DISEASE AGGREGATION RESULTS (Baseline Mode)   ┃${NC}"
    echo -e "   ${GREEN}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"
    echo ""
    echo "3. Verify weighted class scores are displayed"
    echo ""
    echo -e "${YELLOW}💡 Tip: To enable Hybrid mode, see README.md${NC}"
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              ❌ Flashing Failed!                           ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "1. Check if device is in bootloader mode"
    echo "2. Try holding BOOT button while connecting"
    echo "3. Verify port permissions: sudo chmod 666 $PORT"
    echo "4. Check USB cable (must be data cable, not charge-only)"
    exit 1
fi
