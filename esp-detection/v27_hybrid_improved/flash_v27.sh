#!/bin/bash
# Flash script for ESP32-S3 Grape Leaf Detection v27

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ESP32-S3 Grape Leaf Detection Firmware v27  ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"

PORT="${1:-/dev/ttyUSB0}"

if ! command -v esptool.py &> /dev/null; then
    echo -e "${RED}Error: esptool.py not found${NC}"
    echo "Install with: pip install esptool"
    exit 1
fi

if [ ! -e "$PORT" ]; then
    echo -e "${YELLOW}Warning: Port $PORT not found${NC}"
    echo "Available ports:"
    ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "No serial ports found"
    echo ""
    read -p "Enter port (or press Enter for /dev/ttyUSB0): " user_port
    PORT="${user_port:-/dev/ttyUSB0}"
fi

if [ ! -f "bootloader.bin" ] || [ ! -f "partition-table.bin" ] || [ ! -f "grape_leaf_detect.bin" ]; then
    echo -e "${RED}Error: Binary files not found${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Flashing to: $PORT${NC}"
echo ""
sleep 2

echo -e "${GREEN}Starting flash...${NC}"
python3 -m esptool --chip esp32s3 -p "$PORT" -b 460800 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 8MB --flash_freq 80m \
  0x0 bootloader.bin \
  0x8000 partition-table.bin \
  0x10000 grape_leaf_detect.bin

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Flash completed successfully! ✓${NC}"
    echo ""
    echo "Monitor with: screen $PORT 115200"
else
    echo -e "${RED}Flash failed! ✗${NC}"
    exit 1
fi
