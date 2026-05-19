╔═══════════════════════════════════════════════════════════════════════════╗
║                ESP32-S3 GRAPE LEAF DETECTION v6 - COMPLETE               ║
║                        WITH MODEL PARTITION FIX                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

📦 PACKAGE: grape_leaf_detection_esp32s3_flash_package_v6_WITH_MODEL.tar.gz
   MD5:     a95f06324e49e0f6ca40f0a7fdd53a5a
   Size:    ~1.3 MB

═══════════════════════════════════════════════════════════════════════════
🔧 WHAT'S NEW IN v6 - CRITICAL FIX
═══════════════════════════════════════════════════════════════════════════

✓ MODEL FILE NOW INCLUDED: espdet_pico_320_320_grape_leaf.espdl (479 KB)
✓ FLASH SCRIPTS UPDATED: Now flash model to partition at 0x310000
✓ PARTITION-TO-PSRAM: Code reads model from partition into PSRAM
✓ MEMORY LOCATION: Model initialized from writable PSRAM memory

Previous versions (v2-v5) FAILED because:
  ❌ Model file was NOT flashed to the espdet_det partition
  ❌ Partition was empty, causing undefined behavior during read
  ❌ Cache write-back error occurred when accessing uninitialized partition

═══════════════════════════════════════════════════════════════════════════
📦 PACKAGE CONTENTS
═══════════════════════════════════════════════════════════════════════════

├── bootloader.bin                           (23 KB)  @ 0x0
├── partition-table.bin                      (3 KB)   @ 0xf000
├── grape_leaf_detect.bin                    (1.9 MB) @ 0x10000
├── espdet_pico_320_320_grape_leaf.espdl     (479 KB) @ 0x310000  ← NEW!
├── flash_windows.bat                        ← Flashes all 4 files
├── flash_linux.sh                           ← Flashes all 4 files
└── FLASH_INSTRUCTIONS_WINDOWS.txt

═══════════════════════════════════════════════════════════════════════════
⚡ QUICK START
═══════════════════════════════════════════════════════════════════════════

WINDOWS:
  1. Extract package
  2. Open Command Prompt in extracted folder
  3. Run: flash_windows.bat COM3
     (Replace COM3 with your port)

LINUX/MAC:
  1. Extract package
  2. cd to extracted folder
  3. Run: ./flash_linux.sh /dev/ttyUSB0
     (Replace with your port)

═══════════════════════════════════════════════════════════════════════════
🔍 EXPECTED SERIAL OUTPUT (Success)
═══════════════════════════════════════════════════════════════════════════

I (1443) grape_leaf_detect: 🧠 Initializing Detection Model...
I (1443) grape_leaf_detect: Loading model from partition...
I (1450) grape_leaf_detect: Found espdet_det partition at offset 0x310000, size 524288
I (1455) grape_leaf_detect: Allocating 490888 bytes in PSRAM for model...
I (1460) grape_leaf_detect: ✓ Model copied to PSRAM (490888 bytes)
I (1850) grape_leaf_detect: ✓ Model initialized successfully
I (1855) grape_leaf_detect: 📸 Starting detection loop...

═══════════════════════════════════════════════════════════════════════════
📝 MANUAL FLASH (If scripts fail)
═══════════════════════════════════════════════════════════════════════════

esptool.py --chip esp32s3 --port COM3 --baud 460800 \
    --before default_reset --after hard_reset \
    write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB \
    0x0 bootloader.bin \
    0xf000 partition-table.bin \
    0x10000 grape_leaf_detect.bin \
    0x310000 espdet_pico_320_320_grape_leaf.espdl

═══════════════════════════════════════════════════════════════════════════
⚙️ TECHNICAL DETAILS
═══════════════════════════════════════════════════════════════════════════

Partition Table:
  • nvs         (24 KB)  @ 0x9000   - WiFi/BT data
  • phy_init    (4 KB)   @ 0xf000   - RF calibration
  • factory     (3 MB)   @ 0x10000  - Application
  • espdet_det  (512 KB) @ 0x310000 - Model storage ← MODEL HERE!

Model Loading Flow:
  1. esp_partition_find_first("espdet_det") - Locate partition
  2. heap_caps_malloc(SPIRAM | 8BIT) - Allocate PSRAM buffer
  3. esp_partition_read() - Copy 479 KB from flash to PSRAM
  4. new dl::Model(PSRAM_ptr, "grape_leaf", MEMORY) - Init from RAM
  5. m_model->minimize() - Optimize (no flash write-back!)

Memory Configuration:
  • FLASH_PARTITION mode enabled
  • MODEL_LOCATION = MEMORY (not FLASH_RODATA)
  • SPIRAM cache: instructions + rodata enabled
  • PSRAM: 8 MB available for model + frame buffers

═══════════════════════════════════════════════════════════════════════════
🐛 TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════

Problem: Still crashes with "Cache disabled but cached memory region accessed"
  → Ensure v6 package was used (includes model file)
  → Verify all 4 files were flashed (check flash log)
  → Check bootloader shows: "3 espdet_det Unknown data 01 82 00310000..."

Problem: "Partition not found"
  → Reflash partition-table.bin at 0xf000
  → Erase flash first: esptool.py --port COM3 erase_flash

Problem: Flash fails
  → Hold BOOT button, press RST, release RST, release BOOT
  → Try lower baud: change 460800 to 115200
  → Check USB cable (must support data, not just power)

═══════════════════════════════════════════════════════════════════════════
📊 BUILD INFO
═══════════════════════════════════════════════════════════════════════════

Compiled:       Jan 8 2026 23:15:22
ESP-IDF:        v5.3.3
Commit:         15d150d-dirty
Model:          espdet_pico 320x320 grape_leaf
Model Size:     479 KB (490,888 bytes)
Target:         ESP32-S3 (QFN56, 8MB PSRAM, 16MB Flash)
Camera:         OV3660 VGA (640×480)

═══════════════════════════════════════════════════════════════════════════

✓ This version should WORK - model is now properly flashed to partition!

═══════════════════════════════════════════════════════════════════════════
