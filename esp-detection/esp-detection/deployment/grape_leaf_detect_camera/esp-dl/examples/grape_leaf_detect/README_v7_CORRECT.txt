╔═══════════════════════════════════════════════════════════════════════════╗
║           ESP32-S3 GRAPE LEAF DETECTION v7 - CORRECT BINARIES           ║
║                   COMPILED: Jan 8 2026 23:15:22                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

📦 PACKAGE: grape_leaf_detection_esp32s3_v7_CORRECT_BINARIES.tar.gz
   MD5:     a343a56a3864f498300454d66bc0a549
   Size:    938 KB

═══════════════════════════════════════════════════════════════════════════
⚠️  WHY v7? v6 HAD WRONG BINARIES!
═══════════════════════════════════════════════════════════════════════════

v6 package contained OLD binaries from 22:29 build (before partition fix)
v7 package contains CORRECT binaries from 23:15 build (with partition fix)

You MUST use v7 - v6 will continue to crash!

═══════════════════════════════════════════════════════════════════════════
⚡ FLASH INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════

WINDOWS (from extracted folder):
  flash_windows.bat COM6

MANUAL (one-line command in extracted folder):
  esptool.py --chip esp32s3 --port COM6 --baud 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB 0x0 bootloader.bin 0xf000 partition-table.bin 0x10000 grape_leaf_detect.bin 0x310000 espdet_pico_320_320_grape_leaf.espdl

═══════════════════════════════════════════════════════════════════════════
✅ EXPECTED OUTPUT (Verify compile time!)
═══════════════════════════════════════════════════════════════════════════

I (825) app_init: Compile time:     Jan  8 2026 23:15:22  ← MUST BE 23:15!
I (1506) grape_leaf_detect: 🧠 Initializing Detection Model...
I (1516) grape_leaf_detect: Loading model from partition...
I (1520) grape_leaf_detect: Found espdet_det partition at offset 0x310000, size 524288
I (1525) grape_leaf_detect: Allocating 490888 bytes in PSRAM for model...
I (1530) grape_leaf_detect: ✓ Model copied to PSRAM (490888 bytes)
I (1920) grape_leaf_detect: ✓ Model initialized successfully

═══════════════════════════════════════════════════════════════════════════
📦 PACKAGE CONTENTS
═══════════════════════════════════════════════════════════════════════════

├── bootloader.bin                           (23 KB)  @ 0x0
├── partition-table.bin                      (3 KB)   @ 0xf000
├── grape_leaf_detect.bin                    (1.5 MB) @ 0x10000
├── espdet_pico_320_320_grape_leaf.espdl     (479 KB) @ 0x310000
├── flash_windows.bat
└── flash_linux.sh

All binaries built: Jan 8 2026 23:15-23:18 with PARTITION-TO-PSRAM fix

═══════════════════════════════════════════════════════════════════════════
🔍 HOW TO VERIFY YOU'RE USING THE RIGHT BINARY
═══════════════════════════════════════════════════════════════════════════

After flashing, check serial output for:
  1. Compile time MUST be: "Jan  8 2026 23:15:22" (NOT 22:29:10)
  2. You MUST see: "Loading model from partition..."
  3. You MUST see: "Found espdet_det partition..."
  4. You MUST see: "✓ Model copied to PSRAM"

If you see compile time 22:29:10 → You flashed the WRONG package!

═══════════════════════════════════════════════════════════════════════════

✓ THIS IS THE CORRECT PACKAGE - v7 with proper binaries from 23:15 build!

═══════════════════════════════════════════════════════════════════════════
