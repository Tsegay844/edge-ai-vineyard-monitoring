# 🚀 ESP32-S3 Grape Leaf Detection - Windows Flash Guide

## 📦 What's Included

- OV2660 Camera (640×480) on Freenove ESP32-S3
- AI detection model (grape leaf disease)
- 5-minute capture intervals
- Top 10 detections cropped and saved to flash
- Fixed ~40KB storage (replaces old crops)

---

## ⚠️ CAMERA FIX APPLIED (Dec 27, 2025)

**Problem:** Cache error during camera initialization (address 0x3c150d64)

**Root Cause:** Camera driver allocated 2 framebuffers in PSRAM before system fully stabilized

**Solution Applied:**
- ✅ Added 100ms delay before camera init (PSRAM stabilization)
- ✅ Reduced framebuffers: 2 → 1 (less memory pressure)
- ✅ Changed to CAMERA_GRAB_LATEST mode (more stable)
- ✅ Kept PSRAM for framebuffers (confirmed working with MicroPython)

**Package:** `esp32_flash_fix_20251227_225636.tar.gz`

---

## ⚡ Quick Flash (Recommended)

**Download fixed firmware package:**

```powershell
# 1. Download esp32_flash_fix_20251227_225636.tar.gz to Windows
# 2. Extract to folder
# 3. Open ESP-IDF PowerShell Terminal
# 4. Navigate to extracted folder: cd D:\path\to\extracted_folder
# 5. Flash:

python -m esptool --chip esp32s3 -b 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 8MB --flash_freq 80m 0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 grape_leaf_detect.bin
```

**Replace COM port if needed:**
```powershell
python -m esptool -p COM6 --chip esp32s3 -b 460800 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_size 8MB --flash_freq 80m 0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 grape_leaf_detect.bin
```

**That's it!** Skip to "Verify" section below.

## to monitor 
``
python -m serial.tools.miniterm COM6 115200
``

---

## 🔨 Build from Source (Optional)

Only if you need to modify code:

```powershell
# 1. Activate ESP-IDF
C:\Espressif\frameworks\esp-idf-v5.3.3\export.ps1

# 2. Navigate to project
cd edge-ai-vineyard-monitoring\esp-detection\esp-detection\deployment\grape_leaf_detect_camera\esp-dl\examples\grape_leaf_detect

# 3. Clean and build
idf.py fullclean
idf.py set-target esp32s3
idf.py build

# 4. Flash
idf.py -p COM6 flash monitor
```

**Expected binary size:** ~2.0 MB (0x1e9890 bytes)

---

## ✅ Verify Success

After flashing, serial monitor (115200 baud) should show:

```
╔════════════════════════════════════════════════╗
║    GRAPE LEAF DETECTION - CAMERA + CROP       ║
║ Chip: ESP32-s3                                 
║ Camera: OV2660 (640x480 VGA)                   
║ Free PSRAM: 8355583 bytes                      
╚════════════════════════════════════════════════╝

📷 Initializing Camera (OV2660)...
✓ Camera initialized successfully

🧠 Initializing Detection Model...
✓ Model initialized in 234 ms

🔄 Starting Detection Loop (capture every 5 minutes)...
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| **"esptool not found"** | `pip install esptool` |
| **"Failed to connect"** | Hold BOOT button, install CP210x/CH340 drivers |
| **"Port busy"** | Close Arduino IDE, PuTTY, other serial monitors |
| **Wrong COM port** | Check Device Manager → Ports (COM & LPT) |
| **GPIO warnings** | Normal for unused pins (PWDN/RESET = -1) |

---

## 📊 System Behavior

**Every 5 minutes:**
1. 📸 Capture image (640×480, ~30ms)
2. 🧠 Detect objects (~150ms)
3. ✂️ Crop top 10 detections
4. 🗑️ Clear old crops (c0-c9)
5. 💾 Save new crops to flash
6. ⏸️ Sleep 5 minutes

**Storage:**
- Namespace: `crops`
- Keys: `c0`, `c1`, ..., `c9`
- Size: ~40KB total (fixed)
- Behavior: Replaces old crops each cycle

**Performance:**
- Active: 0.5 sec (0.17% duty)
- Sleep: 299.5 sec (99.83%)
- Flash: 76% free (6.2 MB available)

---

**Built:** December 27, 2025  
**ESP-IDF:** v5.5.2 (Linux) / v5.3.3 (Windows)  
**Target:** ESP32-S3