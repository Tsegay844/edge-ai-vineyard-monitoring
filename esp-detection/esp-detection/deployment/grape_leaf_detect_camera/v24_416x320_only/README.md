# ESP32-S3 Grape Leaf Disease Detection - v24 416×320 ONLY

**Build Date:** January 13, 2026 20:02  
**Binary Size:** 4.2 MB  
**Status:** PRODUCTION - 416×320 MODEL ONLY

---

## 🎯 What's New in v24:

### ✅ **CLEAN CONFIGURATION - 416×320 MODEL ONLY**

All 320×320 model references removed from codebase. This is now a **single-model** deployment:

#### 1. **Simplified Configuration**
   - **Before (v23):** Both 320×320 and 416×320 models in config (320×320 disabled by default)
   - **After (v24):** Only 416×320 model exists in configuration
   - **Benefit:** No confusion, cleaner build, guaranteed correct model

#### 2. **416×320 Detection Model Features**
   - **Input Size:** 416×320 (aspect ratio 13:10 ≈ 1.3)
   - **Camera:** 640×480 (aspect ratio 4:3 ≈ 1.333)
   - **Padding:** Only 8px vertical (2.5% waste!)
   - **Pixels:** 133,120 (30% more than 320×320)
   - **Accuracy:** Better for complex natural scenes with overlapping leaves

#### 3. **Configuration Changes**
   - Removed `ESPDET_PICO_320_320_GRAPE_LEAF` from Kconfig
   - Removed `ESPDET_PICO_320_320_GRAPE_LEAF` enum from espdet_detect.hpp
   - Removed 320×320 case from espdet_detect.cpp
   - Removed 320×320 conditional from CMakeLists.txt
   - **Result:** Cleaner, simpler, impossible to select wrong model

---

## 📐 Letterbox Preprocessing Details:

```
Camera Input: 640×480
Model Input:  416×320

Scale Calculation:
  scale_x = 416/640 = 0.65
  scale_y = 320/480 = 0.667
  scale = min(0.65, 0.667) = 0.65  ← Width limiting

Scaled Dimensions:
  width:  640 × 0.65 = 416 ✅ (perfect fit!)
  height: 480 × 0.65 = 312 ❌ (needs 320)

Padding:
  pad_h = 320 - 312 = 8 pixels
  border_top = 4 pixels (gray RGB 114,114,114)
  border_bottom = 4 pixels

Result: Only 2.5% padding waste!
```

---

## 📊 Expected Performance:

```
Stage              Time     Notes
─────────────────────────────────────────
Camera Capture     ~80ms    640×480 JPEG
JPEG Decode        ~60ms    → RGB888
Preprocessing       22ms    Resize 0.65× + 8px pad
Detection          330ms    416×320 input ← Should see this!
Classification    1050ms    MobileNet bottleneck
─────────────────────────────────────────
TOTAL             ~1542ms   
```

**Key Metric:** Detection should be **~320-340ms** (not 283ms like 320×320!)

---

## 🚀 Flash Instructions (Linux/Mac):

```bash
chmod +x flash_v24.sh
./flash_v24.sh
```

Or manually:
```bash
esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 erase_flash

esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 16MB --flash_freq 80m \
  0x0 bootloader.bin \
  0x8000 partition-table.bin \
  0x10000 grape_leaf_detect.bin
```

---

## 🪟 Flash Instructions (Windows):

```cmd
flash_v24.bat
```

Or manually:
```cmd
esptool.py --chip esp32s3 --port COM6 --baud 921600 erase_flash

esptool.py --chip esp32s3 --port COM6 --baud 921600 ^
  --before default_reset --after hard_reset write_flash ^
  --flash_mode dio --flash_size 16MB --flash_freq 80m ^
  0x0 bootloader.bin ^
  0x8000 partition-table.bin ^
  0x10000 grape_leaf_detect.bin
```

---

## 📡 Monitor Serial Output:

```bash
# Linux/Mac
python -m serial.tools.miniterm /dev/ttyUSB0 115200

# Windows
python -m serial.tools.miniterm COM6 115200
```

---

## 🔍 What to Look For in Serial Output:

### **Detection Timing (CRITICAL VERIFICATION):**

```
I (2889) grape_leeaf_DD: Detected 7 objects (330 ms)  ← Should be 320-340ms!
```

**NOT** 283ms like the old 320×320 model!

If you see **283ms**, the old model was flashed. Reflash this v24 package.

### **Model Loading:**

```
I (1619) grape_leeaf_DD: Detection model initialized in 140 ms
```

### **Performance Summary:**

```
⏱️  Performance:
    Capture:   70 ms
    Decode:    52 ms
    Detection: 330 ms  ← VERIFY THIS!
    Disease:   33 ms (classification pipeline)
    TOTAL:     485 ms (2.06 FPS)
    Free PSRAM: 4130 KB
```

---

## 📦 Package Contents:

```
v24_416x320_only/
├── README.md                    (this file)
├── bootloader.bin               (23 KB) @ 0x0
├── partition-table.bin          (3 KB)  @ 0x8000
├── grape_leaf_detect.bin        (4.2 MB) @ 0x10000
├── flash_v24.sh                 (Linux/Mac flash script)
└── flash_v24.bat                (Windows flash script)
```

**Model Embedded in grape_leaf_detect.bin:**
- espdet_pico_416_320_grape_leaf.espdl (~550KB) - ONLY MODEL!
- mobilenetv2_128_grape_leaf.espdl (2.3MB)

---

## 🎓 Thesis Notes:

### Changes from v23:

| Aspect | v23 | v24 |
|--------|-----|-----|
| **320×320 Model** | In config (disabled) | Completely removed |
| **416×320 Model** | Default option | Only option |
| **Configuration** | Dual-model | Single-model |
| **Enum Value** | 1 | 0 (renumbered) |
| **Build Clarity** | Can confuse | Crystal clear |

### Why Remove 320×320?

1. **Simplicity:** One model = no confusion
2. **Accuracy:** 416×320 is better for complex natural scenes
3. **Minimal Overhead:** Only 45ms slower, 30% more pixels
4. **Thesis Focus:** Show optimal solution, not all options

---

## ⚠️ Verification Checklist:

After flashing v24, verify:

- ✅ **Detection time:** ~320-340ms (NOT 283ms!)
- ✅ **Binary size:** 4.2 MB
- ✅ **Compile time:** Jan 13 2026 20:02
- ✅ **Free PSRAM:** ~4100-4200 KB
- ✅ **Model loads:** "Detection model initialized in ~140 ms"

If detection is 283ms → Wrong binary, reflash v24!

---

## 🐛 Troubleshooting:

**"Detection time is 283ms, not 330ms!"**
- You flashed the wrong binary (v22 or earlier)
- Solution: Re-flash this v24 package
- Verify binary date: Jan 13 2026 20:02

**"Model not found" error:**
- Package corruption during transfer
- Solution: Re-download/extract package, reflash

**Build from source showing 283ms:**
- sdkconfig cached old config
- Solution: `idf.py fullclean && idf.py build`

---

## 📝 Version History:

- **v22 (Jan 10):** Timing breakdown with 320×320 model
- **v23 (Jan 13):** Added 416×320 as optional model
- **v24 (Jan 13):** Removed 320×320, 416×320 only ← **YOU ARE HERE**

---

Built with ESP-IDF v5.3.3 and ESP-DL v3.2.2  
For thesis: Edge AI Vineyard Monitoring System  
ESP32-S3 QFN56 v0.2 | 8MB Octal PSRAM | 16MB Flash
