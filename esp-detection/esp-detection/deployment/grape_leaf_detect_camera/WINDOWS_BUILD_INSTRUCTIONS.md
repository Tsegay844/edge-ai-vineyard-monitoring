# Windows Build Instructions - Camera + Crop Detection

## 📦 Package Contents

This deployment package includes:
- ✅ **Camera integration** (OV2660 on Freenove ESP32-S3)
- ✅ **5-minute capture intervals** (300 seconds)
- ✅ **Top 10 bounding box cropping** (sorted by confidence)
- ✅ **Flash NVS storage** (crops replaced each cycle)
- ✅ **Fixed model embedding** (works on Windows)

---

## 🚀 Quick Start (Windows)

### 1. Extract Package

```cmd
cd C:\esp32_projects
tar -xzf grape_leaf_camera_crop_deployment.tar.gz
cd grape_leaf_detect_camera\esp-dl\examples\grape_leaf_detect
```

### 2. Clean Build

```cmd
REM Clean everything
rmdir /s /q build
del sdkconfig 2>nul
del sdkconfig.old 2>nul

REM Set target (critical!)
idf.py set-target esp32s3
```

### 3. Build

```cmd
idf.py build
```

**MUST SEE these messages:**
```
-- 🔧 DIRECT EMBED: Found model at ...
-- 🔧 DIRECT EMBED: Will embed ... as RODATA
```

**Verify firmware size at end:**
```
Total sizes:
 DRAM .data size:   xxxxx bytes
 DRAM .bss  size:   xxxxx bytes
Used static DRAM:   xxxxx bytes (  xxxxx available, x.x% used)
```

Firmware binary should be **~1.75MB**, NOT 1.27MB.

### 4. Flash

```cmd
idf.py -p COM6 flash monitor
```

Replace `COM6` with your ESP32's COM port.

---

## ✅ Expected Boot Output

```
I (xxx) grape_leaf_detect: ╔════════════════════════════════════════════════╗
I (xxx) grape_leaf_detect: ║    GRAPE LEAF DETECTION - CAMERA + CROP       ║
I (xxx) grape_leaf_detect: ╠════════════════════════════════════════════════╣
I (xxx) grape_leaf_detect: ║ Chip: ESP32-s3
I (xxx) grape_leaf_detect: ║ PSRAM: Yes
I (xxx) grape_leaf_detect: ║ Camera: OV2660 (640x480 VGA)
I (xxx) grape_leaf_detect: ╚════════════════════════════════════════════════╝
I (xxx) grape_leaf_detect: 📷 Initializing Camera (OV2660)...
I (xxx) grape_leaf_detect: ✓ Camera initialized successfully
I (xxx) grape_leaf_detect: 🧠 Initializing Detection Model...
W (xxx) FbsLoader: There is only one model in the flatbuffers...
I (xxx) grape_leaf_detect: ✓ Model initialized in ~200-300ms
I (xxx) grape_leaf_detect: 🔄 Starting Detection Loop (capture every 5 minutes)...
```

**NOT this:**
```
Guru Meditation Error: Cache disabled but cached memory region accessed
```

---

## 🔧 Troubleshooting

### Problem: "Model file not found" error

**Solution:** Model file is at:
```
esp-dl\models\grape_leaf_detect\models\s3\espdet_pico_224_224_grape_leaf.espdl
```

Verify it exists (should be ~486KB).

### Problem: Firmware size only 1.27MB

**Cause:** Model not embedded.

**Solution:**
1. Check CMakeLists.txt has the simplified embedding code
2. Rebuild from scratch: `rmdir /s /q build && idf.py set-target esp32s3 && idf.py build`
3. Must see "🔧 DIRECT EMBED" messages during build

### Problem: Camera GPIO errors

```
E (xxx) gpio: gpio_set_level(226): GPIO output gpio_num error
```

**This is NORMAL** - These are warnings for optional power pins (PWDN/RESET) set to -1. Camera will work fine.

### Problem: "Cache disabled but cached memory region accessed"

**Cause:** Model binary not in firmware.

**Solution:** 
1. Verify model file exists
2. Full clean rebuild
3. Check firmware size is ~1.75MB

---

## 📊 System Behavior

### Capture Cycle (Every 5 Minutes)

1. **Camera captures** 640x480 JPEG (~20-50ms)
2. **JPEG decodes** to RGB888 (~80-100ms)
3. **Model runs inference** (~130-150ms)
4. **Detections sorted** by confidence
5. **Top 10 crops extracted** and encoded to JPEG
6. **Old crops cleared** from flash
7. **New crops saved** to flash NVS (keys: c0-c9)
8. **Wait 5 minutes** (300 seconds)
9. **Repeat**

### Flash Storage

- **Namespace:** `crops`
- **Keys:** `c0`, `c1`, `c2`, ... `c9`
- **Behavior:** Replaced each cycle (not accumulated)
- **Persistence:** S