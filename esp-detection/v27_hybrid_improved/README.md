# ESP32-S3 Grape Leaf Disease Detection Firmware v27
## Hybrid Aggregation with Improved Preprocessing

**Build Date:** January 14, 2026  
**Version:** v27  
**Target:** ESP32-S3 (QFN56, 8MB PSRAM, 16MB Flash)  
**ESP-IDF:** v5.3.3  

---

## 🎯 Key Features

### 1. **Adaptive Classification Threshold**
- **Threshold:** 0.3 confidence (optimized for 416×320 detection model)
- **Result:** Classifies 5-7 leaves per frame (vs 2-3 with 0.5 threshold)

### 2. **Letterbox Preprocessing**
- Aspect ratio preservation for crop preprocessing
- Square bbox expansion centered on leaf
- Gray padding (127) for out-of-bounds pixels
- Better MobileNetV2 input quality

### 3. **Fixed Filtering Logic**
- Correctly collects only high-confidence detections (≥0.3)
- No index mismatches between detection and classification
- Processes exactly the leaves that meet threshold

### 4. **Hybrid Aggregation**
- **Entropy Weighting:** Weights by prediction certainty
- **Spatial Weighting:** Weights by bbox quality (size, position, detection confidence)
- More accurate final disease assessment

---

## 📦 Package Contents

```
v27_hybrid_improved/
├── bootloader.bin          (23 KB)
├── partition-table.bin     (3 KB)
├── grape_leaf_detect.bin   (4.2 MB)
├── README.md              (this file)
├── flash_v27.sh           (Linux/Mac)
└── flash_v27.bat          (Windows)
```

---

## ⚡ Flash Instructions

### Linux/Mac:
```bash
cd v27_hybrid_improved
chmod +x flash_v27.sh
./flash_v27.sh
```

### Windows:
```cmd
cd v27_hybrid_improved
flash_v27.bat
```

### Manual Flash:
```bash
python -m esptool --chip esp32s3 -p /dev/ttyUSB0 -b 460800 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 8MB --flash_freq 80m \
  0x0 bootloader.bin \
  0x8000 partition-table.bin \
  0x10000 grape_leaf_detect.bin
```

---

## 🔍 Monitor Output

```bash
idf.py monitor -p /dev/ttyUSB0
# or
screen /dev/ttyUSB0 115200
```

---

## 📝 Expected Output

```
Detected 10 objects (360 ms)
Running disease classification on top 7 detections:
  [0] Bbox: [120,80,220,180], Leaf_Confidence: 0.850
      Disease Probability:
         • healthy: 75.00%
         • esca: 15.00%
  ...

WEIGHTED DISEASE AGGREGATION RESULTS:
  Method: HYBRID (Entropy + Spatial Weighting)
  Final Disease Distribution:
    • healthy: 45.20%
    • esca: 30.50%
    • black_rot: 20.30%
```

---

**Built with:** ESP-IDF v5.3.3 | ESP-DL 3.2.2 | Clean build (no cache)
