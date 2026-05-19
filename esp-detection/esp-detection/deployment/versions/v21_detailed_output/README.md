# ESP32-S3 Grape Leaf Disease Detection - v21 DETAILED OUTPUT

**Build Date:** January 10, 2026 14:59  
**Binary Size:** 4.2 MB  
**Compile Time:** Jan 10 2026 14:59:xx  
**Clean Build:** YES (full rebuild from scratch)

## 🆕 What's New in v21

This version displays **ALL disease classification probabilities** above 10% confidence threshold, not just the top prediction.

### Output Format Change:

**Before (v19/v20):**
```
[0] Bbox: [213,125,251,165], Conf: 0.731
    → esca (100.00%)
```

**After (v21):**
```
[0] Bbox: [213,125,251,165], Conf: 0.731
    Disease probabilities (confidence > 10%):
       • esca: 85.23%
       • black_rot: 12.45%
       • healthy: 2.10%
```

This allows you to see:
- **All competing disease classes** (not just the winner)
- **Confidence distribution** across classes
- **Ambiguous cases** where multiple diseases have similar probabilities

## 📦 Package Contents

- `bootloader.bin` (23 KB) - ESP32-S3 bootloader
- `partition-table.bin` (3 KB) - Partition layout
- `grape_leaf_detect.bin` (4.2 MB) - Main application firmware
- `flash_v21.bat` - Windows flash script
- `flash_v21.sh` - Linux/Mac flash script

## 🔧 Hardware Requirements

- **Board:** ESP32-S3 QFN56 (revision v0.2)
- **PSRAM:** 8MB Octal PSRAM (AP_3v3)
- **Flash:** 16MB
- **Camera:** OV3660 (640×480 JPEG)

## ⚡ Flash Instructions

### Windows:
```batch
cd v21_detailed_output
flash_v21.bat
```

### Linux/Mac:
```bash
cd v21_detailed_output
chmod +x flash_v21.sh
./flash_v21.sh
```

### Manual Flash:
```bash
esptool.py --chip esp32s3 --port COM6 --baud 921600 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 16MB --flash_freq 80m \
  0x0 bootloader.bin \
  0x8000 partition-table.bin \
  0x10000 grape_leaf_detect.bin
```

## 📊 Expected Serial Output

```
I (1161) app_init: Compile time:     Jan 10 2026 14:59:xx

I (1573) DiseaseClassifier: Loading MobileNetV2 model: mobilenetv2_128_grape_leaf.espdl
I (1903) DiseaseClassifier: ✓ Model loaded and validated successfully
I (1923) grape_leaf_detect: ✓ Disease classifier initialized in 349 ms

I (904183) grape_leaf_detect: 🔬 Running disease classification on top 3 detections:
I (904193) grape_leaf_detect:   [0] Bbox: [213,125,251,165], Conf: 0.731
I (904203) grape_leaf_detect:       Disease probabilities (confidence > 10%):
I (904213) grape_leaf_detect:          • esca: 85.23%
I (904223) grape_leaf_detect:          • black_rot: 12.45%
I (904733) grape_leaf_detect:   [1] Bbox: [67,75,100,110], Conf: 0.679
I (904743) grape_leaf_detect:       Disease probabilities (confidence > 10%):
I (904753) grape_leaf_detect:          • esca: 92.10%
I (905273) grape_leaf_detect:   [2] Bbox: [216,117,256,151], Conf: 0.651
I (905283) grape_leaf_detect:       Disease probabilities (confidence > 10%):
I (905293) grape_leaf_detect:          • esca: 78.56%
I (905303) grape_leaf_detect:          • healthy: 18.32%

I (905803) grape_leaf_detect: ✅ FINAL DIAGNOSIS: esca (85.2% confidence)
```

## 🔍 Verification

To verify you have the correct v21 build:
1. Check serial output for compile time: `Jan 10 2026 14:59:xx`
2. Look for new disease probability format (not just top class)
3. Binary file timestamp: Jan 10 15:00

## 🔍 Use Cases

This detailed output is useful for:
- **Model debugging:** See if model is confident or uncertain
- **Mixed symptoms:** Detect leaves showing multiple disease indicators
- **Threshold tuning:** Determine optimal confidence thresholds
- **False positive analysis:** Identify when healthy leaves get misclassified

## 🧬 Disease Classes

1. **healthy** - No disease symptoms
2. **black_rot** - Black Rot fungal infection
3. **esca** - Esca/Black Measles disease
4. **leaf_blight** - Leaf Blight (Isariopsis)

## ⚙️ System Configuration

- **Detection Model:** espdet_pico_320_320_grape_leaf (479KB YOLO-based)
- **Classification Model:** mobilenetv2_128_grape_leaf (2.3MB INT8 quantized)
- **Top-K Detections:** 3 (highest confidence crops)
- **Detection Threshold:** 0.45
- **Classification Threshold:** 0.10 (10% - shows all significant classes)

## 📈 Performance

- **Detection:** ~280ms per frame
- **Disease Classification:** ~535ms per crop (3 crops = ~1.6s)
- **Total Pipeline:** ~2 seconds per frame (~0.5 FPS with classification)
- **Detection Only:** ~2.5 FPS

## 📝 Version History

- **v17:** Initial dual-model release (MobileNetV2 missing)
- **v18:** Added debug logging, graceful error handling
- **v19:** Fixed model name extension, build cache issue
- **v20:** Fixed pointer bug in dl::Model constructor
- **v21:** **Added detailed disease probability output (threshold: 10%, CLEAN BUILD)**

## 📍 File Location

**Server Path:**
```
/home/ubuntu/edge-ai-vineyard-monitoring/esp-detection/esp-detection/
deployment/grape_leaf_detect_camera/v21_detailed_output/
```

**Archive:**
```
grape_leaf_detection_esp32s3_v21_detailed_output.tar.gz
```
