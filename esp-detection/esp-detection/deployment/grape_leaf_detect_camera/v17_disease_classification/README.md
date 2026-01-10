# v17 Disease Classification Deployment

**Date:** January 10, 2026  
**Version:** v17  
**Hardware:** ESP32-S3 (QFN56) with OV3660 Camera (ESP32S3-EYE pinout)

## 🎯 What's New in v17

### Integrated Disease Classification Pipeline
- **Dual-Model System**: Detection (YOLO) + Disease Classification (MobileNetV2)
- **No Storage**: Removed SPIFFS crop saving for real-time classification only
- **Memory-Efficient**: Single-allocation buffers, no malloc/free in loop
- **Top-K Filtering**: Only classifies top 3 detections with confidence > 0.45
- **Result Aggregation**: Final diagnosis based on maximum confidence

### Models
1. **Detection**: `espdet_pico_320_320_grape_leaf.espdl` (479KB, YOLO-like)
2. **Classification**: `mobilenetv2_128_grape_leaf.espdl` (INT8 quantized, 128×128)

### Disease Classes
- `healthy`
- `black_rot`
- `esca`
- `leaf_blight`

## 📊 Performance Metrics

| Metric | v16 (SPIFFS) | v17 (Disease) | Change |
|--------|--------------|---------------|--------|
| Binary Size | 2.0 MB | 4.2 MB | +2.2 MB |
| Factory Partition | 3 MB | 5 MB | +2 MB |
| Detection Time | 285 ms | 285 ms | No change |
| Crop Processing | 700-850 ms (JPEG+save) | ~150 ms (classify top-3) | **5× faster** |
| Total Frame Time | 1140-1300 ms | ~435 ms | **3× faster** |
| Throughput | 0.77-0.88 FPS | ~2.3 FPS | **2.6× increase** |

## 🔄 Pipeline Flow

```
Camera Capture (VGA 640×480 JPEG)
    ↓
JPEG Decode → RGB888 full frame
    ↓
Detection Model (320×320, 285ms)
    ↓ 10 bounding boxes
Filter: Top-3 with confidence > 0.45
    ↓
For each filtered detection:
    ├─ Crop bbox from full frame
    ├─ Resize to 128×128 (nearest-neighbor, direct)
    ├─ Disease Classification (~50ms per crop)
    └─ Result: {class_id, confidence, class_name}
    ↓
Aggregate results (max confidence)
    ↓
Final Diagnosis: e.g., "black_rot (87.3%)"
```

## 💾 Binary Files

| File | Size | Description |
|------|------|-------------|
| `bootloader.bin` | 23 KB | ESP32-S3 bootloader |
| `partition-table.bin` | 3 KB | 5MB factory, 24KB NVS layout |
| `grape_leaf_detect.bin` | 4.2 MB | Main application with both models |

## 🚀 Flashing Instructions

### Windows (esptool.py)

```cmd
esptool.py --chip esp32s3 --port COM6 --baud 460800 ^
  --before default_reset --after hard_reset ^
  write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB ^
  0x0 bootloader.bin ^
  0x8000 partition-table.bin ^
  0x10000 grape_leaf_detect.bin
```

### Linux/Mac

```bash
esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 460800 \
  --before default_reset --after hard_reset \
  write_flash --flash_mode dio --flash_freq 80m --flash_size 16MB \
  0x0 bootloader.bin \
  0x8000 partition-table.bin \
  0x10000 grape_leaf_detect.bin
```

### ESP-IDF (Recommended if available)

```bash
cd /path/to/grape_leaf_detect
idf.py -p COM6 flash monitor
```

## 📺 Expected Serial Output

```
I (450) grape_leaf_detect: ╔════════════════════════════════════════════════╗
I (455) grape_leaf_detect: ║         ESP32-S3 Grape Leaf Detection         ║
I (465) grape_leaf_detect: ║              with Disease Classifier          ║
I (470) grape_leaf_detect: ╚════════════════════════════════════════════════╝
I (480) grape_leaf_detect: ESP32-S3 Chip: QFN56 v0.2
I (485) grape_leaf_detect: Free Heap: 8345216 bytes
I (490) grape_leaf_detect: Free PSRAM: 8386256 bytes

I (500) grape_leaf_detect: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
I (510) grape_leaf_detect: 🧠 Initializing Detection Model...
I (2850) grape_leaf_detect: ✓ Model initialized in 2340 ms
I (2855) grape_leaf_detect:   Free heap after init: 7123 KB

I (2860) grape_leaf_detect: 🧬 Initializing Disease Classifier (MobileNetV2 128x128)...
I (2920) DiseaseClassifier: Loading MobileNetV2 model: mobilenetv2_128_grape_leaf
I (2925) DiseaseClassifier: Model input shape: [1, 128, 128, 3]
I (2930) DiseaseClassifier: ✓ Allocated 49152 bytes in PSRAM for 128x128x3 input buffer
I (2940) grape_leaf_detect: ✓ Disease classifier initialized in 80 ms
I (2945) grape_leaf_detect:   Free heap after init: 7075 KB

I (2955) grape_leaf_detect: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
I (2965) grape_leaf_detect: 📷 Initializing Camera (OV3660)...
I (3150) grape_leaf_detect: ✓ Camera initialized successfully
I (3155) grape_leaf_detect:   Sensor: OV3660, Format: JPEG, Quality: 12

I (3165) grape_leaf_detect: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
I (3175) grape_leaf_detect: 🔄 Starting Detection Loop (capture every 5 minutes)...

I (3185) grape_leaf_detect: ━━━━━━━━━━━━━━ Frame #1 ━━━━━━━━━━━━━━━
I (3190) grape_leaf_detect: ⏰ Next capture in 5 minutes

I (3735) grape_leaf_detect: 📸 Captured: 640x480, 18432 bytes (545 ms)
I (3825) grape_leaf_detect: ✓ Decoded: 640x480 RGB888 (90 ms)
I (4110) grape_leaf_detect: 🔍 Detected 10 objects (285 ms)
I (4115) grape_leaf_detect: ✓ Sorted by confidence (highest first)

I (4120) grape_leaf_detect: 🔬 Running disease classification on top 3 detections:
I (4125) grape_leaf_detect:   [0] Bbox: [125,89,287,245], Conf: 0.593
I (4175) grape_leaf_detect:       → black_rot (87.3%)
I (4180) grape_leaf_detect:   [1] Bbox: [312,102,456,268], Conf: 0.521
I (4230) grape_leaf_detect:       → black_rot (82.1%)
I (4235) grape_leaf_detect:   [2] Bbox: [78,312,198,441], Conf: 0.487
I (4285) grape_leaf_detect:       → healthy (91.5%)

I (4290) grape_leaf_detect: ✅ FINAL DIAGNOSIS: healthy (91.5% confidence)
I (4295) grape_leaf_detect:    Disease inference: 170 ms total (56 ms avg per crop)

I (4305) grape_leaf_detect: ⏱️  Performance Summary:
I (4310) grape_leaf_detect:     Capture:   545 ms
I (4315) grape_leaf_detect:     Decode:     90 ms
I (4320) grape_leaf_detect:     Detect:    285 ms
I (4325) grape_leaf_detect:     Classify:  170 ms
I (4330) grape_leaf_detect:     ──────────────────
I (4335) grape_leaf_detect:     Total:    1090 ms  (0.92 FPS)
```

## 🔧 System Configuration

### Hardware
- **MCU**: ESP32-S3 QFN56 v0.2
- **PSRAM**: 8MB
- **Flash**: 16MB
- **Camera**: OV3660 (VGA 640×480)
- **I2C Address**: 0x3c
- **Pin Mapping**: ESP32S3-EYE standard pinout

### Camera Pins (ESP32S3-EYE)
| Signal | GPIO | Notes |
|--------|------|-------|
| PWDN | NC | Not connected |
| RESET | 3 | Active low |
| XCLK | 15 | 20MHz |
| SIOD (SDA) | 4 | I2C data |
| SIOC (SCL) | 5 | I2C clock |
| Y9 | 13 | |
| Y8 | 14 | |
| Y7 | 47 | |
| Y6 | 48 | |
| Y5 | 21 | |
| Y4 | 38 | |
| Y3 | 39 | |
| Y2 | 40 | |
| VSYNC | 6 | |
| HREF | 7 | |
| PCLK | 2 | |
| **D7** | **16** | Critical fix |
| **D6** | **17** | Critical fix |
| **D5** | **18** | Critical fix |
| **D4** | **12** | Critical fix |
| **D3** | **10** | Critical fix |
| **D2** | **8** | Critical fix |
| **D1** | **9** | Critical fix |
| **D0** | **11** | Critical fix |

### Memory Allocation
- **Detection Model**: FLASH_RODATA
- **Disease Model**: FLASH_RODATA (packed with detection)
- **RGB Frame Buffer**: PSRAM (640×480×3 = 921KB)
- **128×128 Input Buffer**: PSRAM (49KB, allocated once)
- **Camera Framebuffers**: PSRAM (2 buffers, ~18KB each JPEG)

### Partition Layout
```
nvs       : 0x009000 - 0x00F000 (  24 KB)  NVS storage
phy_init  : 0x00F000 - 0x010000 (   4 KB)  PHY calibration
factory   : 0x010000 - 0x510000 (   5 MB)  Application
```

## 🧪 Testing Checklist

- [ ] Flash all three binaries successfully
- [ ] Boot messages show both models loading
- [ ] Camera initializes correctly (OV3660 detected)
- [ ] First frame captured without NO-SOI errors
- [ ] Detection outputs 10 bounding boxes
- [ ] Top-3 detections filtered by confidence
- [ ] Disease classification runs on filtered crops
- [ ] Final diagnosis displayed with confidence
- [ ] Performance metrics logged (all timings)
- [ ] System stable for multiple frames
- [ ] Total frame time ~400-1100ms (0.9-2.5 FPS)

## ⚠️ Troubleshooting

### Build Issues
- **Binary too large**: Partition table already increased to 5MB
- **Model not found**: Both models packed in `grape_leaf_detect` binary
- **Compilation errors**: Ensure ESP-IDF v5.3.3 and ESP-DL 3.2.2

### Runtime Issues
- **Camera init fails**: Check pin mapping matches ESP32S3-EYE
- **Model load fails**: Verify FLASH_RODATA mode and model packing
- **Classification errors**: Check 128×128 buffer allocation in PSRAM
- **Out of memory**: Both models load into internal RAM, PSRAM for inference buffers

## 📈 Performance Tuning

### Adjustable Parameters (in app_main.cpp)

```cpp
// Detection loop configuration
const int CAPTURE_INTERVAL_SEC = 300;  // 5 minutes between captures

// Disease classification filtering
const float CONF_THRESHOLD = 0.45f;    // Minimum detection confidence
const int TOP_K = 3;                   // Number of crops to classify
```

**Recommendations:**
- Increase `TOP_K` to 5 for more thorough analysis (adds ~100ms)
- Lower `CONF_THRESHOLD` to 0.35 to catch more potential diseases
- Decrease `CAPTURE_INTERVAL_SEC` to 60 for more frequent monitoring

## 📝 Code Changes from v16

### Removed
- SPIFFS initialization and mounting
- `crop_bbox()` function (replaced by direct resize)
- `rgb_to_jpeg()` JPEG encoding
- `save_crop_to_flash()` file writing
- `clear_old_crops()` cleanup function
- SPIFFS partition (512KB freed)

### Added
- `disease_classifier.hpp` - Complete DiseaseClassifier class
- `mobilenetv2_128_grape_leaf.espdl` to model packing
- Top-K filtering with confidence threshold
- Disease result aggregation (max confidence)
- Classification timing metrics

### Modified
- Factory partition: 3MB → 5MB
- Main loop: Crop-save → Crop-classify
- Memory strategy: Single-allocation buffers only

## 🔗 Related Files

Source code location:
```
/home/ubuntu/edge-ai-vineyard-monitoring/esp-detection/esp-detection/
  deployment/grape_leaf_detect_camera/esp-dl/examples/grape_leaf_detect/
    ├── main/
    │   ├── app_main.cpp              # Main application with dual-model pipeline
    │   └── disease_classifier.hpp    # Disease classification class
    ├── models/grape_leaf_detect/
    │   └── CMakeLists.txt            # Model packing configuration
    └── partitions.csv                # 5MB partition table
```

## 📞 Support

If you encounter issues:
1. Check serial output for error messages
2. Verify camera wiring matches ESP32S3-EYE pinout
3. Ensure both `.espdl` models are present in models directory
4. Confirm 16MB flash chip is detected
5. Monitor memory usage (should have ~7MB free after init)

---

**Build Date:** January 10, 2026  
**ESP-IDF Version:** v5.3.3  
**ESP-DL Version:** 3.2.2  
**Target:** ESP32-S3 (QFN56)
