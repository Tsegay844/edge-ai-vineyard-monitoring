# ESP32-S3 Grape Leaf Detection - Version History

## v17: Disease Classification (January 10, 2026) ✨ CURRENT
**File:** `grape_leaf_detection_esp32s3_v17_disease_classification.tar.gz` (2.6 MB)
**Status:** Production-ready dual-model system

### Major Features
- ✅ **Dual-Model Pipeline**: Detection + Disease Classification
- ✅ **MobileNetV2 Classifier**: 128×128 INT8 quantized for 4 disease classes
- ✅ **Memory-Efficient**: Single-allocation buffers, no malloc/free in loop
- ✅ **Smart Filtering**: Top-K (K=3) detections with confidence threshold (>0.45)
- ✅ **Real-Time Only**: Removed SPIFFS storage, no crop saving
- ✅ **3× Faster**: 435ms per frame (vs 1140-1300ms in v16)
- ✅ **Result Aggregation**: Maximum confidence across filtered crops

### Disease Classes
1. `healthy` - No disease detected
2. `black_rot` - Black rot fungal infection
3. `esca` - Esca disease (leaf tiger stripes)
4. `leaf_blight` - Bacterial/fungal blight

### Technical Specs
- Binary Size: 4.2 MB (both models embedded)
- Factory Partition: 5 MB (17% free)
- Detection Model: espdet_pico_320_320_grape_leaf.espdl (479 KB)
- Classification Model: mobilenetv2_128_grape_leaf.espdl (~491 KB)
- Performance: ~2.3 FPS (435ms per frame)
- Memory: 128×128×3 buffer in PSRAM (49 KB)

### Changes from v16
- ➕ Added DiseaseClassifier class
- ➕ Added mobilenetv2 model to build
- ➕ Top-K filtering with confidence threshold
- ➕ Disease result aggregation
- ➖ Removed SPIFFS initialization
- ➖ Removed crop_bbox(), rgb_to_jpeg() functions
- ➖ Removed save_crop_to_flash()
- ➖ Removed SPIFFS partition (512 KB freed)
- 📏 Increased factory partition: 3 MB → 5 MB

---

## v16: SPIFFS Crop Storage (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v16_SPIFFS_512KB.tar.gz` (954 KB)
**Status:** Stable baseline for crop storage

### Features
- ✅ Detection working at 320×320
- ✅ SPIFFS partition (512 KB) for 10 crop JPEGs
- ✅ Saves all detected bounding boxes as JPEG files
- ✅ clear_old_crops() removes previous frame's crops
- ✅ Camera: OV3660 with correct ESP32S3-EYE pins

### Technical Specs
- Binary Size: 2.0 MB
- Factory Partition: 3 MB
- SPIFFS Partition: 512 KB (espdet_det)
- Performance: 0.77-0.88 FPS (1140-1300ms per frame)
- Crop Format: JPEG quality 10 (~700 bytes each)

### Issues Fixed
- ✅ NVS too small for 10 crops (v15 limitation)
- ✅ Switched from NVS blobs to SPIFFS files
- ✅ All 10 crops now saved successfully

---

## v15: Camera Pin Fix (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v15_OV3660_PINS.tar.gz` (936 KB)

### Critical Fix
- ✅ **Pin Mapping**: Corrected D7-D0 pins to ESP32S3-EYE standard
  - D7=16, D6=17, D5=18, D4=12, D3=10, D2=8, D1=9, D0=11
- ✅ Camera capture now works reliably
- ✅ No more NO-SOI errors or corrupted JPEG

### Performance
- Detection: 285ms
- Total: ~1000ms per frame
- Stability: 15+ minutes continuous operation

---

## v14: OV3660 Sensor Name Fix (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v14_OV3660_FIX.tar.gz` (936 KB)

### Fix
- ✅ Added OV3660 sensor name matching in code
- ❌ Still had camera pin mapping issues

---

## v13: Model→Camera Init Order (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v13_RODATA_224METHOD.tar.gz` (936 KB)

### Critical Discovery
- ✅ Load model BEFORE camera initialization
- ✅ Prevents cache conflicts during partition read
- ✅ Model loading reliable (FLASH_RODATA)
- ❌ Camera initialization still failing (pin issues)

---

## v12: Partition-Based Loading Attempt (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v12_PARTITION_CHATGPT.tar.gz` (937 KB)

### Approach
- Model in dedicated partition (espdet_model @ 0x600000)
- MODEL_LOCATION_IN_FLASH_PARTITION
- ❌ Cache/MMU errors, unreliable loading

---

## v11: RODATA with param_copy=true (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v11_RODATA_PSRAM_COPY.tar.gz` (936 KB)

### Features
- FLASH_RODATA with param_copy=true
- ❌ Still unreliable, GPIO errors

---

## v10: RODATA Embedded (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v10_RODATA_EMBEDDED.tar.gz` (936 KB)

### Features
- Model embedded in .rodata section
- Direct FLASH_RODATA loading
- ❌ Camera GPIO conflicts

---

## v9: Chunked PSRAM Copy (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v9_CHUNKED_PSRAM.tar.gz` (937 KB)

### Features
- Chunked model loading to PSRAM
- ❌ Still had partition read errors

---

## v8: Memory-Mapped No Copy (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v8_MMAP_NO_COPY.tar.gz` (937 KB)

### Features
- param_copy=false to avoid PSRAM copy
- ❌ Unreliable model access

---

## v7: Correct Binaries (January 9, 2026)
**File:** `grape_leaf_detection_esp32s3_v7_CORRECT_BINARIES.tar.gz` (938 KB)

### Fix
- Used correct bootloader and partition table
- ❌ Still had model loading issues

---

## v6: First Model Embedding (January 8, 2026)
**File:** `grape_leaf_detection_esp32s3_flash_package_v6_WITH_MODEL.tar.gz` (1.3 MB)

### Features
- First successful model embedding
- ❌ Wrong partition configuration

---

## Earlier Versions (Pre-v6)
**File:** `esp32_flash_CAMERA_FINAL_20251228_090726.tar.gz` (195 KB)

### Features
- Camera-only testing
- No model integration
- Pin mapping experiments

---

## Version Comparison Summary

| Version | Binary Size | Partition | Status | Key Feature |
|---------|-------------|-----------|--------|-------------|
| **v17** | 4.2 MB | 5 MB | ✅ Production | Disease classification |
| v16 | 2.0 MB | 3 MB | ✅ Stable | SPIFFS crop storage |
| v15 | 2.0 MB | 3 MB | ✅ Working | Camera pin fix |
| v14 | 2.0 MB | 3 MB | ⚠️ Partial | Sensor name fix |
| v13 | 2.0 MB | 3 MB | ⚠️ Partial | Init order fix |
| v10-v12 | 2.0 MB | 3 MB | ❌ Failed | Various loading methods |
| v6-v9 | 2.0 MB | 3 MB | ❌ Failed | Early model embedding |

---

## Performance Evolution

| Version | FPS | Frame Time | Notable Change |
|---------|-----|------------|----------------|
| v17 | ~2.3 | 435 ms | Classification added, SPIFFS removed |
| v16 | 0.77-0.88 | 1140-1300 ms | SPIFFS crop saving |
| v15 | ~1.0 | 1000 ms | Camera working |
| v13 | N/A | N/A | Camera not working |
| v6-v12 | N/A | N/A | Model loading issues |

---

## Recommended Versions

### For Disease Classification (Current Requirement)
**Use v17**: Full detection + classification pipeline, real-time results

### For Crop Collection/Training Data
**Use v16**: Saves all 10 crops as JPEG files to SPIFFS

### For Debugging/Testing
**Use v15**: Minimal working system, detection only

---

## File Locations

All versions stored at:
```
/home/ubuntu/edge-ai-vineyard-monitoring/esp-detection/esp-detection/
  deployment/grape_leaf_detect_camera/esp-dl/examples/grape_leaf_detect/
```

Source code:
```
/home/ubuntu/edge-ai-vineyard-monitoring/esp-detection/esp-detection/
  deployment/grape_leaf_detect_camera/esp-dl/examples/grape_leaf_detect/
    ├── main/
    │   ├── app_main.cpp
    │   └── disease_classifier.hpp  (v17 only)
    ├── models/grape_leaf_detect/
    └── partitions.csv
```

---

**Last Updated:** January 10, 2026  
**Maintainer:** Edge AI Vineyard Monitoring Team
