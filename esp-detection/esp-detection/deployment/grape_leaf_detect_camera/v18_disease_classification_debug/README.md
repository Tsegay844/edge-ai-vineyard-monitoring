# ESP32-S3 Grape Leaf Detection v18 (Debug Enhanced)

**Firmware Version**: v18_disease_classification_debug  
**Build Date**: January 10, 2026  
**Hardware**: ESP32-S3 QFN56 v0.2 with 8MB PSRAM + OV3660 Camera  

## 🆕 What's New in v18

### Enhanced Debugging & Diagnostics
- **Comprehensive Model Loading Logs**: Real-time visibility into model embedding status
  - Packed binary start/end pointers displayed
  - Binary size verification (should be ~2.8 MB with both models)
  - First 32 bytes hex dump for format validation
  - Step-by-step model initialization tracking
  
- **Graceful Error Handling**: System continues operating even if disease classifier fails
  - Detection model always works (priority functionality)
  - Disease classifier failure logged but doesn't crash system
  - Clear warnings when running in detection-only mode
  
- **Runtime Diagnostics**: Detailed logging for troubleshooting
  - PSRAM allocation status
  - Model input shape validation
  - Free heap monitoring at each init stage

### Technical Improvements from v17
- Removed C++ exception handling (not supported in ESP-IDF)
- Added explicit binary symbol checks (`_binary_grape_leaf_detect_espdl_start/end`)
- Improved error messages with actionable suggestions
- Memory allocation debugging for 128×128 input buffer

## 📦 Package Contents

```
v18_disease_classification_debug/
├── bootloader.bin          (23 KB)  - ESP32-S3 bootloader
├── partition-table.bin     (3 KB)   - Partition layout (5MB factory)
├── grape_leaf_detect.bin   (4.3 MB) - Main application with BOTH models
├── flash_v18.sh            - Linux/Mac flash script
├── flash_v18.bat           - Windows flash script
└── README.md               - This file
```

**Total Binary Size**: 4.27 MB (83% of 5MB partition, 17% free)

## 🔧 Hardware Requirements

- **MCU**: ESP32-S3-WROOM-1 (QFN56 package, revision v0.2+)
- **PSRAM**: 8MB Octal PSRAM (AP_3v3, required for dual-model operation)
- **Flash**: 16MB external SPI flash
- **Camera**: OV3660 (configured for 640×480 JPEG capture, decoded to RGB888)
- **Board**: ESP32-S3-EYE compatible pinout

### Camera Pinout (ESP32-S3-EYE)
```
SIOD/SDA  → GPIO4   (I2C Data)
SIOC/SCL  → GPIO5   (I2C Clock)
VSYNC     → GPIO6
HREF      → GPIO7
PCLK      → GPIO13
XCLK      → GPIO15  (20MHz)
D0-D7     → GPIO11, GPIO9, GPIO8, GPIO47, GPIO48, GPIO16, GPIO18, GPIO17
RESET     → GPIO-1  (not used)
PWDN      → GPIO-1  (not used)
```

## 🚀 Quick Start

### Windows (CMD/PowerShell)
```cmd
cd v18_disease_classification_debug
flash_v18.bat COM6
```

### Linux/Mac (Bash)
```bash
cd v18_disease_classification_debug
chmod +x flash_v18.sh
./flash_v18.sh /dev/ttyUSB0
```

### Monitor Serial Output
```bash
python -m serial.tools.miniterm COM6 115200   # Windows
python -m serial.tools.miniterm /dev/ttyUSB0 115200   # Linux/Mac
```

**Quit Miniterm**: `Ctrl+]`

## 🔍 Expected Debug Output (v18)

### Successful Initialization
```
I (1276) grape_leaf_detect: ╔════════════════════════════════════════════════╗
I (1286) grape_leaf_detect: ║    GRAPE LEAF DETECTION - CAMERA + CROP       ║
I (1296) grape_leaf_detect: ╠════════════════════════════════════════════════╣
I (1316) grape_leaf_detect: ║ Chip: ESP32-esp32s3                           
I (1326) grape_leaf_detect: ║ Cores: 2                                      
I (1336) grape_leaf_detect: ║ Flash: 16MB external                          
I (1346) grape_leaf_detect: ║ PSRAM: No                                     
I (1356) grape_leaf_detect: ║ Camera: OV2660 (320x240)                      
I (1366) grape_leaf_detect: ║ Free Heap: 8638888 bytes                      
I (1376) grape_leaf_detect: ║ Free PSRAM: 8385720 bytes                     
I (1416) grape_leaf_detect: 🧠 Initializing Detection Model...
I (1556) grape_leaf_detect: ✓ Model initialized in 140 ms
I (1566) grape_leaf_detect: 🧬 Initializing Disease Classifier...
I (1576) DiseaseClassifier: Loading MobileNetV2 model: mobilenetv2_128_grape_leaf
I (1580) DiseaseClassifier: 📦 Packed binary found:
I (1585) DiseaseClassifier:    Start: 0x3c140080
I (1590) DiseaseClassifier:    End:   0x3c3cc080
I (1595) DiseaseClassifier:    Size:  2883584 bytes (2.75 MB)
I (1600) DiseaseClassifier:    Header (first 32 bytes):
I (1605) DiseaseClassifier:    0000: 50 44 4c 32 02 00 00 00 70 00 00 00 20 00 00 00
I (1615) DiseaseClassifier:    0010: 24 00 00 00 f0 78 07 00 44 00 00 00 20 00 00 00
I (1625) DiseaseClassifier: 🔄 Creating dl::Model instance...
I (1630) DiseaseClassifier: ✓ Model instance created successfully
I (1635) DiseaseClassifier: 🔄 Getting model input shape...
I (1640) DiseaseClassifier: ✓ Model input shape: [1, 128, 128, 3]
I (1645) DiseaseClassifier: ✓ Allocated 49152 bytes in PSRAM for input buffer
I (1650) DiseaseClassifier: ✅ Disease classifier initialization complete!
I (1655) grape_leaf_detect: ✓ Disease classifier initialized in 85 ms
```

### If Disease Model Fails (Graceful Degradation)
```
I (1576) DiseaseClassifier: Loading MobileNetV2 model: mobilenetv2_128_grape_leaf
I (1585) DiseaseClassifier: 📦 Packed binary found:
I (1590) DiseaseClassifier:    Size:  479872 bytes (0.46 MB)  ⚠️ TOO SMALL!
E (1595) DiseaseClassifier: ❌ Packed binary is empty! Model not embedded correctly.
W (1600) grape_leaf_detect: ⚠️  Disease classifier initialization FAILED
W (1605) grape_leaf_detect:    System will continue with DETECTION ONLY
W (1610) grape_leaf_detect:    Check logs above for details
I (1615) grape_leaf_detect: 📷 Initializing Camera...
[System continues with detection-only mode]
```

## 🧪 Models Included

### 1. Detection Model (YOLO-based)
- **File**: `espdet_pico_320_320_grape_leaf.espdl`
- **Size**: 479 KB (INT8 quantized)
- **Input**: 320×320×3 RGB888
- **Output**: Bounding boxes + confidence scores
- **Inference Time**: ~285 ms
- **Trained On**: 2,439 images (Roboflow dataset v1)

### 2. Disease Classification Model (MobileNetV2)
- **File**: `mobilenetv2_128_grape_leaf.espdl`
- **Size**: 2.3 MB (INT8 quantized)
- **Input**: 128×128×3 RGB888 (from detection crops)
- **Output**: 4-class softmax (healthy, black_rot, esca, leaf_blight)
- **Inference Time**: ~50 ms per crop
- **Trained On**: 5,171 disease images

**Packing**: Both models packed into single `grape_leaf_detect.espdl` (2.8 MB total)

## ⚙️ System Configuration

### Partition Layout
```
0x009000 - 0x00F000  NVS          (24 KB)
0x00F000 - 0x010000  phy_init     (4 KB)
0x010000 - 0x510000  factory      (5 MB) ← Main app
```

### Memory Usage
- **Binary Size**: 4.27 MB (0x4277d0)
- **Free Space**: 888 KB (17% of partition)
- **PSRAM Usage**:
  - Detection buffer: ~350 KB
  - Disease classifier: 49 KB (128×128×3)
  - JPEG decoder: Dynamic
- **Heap**: ~260 KB internal RAM

### Camera Configuration
- **Resolution**: VGA 640×480 (JPEG compressed)
- **Pixel Format**: JPEG → RGB888 decode
- **Frame Buffer**: 1 buffer in PSRAM
- **JPEG Quality**: 10 (medium compression)
- **Frequency**: 20 MHz XCLK

## 📊 Performance Metrics

### Inference Pipeline
```
Camera Capture    →  20-40 ms   (JPEG encoding)
JPEG Decode       →  80-120 ms  (to RGB888)
Detection         →  285 ms     (YOLO inference)
Top-K Filter      →  <1 ms      (conf > 0.45)
Disease Classify  →  50 ms/crop (×3 crops = 150 ms)
────────────────────────────────────────────────────
Total             →  535-595 ms (~2 FPS)
```

### Resource Efficiency
- **FPS**: ~2 frames/second with full pipeline
- **Power**: ~350 mA @ 5V (1.75W with camera active)
- **Latency**: <600 ms from capture to diagnosis

## 🐛 Troubleshooting (v18 Debug Features)

### 1. Disease Model Not Loading
**Check Serial Output For**:
```
I (1585) DiseaseClassifier: 📦 Packed binary found:
I (1590) DiseaseClassifier:    Size:  XXXXXX bytes
```

**Expected**: Size should be ~2,883,584 bytes (2.75-2.8 MB)  
**If Smaller**: Model not packed correctly, detection-only mode activated

**Header Validation**:
```
I (1605) DiseaseClassifier:    0000: 50 44 4c 32 ...
                                      ^^^^^^^^^
                                      "PDL2" magic bytes (correct)
```

### 2. PSRAM Allocation Failed
```
E (1645) DiseaseClassifier: ❌ Failed to allocate 49152 bytes in PSRAM
E (1650) DiseaseClassifier:    Free PSRAM: XXXXX bytes
```
**Solution**: Check that board has 8MB PSRAM (not 2MB variant)

### 3. Model Name Mismatch
```
E (1630) DiseaseClassifier: ❌ Failed to create model
E (1635) DiseaseClassifier:    Possible causes:
E (1640) DiseaseClassifier:    1. Model 'mobilenetv2_128_grape_leaf' not found
```
**Solution**: Verify model filename in `pack_espdl_models.py` output

### 4. System Crash on Boot
**v18 Fix**: System should NOT crash if disease model fails
- Detection continues working
- Only disease classification disabled
- Check for warning logs instead of assert failures

### Common Issues

**COM Port Not Found (Windows)**:
```bash
# Check Device Manager → Ports (COM & LPT)
# Install CP210x drivers if needed
```

**Permission Denied (Linux)**:
```bash
sudo usermod -a -G dialout $USER
# Log out and back in
```

**Slow Flash Speed**:
```bash
# Reduce baud rate in flash script:
# Change 921600 → 460800 or 115200
```

## 📈 Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **v18** | Jan 10, 2026 | Enhanced debug logging, graceful error handling, binary verification |
| v17 | Jan 10, 2026 | Added MobileNetV2 disease classifier (initial version had missing model bug) |
| v16 | Jan 9, 2026 | Detection-only baseline (320×320 YOLO) |

## 🔗 Technical Details

### ESP-DL Model Loading (v18 Implementation)
```cpp
// Check embedded binary symbols
extern const uint8_t grape_leaf_detect_espdl_start[] 
    asm("_binary_grape_leaf_detect_espdl_start");
extern const uint8_t grape_leaf_detect_espdl_end[] 
    asm("_binary_grape_leaf_detect_espdl_end");

size_t size = grape_leaf_detect_espdl_end - grape_leaf_detect_espdl_start;
// Verify size > 2.5 MB (both models present)

// Load model from packed binary
model = new dl::Model("grape_leaf_detect", 
                      "mobilenetv2_128_grape_leaf", 
                      fbs::MODEL_LOCATION_IN_FLASH_RODATA);
```

### Build Configuration
- **ESP-IDF**: v5.3.3
- **ESP-DL**: v3.2.2
- **Compiler**: xtensa-esp32s3-elf-gcc 13.2.0
- **Optimization**: -O2 (balanced speed/size)
- **C++ Exceptions**: Disabled (ESP-IDF default)

## 📝 Notes

- **v18** is a **debug-enhanced** version for troubleshooting model loading issues
- Production deployments should use this version to diagnose any model embedding problems
- Once verified working, can optionally reduce logging verbosity for production
- Graceful degradation ensures system always provides basic detection functionality
- Serial output hex dumps help validate binary embedding at runtime

## 📧 Support

For issues or questions about v18:
- Check serial output for detailed diagnostic messages
- Verify packed binary size is ~2.8 MB
- Confirm header starts with "PDL2" magic bytes
- Test detection-only mode first (should always work)
- Report specific error messages from DiseaseClassifier logs

---

**Build ID**: grape_leaf_detect v18  
**Compile Date**: Jan 10 2026 11:03  
**Git Commit**: c706189-dirty  
**Model Pack**: grape_leaf_detect.espdl (2.8 MB, dual-model)
