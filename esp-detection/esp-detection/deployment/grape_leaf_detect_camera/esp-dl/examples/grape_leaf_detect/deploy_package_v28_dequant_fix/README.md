# ESP32-S3 Grape Leaf Detection - Deployment Package v28

## CRITICAL FIX: Proper INT8 Dequantization

### What Was Wrong

**v27 Bug:** The MobileNetV2 model outputs were **NOT being dequantized properly**. The code was treating quantized INT8 values [-128, 127] as if they were already in the correct float range, leading to:
- All leaves classified as 100% one class (previously all Esca, now all Healthy)
- Softmax receiving wrong input range
- Complete loss of discrimination between classes

**Root Cause:**
```cpp
// WRONG (v27):
logits[i] = (float)output_data[i];  // Just cast INT8 to float

// CORRECT (v28):
float scale = DL_SCALE(output->exponent);  // scale = 2^exponent
logits[i] = (float)output_data[i] * scale;  // Proper dequantization
```

### The Fix

ESP-DL quantized models store an `exponent` field in output tensors. The correct dequantization formula is:

**float_value = int8_value × 2^exponent**

Without this scaling, the INT8 range [-128, 127] doesn't map correctly to the logit range, causing softmax to produce meaningless probabilities.

---

## Package Contents

- `bootloader.bin` (23 KB) - ESP32-S3 bootloader
- `partition-table.bin` (3 KB) - Partition layout
- `grape_leaf_detect.bin` (4.4 MB) - Main application **with dequantization fix**
- `flash_v28.sh` - Linux/Mac flash script
- `flash_v28.bat` - Windows flash script
- `README.md` - This file

---

## Quick Start

### Linux/Mac
```bash
chmod +x flash_v28.sh
sudo ./flash_v28.sh /dev/ttyUSB0

# Monitor output
python3 -m serial.tools.miniterm /dev/ttyUSB0 115200
```

### Windows
```cmd
flash_v28.bat COM6

# Monitor output
python -m serial.tools.miniterm COM6 115200
```

### Manual Flashing
```bash
python3 -m esptool --chip esp32s3 -p PORT -b 921600 \
    --before=default_reset --after=hard_reset write_flash \
    --flash_mode dio --flash_freq 80m --flash_size 16MB \
    0x0 bootloader.bin \
    0x8000 partition-table.bin \
    0x10000 grape_leaf_detect.bin
```

---

## System Specifications

**Hardware:**
- ESP32-S3 WROOM-1 (240 MHz dual-core)
- 16 MB Flash, 8 MB PSRAM
- OV3660 camera (640×480 JPEG)

**Models:**
- Leaf Detection: ESPDet-Pico (478 KB, ~366ms)
- Disease Classification: MobileNetV2-INT8 (2.73 MB, ~535ms per leaf)

**Performance:**
- Detection: 10 leaves in 366ms
- Classification: ~535ms per leaf (10 leaves = 5.8 seconds)
- Total pipeline: ~6.3 seconds per frame
- Memory: 4.2 MB app, 17% partition free, 3.9 MB PSRAM free

**Camera:**
- Sensor: OV3660
- Resolution: 640×480
- Format: JPEG (quality 12)
- Capture interval: 5 minutes

---

## Disease Classes (Corrected Order)

The class order **matches PyTorch ImageFolder alphabetical order**:

```cpp
0: Black_rot    // Fungal disease
1: Esca         // Trunk disease  
2: Healthy      // No disease
3: Leaf_blight  // Bacterial disease
```

**Note:** Both v27 (class order fix) and v28 (dequantization fix) were required for correct operation.

---

## Expected Output

With proper dequantization, you should see **varied probabilities**:

```
Disease Probability:
   • Black_rot: 5.23%
   • Esca: 12.45%
   • Healthy: 78.32%
   • Leaf_blight: 4.00%
```

**NOT** like v27 where everything was 100% one class:
```
Disease Probability:
   • Healthy: 100.00%  ← WRONG (no discrimination)
```

---

## Debug Logging

Enable debug logs in `disease_classifier.hpp` to see dequantization details:

```
Output exponent: -7, scale: 0.007812
Raw INT8 logits: [23, -15, 87, 12]
Dequantized logits: [0.180, -0.117, 0.680, 0.094]
After softmax: [0.0523, 0.1245, 0.7832, 0.0400]
```

---

## Configuration

Edit `main/app_main.cpp`:

```cpp
#define CAPTURE_INTERVAL_MS (5 * 60 * 1000)  // 5 minutes
#define MAX_DETECTIONS 10                     // Process top 10 leaves
```

Edit `main/camera_config.h` for camera settings.

---

## Monitoring Serial Output

After flashing, connect to serial port (115200 baud) to see real-time detection results.

**Key indicators of correct operation:**
- Varied disease probabilities (not all 100% one class)
- Weighted aggregation results showing distribution
- Performance metrics: ~6.3 seconds total per frame

---

## Troubleshooting

### All leaves still showing 100% one class?
- Check that you flashed v28 (not v27)
- Look for debug logs showing exponent and scale values
- Verify "Compile time: Jan 19 2026" (latest build)

### Flash fails?
- Check port name: `ls /dev/tty*` (Linux) or Device Manager (Windows)
- Try lower baud: 460800 or 115200
- Ensure no other program is using the port
- Press and hold BOOT button during flash if needed

### Model fails to initialize?
- Check free PSRAM: Should be ~8.3 MB at startup
- Verify model size: Should show "2863072 bytes (2.73 MB)"
- Check bootloader time: "Compile time: Jan 19 2026"

### Inference errors?
- Monitor free PSRAM: Should stay above 3.5 MB
- Check detection count: Should find 5-10 leaves typically
- Verify model input shape: [1, 128, 128, 3]

---

## Build Information

- **ESP-IDF:** v5.3.3
- **Compiler:** GCC 13.2.0 (xtensa-esp-elf)
- **Build date:** Jan 19 2026 05:XX:XX
- **Git commit:** 174d0d5-dirty
- **App size:** 4,358,080 bytes (4.4 MB)
- **Free space:** 887,040 bytes (17%)

---

## Verification Checklist

After flashing v28, verify:

- [ ] Bootloader shows "Compile time: Jan 19 2026"
- [ ] MobileNetV2 model loads: "2863072 bytes (2.73 MB)"
- [ ] Camera initializes: "OV3660 camera"
- [ ] Detection finds leaves: "Detected X objects"
- [ ] **Disease probabilities are VARIED (not 100% one class)**
- [ ] Aggregation shows weighted class scores
- [ ] Final diagnosis includes confidence percentage
- [ ] Performance: ~6.3 seconds total per frame
- [ ] Free PSRAM: ~3.9 MB after inference

**Most important:** Disease probabilities should show distribution across classes, not 100% for a single class.

---

## Technical Details

### Dequantization Formula

```
R' = Q × Scale
Scale = 2^Exp

Where:
- R' = float value (logit)
- Q = quantized INT8 value [-128, 127]  
- Exp = exponent from model output tensor
```

### Example Calculation

```
INT8 logit = 87
Exponent = -7
Scale = 2^(-7) = 0.0078125
Float logit = 87 × 0.0078125 = 0.6797
```

Without dequantization, softmax receives values in range [-128, 127] instead of the correct logit range (typically [-10, 10]), causing numerical issues and loss of discrimination.

---

## Previous Issues Resolved

✅ **v27:** Fixed class order mismatch (alphabetical vs manual)  
✅ **v28:** Fixed INT8 dequantization (scale factor missing)  

Both fixes were **critical** - v27 ensured correct class index mapping, v28 ensures correct probability calculations.

---

## Support

For issues or questions, check:
1. Serial monitor output at 115200 baud
2. Debug logs (enable in disease_classifier.hpp)
3. Free memory (PSRAM should stay > 3.5 MB)
4. Model loading (should show "Model loaded and validated successfully")

---

**Version:** 28  
**Date:** January 19, 2026  
**Critical Fix:** INT8 dequantization with proper scale factor  
**Required:** Both v27 class order AND v28 dequantization fixes
