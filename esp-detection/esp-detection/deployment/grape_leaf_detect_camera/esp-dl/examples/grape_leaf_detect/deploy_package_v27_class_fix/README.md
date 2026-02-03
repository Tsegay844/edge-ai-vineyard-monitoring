# ESP32-S3 Grape Leaf Disease Detection v27
## 🔴 CRITICAL UPDATE: Disease Class Order Fixed

**Build Date:** January 19, 2026  
**Version:** v27 (Class Order Fix)  
**Hardware:** ESP32-S3 WROOM-1 (16MB Flash, 8MB PSRAM)  
**Camera:** OV3660 (640×480)  

---

## ⚠️ Critical Bug Fix

**Previous Issue (v24 and earlier):**
- Disease class names were in wrong order
- Model output index 0 (Black_rot) was mapped to "healthy"
- Model output index 2 (Healthy) was mapped to "esca"
- **Result:** ALL disease predictions were incorrect!

**Fixed in v27:**
- Class order now matches training (ImageFolder alphabetical):
  - Index 0: Black_rot ✅
  - Index 1: Esca ✅
  - Index 2: Healthy ✅
  - Index 3: Leaf_blight ✅

---

## 📦 Package Contents

```
deploy_package_v27_class_fix/
├── bootloader.bin           # ESP32-S3 bootloader
├── partition-table.bin      # Partition layout (5MB app)
├── grape_leaf_detect.bin    # Main application (4.3MB)
├── flash_v27.sh            # Linux/Mac flash script
├── flash_v27.bat           # Windows flash script
└── README.md               # This file
```

---

## 🚀 Quick Start

### Linux/Mac:
```bash
chmod +x flash_v27.sh
./flash_v27.sh /dev/ttyUSB0
```

### Windows:
```cmd
flash_v27.bat COM6
```

### Manual Flash:
```bash
python3 -m esptool --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 16MB --flash_freq 80m \
  0x0 bootloader.bin \
  0x8000 partition-table.bin \
  0x10000 grape_leaf_detect.bin
```

---

## 📊 System Specifications

### Models
- **Detection:** ESPDet-Pico (416×320, INT8, 478KB)
- **Classification:** MobileNetV2 (128×128, INT8, 2.73MB)
- **Total Model Size:** 3.2MB (embedded in firmware)

### Performance
- **Detection:** ~280ms per frame
- **Classification:** ~535ms per leaf (avg)
- **Full Pipeline:** ~2 seconds (10 leaves)
- **Capture Interval:** 5 minutes (configurable)

### Memory
- **Flash:** 16MB (5MB app partition)
- **PSRAM:** 8MB (models + frame buffers)
- **Firmware Size:** 4.3MB (includes both models)

### Camera
- **Sensor:** OV3660
- **Resolution:** 640×480 (VGA)
- **Format:** JPEG (quality 12)
- **Frame Buffer:** 2× 60KB (PSRAM)

---

## 🔧 Configuration

### Disease Classes (Corrected Order)
```cpp
0: Black_rot
1: Esca  
2: Healthy
3: Leaf_blight
```

### Detection Settings
- Confidence threshold: 0.25
- Top-K leaves: 10
- Aggregation: Hybrid (detection + entropy + spatial)

### Capture Settings
- Interval: 300 seconds (5 minutes)
- Auto exposure: Enabled
- Auto white balance: Enabled

---

## 📝 Monitoring Output

After flashing, monitor serial output:

### Linux/Mac:
```bash
python3 -m serial.tools.miniterm /dev/ttyUSB0 115200
```

### Windows:
```cmd
python -m serial.tools.miniterm COM6 115200
```

---

## 🐛 Troubleshooting

### Flash fails
1. Check USB connection
2. Verify correct port (Linux: `/dev/ttyUSB*`, Windows: `COM*`)
3. Install esptool: `pip3 install esptool`
4. Hold BOOT button during flash if auto-reset fails

### No output after flash
1. Press RESET button on ESP32
2. Check serial port settings (115200 baud)
3. Try different USB cable (data + power)

### Model initialization fails
1. Verify 8MB PSRAM is detected
2. Check firmware size < 5MB (partition limit)
3. Ensure correct ESP32-S3 variant (not ESP32-S2)

---

## ⚙️ Build Information

**Built with:**
- ESP-IDF: v5.3.3
- Compiler: GCC 13.2.0 (Xtensa ESP32-S3)
- ESP-DL: Latest (custom ESPDet-Pico + MobileNetV2)
- Build date: January 19, 2026

**Binary sizes:**
- Bootloader: 23.1 KB
- Partition table: 3 KB
- Application: 4.3 MB

**Warnings (safe to ignore):**
- Unused variables in disease_classifier.hpp (optimization opportunity)
- Missing field initializers (explicit defaults)

---

## 📞 Support

**Critical Fix:** This version corrects the disease classification bug present in all previous versions (v1-v26). **Update immediately** if using older firmware.

**For issues:**
1. Check this README
2. Verify hardware connections
3. Review serial monitor output
4. Confirm correct firmware version (v27)

---

## ✅ Verification Checklist

After flashing v27:
- [ ] Serial monitor shows system info at boot
- [ ] Camera initializes successfully (OV3660 detected)
- [ ] Detection model loads (ESPDet-Pico)
- [ ] Disease classifier loads (MobileNetV2)
- [ ] PSRAM: ~4130 KB free after init
- [ ] Disease classes: Black_rot, Esca, Healthy, Leaf_blight (in order)
- [ ] Predictions make sense (not all "Healthy" or "Esca")

---

**END OF README**
