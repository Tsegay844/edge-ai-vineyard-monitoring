# 🚀 FULL RUNNING INSTRUCTIONS - ESP32-S3 Grape Leaf Detection

**Project:** Edge AI Vineyard Monitoring System  
**Target:** ESP32-S3 with OV2660 Camera  
**Mode:** 5-Minute Replace Mode  
**Date:** December 27, 2025

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [ESP-IDF Setup](#esp-idf-setup)
4. [Project Structure](#project-structure)
5. [Building the Firmware](#building-the-firmware)
6. [Flashing to ESP32-S3](#flashing-to-esp32-s3)
7. [Monitoring Output](#monitoring-output)
8. [Understanding the System](#understanding-the-system)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Configuration](#advanced-configuration)

---

## 🎯 System Overview

### What This System Does

Your ESP32-S3 runs a complete AI-powered grape leaf detection system with these features:

**5-Minute Replace Mode:**
```
Boot → Initialize Camera & AI Model
  ↓
Every 5 minutes:
  1. 📸 Capture image (640x480) from OV2660 camera
  2. 🧠 Run grape leaf detection AI model
  3. 📊 Sort detections by confidence (highest first)
  4. 🗑️  Clear old crops (erase c0-c9 from flash)
  5. ✂️  Crop top 10 bounding boxes
  6. 💾 Save crops as c0, c1, c2...c9 (overwrites old ones)
  7. ⏸️  Sleep for 300 seconds
  ↓
Loop forever
```

**Key Features:**
- ✅ Runs completely offline (no internet needed)
- ✅ Power efficient (0.17% duty cycle)
- ✅ Fixed storage (~40KB for crops)
- ✅ Real-time inference (~130-150ms)
- ✅ Automatic old data cleanup

---

## 📦 Prerequisites

### Hardware
- ✅ **ESP32-S3 Development Board** (Freenove ESP32-S3 WROOM-1 recommended)
- ✅ **OV2660 Camera Module** (connected to ESP32-S3)
- ✅ **USB-C Cable** (for programming and power)
- ✅ **Computer** (Linux/Windows/Mac with USB port)

### Software

#### Ubuntu/Linux
```bash
# 1. Git
sudo apt-get update
sudo apt-get install git

# 2. Python 3.8+
python3 --version  # Should be 3.8 or higher

# 3. Build tools
sudo apt-get install git wget flex bison gperf python3 python3-pip python3-venv \
    cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
```

#### Windows
- Git for Windows
- Python 3.8+
- ESP-IDF installer (see next section)

---

## 🛠️ ESP-IDF Setup

### Option 1: Linux/Ubuntu (Recommended)

#### Step 1: Clone ESP-IDF v5.5.2
```bash
cd ~
git clone --recursive --branch v5.5.2 https://github.com/espressif/esp-idf.git
cd esp-idf
```

#### Step 2: Install ESP-IDF Tools
```bash
./install.sh esp32,esp32s3
```

This will download (~2GB) and install:
- xtensa-esp-elf toolchain
- riscv32-esp-elf toolchain  
- OpenOCD debugger
- Other ESP32 tools

**Time:** ~10-15 minutes depending on internet speed

#### Step 3: Set Up Environment
```bash
# Add to ~/.bashrc for permanent setup
echo 'alias get_idf=". $HOME/esp-idf/export.sh"' >> ~/.bashrc
source ~/.bashrc

# Activate ESP-IDF environment
get_idf
```

#### Step 4: Verify Installation
```bash
idf.py --version
```

**Expected output:**
```
ESP-IDF v5.5.2
```

### Option 2: Windows

1. Download ESP-IDF installer from: https://dl.espressif.com/dl/esp-idf/
2. Run installer and select ESP32-S3 target
3. Open "ESP-IDF PowerShell" or "ESP-IDF Command Prompt"
4. Verify: `idf.py --version`

---

## 📁 Project Structure

```
edge-ai-vineyard-monitoring/
├── esp-detection/
│   └── esp-detection/
│       ├── deployment/
│       │   └── grape_leaf_detect_camera/
│       │       ├── grape_leaf_espdet_224x224_esp32s3.espdl  ← AI Model (491KB)
│       │       ├── grape_leaf_espdet_224x224_esp32s3.info
│       │       ├── grape_leaf_espdet_224x224_esp32s3.json
│       │       └── esp-dl/
│       │           └── examples/
│       │               └── grape_leaf_detect/              ← BUILD THIS
│       │                   ├── main/
│       │                   │   ├── app_main.cpp            ← Main code (414 lines)
│       │                   │   ├── CMakeLists.txt
│       │                   │   └── idf_component.yml
│       │                   ├── CMakeLists.txt
│       │                   ├── sdkconfig.defaults.esp32s3
│       │                   ├── partitions.csv
│       │                   └── build/                      ← Created after build
│       │                       └── grape_leaf_detect.bin   ← Flash this
│       ├── train.py
│       ├── deploy/
│       └── README.md
└── YOLO/
```

**Key Files:**
- `app_main.cpp` - Your 5-minute replace mode implementation
- `*.espdl` - AI model file (quantized for ESP32-S3)
- `sdkconfig.defaults.esp32s3` - ESP32-S3 configuration
- `partitions.csv` - Memory layout

---

## 🔨 Building the Firmware

### Step 1: Clone Your Repository

```bash
cd ~
git clone https://github.com/Tsegay844/edge-ai-vineyard-monitoring.git
cd edge-ai-vineyard-monitoring
```

### Step 2: Navigate to Project

```bash
cd esp-detection/esp-detection/deployment/grape_leaf_detect_camera/esp-dl/examples/grape_leaf_detect
```

### Step 3: Activate ESP-IDF Environment

```bash
# Linux/Mac
source ~/esp-idf/export.sh

# Windows (ESP-IDF PowerShell)
%userprofile%\esp\esp-idf\export.bat
```

### Step 4: Set Target to ESP32-S3

```bash
idf.py set-target esp32s3
```

**Expected output:**
```
Setting target to 'esp32s3'...
```

### Step 5: Build the Project

```bash
idf.py build
```

**Build time:** 3-5 minutes (first build)

**Expected output (final lines):**
```
Project build complete. To flash, run:
 idf.py -p (PORT) flash
or
 idf.py -p (PORT) flash monitor
```

**Build artifacts created:**
- `build/grape_leaf_detect.bin` - Main firmware
- `build/bootloader/bootloader.bin` - Bootloader
- `build/partition_table/partition-table.bin` - Partition table

---

## ⚡ Flashing to ESP32-S3

### 🔄 Build on Linux, Flash on Windows

If you build on Linux but need to flash on Windows:

#### Step 1: Build on Linux (Current Location)

```bash
cd ~/edge-ai-vineyard-monitoring/esp-detection/esp-detection/deployment/grape_leaf_detect_camera/esp-dl/examples/grape_leaf_detect
source ~/esp-idf/export.sh
idf.py set-target esp32s3
idf.py build
```

#### Step 2: Download Build Artifacts

The build artifacts have been packaged into a compressed archive:

**📦 Build Package:**
```
/home/ubuntu/esp32_build_20251227_134228.tar.gz (945 KB)
```

**Contains:**
- `bootloader/bootloader.bin` - ESP32-S3 bootloader
- `partition_table/partition-table.bin` - Partition layout
- `grape_leaf_detect.bin` - Main firmware (1.99 MB)
- `flash_args` - Flash addresses and parameters
- `flasher_args.json` - JSON format flash configuration

**Download from Linux server (run on your Windows machine):**

```bash
# Using SCP
scp ubuntu@YOUR_SERVER_IP:/home/ubuntu/esp32_build_20251227_134228.tar.gz .

# Or using SFTP/WinSCP/FileZilla
# Connect to: YOUR_SERVER_IP
# Navigate to: /home/ubuntu/
# Download: esp32_build_20251227_134228.tar.gz
```

#### Step 3: Flash on Windows

**Option A: Using ESP-IDF (Easiest if you have ESP-IDF installed)**

If you have ESP-IDF installed on Windows (e.g., `C:\Espressif\frameworks\esp-idf-v5.3.3`):

1. Extract the downloaded archive (use 7-Zip)
2. Open **ESP-IDF PowerShell** or **ESP-IDF Command Prompt**
3. Navigate to the extracted folder containing the `.bin` files
4. Connect ESP32-S3 and identify COM port (e.g., COM6)
5. Flash directly:
   ```cmd
   esptool.py -p COM6 -b 460800 --before default_reset --after hard_reset --chip esp32s3 write_flash @flash_args
   ```

**Option B: Using esptool.py (Without ESP-IDF)**

1. Install Python 3.8+ on Windows
2. Install esptool:
   ```cmd
   pip install esptool
   ```

3. Connect ESP32-S3 and identify COM port (e.g., COM6)

4. Flash using explicit addresses:
   ```cmd
   esptool.py -p COM6 -b 460800 --before default_reset --after hard_reset --chip esp32s3 write_flash --flash_mode dio --flash_size 8MB --flash_freq 80m 0x0 bootloader/bootloader.bin 0x8000 partition_table/partition-table.bin 0x10000 grape_leaf_detect.bin
   ```

**Flash addresses (from build/flash_args):**
```
0x0      → bootloader/bootloader.bin
0x8000   → partition_table/partition-table.bin
0x10000  → grape_leaf_detect.bin
```

---

### 💻 Direct Flashing (Linux or Windows)

#### Step 1: Connect ESP32-S3

1. Connect ESP32-S3 to computer via USB-C cable
2. Identify the port:

**Linux:**
```bash
ls /dev/ttyUSB* /dev/ttyACM*
# Usually: /dev/ttyUSB0 or /dev/ttyACM0
```

**Windows:**
```cmd
# Check Device Manager → Ports (COM & LPT)
# Usually: COM3, COM4, COM5, etc.
```

**Mac:**
```bash
ls /dev/cu.*
# Usually: /dev/cu.usbserial-*
```

### Step 2: Flash Firmware

**Linux/Mac:**
```bash
idf.py -p /dev/ttyUSB0 flash
```

**Windows:**
```cmd
idf.py -p COM3 flash
```

**Flash time:** ~30-60 seconds

**Expected output:**
```
Detecting chip type... ESP32-S3
Chip is ESP32-S3 (revision v0.1)
Features: WiFi, BLE
Crystal is 40MHz
...
Hash of data verified.

Leaving...
Hard resetting via RTS pin...
```

### Step 3: Flash and Monitor (One Command)

To flash and immediately see output:

**Linux:**
```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

**Windows:**
```cmd
idf.py -p COM6 flash monitor
```

**To exit monitor:** Press `Ctrl+]`

---

## 📺 Monitoring Output

### Start Monitor

**Linux:**
```bash
idf.py -p /dev/ttyUSB0 monitor
```

**Windows:**
```cmd
idf.py -p COM3 monitor
```

### Expected Boot Output

```
╔════════════════════════════════════════════════╗
║    GRAPE LEAF DETECTION - CAMERA + CROP       ║
╠════════════════════════════════════════════════╣
║ Chip: ESP32-s3                                 
║ Cores: 2                                       
║ Silicon Rev: 0                                 
║ Flash: 8MB external                            
║ PSRAM: Yes                                     
║ Camera: OV2660 (640x480 VGA)                   
║ Free Heap: 341248 bytes                        
║ Free PSRAM: 8355583 bytes                      
╚════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📷 Initializing Camera (OV2660)...
✓ Camera initialized successfully

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 Initializing Detection Model...
✓ Model initialized in 234 ms
  Free heap after init: 298 KB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 Starting Detection Loop (capture every 5 minutes)...
```

### Normal Operation Output

```
╔════════════════ FRAME 1 ════════════════╗
📅 Time: 0 seconds since boot
⏰ Next capture in 5 minutes

📸 Captured: 640x480, 28456 bytes (46 ms)
  Frame 1 saved: 28456 bytes
🖼️  Decoded: 640x480 RGB888 (90 ms)
🔍 Detected 10 objects (137 ms)
✓ Sorted by confidence (highest first)

🗑️  Cleared old crops from flash

📦 Processing top 10 detections:
  [0] Confidence: 0.923, BBox: [120,45,240,180]
    ✓ Crop 0 saved: 4523 bytes [c0]
  [1] Confidence: 0.891, BBox: [300,120,380,220]
    ✓ Crop 1 saved: 3456 bytes [c1]
  [2] Confidence: 0.874, BBox: [450,80,520,160]
    ✓ Crop 2 saved: 2987 bytes [c2]
  ...
  [9] Confidence: 0.612, BBox: [200,300,280,380]
    ✓ Crop 9 saved: 3124 bytes [c9]

✓ Saved 10/10 crops (156 ms)
💾 Flash now contains ONLY the latest 10 crops (c0-c9)

⏱️  Performance:
    Capture:   46 ms
    Decode:    90 ms
    Inference: 137 ms
    Crop+Save: 156 ms
    TOTAL:     429 ms (2.33 FPS)
    Free PSRAM: 8120 KB
╚══════════════════════════════════════════════╝

⏸️  Sleeping for 5 minutes...
💤 Next capture at: ~300 seconds
```

---

## 🧠 Understanding the System

### Main Code Structure (app_main.cpp)

```cpp
// 1. Camera Configuration (Lines 19-48)
static camera_config_t camera_config = {
    // OV2660 pin configuration for Freenove ESP32-S3
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_VGA,  // 640x480
};

// 2. Key Functions:

// Crop bounding box from image
uint8_t* crop_bbox(const dl::image::img_t &img, int x1, int y1, int x2, int y2);

// Convert RGB to JPEG
uint8_t* rgb_to_jpeg(uint8_t *rgb_data, int width, int height, size_t &jpeg_len);

// 🗑️ Clear old crops from flash (YOUR FEATURE!)
esp_err_t clear_old_crops() {
    // Erases c0, c1, c2...c9 from NVS flash
}

// Save crop to flash with simple key
esp_err_t save_crop_to_flash(const uint8_t *data, size_t len, int crop_idx);

// 3. Main Loop (Lines 220-414)
void app_main(void) {
    // Initialize NVS, camera, model
    
    const int CAPTURE_INTERVAL_SEC = 300;  // 5 minutes
    
    while (true) {
        // 1. Capture frame
        // 2. Decode JPEG
        // 3. Run detection
        // 4. Sort by confidence
        // 5. clear_old_crops()  ← YOUR FEATURE
        // 6. Crop and save top 10
        // 7. Sleep 5 minutes
    }
}
```

### Flash Storage (NVS)

**Namespace: "crops"**
```
c0 → Latest crop 0 (highest confidence) - JPEG ~4KB
c1 → Latest crop 1                      - JPEG ~3KB
c2 → Latest crop 2                      - JPEG ~5KB
...
c9 → Latest crop 9                      - JPEG ~4KB

Total: ~40KB (fixed, never grows)
```

**Namespace: "frames"**
```
frame_1 → Full captured frame 1 - JPEG ~28KB
frame_2 → Full captured frame 2 - JPEG ~29KB
...
```

### Performance Metrics

| Operation | Time | % of Cycle |
|-----------|------|------------|
| Camera capture | ~40-50ms | 0.013% |
| JPEG decode | ~80-100ms | 0.027% |
| AI inference | ~130-150ms | 0.043% |
| Crop & save (×10) | ~100-200ms | 0.050% |
| **Active total** | ~400-500ms | **0.17%** |
| Sleep | 300 seconds | 99.83% |
| **Full cycle** | 300.5 seconds | 100% |

**Power efficiency:** Active only 0.5s every 5 minutes!

---

## 🔧 Troubleshooting

### Build Errors

#### Error: "No module named 'serial'"
```bash
pip install pyserial
```

#### Error: "Command 'ninja' not found"
```bash
# Ubuntu
sudo apt-get install ninja-build

# Windows
# Reinstall ESP-IDF using installer
```

#### Error: "Target 'esp32s3' not supported"
```bash
# Reinstall ESP-IDF tools for ESP32-S3
cd ~/esp-idf
./install.sh esp32s3
```

### Flash Errors

#### Error: "Failed to connect to ESP32-S3"
1. Check USB cable (use data cable, not charge-only)
2. Hold "BOOT" button while connecting
3. Try different USB port
4. Check port permissions (Linux):
   ```bash
   sudo usermod -a -G dialout $USER
   # Logout and login again
   ```

#### Error: "A fatal error occurred: MD5 checksum does not match"
```bash
# Clean and rebuild
idf.py fullclean
idf.py build
idf.py flash
```

### Runtime Errors

#### Camera not detected
- Check camera ribbon cable connection
- Verify pin configuration in app_main.cpp matches your board
- Try re-seating the camera module

#### Model initialization failed
- Check if model file is embedded (should show in build log)
- Verify `sdkconfig.defaults.esp32s3` has correct model config:
  ```
  CONFIG_FLASH_ESPDET_PICO_224_224_GRAPE_LEAF=y
  CONFIG_ESPDET_DETECT_MODEL_IN_FLASH_RODATA=y
  ```

#### Out of memory errors
- ESP32-S3 needs PSRAM enabled
- Check `sdkconfig.defaults.esp32s3`:
  ```
  CONFIG_SPIRAM=y
  CONFIG_SPIRAM_BOOT_INIT=y
  ```

---

## ⚙️ Advanced Configuration

### Adjust Capture Interval

Edit `app_main.cpp` line 264:

```cpp
// Change from 5 minutes to your desired interval
const int CAPTURE_INTERVAL_SEC = 300;   // 5 minutes
// const int CAPTURE_INTERVAL_SEC = 60;    // 1 minute
// const int CAPTURE_INTERVAL_SEC = 600;   // 10 minutes
// const int CAPTURE_INTERVAL_SEC = 3600;  // 1 hour
```

Rebuild and flash.

### Change Number of Crops Saved

Edit `app_main.cpp` line 263:

```cpp
const int MAX_CROPS_PER_FRAME = 10;  // Save top 10
// const int MAX_CROPS_PER_FRAME = 5;   // Save top 5
// const int MAX_CROPS_PER_FRAME = 20;  // Save top 20
```

Also update `clear_old_crops()` function line 127 to match.

### Adjust JPEG Quality

Lower quality = smaller file size

Edit `app_main.cpp` line 96:

```cpp
// Quality 1-100 (lower = smaller, lower quality)
dl::image::jpeg_img_t jpeg_result = dl::image::sw_encode_jpeg(crop_img, MALLOC_CAP_SPIRAM, 10);
// Change 10 to desired quality (5-20 recommended)
```

### Enable WiFi Upload

To upload crops to a server, add WiFi code after line 220:

```cpp
#include "esp_wifi.h"
#include "esp_http_client.h"

// After NVS init:
// 1. Initialize WiFi
// 2. Connect to AP
// 3. Upload crops using HTTP POST
```

---

## 📚 Additional Resources

### ESP-IDF Documentation
- Getting Started: https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/
- API Reference: https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-reference/

### ESP-DL (Deep Learning Library)
- GitHub: https://github.com/espressif/esp-dl
- Examples: https://github.com/espressif/esp-dl/tree/master/examples

### Hardware
- ESP32-S3 Datasheet: https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf
- Freenove ESP32-S3: https://github.com/Freenove/Freenove_ESP32_S3_WROOM_Board

---

## 🎯 Quick Command Reference

### Build on Linux, Flash on Windows Workflow

```bash
# === ON LINUX (Build) ===

# 1. Setup environment
source ~/esp-idf/export.sh

# 2. Navigate to project
cd ~/edge-ai-vineyard-monitoring/esp-detection/esp-detection/deployment/grape_leaf_detect_camera/esp-dl/examples/grape_leaf_detect

# 3. Clean previous build (optional)
idf.py fullclean

# 4. Set target
idf.py set-target esp32s3

# 5. Build
idf.py build

# 6. Package build artifacts
cd build
tar -czf ~/esp32_build_$(date +%Y%m%d_%H%M%S).tar.gz \
    bootloader/bootloader.bin \
    partition_table/partition-table.bin \
    grape_leaf_detect.bin \
    flash_args \
    flasher_args.json

# Current build: esp32_build_20251227_134228.tar.gz
# Download from: /home/ubuntu/esp32_build_20251227_134228.tar.gz
```

```cmd
REM === ON WINDOWS (Flash) ===

REM 1. Extract downloaded tar.gz (use 7-Zip or similar)

REM 2. Install esptool if not installed
pip install esptool

REM 3. Check COM port (Device Manager)
REM Example: COM6

REM 4. Flash to ESP32-S3
esptool.py -p COM6 -b 460800 --before default_reset --after hard_reset --chip esp32s3 write_flash --flash_mode dio --flash_size 8MB --flash_freq 80m 0x0 bootloader/bootloader.bin 0x8000 partition_table/partition-table.bin 0x10000 grape_leaf_detect.bin

REM 5. Monitor output (optional, requires ESP-IDF on Windows)
idf.py -p COM6 monitor

REM Or use any serial monitor at 115200 baud
```

### Standard Commands (Linux)

```bash
# Setup ESP-IDF environment
source ~/esp-idf/export.sh

# Navigate to project
cd ~/edge-ai-vineyard-monitoring/esp-detection/esp-detection/deployment/grape_leaf_detect_camera/esp-dl/examples/grape_leaf_detect

# Set target
idf.py set-target esp32s3



# Build
idf.py build

# Flash (if on Linux)
idf.py -p /dev/ttyUSB0 flash

# Monitor (if on Linux)
idf.py -p /dev/ttyUSB0 monitor

# Flash and monitor (one command)
idf.py -p /dev/ttyUSB0 flash monitor

# Clean build
idf.py fullclean

# Size analysis
idf.py size

# Menuconfig (advanced)
idf.py menuconfig
```

---

## ✅ Success Checklist

- [ ] ESP-IDF v5.5.2 installed
- [ ] Tools installed (`./install.sh esp32s3`)
- [ ] Environment activated (`source ~/esp-idf/export.sh`)
- [ ] Repository cloned
- [ ] Project builds successfully
- [ ] ESP32-S3 connected and detected
- [ ] Firmware flashed successfully
- [ ] Serial monitor shows boot messages
- [ ] Camera initialized
- [ ] Model loaded successfully
- [ ] First detection cycle completed
- [ ] Crops saved to flash
- [ ] System sleeping for 5 minutes

---

## 📞 Support

**GitHub Repository:** https://github.com/Tsegay844/edge-ai-vineyard-monitoring

**Issues:** Report bugs or ask questions in GitHub Issues

**ESP-IDF Forum:** https://esp32.com/

---

**Created:** December 27, 2025  
**Version:** 1.0  
**Author:** Edge AI Vineyard Monitoring Team  
**License:** MIT
