# 🚀 LOCAL BUILD & FLASH INSTRUCTIONS - Windows VS Code

**Project:** ESP32-S3 Grape Leaf Detection  
**Platform:** Windows with ESP-IDF v5.3.3  
**IDE:** Visual Studio Code  
**Date:** December 27, 2025

---

## 📋 Prerequisites

✅ ESP-IDF v5.3.3 installed at `C:\Espressif\frameworks\esp-idf-v5.3.3`  
✅ Project cloned to local Windows machine  
✅ VS Code installed  
✅ ESP32-S3 board with USB cable  

---

## 📍 CRITICAL: What Folder to Build?

**The build folder is DEEP inside the project!**

Your cloned project structure looks like this:

```
📁 edge-ai-vineyard-monitoring/          ← You cloned this
   ├─ 📁 dd_cnn/                          ← Other stuff
   ├─ 📁 YOLO/                            ← Other stuff
   └─ 📁 esp-detection/                   ← Go here
      └─ 📁 esp-detection/                ← Then here
         ├─ 📄 train.py
         ├─ 📄 FULL_RUNNING_INSTRUCTIONS.md
         └─ 📁 deployment/                ← Then here
            └─ 📁 grape_leaf_detect_camera/  ← Then here
               ├─ 📄 grape_leaf_espdet_224x224_esp32s3.espdl
               └─ 📁 esp-dl/               ← Then here
                  └─ 📁 examples/          ← Then here
                     └─ 📂 grape_leaf_detect/  ← OPEN THIS IN VS CODE!
                        ├─ 📁 main/
                        │  └─ 📄 app_main.cpp (your code!)
                        ├─ 📄 CMakeLists.txt
                        ├─ 📄 sdkconfig.defaults.esp32s3
                        └─ 📄 partitions.csv
```

**Full Windows Path Example:**
```
C:\Users\YourName\Documents\edge-ai-vineyard-monitoring\esp-detection\esp-detection\deployment\grape_leaf_detect_camera\esp-dl\examples\grape_leaf_detect
```

**This is the folder you open in VS Code to build!**

---

## 🔧 Part 1: VS Code Setup (One-Time)

### Step 1: Install ESP-IDF Extension

1. Open VS Code
2. Press `Ctrl+Shift+X` to open Extensions
3. Search for: **ESP-IDF**
4. Install: **ESP-IDF** by Espressif Systems
5. Restart VS Code

### Step 2: Configure ESP-IDF Extension

1. Press `Ctrl+Shift+P` to open Command Palette
2. Type: `ESP-IDF: Configure ESP-IDF Extension`
3. Select: **Use Existing Setup**

4. Configure paths:
   ```
   ESP-IDF Path: C:\Espressif\frameworks\esp-idf-v5.3.3
   Python Path: C:\Espressif\python_env\idf5.3_py3.11_env\Scripts\python.exe
   Tools Path: C:\Espressif
   ```
   
5. Click **Save**

### Step 3: Open Project Folder

**⚠️ CRITICAL: You must open the EXACT folder shown below!**

In VS Code:
1. Click **File** → **Open Folder**
2. Navigate through your folders:
   ```
   📁 edge-ai-vineyard-monitoring/
      └─ 📁 esp-detection/
         └─ 📁 esp-detection/
            └─ 📁 deployment/
               └─ 📁 grape_leaf_detect_camera/
                  └─ 📁 esp-dl/
                     └─ 📁 examples/
                        └─ 📂 grape_leaf_detect/  ← SELECT THIS FOLDER!
                           ├─ 📁 main/
                           ├─ CMakeLists.txt     ← You should see this file
                           └─ sdkconfig.defaults.esp32s3
   ```

3. **Select the `grape_leaf_detect` folder** and click **Select Folder**

**How to verify you opened the correct folder:**
- Look at VS Code's Explorer sidebar
- You should see files directly:
  - `CMakeLists.txt`
  - `partitions.csv`
  - `sdkconfig.defaults.esp32s3`
  - `main/` folder
  
**❌ WRONG:** If you see `esp-detection/`, `YOLO/`, `dd_cnn/` folders at the top level, you opened the wrong folder!

**✅ CORRECT:** You see `main/`, `CMakeLists.txt`, `partitions.csv` at the top level

---

## 🏗️ Part 2: Build the Project

**Before building, make sure you opened the CORRECT folder!**  
👉 VS Code Explorer should show: `main/`, `CMakeLists.txt`, `partitions.csv`

If not, go back to Part 1, Step 3 and open the correct folder.

---

### Method 1: Using VS Code Status Bar (Easiest)

After opening the project, you'll see buttons at the bottom of VS Code:

1. Click **🎯 Set Target** → Select **esp32s3**
2. Click **🔨 Build** button
3. Wait 3-5 minutes for first build

### Method 2: Using Command Palette

1. Press `Ctrl+Shift+P`
2. Type: `ESP-IDF: Set Espressif Device Target`
3. Select: **esp32s3**

4. Press `Ctrl+Shift+P` again
5. Type: `ESP-IDF: Build your Project`
6. Wait for build to complete

### Method 3: Using ESP-IDF Terminal

1. Press `Ctrl+Shift+P`
2. Type: `ESP-IDF: Open ESP-IDF Terminal`
3. Run commands:
   ```cmd
   idf.py set-target esp32s3
   idf.py build
   ```

### ✅ Build Success

You should see:
```
Project build complete. To flash, run:
 idf.py -p (PORT) flash
or
 idf.py -p (PORT) flash monitor
```

Build output location:
```
build\grape_leaf_detect.bin (2.0 MB)
build\bootloader\bootloader.bin
build\partition_table\partition-table.bin
```

---

## ⚡ Part 3: Flash to ESP32-S3

### Step 1: Connect ESP32-S3

1. Connect ESP32-S3 to Windows via USB-C cable
2. Check **Device Manager** → **Ports (COM & LPT)**
3. Note the COM port (e.g., **COM6**)

**No COM port?** Install drivers:
- CP210x USB Driver (Silicon Labs)
- Or CH340 Driver
- Download from manufacturer's website

### Step 2: Select COM Port in VS Code

**Option A: Click Port in Status Bar**
- Bottom status bar shows port (e.g., COM6)
- Click to change if needed

**Option B: Command Palette**
1. Press `Ctrl+Shift+P`
2. Type: `ESP-IDF: Select Port to Use`
3. Select your COM port (e.g., **COM6**)

### Step 3: Flash the Device

**Method 1: Status Bar (Easiest)**
1. Click **⚡ Flash** button in bottom status bar
2. Wait ~30-60 seconds

**Method 2: Build, Flash and Monitor (All-in-One)**
1. Click **🔧 Build, Flash and Monitor** button
2. Builds → Flashes → Opens Monitor automatically

**Method 3: Command Palette**
1. Press `Ctrl+Shift+P`
2. Type: `ESP-IDF: Flash your Project`
3. Confirm port if asked

**Method 4: Terminal**
1. Open ESP-IDF Terminal (`Ctrl+Shift+P` → `ESP-IDF: Open ESP-IDF Terminal`)
2. Run:
   ```cmd
   idf.py -p COM6 flash
   ```

### ✅ Flash Success

You should see:
```
Connecting...
Detecting chip type... ESP32-S3
Chip is ESP32-S3 (revision v0.1)
Features: WiFi, BLE
...
Hash of data verified.
Leaving...
Hard resetting via RTS pin...
```

---

## 📺 Part 4: Monitor Serial Output

### Open Serial Monitor

**Method 1: Status Bar**
- Click **📺 Monitor** button

**Method 2: Command Palette**
1. Press `Ctrl+Shift+P`
2. Type: `ESP-IDF: Monitor your Device`

**Method 3: Terminal**
```cmd
idf.py -p COM6 monitor
```

**To exit monitor:** Press `Ctrl+]`

### Expected Output

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

---

## 🎯 Quick Workflow Summary

### One-Time Setup
1. Install ESP-IDF Extension in VS Code
2. Configure ESP-IDF paths
3. Open correct project folder

### Every Build/Flash
1. **Open project folder** (grape_leaf_detect)
2. **Connect ESP32-S3** → Note COM port
3. **Click 🔧 Build, Flash and Monitor** in status bar
4. **Done!** Monitor shows output

---

## 🔧 Troubleshooting

### Error: "ESP-IDF not configured"
**Solution:**
1. `Ctrl+Shift+P` → `ESP-IDF: Configure ESP-IDF Extension`
2. Select **Use Existing Setup**
3. Enter correct paths

### Error: "Target not set" or "Wrong chip type"
**Solution:**
1. `Ctrl+Shift+P` → `ESP-IDF: Set Espressif Device Target`
2. Select **esp32s3**
3. Rebuild

### Error: "Failed to connect to ESP32-S3"
**Solutions:**
- Hold **BOOT** button while connecting
- Install USB drivers (CP210x or CH340)
- Try different USB cable (data cable, not charge-only)
- Try different USB port
- Check Device Manager for COM port

### Error: "Port busy" or "Permission denied"
**Solutions:**
- Close Arduino IDE, PuTTY, or other serial monitors
- Unplug and replug USB cable
- Restart VS Code

### Error: "Build failed" or "CMake error"
**Solutions:**
1. Clean build: `Ctrl+Shift+P` → `ESP-IDF: Full Clean`
2. Rebuild: `Ctrl+Shift+P` → `ESP-IDF: Build your Project`
3. Check that you opened the **correct folder** (grape_leaf_detect)

### Error: "Python not found"
**Solution:**
- Reconfigure extension with correct Python path
- Usually: `C:\Espressif\python_env\idf5.3_py3.11_env\Scripts\python.exe`

### Monitor shows garbage/strange characters
**Solutions:**
- Press `Ctrl+T` in monitor, then `Ctrl+R` to reset ESP32
- Check baud rate (should be 115200)
- Unplug/replug USB

---

## 📁 Project Structure Reference

```
edge-ai-vineyard-monitoring/
└── esp-detection/
    └── esp-detection/
        └── deployment/
            └── grape_leaf_detect_camera/
                ├── grape_leaf_espdet_224x224_esp32s3.espdl  ← AI Model
                └── esp-dl/
                    └── examples/
                        └── grape_leaf_detect/  ← OPEN THIS FOLDER
                            ├── main/
                            │   ├── app_main.cpp  ← Your code
                            │   ├── CMakeLists.txt
                            │   └── idf_component.yml
                            ├── CMakeLists.txt
                            ├── sdkconfig.defaults.esp32s3
                            ├── partitions.csv
                            └── build/  ← Created after build
                                ├── grape_leaf_detect.bin
                                ├── bootloader/
                                └── partition_table/
```

---

## 🎮 VS Code Status Bar Reference

After opening project, you'll see:

```
🎯 esp32s3 | 🔌 COM6 | 🔨 Build | ⚡ Flash | 📺 Monitor | 🔧 Build/Flash/Monitor | 🧹 Clean
```

- **🎯 esp32s3** - Target device (click to change)
- **🔌 COM6** - Serial port (click to change)
- **🔨 Build** - Build project
- **⚡ Flash** - Flash to device
- **📺 Monitor** - Serial monitor
- **🔧 Build/Flash/Monitor** - Do all three
- **🧹 Clean** - Full clean

---

## ⚡ Quick Commands Cheat Sheet

### Command Palette (`Ctrl+Shift+P`)

```
ESP-IDF: Configure ESP-IDF Extension
ESP-IDF: Set Espressif Device Target
ESP-IDF: Select Port to Use
ESP-IDF: Build your Project
ESP-IDF: Flash your Project
ESP-IDF: Monitor your Device
ESP-IDF: Build, Flash and Monitor
ESP-IDF: Full Clean
ESP-IDF: Open ESP-IDF Terminal
```

### Terminal Commands

```cmd
# Set target
idf.py set-target esp32s3

# Build
idf.py build

# Flash
idf.py -p COM6 flash

# Monitor
idf.py -p COM6 monitor

# Flash and monitor
idf.py -p COM6 flash monitor

# Clean
idf.py fullclean

# Size analysis
idf.py size
```

---

## ✅ Success Checklist

- [ ] ESP-IDF Extension installed in VS Code
- [ ] ESP-IDF paths configured correctly
- [ ] Opened correct folder (grape_leaf_detect)
- [ ] Target set to esp32s3
- [ ] Build completed successfully (2.0 MB binary)
- [ ] ESP32-S3 connected (COM port visible)
- [ ] Flash completed successfully
- [ ] Serial monitor shows boot messages
- [ ] Camera initialized
- [ ] Model loaded
- [ ] Detection loop started

---

## 🎯 System Behavior

### 5-Minute Replace Mode

Every 5 minutes:
1. 📸 Capture image (640×480)
2. 🧠 Run AI detection (~130-150ms)
3. 📊 Sort by confidence
4. 🗑️ Clear old crops (c0-c9)
5. ✂️ Crop top 10 detections
6. 💾 Save as c0, c1...c9 (overwrites old)
7. ⏸️ Sleep for 5 minutes

### Performance
- Active: 0.5 seconds (0.17% duty cycle)
- Sleep: 299.5 seconds (99.83%)
- Storage: Fixed ~40KB (never grows)

---

## 📞 Support

**GitHub Repository:** https://github.com/Tsegay844/edge-ai-vineyard-monitoring

**Full Manual:** See FULL_RUNNING_INSTRUCTIONS.md

**ESP-IDF Docs:** https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/

---

**Created:** December 27, 2025  
**Version:** 1.0 (Local Windows Build)  
**ESP-IDF:** v5.3.3  
**Platform:** Windows + VS Code






Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

C:\Espressif\frameworks\esp-idf-v5.3.3\export.ps1