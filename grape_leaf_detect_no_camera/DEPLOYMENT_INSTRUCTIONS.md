
# ESP32 Deployment Instructions

## Overview
Your custom object detection model for 'grape_leaf' has been successfully prepared for deployment on ESP32S3.

## Prerequisites

1. **ESP-IDF Installation**
   - Install ESP-IDF v5.3 or later
   - Follow: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/
   - Verify installation: `idf.py --version`

2. **Hardware Requirements**
   - ESP32S3 development board
   - USB cable for programming
   - Camera module (if using live detection)

## Project Structure

```
grape_leaf_detect/
├── main/
│   ├── app_main.cpp         # Main application code
│   ├── test_image.jpg       # Test image for inference
│   └── CMakeLists.txt
├── CMakeLists.txt
├── sdkconfig.defaults.esp32s3
└── partitions.csv

grape_leaf_detect/
├── models/
│   ├── s3/                  # ESP32-S3 models
│   └── p4/                  # ESP32-P4 models
├── espdet_detect.cpp        # Detection implementation
└── espdet_detect.hpp        # Detection interface
```

## Build and Flash Instructions

### Step 1: Navigate to Project Directory
```bash
cd deployment/grape_leaf_detect/esp-dl/examples/grape_leaf_detect
```

### Step 2: Set Target Chip
```bash
idf.py set-target esp32s3
```

### Step 3: Configure Project (Optional)
```bash
idf.py menuconfig
```

You can configure:
- Serial port settings
- WiFi settings (if needed)
- Camera settings (if using live detection)
- Model-specific parameters

### Step 4: Build Project
```bash
idf.py build
```

This will compile the project and generate the firmware binary.

### Step 5: Flash to Device
Connect your ESP32S3 board via USB and run:
```bash
idf.py flash
```

### Step 6: Monitor Output
```bash
idf.py flash monitor
```

Or use combined command:
```bash
idf.py build flash monitor
```

To exit monitor: Press `Ctrl+]`

## Expected Output

After flashing, you should see output similar to:
```
I (xxx) main: Initializing grape_leaf detection
I (xxx) main: Loading model...
I (xxx) main: Model loaded successfully
I (xxx) main: Running inference on test image...
I (xxx) main: Detection results:
I (xxx) main:   Object 0: class=grape_leaf, confidence=0.95, bbox=[x, y, w, h]
I (xxx) main:   Object 1: class=grape_leaf, confidence=0.87, bbox=[x, y, w, h]
I (xxx) main: Inference complete!
```

## Customization

### Using Your Own Test Images

1. Convert image to C array:
   ```bash
   python tools/image_to_array.py your_image.jpg
   ```

2. Replace the image array in `main/app_main.cpp`

### Integrating with Camera

The example uses a static test image. To use live camera:

1. Enable camera component in `CMakeLists.txt`
2. Initialize camera in `app_main.cpp`
3. Replace static image buffer with camera frame buffer
4. Run detection in a loop

Example camera integration:
```cpp
#include "esp_camera.h"

// Initialize camera
camera_config_t config = {...};
esp_camera_init(&config);

// Capture and detect in loop
while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    // Run detection on fb->buf
    esp_camera_fb_return(fb);
    vTaskDelay(100 / portTICK_PERIOD_MS);
}
```

### Adjusting Detection Parameters

Edit `espdet_detect.hpp` to modify:
- Confidence threshold: `float conf_threshold = 0.25;`
- IoU threshold: `float iou_threshold = 0.45;`
- Maximum detections: `int max_detections = 100;`

## Performance Tuning

### Optimize Inference Speed
1. Reduce input resolution (if acceptable)
2. Adjust confidence threshold
3. Use ESP32-P4 instead of ESP32-S3 (faster)
4. Enable CPU frequency boost in menuconfig

### Reduce Memory Usage
1. Use 8-bit quantization (already default)
2. Reduce batch size if processing multiple images
3. Adjust partition table if needed

## Troubleshooting

### Build Errors
- Ensure ESP-IDF v5.3+ is installed
- Check `idf.py --version`
- Clean and rebuild: `idf.py fullclean && idf.py build`

### Flash Errors
- Check USB connection
- Identify serial port: `idf.py -p PORT flash`
- Hold BOOT button during flash if needed

### Runtime Errors
- Check partition size (model may be too large)
- Verify correct target chip selected
- Check power supply (some boards need external power)

### Performance Issues
- Monitor CPU usage
- Check memory usage: `esp_get_free_heap_size()`
- Reduce input size if needed
- Verify quantized model is loaded (not FP32)

## Additional Resources

- ESP-IDF Documentation: https://docs.espressif.com/projects/esp-idf/
- ESP-DL Repository: https://github.com/espressif/esp-dl
- ESP-Detection Repository: https://github.com/espressif/esp-detection
- Espressif Forum: https://esp32.com/

## Support

For issues related to:
- Model training/quantization: Check ESP-Detection repository
- ESP32 deployment: Check ESP-DL repository  
- Hardware/ESP-IDF: Check Espressif documentation

## Next Steps

1. Test the model with various images
2. Integrate with camera for live detection
3. Add wireless communication (WiFi/BLE) for results
4. Implement trigger actions based on detections
5. Optimize performance for your use case

Good luck with your deployment!
