#include "espdet_detect.hpp"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_chip_info.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_flash.h"
#include "dl_image_jpeg.hpp"
#include "nvs_flash.h"
#include <cstring>

static const char *TAG = "grape_leaf_camera_ai";

// ========== FREENOVE ESP32-S3 CAMERA PINS (OV3660) ==========
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5

#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11

#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

// Arduino-style camera configuration (proven to work!)
static camera_config_t camera_config = {
    .pin_pwdn  = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,

    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,

    .xclk_freq_hz = 20000000,              // 20MHz (Arduino working config)
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG,        // ✅ JPEG mode (Arduino uses this successfully!)
    .frame_size = FRAMESIZE_QVGA,          // 320x240 QVGA
    .jpeg_quality = 12,                    // Quality 12 (lower = better quality, larger size)
    .fb_count = 2,                         // Double buffering (Arduino style)
    .fb_location = CAMERA_FB_IN_PSRAM,     // PSRAM framebuffers (Arduino uses this!)
    .grab_mode = CAMERA_GRAB_LATEST,       // ✅ GRAB_LATEST (Arduino mode - key difference!)
    .sccb_i2c_port = 1                     // I2C port 1 for camera
};

// Crop bounding box from RGB888 image
static uint8_t* crop_bbox_rgb888(const uint8_t *img, int img_w, int img_h, 
                                 int x1, int y1, int x2, int y2, 
                                 int *crop_w, int *crop_h) {
    // Clamp coordinates
    x1 = (x1 < 0) ? 0 : (x1 >= img_w) ? img_w - 1 : x1;
    y1 = (y1 < 0) ? 0 : (y1 >= img_h) ? img_h - 1 : y1;
    x2 = (x2 < 0) ? 0 : (x2 >= img_w) ? img_w - 1 : x2;
    y2 = (y2 < 0) ? 0 : (y2 >= img_h) ? img_h - 1 : y2;
    
    // Ensure x1 < x2, y1 < y2
    if (x1 > x2) { int temp = x1; x1 = x2; x2 = temp; }
    if (y1 > y2) { int temp = y1; y1 = y2; y2 = temp; }
    
    *crop_w = x2 - x1;
    *crop_h = y2 - y1;
    
    if (*crop_w <= 0 || *crop_h <= 0) {
        ESP_LOGE(TAG, "Invalid crop dimensions: %dx%d", *crop_w, *crop_h);
        return nullptr;
    }
    
    // Allocate crop buffer in PSRAM (3 bytes per pixel for RGB888)
    uint8_t *crop = (uint8_t *)heap_caps_malloc(*crop_w * *crop_h * 3, MALLOC_CAP_SPIRAM);
    if (!crop) {
        ESP_LOGE(TAG, "Failed to allocate crop buffer (%d bytes)", *crop_w * *crop_h * 3);
        return nullptr;
    }
    
    // Copy rows
    for (int y = 0; y < *crop_h; y++) {
        const uint8_t *src_row = img + ((y1 + y) * img_w + x1) * 3;
        uint8_t *dst_row = crop + y * (*crop_w) * 3;
        memcpy(dst_row, src_row, *crop_w * 3);
    }
    
    return crop;
}

// Save crop to NVS flash
static esp_err_t save_crop_to_flash(const uint8_t *jpg_data, size_t jpg_len, int crop_idx) {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open("crops", NVS_READWRITE, &nvs);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS: %s", esp_err_to_name(err));
        return err;
    }
    
    char key[8];
    snprintf(key, sizeof(key), "c%d", crop_idx);
    
    err = nvs_set_blob(nvs, key, jpg_data, jpg_len);
    if (err == ESP_OK) {
        err = nvs_commit(nvs);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "    ✓ Crop %d saved: %zu bytes [%s]", crop_idx, jpg_len, key);
        }
    } else {
        ESP_LOGE(TAG, "    ✗ Failed to save crop %d: %s", crop_idx, esp_err_to_name(err));
    }
    
    nvs_close(nvs);
    return err;
}

extern "C" void app_main(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // ========== System Information ==========
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    ESP_LOGI(TAG, "╔════════════════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   GRAPE LEAF DETECTION - ARDUINO CAMERA MODE  ║");
    ESP_LOGI(TAG, "╠════════════════════════════════════════════════╣");
    ESP_LOGI(TAG, "║ Chip: ESP32-S3");
    ESP_LOGI(TAG, "║ Cores: %d", chip_info.cores);
    ESP_LOGI(TAG, "║ Silicon Rev: %d", chip_info.revision);
    uint32_t flash_size;
    esp_flash_get_size(NULL, &flash_size);
    ESP_LOGI(TAG, "║ Flash: %uMB %s", 
             (unsigned int)(flash_size / (1024 * 1024)),
             (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");
    ESP_LOGI(TAG, "║ Free Heap: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));
    ESP_LOGI(TAG, "║ Free PSRAM: %u KB", (unsigned int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(TAG, "╚════════════════════════════════════════════════╝");

    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "🧠 Initializing Detection Model (BEFORE camera)...");

    int64_t init_start = esp_timer_get_time();
    ESPDetDetect *detect = new ESPDetDetect();
    if (!detect) {
        ESP_LOGE(TAG, "Failed to create detection model!");
        return;
    }
    int64_t init_time = (esp_timer_get_time() - init_start) / 1000;

    ESP_LOGI(TAG, "✓ Model initialization complete (%lld ms)", init_time);
    ESP_LOGI(TAG, "  Free heap after init: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));

    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "📷 Initializing Camera (Arduino JPEG mode)...");

    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization failed: 0x%x", err);
        ESP_LOGE(TAG, "Possible issues:");
        ESP_LOGE(TAG, "  - Camera not connected");
        ESP_LOGE(TAG, "  - Wrong GPIO pins");
        ESP_LOGE(TAG, "  - Insufficient memory");
        return;
    }

    // Get camera sensor for OV3660 adjustments
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor != NULL && sensor->id.PID == OV3660_PID) {
        sensor->set_vflip(sensor, 1);        // Flip vertically
        sensor->set_brightness(sensor, 1);    // Increase brightness
        sensor->set_saturation(sensor, -2);   // Lower saturation
        ESP_LOGI(TAG, "✓ Camera initialized: OV3660 (JPEG mode, PSRAM, GRAB_LATEST)");
    } else {
        ESP_LOGI(TAG, "✓ Camera initialized (JPEG mode, PSRAM, GRAB_LATEST)");
    }

    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "🔄 Starting Detection Loop...");

    int frame_num = 0;
    while (true) {
        frame_num++;
        int64_t loop_start = esp_timer_get_time();
        
        ESP_LOGI(TAG, "\n╔════════════════ FRAME %d ════════════════╗", frame_num);
        
        // ========== Step 1: Capture JPEG frame ==========
        int64_t cap_start = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t cap_time = (esp_timer_get_time() - cap_start) / 1000;
        
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed!");
            continue;
        }
        
        ESP_LOGI(TAG, "📸 Captured: %dx%d, %zu bytes (%lld ms) - JPEG format", 
                 fb->width, fb->height, fb->len, cap_time);

        // ========== Step 2: Decode JPEG → RGB888 ==========
        int64_t decode_start = esp_timer_get_time();
        dl::image::jpeg_img_t jpeg_img = {.data = (void *)fb->buf, .data_len = fb->len};
        auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
        int64_t decode_time = (esp_timer_get_time() - decode_start) / 1000;
        
        if (!img.data) {
            ESP_LOGE(TAG, "JPEG decode failed!");
            esp_camera_fb_return(fb);
            continue;
        }
        
        ESP_LOGI(TAG, "✅ JPEG decoded: %dx%d RGB888 (%lld ms)", img.width, img.height, decode_time);

        // ========== Step 3: Run AI Detection ==========
        int64_t det_start = esp_timer_get_time();
        std::vector<dl::detect::result_t> &results = detect->run(img);
        int64_t det_time = (esp_timer_get_time() - det_start) / 1000;
        
        ESP_LOGI(TAG, "🔍 Detected %zu objects (%lld ms)", results.size(), det_time);

        // ========== Step 4: Sort by confidence ==========
        if (!results.empty()) {
            std::sort(results.begin(), results.end(), 
                     [](const dl::detect::result_t &a, const dl::detect::result_t &b) {
                         return a.score > b.score;
                     });
            ESP_LOGI(TAG, "✓ Sorted by confidence (highest first)");

            // ========== Step 5: Process top 10 detections ==========
            int num_crops = (results.size() > 10) ? 10 : results.size();
            ESP_LOGI(TAG, "\n📦 Processing top %d detections:", num_crops);
            
            int64_t crop_start = esp_timer_get_time();
            for (int i = 0; i < num_crops; i++) {
                const auto &det = results[i];
                ESP_LOGI(TAG, "  [%d] Confidence: %.3f, BBox: [x1:%d, y1:%d, x2:%d, y2:%d]",
                         i, det.score, det.box[0], det.box[1], det.box[2], det.box[3]);

                // Crop detection
                int crop_w, crop_h;
                uint8_t *crop_rgb = crop_bbox_rgb888((uint8_t *)img.data, img.width, img.height,
                                                     det.box[0], det.box[1], det.box[2], det.box[3],
                                                     &crop_w, &crop_h);
                if (crop_rgb) {
                    // Convert crop to JPEG
                    uint8_t *jpg_buf = nullptr;
                    size_t jpg_len = 0;
                    
                    bool success = fmt2jpg(crop_rgb, crop_w * crop_h * 3, crop_w, crop_h, 
                                          PIXFORMAT_RGB888, 90, &jpg_buf, &jpg_len);
                    
                    if (success && jpg_buf) {
                        // Save to flash
                        save_crop_to_flash(jpg_buf, jpg_len, i);
                        free(jpg_buf);
                    }
                    
                    heap_caps_free(crop_rgb);
                }
            }
            int64_t crop_time = (esp_timer_get_time() - crop_start) / 1000;
            ESP_LOGI(TAG, "  ✓ Saved %d crops to flash (%lld ms)", num_crops, crop_time);
        } else {
            ESP_LOGI(TAG, "  No detections found");
        }

        // Cleanup
        dl::tool::free_image(img);
        esp_camera_fb_return(fb);

        // ========== Performance Summary ==========
        int64_t loop_time = (esp_timer_get_time() - loop_start) / 1000;
        ESP_LOGI(TAG, "\n⏱️  Performance:");
        ESP_LOGI(TAG, "    Capture:  %lld ms", cap_time);
        ESP_LOGI(TAG, "    Decode:   %lld ms", decode_time);
        ESP_LOGI(TAG, "    Inference: %lld ms", det_time);
        ESP_LOGI(TAG, "    TOTAL:    %lld ms (%.2f FPS)", loop_time, 1000.0 / loop_time);
        ESP_LOGI(TAG, "    Free PSRAM: %u KB", (unsigned int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
        ESP_LOGI(TAG, "╚══════════════════════════════════════════════╝\n");

        // Sleep 30 seconds between captures
        ESP_LOGI(TAG, "⏸️  Sleeping for 30 seconds...\n");
        vTaskDelay(30000 / portTICK_PERIOD_MS);
    }
}

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_chip_info.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_flash.h"
#include "dl_image_jpeg.hpp"
#include "nvs_flash.h"
#include <cstring>

static const char *TAG = "grape_leaf_camera_ai";

// ========== FREENOVE ESP32-S3 CAMERA PINS (OV3660) ==========
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5

#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11

#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

// Arduino-style camera configuration (proven to work!)
static camera_config_t camera_config = {
    .pin_pwdn  = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,

    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,

    .xclk_freq_hz = 20000000,              // 20MHz (Arduino working config)
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG,        // ✅ JPEG mode (Arduino uses this successfully!)
    .frame_size = FRAMESIZE_QVGA,          // 320x240 QVGA
    .jpeg_quality = 12,                    // Quality 12 (lower = better quality, larger size)
    .fb_count = 2,                         // Double buffering (Arduino style)
    .fb_location = CAMERA_FB_IN_PSRAM,     // PSRAM framebuffers (Arduino uses this!)
    .grab_mode = CAMERA_GRAB_LATEST,       // ✅ GRAB_LATEST (Arduino mode - key difference!)
    .sccb_i2c_port = 1                     // I2C port 1 for camera
};

// Convert RGB888 to RGB565 (for model input if needed)
static void rgb888_to_rgb565(const uint8_t *src, uint16_t *dst, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        uint8_t r = src[i * 3 + 0];
        uint8_t g = src[i * 3 + 1];
        uint8_t b = src[i * 3 + 2];
        dst[i] = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
    }
}

// Crop bounding box from RGB888 image
static uint8_t* crop_bbox_rgb888(const uint8_t *img, int img_w, int img_h, 
                                 int x1, int y1, int x2, int y2, 
                                 int *crop_w, int *crop_h) {
    // Clamp coordinates
    x1 = (x1 < 0) ? 0 : (x1 >= img_w) ? img_w - 1 : x1;
    y1 = (y1 < 0) ? 0 : (y1 >= img_h) ? img_h - 1 : y1;
    x2 = (x2 < 0) ? 0 : (x2 >= img_w) ? img_w - 1 : x2;
    y2 = (y2 < 0) ? 0 : (y2 >= img_h) ? img_h - 1 : y2;
    
    // Ensure x1 < x2, y1 < y2
    if (x1 > x2) { int temp = x1; x1 = x2; x2 = temp; }
    if (y1 > y2) { int temp = y1; y1 = y2; y2 = temp; }
    
    *crop_w = x2 - x1;
    *crop_h = y2 - y1;
    
    if (*crop_w <= 0 || *crop_h <= 0) {
        ESP_LOGE(TAG, "Invalid crop dimensions: %dx%d", *crop_w, *crop_h);
        return nullptr;
    }
    
    // Allocate crop buffer in PSRAM (3 bytes per pixel for RGB888)
    uint8_t *crop = (uint8_t *)heap_caps_malloc(*crop_w * *crop_h * 3, MALLOC_CAP_SPIRAM);
    if (!crop) {
        ESP_LOGE(TAG, "Failed to allocate crop buffer (%d bytes)", *crop_w * *crop_h * 3);
        return nullptr;
    }
    
    // Copy rows
    for (int y = 0; y < *crop_h; y++) {
        const uint8_t *src_row = img + ((y1 + y) * img_w + x1) * 3;
        uint8_t *dst_row = crop + y * (*crop_w) * 3;
        memcpy(dst_row, src_row, *crop_w * 3);
    }
    
    return crop;
}

// Save crop to NVS flash
static esp_err_t save_crop_to_flash(const uint8_t *jpg_data, size_t jpg_len, int crop_idx) {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open("crops", NVS_READWRITE, &nvs);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS: %s", esp_err_to_name(err));
        return err;
    }
    
    char key[8];
    snprintf(key, sizeof(key), "c%d", crop_idx);
    
    err = nvs_set_blob(nvs, key, jpg_data, jpg_len);
    if (err == ESP_OK) {
        err = nvs_commit(nvs);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "    ✓ Crop %d saved: %zu bytes [%s]", crop_idx, jpg_len, key);
        }
    } else {
        ESP_LOGE(TAG, "    ✗ Failed to save crop %d: %s", crop_idx, esp_err_to_name(err));
    }
    
    nvs_close(nvs);
    return err;
}

extern "C" void app_main(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // ========== System Information ==========
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    ESP_LOGI(TAG, "╔════════════════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   GRAPE LEAF DETECTION - ARDUINO CAMERA MODE  ║");
    ESP_LOGI(TAG, "╠════════════════════════════════════════════════╣");
    ESP_LOGI(TAG, "║ Chip: ESP32-S3");
    ESP_LOGI(TAG, "║ Cores: %d", chip_info.cores);
    ESP_LOGI(TAG, "║ Silicon Rev: %d", chip_info.revision);
    uint32_t flash_size;
    esp_flash_get_size(NULL, &flash_size);
    ESP_LOGI(TAG, "║ Flash: %uMB %s", 
             (unsigned int)(flash_size / (1024 * 1024)),
             (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");
    ESP_LOGI(TAG, "║ Free Heap: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));
    ESP_LOGI(TAG, "║ Free PSRAM: %u KB", (unsigned int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(TAG, "╚════════════════════════════════════════════════╝");

    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "🧠 Initializing Detection Model (BEFORE camera)...");

    int64_t init_start = esp_timer_get_time();
    ESPDetDetect *detect = new ESPDetDetect();
    if (!detect) {
        ESP_LOGE(TAG, "Failed to create detection model!");
        return;
    }
    int64_t init_time = (esp_timer_get_time() - init_start) / 1000;

    ESP_LOGI(TAG, "✓ Model initialization complete (%lld ms)", init_time);
    ESP_LOGI(TAG, "  Free heap after init: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));

    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "📷 Initializing Camera (Arduino JPEG mode)...");

    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization failed: 0x%x", err);
        ESP_LOGE(TAG, "Possible issues:");
        ESP_LOGE(TAG, "  - Camera not connected");
        ESP_LOGE(TAG, "  - Wrong GPIO pins");
        ESP_LOGE(TAG, "  - Insufficient memory");
        return;
    }

    // Get camera sensor for OV3660 adjustments
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor != NULL && sensor->id.PID == OV3660_PID) {
        sensor->set_vflip(sensor, 1);        // Flip vertically
        sensor->set_brightness(sensor, 1);    // Increase brightness
        sensor->set_saturation(sensor, -2);   // Lower saturation
        ESP_LOGI(TAG, "✓ Camera initialized: OV3660 (JPEG mode, PSRAM, GRAB_LATEST)");
    } else {
        ESP_LOGI(TAG, "✓ Camera initialized (JPEG mode, PSRAM, GRAB_LATEST)");
    }

    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "🔄 Starting Detection Loop...");

    int frame_num = 0;
    while (true) {
        frame_num++;
        int64_t loop_start = esp_timer_get_time();
        
        ESP_LOGI(TAG, "\n╔════════════════ FRAME %d ════════════════╗", frame_num);
        
        // ========== Step 1: Capture JPEG frame ==========
        int64_t cap_start = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t cap_time = (esp_timer_get_time() - cap_start) / 1000;
        
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed!");
            continue;
        }
        
        ESP_LOGI(TAG, "📸 Captured: %dx%d, %zu bytes (%lld ms) - JPEG format", 
                 fb->width, fb->height, fb->len, cap_time);

        // ========== Step 2: Decode JPEG → RGB888 ==========
        int64_t decode_start = esp_timer_get_time();
        dl::image::jpeg_img_t jpeg_img = {.data = (void *)fb->buf, .data_len = fb->len};
        auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
        int64_t decode_time = (esp_timer_get_time() - decode_start) / 1000;
        
        if (!img.data) {
            ESP_LOGE(TAG, "JPEG decode failed!");
            esp_camera_fb_return(fb);
            continue;
        }
        
        ESP_LOGI(TAG, "✅ JPEG decoded: %dx%d RGB888 (%lld ms)", img.width, img.height, decode_time);

        // ========== Step 3: Run AI Detection ==========
        int64_t det_start = esp_timer_get_time();
        std::vector<dl::detect::result_t> &results = detect->run(img);
        int64_t det_time = (esp_timer_get_time() - det_start) / 1000;
        
        ESP_LOGI(TAG, "🔍 Detected %zu objects (%lld ms)", results.size(), det_time);

        // ========== Step 4: Sort by confidence ==========
        if (!results.empty()) {
            std::sort(results.begin(), results.end(), 
                     [](const dl::detect::result_t &a, const dl::detect::result_t &b) {
                         return a.score > b.score;
                     });
            ESP_LOGI(TAG, "✓ Sorted by confidence (highest first)");

            // ========== Step 5: Process top 10 detections ==========
            int num_crops = (results.size() > 10) ? 10 : results.size();
            ESP_LOGI(TAG, "\n📦 Processing top %d detections:", num_crops);
            
            int64_t crop_start = esp_timer_get_time();
            for (int i = 0; i < num_crops; i++) {
                const auto &det = results[i];
                ESP_LOGI(TAG, "  [%d] Confidence: %.3f, BBox: [x1:%d, y1:%d, x2:%d, y2:%d]",
                         i, det.score, det.box[0], det.box[1], det.box[2], det.box[3]);

                // Crop detection
                int crop_w, crop_h;
                uint8_t *crop_rgb = crop_bbox_rgb888((uint8_t *)img.data, img.width, img.height,
                                                     det.box[0], det.box[1], det.box[2], det.box[3],
                                                     &crop_w, &crop_h);
                if (crop_rgb) {
                    // Convert crop to JPEG
                    uint8_t *jpg_buf = nullptr;
                    size_t jpg_len = 0;
                    
                    bool success = fmt2jpg(crop_rgb, crop_w * crop_h * 3, crop_w, crop_h, 
                                          PIXFORMAT_RGB888, 90, &jpg_buf, &jpg_len);
                    
                    if (success && jpg_buf) {
                        // Save to flash
                        save_crop_to_flash(jpg_buf, jpg_len, i);
                        free(jpg_buf);
                    }
                    
                    heap_caps_free(crop_rgb);
                }
            }
            int64_t crop_time = (esp_timer_get_time() - crop_start) / 1000;
            ESP_LOGI(TAG, "  ✓ Saved %d crops to flash (%lld ms)", num_crops, crop_time);
        } else {
            ESP_LOGI(TAG, "  No detections found");
        }

        // Cleanup
        dl::tool::free_image(img);
        esp_camera_fb_return(fb);

        // ========== Performance Summary ==========
        int64_t loop_time = (esp_timer_get_time() - loop_start) / 1000;
        ESP_LOGI(TAG, "\n⏱️  Performance:");
        ESP_LOGI(TAG, "    Capture:  %lld ms", cap_time);
        ESP_LOGI(TAG, "    Decode:   %lld ms", decode_time);
        ESP_LOGI(TAG, "    Inference: %lld ms", det_time);
        ESP_LOGI(TAG, "    TOTAL:    %lld ms (%.2f FPS)", loop_time, 1000.0 / loop_time);
        ESP_LOGI(TAG, "    Free PSRAM: %u KB", (unsigned int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
        ESP_LOGI(TAG, "╚══════════════════════════════════════════════╝\n");

        // Sleep 30 seconds between captures
        ESP_LOGI(TAG, "⏸️  Sleeping for 30 seconds...\n");
        vTaskDelay(30000 / portTICK_PERIOD_MS);
    }
}
