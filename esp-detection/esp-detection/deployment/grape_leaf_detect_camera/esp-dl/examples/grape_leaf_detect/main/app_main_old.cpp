#include "espdet_detect.hpp"
#include "dl_image_jpeg.hpp"
#include "esp_log.h"
#include "bsp/esp-bsp.h"
#include "esp_system.h"
#include "esp_chip_info.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_flash.h"
#include "esp_camera.h"
#include "nvs_flash.h"
#include "nvs.h"
#include <algorithm>
#include <vector>

const char *TAG = "grape_leaf_detect";

// Freenove ESP32-S3 WROOM-1 Camera Configuration (OV2660)
static camera_config_t camera_config = {
    .pin_pwdn = -1,
    .pin_reset = -1,
    .pin_xclk = 15,
    .pin_sccb_sda = 4,
    .pin_sccb_scl = 5,
    .pin_d7 = 18,
    .pin_d6 = 12,
    .pin_d5 = 10,
    .pin_d4 = 8,
    .pin_d3 = 9,
    .pin_d2 = 11,
    .pin_d1 = -1,
    .pin_d0 = -1,
    .pin_vsync = 6,
    .pin_href = 7,
    .pin_pclk = 13,
    
    .xclk_freq_hz = 20000000,           // 20MHz (Arduino working config)
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    
    .pixel_format = PIXFORMAT_RGB565,   // Use RGB565 instead of broken JPEG
    .frame_size = FRAMESIZE_QQVGA,      // 160x120 - reduced to minimize PSRAM DMA
    .jpeg_quality = 10,                 // Not used for RGB565
    .fb_count = 2,                      // Double buffering may reduce cache conflicts
    .fb_location = CAMERA_FB_IN_PSRAM,  // PSRAM framebuffers (after model loaded)
    .grab_mode = CAMERA_GRAB_LATEST,    // Arduino uses LATEST with PSRAM
    .sccb_i2c_port = 0
};

// Convert RGB565 to RGB888
void rgb565_to_rgb888(uint16_t *rgb565, uint8_t *rgb888, int pixels) {
    for (int i = 0; i < pixels; i++) {
        uint16_t pixel = rgb565[i];
        // Extract RGB565 components and scale to RGB888
        rgb888[i*3 + 0] = ((pixel >> 11) & 0x1F) * 255 / 31;  // Red (5 bits)
        rgb888[i*3 + 1] = ((pixel >> 5) & 0x3F) * 255 / 63;   // Green (6 bits)
        rgb888[i*3 + 2] = (pixel & 0x1F) * 255 / 31;          // Blue (5 bits)
    }
}

// Crop a bounding box from RGB888 image
uint8_t* crop_bbox(const dl::image::img_t &img, int x1, int y1, int x2, int y2, int &crop_width, int &crop_height) {
    // Clamp coordinates to image bounds
    x1 = std::max(0, std::min(x1, (int)img.width - 1));
    y1 = std::max(0, std::min(y1, (int)img.height - 1));
    x2 = std::max(0, std::min(x2, (int)img.width));
    y2 = std::max(0, std::min(y2, (int)img.height));
    
    crop_width = x2 - x1;
    crop_height = y2 - y1;
    
    if (crop_width <= 0 || crop_height <= 0) {
        ESP_LOGW(TAG, "Invalid crop dimensions: %dx%d", crop_width, crop_height);
        return nullptr;
    }
    
    // Allocate crop buffer in PSRAM
    size_t crop_size = crop_width * crop_height * 3;  // RGB888
    uint8_t *crop_data = (uint8_t*)heap_caps_malloc(crop_size, MALLOC_CAP_SPIRAM);
    if (!crop_data) {
        ESP_LOGE(TAG, "Failed to allocate %d bytes for crop", crop_size);
        return nullptr;
    }
    
    // Copy rows from source image
    for (int y = 0; y < crop_height; y++) {
        int src_offset = ((y1 + y) * img.width + x1) * 3;
        int dst_offset = y * crop_width * 3;
        memcpy(crop_data + dst_offset, (uint8_t*)img.data + src_offset, crop_width * 3);
    }
    
    return crop_data;
}

// Convert RGB crop to JPEG
uint8_t* rgb_to_jpeg(uint8_t *rgb_data, int width, int height, size_t &jpeg_len) {
    // Create image struct for encoder
    dl::image::img_t crop_img;
    crop_img.width = width;
    crop_img.height = height;
    crop_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    crop_img.data = rgb_data;
    
    // Encode to JPEG (quality 10 for small size)
    dl::image::jpeg_img_t jpeg_result = dl::image::sw_encode_jpeg(crop_img, MALLOC_CAP_SPIRAM, 10);
    
    if (jpeg_result.data == nullptr || jpeg_result.data_len == 0) {
        ESP_LOGE(TAG, "JPEG encoding failed");
        return nullptr;
    }
    
    // Copy the encoded JPEG data
    uint8_t *jpeg_buf = (uint8_t*)malloc(jpeg_result.data_len);
    if (!jpeg_buf) {
        ESP_LOGE(TAG, "Failed to allocate JPEG buffer");
        free(jpeg_result.data);
        return nullptr;
    }
    
    memcpy(jpeg_buf, jpeg_result.data, jpeg_result.data_len);
    jpeg_len = jpeg_result.data_len;
    
    // Free the original encoded data
    free(jpeg_result.data);
    
    return jpeg_buf;
}

// Clear all old crops from flash NVS (to replace with new ones)
esp_err_t clear_old_crops() {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open("crops", NVS_READWRITE, &nvs);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS for clearing: %s", esp_err_to_name(err));
        return err;
    }
    
    // Erase all crops (c0 through c9)
    for (int i = 0; i < 10; i++) {
        char key[8];
        snprintf(key, sizeof(key), "c%d", i);
        nvs_erase_key(nvs, key);  // Ignore errors if key doesn't exist
    }
    
    err = nvs_commit(nvs);
    nvs_close(nvs);
    
    ESP_LOGI(TAG, "🗑️  Cleared old crops from flash");
    return err;
}

// Save crop to flash NVS
esp_err_t save_crop_to_flash(const uint8_t *data, size_t len, int crop_idx) {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open("crops", NVS_READWRITE, &nvs);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS: %s", esp_err_to_name(err));
        return err;
    }
    
    // Simple key: c0, c1, c2, ... c9 (always overwrite)
    char key[8];
    snprintf(key, sizeof(key), "c%d", crop_idx);
    
    // Save blob
    err = nvs_set_blob(nvs, key, data, len);
    if (err == ESP_OK) {
        err = nvs_commit(nvs);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "    ✓ Crop %d saved: %d bytes [%s]", crop_idx, len, key);
        }
    } else {
        ESP_LOGE(TAG, "    ✗ Failed to save crop %d: %s", crop_idx, esp_err_to_name(err));
    }
    
    nvs_close(nvs);
    return err;
}

// Save full frame to flash NVS
esp_err_t save_frame_to_flash(const uint8_t *data, size_t len, int frame_num) {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open("frames", NVS_READWRITE, &nvs);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open frames NVS: %s", esp_err_to_name(err));
        return err;
    }
    
    char key[32];
    snprintf(key, sizeof(key), "frame_%d", frame_num);
    
    err = nvs_set_blob(nvs, key, data, len);
    if (err == ESP_OK) {
        err = nvs_commit(nvs);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "  Frame %d saved: %d bytes", frame_num, len);
        }
    }
    
    nvs_close(nvs);
    return err;
}

static void print_system_info()
{
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    uint32_t flash_size;
    esp_flash_get_size(NULL, &flash_size);
    
    ESP_LOGI(TAG, "╔════════════════════════════════════════════════╗");
    ESP_LOGI(TAG, "║    GRAPE LEAF DETECTION - CAMERA + CROP       ║");
    ESP_LOGI(TAG, "╠════════════════════════════════════════════════╣");
    ESP_LOGI(TAG, "║ Chip: ESP32-%s                                 ", CONFIG_IDF_TARGET);
    ESP_LOGI(TAG, "║ Cores: %d                                      ", chip_info.cores);
    ESP_LOGI(TAG, "║ Silicon Rev: %d                                ", chip_info.revision);
    ESP_LOGI(TAG, "║ Flash: %uMB %s                                 ", 
             (unsigned int)(flash_size / (1024 * 1024)),
             (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");
    ESP_LOGI(TAG, "║ PSRAM: %s                                      ",
             (chip_info.features & CHIP_FEATURE_EMB_PSRAM) ? "Yes" : "No");
    ESP_LOGI(TAG, "║ Camera: OV2660 (640x480 VGA)                   ");
    ESP_LOGI(TAG, "║ Free Heap: %u bytes                            ", 
             (unsigned int)esp_get_free_heap_size());
    ESP_LOGI(TAG, "║ Free PSRAM: %u bytes                           ", 
             (unsigned int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "╚════════════════════════════════════════════════╝");
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
    
    print_system_info();

#if CONFIG_ESPDET_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    // ========== Model Initialization FIRST (before camera!) ========== 
    // ✅ FIX: Load model BEFORE camera to avoid cache conflicts
    // No-camera test proved model loading works fine - camera init interferes with flash access
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "🧠 Initializing Detection Model (BEFORE camera)...");
    
    int64_t init_start = esp_timer_get_time();
    ESPDetDetect *detect = new ESPDetDetect();
    int64_t init_time = (esp_timer_get_time() - init_start) / 1000;
    
    ESP_LOGI(TAG, "✓ Model initialization complete (%lld ms)", init_time);
    ESP_LOGI(TAG, "  Free heap after init: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));

    // ========== Camera Initialization (AFTER model) ==========
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "📷 Initializing Camera (Freenove ESP32-S3 config)...");
    
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization failed: 0x%x", err);
        return;
    }
    
    // Apply Arduino working sensor settings
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor != NULL) {
        // Arduino working config for OV3660
        if (sensor->id.PID == OV3660_PID) {
            sensor->set_vflip(sensor, 1);
            sensor->set_brightness(sensor, 1);
            sensor->set_saturation(sensor, -2);  // Arduino uses -2 for OV3660
        }
        ESP_LOGI(TAG, "✓ Camera initialized (RGB565+DRAM - model loaded first)");
    }
    
    // ========== Main Detection Loop ==========
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "🔄 Starting Camera Test Loop (capture every 30 seconds)...\n");
    
    int frame_count = 0;
    const int MAX_CROPS_PER_FRAME = 10;
    const int CAPTURE_INTERVAL_SEC = 30;  // Test: 30 seconds
    
    while (true) {
        frame_count++;
        int64_t frame_start = esp_timer_get_time();
        
        ESP_LOGI(TAG, "╔════════════════ FRAME %d ════════════════╗", frame_count);
        ESP_LOGI(TAG, "📅 Time: %lld seconds since boot", esp_timer_get_time() / 1000000);
        ESP_LOGI(TAG, "⏰ Next capture in %d minutes\n", CAPTURE_INTERVAL_SEC / 60);
        
        // ========== Capture Image ==========
        int64_t capture_start = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t capture_time = (esp_timer_get_time() - capture_start) / 1000;
        
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed!");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        
        ESP_LOGI(TAG, "📸 Captured: %dx%d, %u bytes (%lld ms) - RGB565 format", 
                 fb->width, fb->height, fb->len, capture_time);
        
        // ========== Convert RGB565 to RGB888 ==========
        int64_t convert_start = esp_timer_get_time();
        int pixels = fb->width * fb->height;
        size_t rgb888_size = pixels * 3;
        
        uint8_t *rgb888_data = (uint8_t*)heap_caps_malloc(rgb888_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!rgb888_data) {
            ESP_LOGE(TAG, "Failed to allocate %d bytes for RGB888 conversion", rgb888_size);
            esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        
        rgb565_to_rgb888((uint16_t*)fb->buf, rgb888_data, pixels);
        int64_t convert_time = (esp_timer_get_time() - convert_start) / 1000;
        
        ESP_LOGI(TAG, "✅ RGB565→RGB888 conversion complete (%lld ms)", convert_time);
        
        // Create image structure for AI model
        dl::image::img_t img;
        img.data = rgb888_data;
        img.width = fb->width;
        img.height = fb->height;
        img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
        
        // Return camera framebuffer (RGB565 no longer needed)
        esp_camera_fb_return(fb);
        
        // ========== Model Disabled - Skip Inference ==========
        if (detect == NULL) {
            ESP_LOGI(TAG, "⚠️  AI model disabled - camera test only");
            ESP_LOGI(TAG, "✅ Camera capture successful! RGB565→RGB888 conversion works.");
            
            // Free RGB888 converted image
            if (rgb888_data) {
                free(rgb888_data);
            }
            
            int64_t total_time = (esp_timer_get_time() - frame_start) / 1000;
            ESP_LOGI(TAG, "⏱️  Total time: %lld ms (capture + convert)", total_time);
            ESP_LOGI(TAG, "╚══════════════════════════════════════════════╝\n");
            
            // Wait 30 seconds before next test
            vTaskDelay(pdMS_TO_TICKS(30000));
            continue;
        }
        
        // ========== Run Inference (if model enabled) ==========
        int64_t inference_start = esp_timer_get_time();
        auto &detect_results = detect->run(img);
        int64_t inference_time = (esp_timer_get_time() - inference_start) / 1000;
        
        ESP_LOGI(TAG, "🔍 Detected %d objects (%lld ms)", 
                 detect_results.size(), inference_time);
        
        // ========== Sort by Confidence ==========
        if (detect_results.size() > 0) {
            // Convert list to vector for sorting
            std::vector<dl::detect::result_t> results_vec(detect_results.begin(), detect_results.end());
            
            // Sort detections by score (descending)
            std::sort(results_vec.begin(), results_vec.end(),
                [](const dl::detect::result_t &a, const dl::detect::result_t &b) {
                    return a.score > b.score;
                });
            
            ESP_LOGI(TAG, "✓ Sorted by confidence (highest first)");
            
            // ========== Crop Top 10 Detections ==========
            int crops_to_save = std::min((int)results_vec.size(), MAX_CROPS_PER_FRAME);
            
            // Clear old crops before saving new ones
            clear_old_crops();
            
            ESP_LOGI(TAG, "\n📦 Processing top %d detections:", crops_to_save);
            
            int64_t crop_start = esp_timer_get_time();
            int successful_crops = 0;
            
            for (int i = 0; i < crops_to_save; i++) {
                const auto &res = results_vec[i];
                
                // Extract box coordinates: box[0]=x1, box[1]=y1, box[2]=x2, box[3]=y2
                int x1 = res.box[0];
                int y1 = res.box[1];
                int x2 = res.box[2];
                int y2 = res.box[3];
                
                ESP_LOGI(TAG, "  [%d] Confidence: %.3f, BBox: [%d,%d,%d,%d]",
                         i, res.score, x1, y1, x2, y2);
                
                // Crop bounding box
                int crop_width, crop_height;
                uint8_t *crop_rgb = crop_bbox(img, x1, y1, x2, y2,
                                              crop_width, crop_height);
                
                if (crop_rgb) {
                    // Convert to JPEG
                    size_t jpeg_len;
                    uint8_t *crop_jpeg = rgb_to_jpeg(crop_rgb, crop_width, crop_height, jpeg_len);
                    
                    if (crop_jpeg) {
                        // Save to flash (simplified keys: c0, c1, ..., c9)
                        esp_err_t save_ret = save_crop_to_flash(crop_jpeg, jpeg_len, i);
                        if (save_ret == ESP_OK) {
                            successful_crops++;
                        }
                        
                        free(crop_jpeg);
                    }
                    
                    free(crop_rgb);
                }
            }
            
            int64_t crop_time = (esp_timer_get_time() - crop_start) / 1000;
            ESP_LOGI(TAG, "✓ Saved %d/%d crops (%lld ms)", 
                     successful_crops, crops_to_save, crop_time);
            ESP_LOGI(TAG, "💾 Flash now contains ONLY the latest %d crops (c0-c%d)", 
                     successful_crops, successful_crops - 1);
        } else {
            ESP_LOGI(TAG, "No detections found");
        }
        
        // Free RGB888 converted image
        if (img.data) {
            free(img.data);
        }
        
        // ========== Performance Summary ==========
        int64_t total_time = (esp_timer_get_time() - frame_start) / 1000;
        float fps = 1000.0f / total_time;
        
        ESP_LOGI(TAG, "\n⏱️  Performance:");
        ESP_LOGI(TAG, "    Capture:   %lld ms", capture_time);
        ESP_LOGI(TAG, "    Convert:   %lld ms", convert_time);
        ESP_LOGI(TAG, "    Inference: %lld ms", inference_time);
        if (detect_results.size() > 0) {
            ESP_LOGI(TAG, "    Crop+Save: %lld ms", 
                     (total_time - capture_time - convert_time - inference_time));
        }
        ESP_LOGI(TAG, "    TOTAL:     %lld ms (%.2f FPS)", total_time, fps);
        ESP_LOGI(TAG, "    Free PSRAM: %u KB", 
                 (unsigned int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
        ESP_LOGI(TAG, "╚══════════════════════════════════════════════╝\n");
        
        // Wait 5 minutes before next capture
        ESP_LOGI(TAG, "⏸️  Sleeping for %d minutes...", CAPTURE_INTERVAL_SEC / 60);
        ESP_LOGI(TAG, "💤 Next capture at: ~%lld seconds\n", (esp_timer_get_time() / 1000000) + CAPTURE_INTERVAL_SEC);
        vTaskDelay(pdMS_TO_TICKS(CAPTURE_INTERVAL_SEC * 1000));
    }
}
