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
#include "esp_spiffs.h"
#include <algorithm>
#include <vector>
#include <sys/stat.h>
#include <sys/unistd.h>
#include <dirent.h>

const char *TAG = "grape_leaf_detect";

// Freenove ESP32-S3 WROOM-1 Camera Configuration (OV3660)
static camera_config_t camera_config = {
    .pin_pwdn = GPIO_NUM_NC,
    .pin_reset = GPIO_NUM_NC,
    .pin_xclk = 15,
    .pin_sccb_sda = 4,
    .pin_sccb_scl = 5,
    // Match ESP32S3-EYE wiring from Arduino CameraWebServer
    .pin_d7 = 16,
    .pin_d6 = 17,
    .pin_d5 = 18,
    .pin_d4 = 12,
    .pin_d3 = 10,
    .pin_d2 = 8,
    .pin_d1 = 9,
    .pin_d0 = 11,
    .pin_vsync = 6,
    .pin_href = 7,
    .pin_pclk = 13,
    
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_VGA,        // 640x480
    .jpeg_quality = 12,
    .fb_count = 2,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    .sccb_i2c_port = 0
};

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

// Clear all old crops from SPIFFS (to replace with new ones)
esp_err_t clear_old_crops() {
    // Delete all c*.jpg files in /spiffs
    DIR *dir = opendir("/spiffs");
    if (dir) {
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_name[0] == 'c' && strstr(entry->d_name, ".jpg")) {
                char path[280];
                snprintf(path, sizeof(path), "/spiffs/%s", entry->d_name);
                unlink(path);
            }
        }
        closedir(dir);
    }
    ESP_LOGI(TAG, "🗑️  Cleared old crops from flash");
    return ESP_OK;
}

// Save crop to SPIFFS as JPEG file
esp_err_t save_crop_to_flash(const uint8_t *data, size_t len, int crop_idx) {
    char path[280];
    snprintf(path, sizeof(path), "/spiffs/c%d.jpg", crop_idx);
    
    FILE *f = fopen(path, "wb");
    if (!f) {
        ESP_LOGE(TAG, "    ✗ Failed to open %s for writing", path);
        return ESP_FAIL;
    }
    
    size_t written = fwrite(data, 1, len, f);
    fclose(f);
    
    if (written == len) {
        ESP_LOGI(TAG, "    ✓ Crop %d saved: %d bytes [c%d.jpg]", crop_idx, len, crop_idx);
        return ESP_OK;
    } else {
        ESP_LOGE(TAG, "    ✗ Failed to write crop %d (wrote %d/%d bytes)", crop_idx, written, len);
        return ESP_FAIL;
    }
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
    
    // Initialize SPIFFS for crop storage (512KB partition)
    ESP_LOGI(TAG, "📁 Mounting SPIFFS...");
    esp_vfs_spiffs_conf_t spiffs_conf = {
        .base_path = "/spiffs",
        .partition_label = "espdet_det",
        .max_files = 15,
        .format_if_mount_failed = true
    };
    ret = esp_vfs_spiffs_register(&spiffs_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount SPIFFS: %s", esp_err_to_name(ret));
    } else {
        size_t total = 0, used = 0;
        esp_spiffs_info("espdet_det", &total, &used);
        ESP_LOGI(TAG, "✓ SPIFFS mounted: %d KB total, %d KB used, %d KB free", 
                 total/1024, used/1024, (total-used)/1024);
    }
    
    print_system_info();

#if CONFIG_ESPDET_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    // ========== Model Initialization (BEFORE Camera) ==========
    // CRITICAL: Load model before camera to avoid cache conflicts during partition read
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "🧠 Initializing Detection Model...");
    int64_t init_start = esp_timer_get_time();
    
    ESPDetDetect *detect = new ESPDetDetect();
    
    int64_t init_time = (esp_timer_get_time() - init_start) / 1000;
    ESP_LOGI(TAG, "✓ Model initialized in %lld ms", init_time);
    ESP_LOGI(TAG, "  Free heap after init: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));
    
    // ========== Camera Initialization (AFTER Model) ==========
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "📷 Initializing Camera (OV3660)...");
    
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization failed: 0x%x", err);
        return;
    }
    
    // Get sensor handle and explicitly configure JPEG output for OV3660
    sensor_t *s = esp_camera_sensor_get();
    if (s == NULL) {
        ESP_LOGE(TAG, "Failed to get camera sensor handle");
        return;
    }
    
    // Verify and set JPEG format explicitly
    if (s->pixformat != PIXFORMAT_JPEG) {
        ESP_LOGW(TAG, "Camera not in JPEG mode, setting to JPEG...");
        s->set_pixformat(s, PIXFORMAT_JPEG);
    }
    
    // Optimize JPEG quality for OV3660 (10-12 is good balance)
    s->set_quality(s, 12);
    
    // Enable automatic exposure and white balance for better image quality
    s->set_gain_ctrl(s, 1);      // Auto gain on
    s->set_exposure_ctrl(s, 1);  // Auto exposure on
    s->set_whitebal(s, 1);       // Auto white balance on
    s->set_awb_gain(s, 1);       // Auto white balance gain on
    
    ESP_LOGI(TAG, "✓ Camera initialized successfully");
    ESP_LOGI(TAG, "  Sensor: OV3660, Format: JPEG, Quality: 12");
    
    // ========== Main Detection Loop ==========
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "🔄 Starting Detection Loop (capture every 5 minutes)...\n");
    
    int frame_count = 0;
    const int MAX_CROPS_PER_FRAME = 10;
    const int CAPTURE_INTERVAL_SEC = 300;  // 5 minutes = 300 seconds
    
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
        
        ESP_LOGI(TAG, "📸 Captured: %dx%d, %u bytes (%lld ms)", 
                 fb->width, fb->height, fb->len, capture_time);
        
        // Save full frame to flash
        save_frame_to_flash(fb->buf, fb->len, frame_count);
        
        // ========== Decode JPEG ==========
        int64_t decode_start = esp_timer_get_time();
        dl::image::jpeg_img_t jpeg_img = {.data = (void *)fb->buf, .data_len = fb->len};
        auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
        int64_t decode_time = (esp_timer_get_time() - decode_start) / 1000;
        
        if (!img.data) {
            ESP_LOGE(TAG, "JPEG decode failed!");
            esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        
        ESP_LOGI(TAG, "🖼️  Decoded: %dx%d RGB888 (%lld ms)", 
                 img.width, img.height, decode_time);
        
        // Return frame buffer
        esp_camera_fb_return(fb);
        
        // ========== Run Inference ==========
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
            ESP_LOGI(TAG, "💾 SPIFFS now contains ONLY the latest %d crops (c0.jpg-c%d.jpg)", 
                     successful_crops, successful_crops - 1);
        } else {
            ESP_LOGI(TAG, "No detections found");
        }
        
        // Free decoded image
        if (img.data) {
            free(img.data);
        }
        
        // ========== Performance Summary ==========
        int64_t total_time = (esp_timer_get_time() - frame_start) / 1000;
        float fps = 1000.0f / total_time;
        
        ESP_LOGI(TAG, "\n⏱️  Performance:");
        ESP_LOGI(TAG, "    Capture:   %lld ms", capture_time);
        ESP_LOGI(TAG, "    Decode:    %lld ms", decode_time);
        ESP_LOGI(TAG, "    Inference: %lld ms", inference_time);
        if (detect_results.size() > 0) {
            ESP_LOGI(TAG, "    Crop+Save: %lld ms", 
                     (total_time - capture_time - decode_time - inference_time));
        }
        ESP_LOGI(TAG, "    TOTAL:     %lld ms (%.2f FPS)", total_time, fps);
        ESP_LOGI(TAG, "    Free PSRAM: %u KB", 
                 (unsigned int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
        ESP_LOGI(TAG, "╚══════════════════════════════════════════════╝\n");
        
        // Wait 5 minutes before next capture
        ESP_LOGI(TAG, "⏸️  Sleeping for %d minutes...", CAPTURE_INTERVAL_SEC / 60);
        ESP_LOGI(TAG, "💤 Next capture at: ~%lld seconds\n", (esp_timer_get_time() / 1000000) + CAPTURE_INTERVAL_SEC);
        vTaskDelay(pdMS_TO_TICKS(CAPTURE_INTERVAL_SEC * 1000));  // 5 minutes = 300,000 ms
    }
}
