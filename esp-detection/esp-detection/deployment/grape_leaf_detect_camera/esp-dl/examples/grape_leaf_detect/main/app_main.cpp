#include "espdet_detect.hpp"
#include "disease_classifier.hpp"
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
// Note: esp_spiffs.h removed - no longer saving crops to flash
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

// ========== NOTE: Crop/JPEG functions removed - using disease_classifier->crop_and_resize() instead ==========
// Crops are directly resized to 128x128 for classification without intermediate JPEG encoding

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
    ESP_LOGI(TAG, "║ Camera: OV2660 (320x240)                       ");
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
    
    // ========== SPIFFS removed - no longer saving crops to flash ==========
    // Crops are directly classified without storage
    
    print_system_info();

#if CONFIG_ESPDET_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    // ========== Model Initialization (BEFORE Camera) ==========
    // ⚠️ CRITICAL: DO NOT CHANGE INITIALIZATION ORDER!
    // Models MUST be initialized BEFORE camera to avoid PSRAM fragmentation.
    // Camera init allocates large DMA buffers → fragments PSRAM → model init fails
    // Required order: 1) Detection Model  2) Disease Classifier  3) Camera
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "🧠 Initializing Detection Model...");
    int64_t init_start = esp_timer_get_time();
    
    ESPDetDetect *detect = new ESPDetDetect();
    
    int64_t init_time = (esp_timer_get_time() - init_start) / 1000;
    ESP_LOGI(TAG, "✓ Model initialized in %lld ms", init_time);
    ESP_LOGI(TAG, "  Free heap after init: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));
    
    // ========== Disease Classifier Initialization ==========
    ESP_LOGI(TAG, "\n🧬 Initializing Disease Classifier (MobileNetV2 128x128)...");
    int64_t disease_init_start = esp_timer_get_time();
    
    DiseaseClassifier *disease_classifier = new DiseaseClassifier();
    bool disease_classifier_enabled = false;
    
    if (disease_classifier->init("mobilenetv2_128_grape_leaf.espdl") != ESP_OK) {
        ESP_LOGW(TAG, "⚠️  Disease classifier initialization FAILED");
        ESP_LOGW(TAG, "   System will continue with DETECTION ONLY (no disease classification)");
        ESP_LOGW(TAG, "   Check logs above for details on why model loading failed");
        delete disease_classifier;
        disease_classifier = nullptr;
    } else {
        disease_classifier_enabled = true;
        int64_t disease_init_time = (esp_timer_get_time() - disease_init_start) / 1000;
        ESP_LOGI(TAG, "✓ Disease classifier initialized in %lld ms", disease_init_time);
        ESP_LOGI(TAG, "  Free heap after init: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));
    }
    
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
            
            // ========== Disease Classification on Top-K Detections ==========
            const float CONF_THRESHOLD = 0.45f;
            const int TOP_K = 3;
            
            // Filter by confidence and take top-K
            int num_to_classify = 0;
            for (size_t i = 0; i < results_vec.size() && num_to_classify < TOP_K; i++) {
                if (results_vec[i].score >= CONF_THRESHOLD) {
                    num_to_classify++;
                }
            }
            
            if (num_to_classify == 0) {
                ESP_LOGI(TAG, "No detections above confidence threshold %.2f", CONF_THRESHOLD);
            } else if (!disease_classifier_enabled || disease_classifier == nullptr) {
                ESP_LOGW(TAG, "⚠️  Disease classification SKIPPED (classifier not initialized)");
                ESP_LOGI(TAG, "   Detected %d grape leaves but cannot classify diseases", num_to_classify);
            } else {
                ESP_LOGI(TAG, "\n🔬 Running disease classification on top %d detections:", num_to_classify);
                
                int64_t disease_start = esp_timer_get_time();
                std::vector<DiseaseResult> disease_results;
                
                for (int i = 0; i < num_to_classify; i++) {
                    const auto &res = results_vec[i];
                    
                    // Extract box coordinates
                    int x1 = res.box[0];
                    int y1 = res.box[1];
                    int x2 = res.box[2];
                    int y2 = res.box[3];
                    
                    ESP_LOGI(TAG, "  [%d] Bbox: [%d,%d,%d,%d], Conf: %.3f",
                             i, x1, y1, x2, y2, res.score);
                    
                    // Crop and resize directly into classifier buffer
                    disease_classifier->crop_and_resize(
                        (const uint8_t*)img.data, img.width, img.height,
                        x1, y1, x2, y2);
                    
                    // Run classification
                    DiseaseResult dr = disease_classifier->infer();
                    disease_results.push_back(dr);
                    
                    // Show all classes above threshold
                    ESP_LOGI(TAG, "      Disease probabilities (confidence > 10%%):" );
                    bool found_any = false;
                    for (const auto& p : dr.all_classes) {
                        if (p.second >= 0.10f) {  // 10% threshold
                            const char* cls = DiseaseClassifier::get_class_name(p.first);
                            ESP_LOGI(TAG, "         • %s: %.2f%%", cls, p.second * 100.0f);
                            found_any = true;
                        }
                    }
                    if (!found_any) {
                        ESP_LOGI(TAG, "         (no classes above 10%% threshold)");
                    }
                }
                
                int64_t disease_time = (esp_timer_get_time() - disease_start) / 1000;
                
                // ========== Aggregate Results (Max Confidence) ==========
                DiseaseResult final_result = {-1, 0.0f, "unknown"};
                for (const auto &dr : disease_results) {
                    if (dr.confidence > final_result.confidence) {
                        final_result = dr;
                    }
                }
                
                ESP_LOGI(TAG, "\n✅ FINAL DIAGNOSIS: %s (%.1f%% confidence)",
                         final_result.class_name, final_result.confidence * 100.0f);
                ESP_LOGI(TAG, "   Disease inference: %lld ms total (%lld ms avg per crop)",
                         disease_time, disease_time / num_to_classify);
            }
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
