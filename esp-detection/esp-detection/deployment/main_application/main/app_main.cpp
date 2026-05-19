#include "espdet_detect.hpp" // Detection model (416×320)
#include "disease_classifier.hpp"  // MobileNetV2 classifier (128×128)
#include "disease_aggregator.hpp"  // Professional weighted aggregation module
#include "dl_image_jpeg.hpp" // JPEG decoding
#include "esp_log.h"
#include "bsp/esp-bsp.h" //
#include "esp_system.h"
#include "esp_chip_info.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_flash.h"
#include "esp_camera.h" // Camera driver
#include "nvs_flash.h"
#include "nvs.h"
// Note: esp_spiffs.h removed - no longer saving crops to flash
#include <algorithm>
#include <vector>
#include <sys/stat.h>

#include <sys/unistd.h>
#include <dirent.h>

// Use namespace for cleaner code
using namespace disease_aggregation; 

const char *TAG = "grape_leaf_DD";

// Freenove ESP32-S3 WROOM-1 Camera Configuration (OV3660)
static camera_config_t camera_config = {
    .pin_pwdn = GPIO_NUM_NC,
    .pin_reset = GPIO_NUM_NC,
    .pin_xclk = 15,
    .pin_sccb_sda = 4,
    .pin_sccb_scl = 5,

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
    
    // camera clock frequency parameters
    .xclk_freq_hz = 20000000, // 20 MHz
    .ledc_timer = LEDC_TIMER_0, 
    .ledc_channel = LEDC_CHANNEL_0, 
    
    // image parameters
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_VGA,  // [set to VGA (640x480), QVGA (320x240), SVGA (800x600) also possible]
    .jpeg_quality = 12,
    .fb_count = 2,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    .sccb_i2c_port = 0
};

// NOTE: using disease_classifier->crop_and_resize()
// Crops are directly resized to 128x128 for classification without intermediate JPEG encoding

// ======= Flash save function =======
// Save full frame to flash NVS
// This saves the full JPEG image (640×480, ~50-100KB) to NVS (Non-Volatile Storage) in the 16MB Flash
/*
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
*/

static void print_system_info()
{
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    uint32_t flash_size;
    esp_flash_get_size(NULL, &flash_size);
    
    ESP_LOGI(TAG, "════════════════════════════════════════════════");
    ESP_LOGI(TAG, "    SYSTEM INFORMATION                           ");
    ESP_LOGI(TAG, "════════════════════════════════════════════════");
    ESP_LOGI(TAG, " Chip: %s                                 ", CONFIG_IDF_TARGET);
    ESP_LOGI(TAG, " Cores: %d                                      ", chip_info.cores);
    ESP_LOGI(TAG, " Silicon Rev: %d                                ", chip_info.revision);
    ESP_LOGI(TAG, " Flash: %uMB %s                                 ", 
             (unsigned int)(flash_size / (1024 * 1024)),
             (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded" : "external");
    ESP_LOGI(TAG, " Camera: OV3660 (640x480)                       ");
    ESP_LOGI(TAG, " Free Heap: %u bytes                            ", 
             (unsigned int)esp_get_free_heap_size());
    ESP_LOGI(TAG, " Free PSRAM: %u bytes                           ", 
             (unsigned int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "════════════════════════════════════════════════");
}

/* 
 Main application entry point
 this function initializes NVS, the camera, the detection model, and the disease classifier
 then enters a loop to capture images, run detection, classify diseases, and log results
*/


// Main application entry point

// the extern "C" is required to prevent name mangling for app_main
extern "C" void app_main(void) 
{
    // Initialize NVS
    // Even captured images no saving on the NVS, 
    // The NVS initialization is still required 
    // because ESP-IDF components expect it to exist
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
      
    print_system_info();

    // Mount SD Card if configured to load model from SD card (uncomment the lines below)
    //#if CONFIG_ESPDET_DETECT_MODEL_IN_SDCARD
    //    ESP_ERROR_CHECK(bsp_sdcard_mount());
    //#endif


    // ========== Model Initialization ==========
    // CRITICAL: DO NOT CHANGE INITIALIZATION ORDER!
    // Models MUST be initialized BEFORE camera to avoid PSRAM fragmentation.
    // Camera init allocates large DMA buffers; fragments PSRAM and this causes model init failure.
    // Required initialization order: 1) Detection Model  2) Disease Classifier  3) Camera
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "Initializing Detection Model...");
    int64_t init_start = esp_timer_get_time();

    ESPDetDetect *detect = new ESPDetDetect();

    // Check if model loaded successfully
    if (detect == nullptr) {
        ESP_LOGE(TAG, "FATAL: Detection model initialization FAILED!");
        ESP_LOGE(TAG, "System cannot continue without detection model.");
        return;  // Exit app_main - system won't work
    }

    int64_t init_time = (esp_timer_get_time() - init_start) / 1000;
    ESP_LOGI(TAG, "Detection model initialized in %lld ms", init_time);
    //Internal SRAM heap (512KB)
    // the esp_get_free_heap_size() returns the free heap size in bytes from the internal SRAM
    ESP_LOGI(TAG, "Free heap after init: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));

    // ========== Disease Classifier Initialization ==========
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "\nInitializing Disease Classifier (MobileNetV2 128x128)...");
    int64_t disease_init_start = esp_timer_get_time();

    // Initialize Disease Classifier
    DiseaseClassifier *disease_classifier = new DiseaseClassifier();
    bool disease_classifier_enabled = false;
    
    if (disease_classifier->init("mobilenetv2_128_grape_leaf.espdl") != ESP_OK) {
        ESP_LOGW(TAG, " Disease classifier initialization FAILED");
        ESP_LOGW(TAG, "   System will continue with DETECTION ONLY (no disease classification)");
        ESP_LOGW(TAG, "   Check logs above for details on why model loading failed");
        delete disease_classifier;
        disease_classifier = nullptr;
    } else {
        disease_classifier_enabled = true;
        int64_t disease_init_time = (esp_timer_get_time() - disease_init_start) / 1000;
        ESP_LOGI(TAG, "Disease classifier initialized in %lld ms", disease_init_time);
        ESP_LOGI(TAG, "Free heap after init: %u KB", (unsigned int)(esp_get_free_heap_size() / 1024));
    }
    
    // ========== Camera Initialization (AFTER Model) ==========
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "Initializing Camera (OV3660)...");
    
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
    
    // Optimize JPEG quality for OV3660
    s->set_quality(s, 12);
    
    // Enable automatic exposure and white balance for better image quality
    s->set_gain_ctrl(s, 1);      // Auto gain on
    s->set_exposure_ctrl(s, 1);  // Auto exposure on
    s->set_whitebal(s, 1);       // Auto white balance on
    s->set_awb_gain(s, 1);       // Auto white balance gain on
    
    ESP_LOGI(TAG, "Camera initialized successfully");
    ESP_LOGI(TAG, "  Sensor: OV3660, Format: JPEG, Quality: 12");
    


    // ========== Main Detection Loop ==========
    ESP_LOGI(TAG, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    ESP_LOGI(TAG, "Starting Detection Loop (capture every 5 minutes)...\n");
    
    int frame_count = 0;
    const int CAPTURE_INTERVAL_SEC = 300;  // 5 minutes = 300 seconds
    /*
        Main loop: capture image, run detection, classify diseases, log results
        1. capture image from camera
        2. decode JPEG to RGB888
        3. run detection model
        4. sort detections by confidence
        5. run disease classification on top-K detections
        6. log results and timings
        7. wait for next capture interval
    */
    while (true) {
        frame_count++;
        int64_t frame_start = esp_timer_get_time();
        
        ESP_LOGI(TAG, "════════════════ FRAME %d ════════════════", frame_count);
        ESP_LOGI(TAG, "Time: %lld seconds since boot", esp_timer_get_time() / 1000000);
        ESP_LOGI(TAG, "Next capture in %d minutes\n", CAPTURE_INTERVAL_SEC / 60);
        
        // ========== Capture Image ==========
        int64_t capture_start = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t capture_time = (esp_timer_get_time() - capture_start) / 1000;
        
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed!");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        
        ESP_LOGI(TAG, "Captured: %dx%d, %u bytes (%lld ms)", 
                 fb->width, fb->height, fb->len, capture_time);
        
        // Save full frame to flash
        //save_frame_to_flash(fb->buf, fb->len, frame_count);
        
        // ========== Decode JPEG ==========
        int64_t decode_start = esp_timer_get_time();
        // Decode JPEG to RGB888 using dl::image::sw_decode_jpeg software decoder because Esp32-s3 doesn't have hardware JPEG decoder
        dl::image::jpeg_img_t jpeg_img = {.data = (void *)fb->buf, .data_len = fb->len};
        auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
        int64_t decode_time = (esp_timer_get_time() - decode_start) / 1000;
        
        if (!img.data) {
            ESP_LOGE(TAG, "JPEG decode failed!");
            esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        
        ESP_LOGI(TAG, "Decoded: %dx%d RGB888 (%lld ms)", 
                 img.width, img.height, decode_time);
        
        // Return frame buffer
        esp_camera_fb_return(fb);
        
        // ========== Run Inference ==========
        int64_t inference_start = esp_timer_get_time();
        auto &detect_results = detect->run(img);
        /* Detection result structure:
        results_vec {
            int box[4];      // [x1, y1, x2, y2]
            float score;     // Confidence 0.0-1.0
            int category_id; // Class ID (0 for grape_leaf)
            }*/
        int64_t inference_time = (esp_timer_get_time() - inference_start) / 1000;
        
        ESP_LOGI(TAG, "Detected %d objects (%lld ms)", 
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
            
            ESP_LOGI(TAG, "Sorted by confidence (highest first)");
            
            // ========== Disease Classification on Top-K Detections ==========
            const float CONF_THRESHOLD = 0.25f;  // Lowered from 0.5 to classify more leaves
            const int TOP_K = 10;
            
            // Filter detections: only keep those above threshold, then take top-K
            std::vector<dl::detect::result_t> valid_detections;
            for (const auto &res : results_vec) {
                if (res.score >= CONF_THRESHOLD) {
                    valid_detections.push_back(res);
                    if (valid_detections.size() >= TOP_K) {
                        break;  // Stop after collecting TOP_K valid detections
                    }
                }
            }
            
            int num_to_classify = valid_detections.size();
            
            if (num_to_classify == 0) {
                ESP_LOGI(TAG, "No detections above confidence threshold %.2f", CONF_THRESHOLD);
            } else if (!disease_classifier_enabled || disease_classifier == nullptr) {
                ESP_LOGW(TAG, "Disease classification SKIPPED (classifier not initialized)");
                ESP_LOGI(TAG, "Detected %d grape leaves but cannot classify diseases", num_to_classify);
            } else {
                ESP_LOGI(TAG, "Running disease classification on top %d detections:", num_to_classify);
                
                std::vector<DiseaseResult> disease_results;
                int64_t total_crop_time = 0;
                int64_t total_setup_time = 0;
                int64_t total_inference_time = 0;
                int64_t total_postprocess_time = 0;
                
                /*
                    For each of the top-K detections:
                    1. Crop and resize to 128x128
                    2. Run disease classification
                    3. Log results and timings
                */
                // Store detection confidences and bbox info for weighted aggregation
                std::vector<BboxInfo> bbox_info_list;
                
                for (int i = 0; i < num_to_classify; i++) {
                    const auto &res = valid_detections[i]; // Use filtered detections, not original results_vec
                    // Extract box coordinates
                    int x1 = res.box[0]; // left
                    int y1 = res.box[1]; // top
                    int x2 = res.box[2]; // right
                    int y2 = res.box[3];  // bottom 
                    int bbox_area = (x2 - x1) * (y2 - y1);// calculate bbox area
                    
                    // Store bbox info and detection confidence for aggregation
                    bbox_info_list.emplace_back(x1, y1, x2, y2, res.score);
                    
                    /*
                    Log bounding box info and confidence to be learn which leaf to classify
                    1. Bounding box coordinates [x1,y1,x2,y2]
                    2. Bounding box area in pixels (px²)
                    3. Leaf detection confidence score
                    */
                    ESP_LOGI(TAG, "  [%d] Bbox: [%d,%d,%d,%d] (%d px²), Leaf_Confidence: %.3f",
                             i, x1, y1, x2, y2, bbox_area, res.score); 
                    
                    // Crop and resize (with timing)
                    int64_t crop_time = disease_classifier->crop_and_resize(
                        (const uint8_t*)img.data, img.width, img.height,
                        x1, y1, x2, y2);
                    total_crop_time += crop_time; // accumulate crop time
                    
                    // Run classification (returns timing breakdown)
                    /*
                    infer() returns a DiseaseResult struct containing:
                    struct DiseaseResult {
                        int class_id;            // Predicted class ID
                        float confidence;        // Confidence score of predicted class
                        const char* class_name;  // Human-readable class name
                        int64_t setup_us;        // Time for tensor setup in microseconds
                        int64_t inference_us;    // Time for model inference in microseconds
                        int64_t postprocess_us;  // Time for post-processing in microseconds
                        std::map<int, float> all_classes; // Map of all class IDs to confidence
                    */
                    DiseaseResult dr = disease_classifier->infer();
                    disease_results.push_back(dr);
                    
                    total_setup_time += dr.setup_us;
                    total_inference_time += dr.inference_us;
                    total_postprocess_time += dr.postprocess_us;
                    
                    // Show timing breakdown for this crop
                    // ESP_LOGI(TAG, "      Timing: crop=%lld μs, setup=%lld μs, fwd=%lld μs, post=%lld μs",
                    //         crop_time, dr.setup_us, dr.inference_us, dr.postprocess_us);

                    /*
                    Build result struct (class_id follows ImageFolder order: 0=Black_rot, 1=Esca, 2=Healthy, 3=Leaf_blight)
                        return DiseaseResult {
                            .class_id = 1,              e.g Esca (index 1)
                            .confidence = 1.0,          e.g  100%
                            .class_name = "Esca",
                            .all_classes = [...],        All 4 class probabilities
                            .setup_us = 8,
                            .inference_us = 530331,
                            .postprocess_us = 88
                        };
                    */
                    // Show all classes above threshold
                    ESP_LOGI(TAG, "      Disease Probability:" );
                    bool found_any = false;
                    for (const auto& [class_id, confidence] : dr.all_classes) {
                        if (confidence >= 0.10f) {
                            const char* cls = DiseaseClassifier::get_class_name(class_id);
                            ESP_LOGI(TAG, "         • %s: %.2f%%", cls, confidence * 100.0f);
                            found_any = true;
                        }
                    }
                    if (!found_any) {
                        ESP_LOGI(TAG, "         (no classes above 10%% confidence threshold)");
                    }
                }
                
                // ========== Weighted Aggregation ==========
                // Configure aggregation method
                AggregationConfig agg_config;
                
                // Choose aggregation strategy (modify these flags as needed):
                // BASELINE: Simple detection-confidence weighting
                agg_config.use_entropy_weighting = true;   // Set to true for uncertainty-aware
                agg_config.use_spatial_weighting = true;   // Set to true for spatial quality
                
                // ADVANCED: Hybrid method (uncomment to enable)
                // agg_config.use_entropy_weighting = true;  // Weight by prediction certainty
                // agg_config.use_spatial_weighting = true;  // Weight by bbox quality
                
                // Run aggregation
                AggregationResult agg_result = DiseaseAggregator::aggregate(
                    disease_results,
                    bbox_info_list,
                    img.width,   // 640
                    img.height,  // 480
                    agg_config
                );
                
                // Print detailed results
                DiseaseAggregator::print_results(agg_result, TAG);
                
                /*
                // Total time and breakdown
                int64_t total_time_ms = (total_crop_time + total_setup_time + total_inference_time + total_postprocess_time) / 1000;
                int64_t avg_fwd_ms = (total_inference_time / num_to_classify) / 1000;
                
                ESP_LOGI(TAG, "\n Classification Timing breakdown (%d crops):", num_to_classify);
                ESP_LOGI(TAG, "      Crop+Resize:  %lld ms (%.1f%%)", total_crop_time / 1000,
                         100.0f * total_crop_time / (total_crop_time + total_setup_time + total_inference_time + total_postprocess_time));
                ESP_LOGI(TAG, "      Tensor Setup: %lld ms (%.1f%%)", total_setup_time / 1000,
                         100.0f * total_setup_time / (total_crop_time + total_setup_time + total_inference_time + total_postprocess_time));
                ESP_LOGI(TAG, "      MobileNet:    %lld ms (%.1f%%) ← %lld ms avg/crop", total_inference_time / 1000,
                         100.0f * total_inference_time / (total_crop_time + total_setup_time + total_inference_time + total_postprocess_time),
                         avg_fwd_ms);
                ESP_LOGI(TAG, "      Postprocess:  %lld ms (%.1f%%)", total_postprocess_time / 1000,
                         100.0f * total_postprocess_time / (total_crop_time + total_setup_time + total_inference_time + total_postprocess_time));
                ESP_LOGI(TAG, "      TOTAL:        %lld ms", total_time_ms); */
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
        
        ESP_LOGI(TAG, "\n Performance:");
        ESP_LOGI(TAG, "    Capture:   %lld ms", capture_time);
        ESP_LOGI(TAG, "    Decode:    %lld ms", decode_time);
        ESP_LOGI(TAG, "    Detection: %lld ms", inference_time);
        if (detect_results.size() > 0 && disease_classifier_enabled) {
            ESP_LOGI(TAG, "    Disease:   %lld ms (classification pipeline)", 
                     (total_time - capture_time - decode_time - inference_time));
        }
        ESP_LOGI(TAG, "    TOTAL:     %lld ms (%.2f FPS)", total_time, fps);
            //External PSRAM heap (8MB)
            // the heap_caps_get_free_size(MALLOC_CAP_SPIRAM) returns the free heap size in bytes from the external PSRAM
        ESP_LOGI(TAG, "    Free PSRAM: %u KB", 
                 (unsigned int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
        ESP_LOGI(TAG, "══════════════════════════════════════════════\n");
        
        // Wait 5 minutes before next capture
        ESP_LOGI(TAG, " Sleeping for %d minutes...", CAPTURE_INTERVAL_SEC / 60);
       // ESP_LOGI(TAG, " Next capture at: ~%lld seconds\n", (esp_timer_get_time() / 1000000) + CAPTURE_INTERVAL_SEC);
        vTaskDelay(pdMS_TO_TICKS(CAPTURE_INTERVAL_SEC * 1000));  // 5 minutes = 300,000 ms
    }
}
