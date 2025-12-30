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
#include <algorithm>

static const char *TAG = "grape_leaf_ai";

// Freenove ESP32-S3 Camera Pins
#define CAM_PIN_D0      11
#define CAM_PIN_D1      9
#define CAM_PIN_D2      8
#define CAM_PIN_D3      10
#define CAM_PIN_D4      12
#define CAM_PIN_D5      18
#define CAM_PIN_D6      17
#define CAM_PIN_D7      16
#define CAM_PIN_XCLK    15
#define CAM_PIN_PCLK    13
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5

// Arduino camera config (JPEG + PSRAM + GRAB_LATEST)
static camera_config_t camera_config = {
    .pin_pwdn  = -1,
    .pin_reset = -1,
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
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,
    .jpeg_quality = 12,
    .fb_count = 2,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_LATEST,
    .sccb_i2c_port = 1
};

extern "C" void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    
    ESP_LOGI(TAG, "╔═══════════════════════════════════════════════╗");
    ESP_LOGI(TAG, "║  GRAPE LEAF - ARDUINO CAMERA + ESP-DL MODEL  ║");
    ESP_LOGI(TAG, "╠═══════════════════════════════════════════════╣");
    ESP_LOGI(TAG, "║ ESP32-S3 | Cores:%d | Rev:%d", chip_info.cores, chip_info.revision);
    ESP_LOGI(TAG, "║ Free Heap: %u KB | Free PSRAM: %u KB", 
             (unsigned int)(esp_get_free_heap_size() / 1024),
             (unsigned int)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(TAG, "╚═══════════════════════════════════════════════╝");

    ESP_LOGI(TAG, "\n🧠 Loading AI Model...");
    int64_t init_start = esp_timer_get_time();
    ESPDetDetect *detect = new ESPDetDetect();
    int64_t init_time = (esp_timer_get_time() - init_start) / 1000;
    ESP_LOGI(TAG, "✓ Model loaded (%lld ms)", init_time);

    ESP_LOGI(TAG, "\n📷 Initializing Camera (Arduino JPEG mode)...");
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "❌ Camera init failed: 0x%x", err);
        return;
    }

    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor && sensor->id.PID == OV3660_PID) {
        sensor->set_vflip(sensor, 1);
        sensor->set_brightness(sensor, 1);
        sensor->set_saturation(sensor, -2);
    }
    ESP_LOGI(TAG, "✓ Camera ready (JPEG/PSRAM/GRAB_LATEST)\n");

    int frame_num = 0;
    while (true) {
        frame_num++;
        ESP_LOGI(TAG, "═══ FRAME %d ═══", frame_num);

        // Capture JPEG
        int64_t cap_start = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t cap_time = (esp_timer_get_time() - cap_start) / 1000;
        
        if (!fb) {
            ESP_LOGE(TAG, "Capture failed!");
            vTaskDelay(1000 / portTICK_PERIOD_MS);
            continue;
        }
        
        ESP_LOGI(TAG, "📸 Captured %dx%d (%zu bytes, %lld ms)", 
                 fb->width, fb->height, fb->len, cap_time);

        // Decode JPEG
        int64_t dec_start = esp_timer_get_time();
        dl::image::jpeg_img_t jpeg_img = {.data = fb->buf, .data_len = fb->len};
        auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
        int64_t dec_time = (esp_timer_get_time() - dec_start) / 1000;
        
        if (!img.data) {
            ESP_LOGE(TAG, "Decode failed!");
            esp_camera_fb_return(fb);
            continue;
        }
        ESP_LOGI(TAG, "🖼️  Decoded %dx%d RGB888 (%lld ms)", img.width, img.height, dec_time);

        // Run AI detection
        int64_t det_start = esp_timer_get_time();
        std::list<dl::detect::result_t> &results = detect->run(img);
        int64_t det_time = (esp_timer_get_time() - det_start) / 1000;
        
        ESP_LOGI(TAG, "🔍 Detected %zu objects (%lld ms)", results.size(), det_time);

        if (!results.empty()) {
            int i = 0;
            for (const auto &det : results) {
                ESP_LOGI(TAG, "  [%d] Confidence:%.3f BBox:[%d,%d,%d,%d]",
                         i++, det.score, det.box[0], det.box[1], det.box[2], det.box[3]);
            }
        }

        // Cleanup
        free(img.data);
        esp_camera_fb_return(fb);

        int64_t total_time = (esp_timer_get_time() - cap_start) / 1000;
        ESP_LOGI(TAG, "⏱️  Total: %lld ms (%.2f FPS)\n", total_time, 1000.0/total_time);

        vTaskDelay(30000 / portTICK_PERIOD_MS);
    }
}
