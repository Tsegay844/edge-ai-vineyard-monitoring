#include "espdet_detect.hpp"
#include "esp_log.h"
#include <cstring>
#include "esp_heap_caps.h"
#include "esp_partition.h"

static uint8_t *model_psram_buffer = nullptr;

#if CONFIG_ESPDET_DETECT_MODEL_IN_FLASH_RODATA
extern const uint8_t grape_leaf_detect_espdl[] asm("_binary_grape_leaf_detect_espdl_start");
extern const uint8_t grape_leaf_detect_espdl_end[] asm("_binary_grape_leaf_detect_espdl_end");
static const char *path = (const char *)grape_leaf_detect_espdl;
#elif CONFIG_ESPDET_DETECT_MODEL_IN_FLASH_PARTITION
// Model loaded from partition via esp_partition_read() in ESPDet constructor
#else
#if !defined(CONFIG_BSP_SD_MOUNT_POINT)
#define CONFIG_BSP_SD_MOUNT_POINT "/sdcard"
#endif
#endif

namespace espdet_detect {
ESPDet::ESPDet(const char *model_name)
{
#if CONFIG_ESPDET_DETECT_MODEL_IN_FLASH_PARTITION
    // Load model from FLASH partition to PSRAM using esp_partition_read()
    // This is cache-safe and the IDF-recommended method for large assets
    if (model_psram_buffer == nullptr) {
        ESP_LOGI("grape_leaf_detect", "Loading model from partition to PSRAM...");
        
        // 1. Locate the model partition
        const esp_partition_t *partition = esp_partition_find_first(
            ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_SPIFFS, "espdet_det");
        if (partition == nullptr) {
            ESP_LOGE("grape_leaf_detect", "FATAL: Model partition 'espdet_det' not found!");
            return;
        }
        
        ESP_LOGI("grape_leaf_detect", "Found model partition at 0x%lx, size %lu bytes", 
                 (unsigned long)partition->address, (unsigned long)partition->size);
        
        // 2. Allocate 16-byte aligned PSRAM buffer
        model_psram_buffer = (uint8_t *)heap_caps_aligned_alloc(16, partition->size, 
                                                                  MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (model_psram_buffer == nullptr) {
            ESP_LOGE("grape_leaf_detect", "FATAL: Failed to allocate %lu bytes in PSRAM!", 
                     (unsigned long)partition->size);
            return;
        }
        
        ESP_LOGI("grape_leaf_detect", "Allocated %lu bytes at %p (16-byte aligned)", 
                 (unsigned long)partition->size, model_psram_buffer);
        
        // 3. Read model from partition using cache-safe esp_partition_read()
        esp_err_t err = esp_partition_read(partition, 0, model_psram_buffer, partition->size);
        if (err != ESP_OK) {
            ESP_LOGE("grape_leaf_detect", "FATAL: Failed to read model: %s", esp_err_to_name(err));
            heap_caps_free(model_psram_buffer);
            model_psram_buffer = nullptr;
            return;
        }
        
        ESP_LOGI("grape_leaf_detect", "✓ Model loaded to PSRAM (%lu bytes)", 
                 (unsigned long)partition->size);
    }
    
    // 4. Initialize ESP-DL model with PSRAM buffer
    m_model = new dl::Model((const char *)model_psram_buffer, model_name, 
                           static_cast<fbs::model_location_type_t>(1)); // MEMORY location
#elif !CONFIG_ESPDET_DETECT_MODEL_IN_SDCARD
    m_model =
        new dl::Model(path, model_name, static_cast<fbs::model_location_type_t>(CONFIG_ESPDET_DETECT_MODEL_LOCATION));
#else
    char sd_path[256];
    snprintf(sd_path,
             sizeof(sd_path),
             "%s/%s/%s",
             CONFIG_BSP_SD_MOUNT_POINT,
             CONFIG_ESPDET_DETECT_MODEL_SDCARD_DIR,
             model_name);
    m_model = new dl::Model(sd_path, static_cast<fbs::model_location_type_t>(CONFIG_ESPDET_DETECT_MODEL_LOCATION));
#endif
    m_model->minimize();
#if CONFIG_IDF_TARGET_ESP32P4
    m_image_preprocessor = new dl::image::ImagePreprocessor(m_model, {0, 0, 0}, {255, 255, 255});
#else
    m_image_preprocessor = new dl::image::ImagePreprocessor(
        m_model, {0, 0, 0}, {255, 255, 255}, dl::image::DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
#endif
    m_image_preprocessor->enable_letterbox({114, 114, 114});
    m_postprocessor = new dl::detect::ESPDetPostProcessor(
        m_model, m_image_preprocessor, 0.25, 0.7, 10, {{8, 8, 4, 4}, {16, 16, 8, 8}, {32, 32, 16, 16}});
}

} // namespace espdet_detect

ESPDetDetect::ESPDetDetect(model_type_t model_type) : m_model_type(model_type)
{
    load_model();
}

void ESPDetDetect::load_model()
{
    switch (m_model_type) {
    case model_type_t::ESPDET_PICO_320_320_GRAPE_LEAF:
#if CONFIG_FLASH_ESPDET_PICO_320_320_GRAPE_LEAF || CONFIG_GRAPE_LEAF_DETECT_MODEL_IN_SDCARD
        m_model = new espdet_detect::ESPDet("espdet_pico_320_320_grape_leaf.espdl");
#else
        ESP_LOGE("grape_leaf_detect", "espdet_pico_320_320_grape_leaf is not selected in menuconfig.");
#endif
        break;
    }
}
