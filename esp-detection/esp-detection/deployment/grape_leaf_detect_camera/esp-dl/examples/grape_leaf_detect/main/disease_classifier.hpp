#pragma once

#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "dl_image_jpeg.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

struct DiseaseResult {
    int class_id;
    float confidence;
    const char* class_name;
    std::vector<std::pair<int, float>> all_classes; // All classes with their probabilities
};

class DiseaseClassifier {
private:
    static constexpr const char* TAG = "DiseaseClassifier";
    
    // Disease class names (configurable)
    static constexpr const char* CLASS_NAMES[] = {
        "healthy",
        "black_rot",
        "esca",
        "leaf_blight"
    };
    static constexpr int NUM_CLASSES = sizeof(CLASS_NAMES) / sizeof(CLASS_NAMES[0]);
    
    dl::Model *model;
    uint8_t *input_buffer_128x128;  // PSRAM buffer for 128x128x3 RGB888, allocated once
    
    int input_width;
    int input_height;
    int input_channels;
    
public:
    DiseaseClassifier() : model(nullptr), input_buffer_128x128(nullptr) {}
    
    ~DiseaseClassifier() {
        if (model) {
            delete model;
            model = nullptr;
        }
        if (input_buffer_128x128) {
            heap_caps_free(input_buffer_128x128);
            input_buffer_128x128 = nullptr;
        }
    }
    
    // Static method to get class name by ID
    static const char* get_class_name(int class_id) {
        if (class_id >= 0 && class_id < NUM_CLASSES) {
            return CLASS_NAMES[class_id];
        }
        return "unknown";
    }
    
    // Initialize model and allocate 128x128 buffer
    esp_err_t init(const char *model_name) {
        ESP_LOGI(TAG, "Loading MobileNetV2 model: %s", model_name);
        ESP_LOGI(TAG, "Attempting to load from packed binary: grape_leaf_detect");
        ESP_LOGI(TAG, "Model location: FLASH_RODATA");
        
        // Debug: Check if embedded binary symbols exist
        extern const uint8_t grape_leaf_detect_espdl_start[] asm("_binary_grape_leaf_detect_espdl_start");
        extern const uint8_t grape_leaf_detect_espdl_end[] asm("_binary_grape_leaf_detect_espdl_end");
        
        size_t packed_size = grape_leaf_detect_espdl_end - grape_leaf_detect_espdl_start;
        ESP_LOGI(TAG, "📦 Packed binary found:");
        ESP_LOGI(TAG, "   Start: %p", grape_leaf_detect_espdl_start);
        ESP_LOGI(TAG, "   End:   %p", grape_leaf_detect_espdl_end);
        ESP_LOGI(TAG, "   Size:  %u bytes (%.2f MB)", packed_size, packed_size / (1024.0 * 1024.0));
        
        if (packed_size == 0) {
            ESP_LOGE(TAG, "❌ Packed binary is empty! Model not embedded correctly.");
            return ESP_FAIL;
        }
        
        // Print first 32 bytes of packed binary header
        ESP_LOGI(TAG, "   Header (first 32 bytes):");
        for (int i = 0; i < 32 && i < packed_size; i += 16) {
            ESP_LOGI(TAG, "   %04x: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
                     i,
                     grape_leaf_detect_espdl_start[i+0], grape_leaf_detect_espdl_start[i+1],
                     grape_leaf_detect_espdl_start[i+2], grape_leaf_detect_espdl_start[i+3],
                     grape_leaf_detect_espdl_start[i+4], grape_leaf_detect_espdl_start[i+5],
                     grape_leaf_detect_espdl_start[i+6], grape_leaf_detect_espdl_start[i+7],
                     grape_leaf_detect_espdl_start[i+8], grape_leaf_detect_espdl_start[i+9],
                     grape_leaf_detect_espdl_start[i+10], grape_leaf_detect_espdl_start[i+11],
                     grape_leaf_detect_espdl_start[i+12], grape_leaf_detect_espdl_start[i+13],
                     grape_leaf_detect_espdl_start[i+14], grape_leaf_detect_espdl_start[i+15]);
        }
        
        // Attempt model loading - use pointer to embedded binary (same as detection model)
        ESP_LOGI(TAG, "🔄 Creating dl::Model instance...");
        model = new dl::Model((const char *)grape_leaf_detect_espdl_start, model_name, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
        
        // Check if model loaded successfully by trying to get input
        // If model is invalid, get_input() will return nullptr or crash
        dl::TensorBase *input_info = nullptr;
        bool model_valid = false;
        
        if (model) {
            ESP_LOGI(TAG, "🔄 Validating model by getting input shape...");
            input_info = model->get_input();
            if (input_info && input_info->shape.size() >= 4) {
                model_valid = true;
                ESP_LOGI(TAG, "✓ Model loaded and validated successfully");
            } else {
                ESP_LOGE(TAG, "❌ Model validation failed - get_input() returned invalid tensor");
                delete model;
                model = nullptr;
            }
        }
        
        if (!model_valid || !model) {
            ESP_LOGE(TAG, "❌ Failed to load model");
            ESP_LOGE(TAG, "   Possible causes:");
            ESP_LOGE(TAG, "   1. Model '%s' not found in packed binary", model_name);
            ESP_LOGE(TAG, "   2. Packed binary format invalid");
            ESP_LOGE(TAG, "   3. Model name must include .espdl extension");
            return ESP_FAIL;
        }
        
        // Get input shape
        input_height = input_info->shape[1];
        input_width = input_info->shape[2];
        input_channels = input_info->shape[3];
        
        ESP_LOGI(TAG, "✓ Model input shape: [1, %d, %d, %d]", input_height, input_width, input_channels);
        
        // Validate expected dimensions
        if (input_height != 128 || input_width != 128 || input_channels != 3) {
            ESP_LOGW(TAG, "⚠️  Unexpected input shape! Expected [1, 128, 128, 3]");
        }
        
        // Allocate 128x128x3 RGB888 buffer ONCE in PSRAM (49152 bytes)
        size_t buffer_size = input_height * input_width * input_channels;
        input_buffer_128x128 = (uint8_t*)heap_caps_malloc(buffer_size, MALLOC_CAP_SPIRAM);
        if (!input_buffer_128x128) {
            ESP_LOGE(TAG, "❌ Failed to allocate %d bytes in PSRAM for input buffer", buffer_size);
            ESP_LOGE(TAG, "   Free PSRAM: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
            delete model;
            model = nullptr;
            return ESP_FAIL;
        }
        
        ESP_LOGI(TAG, "✓ Allocated %d bytes in PSRAM for 128x128x3 input buffer", buffer_size);
        ESP_LOGI(TAG, "✅ Disease classifier initialization complete!");
        return ESP_OK;
    }
    
    // Crop bounding box from full frame and resize to 128x128 (nearest neighbor)
    // Writes directly into input_buffer_128x128 (no intermediate allocations)
    void crop_and_resize(const uint8_t *frame, int frame_w, int frame_h,
                         int x1, int y1, int x2, int y2) {
        // Clamp bbox to frame bounds
        x1 = std::max(0, std::min(x1, frame_w - 1));
        y1 = std::max(0, std::min(y1, frame_h - 1));
        x2 = std::max(0, std::min(x2, frame_w));
        y2 = std::max(0, std::min(y2, frame_h));
        
        int bbox_w = x2 - x1;
        int bbox_h = y2 - y1;
        
        // Nearest-neighbor resize from bbox to 128x128
        for (int y = 0; y < 128; y++) {
            int src_y = y1 + (y * bbox_h) / 128;
            for (int x = 0; x < 128; x++) {
                int src_x = x1 + (x * bbox_w) / 128;
                
                // Copy RGB888 pixel (3 bytes)
                int src_offset = (src_y * frame_w + src_x) * 3;
                int dst_offset = (y * 128 + x) * 3;
                
                input_buffer_128x128[dst_offset + 0] = frame[src_offset + 0];  // R
                input_buffer_128x128[dst_offset + 1] = frame[src_offset + 1];  // G
                input_buffer_128x128[dst_offset + 2] = frame[src_offset + 2];  // B
            }
        }
    }
    
    // Run inference on the 128x128 buffer and return disease classification
    DiseaseResult infer() {
        // Get input tensor and set data pointer
        dl::TensorBase *input = model->get_input();
        input->set_element_ptr(input_buffer_128x128);
        
        // Run inference
        model->run();
        
        // Get output tensor (int8 quantized logits)
        dl::TensorBase *output = model->get_output();
        if (!output) {
            ESP_LOGE(TAG, "Failed to get model output");
            return {-1, 0.0f, "error"};
        }
        
        // Extract int8 logits and convert to float
        int8_t *output_data = (int8_t*)output->get_element_ptr();
        int num_classes = output->get_size();  // Total elements (should be 4 for grape diseases)
        
        if (num_classes != NUM_CLASSES) {
            ESP_LOGW(TAG, "Model output size (%d) != expected classes (%d)", num_classes, NUM_CLASSES);
        }
        
        // Dequantize INT8 → float and apply softmax
        std::vector<float> logits(num_classes);
        for (int i = 0; i < num_classes; i++) {
            logits[i] = (float)output_data[i];  // INT8 [-128, 127] to float
        }
        
        softmax(logits.data(), num_classes);
        
        // Find class with maximum probability
        int best_class = 0;
        float best_conf = logits[0];
        for (int i = 1; i < num_classes; i++) {
            if (logits[i] > best_conf) {
                best_conf = logits[i];
                best_class = i;
            }
        }
        
        // Store all class probabilities (sorted by confidence)
        std::vector<std::pair<int, float>> all_probs;
        for (int i = 0; i < num_classes; i++) {
            all_probs.push_back({i, logits[i]});
        }
        // Sort by confidence descending
        std::sort(all_probs.begin(), all_probs.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return {best_class, best_conf, CLASS_NAMES[best_class], all_probs};
    }
    
private:
    // Numerically stable softmax
    void softmax(float *x, int n) {
        // Find max for numerical stability
        float max_val = x[0];
        for (int i = 1; i < n; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            x[i] = std::exp(x[i] - max_val);
            sum += x[i];
        }
        
        // Normalize
        for (int i = 0; i < n; i++) {
            x[i] /= sum;
        }
    }
};
