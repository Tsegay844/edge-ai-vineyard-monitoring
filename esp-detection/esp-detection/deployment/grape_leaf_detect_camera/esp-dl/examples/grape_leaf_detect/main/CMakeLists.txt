set(src_dirs        ./)

set(include_dirs    ./)

set(requires        grape_leaf_detect
                    nvs_flash)

if (IDF_TARGET STREQUAL "esp32s3")
    list(APPEND requires esp32_s3_eye_noglib
                         esp_lcd)
elseif (IDF_TARGET STREQUAL "esp32p4")
    list(APPEND requires esp32_p4_function_ev_board_noglib
                         esp_lcd)
endif()

# ✅ FIX: Use component-based model loading (working approach from grape_leaf_detect_no_camera)
# Model is loaded via grape_leaf_detect component's CMakeLists.txt
# This avoids ESP-IDF v5.3.3 cache bug with direct RODATA embedding
idf_component_register(SRC_DIRS ${src_dirs} 
                       INCLUDE_DIRS ${include_dirs} 
                       REQUIRES ${requires})
