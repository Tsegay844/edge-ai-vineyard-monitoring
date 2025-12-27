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

# DIRECT MODEL EMBEDDING FIX (Simplified for Windows compatibility)
# The model file path (relative to this CMakeLists.txt)
set(model_file ${CMAKE_CURRENT_LIST_DIR}/../../../models/grape_leaf_detect/models/s3/espdet_pico_224_224_grape_leaf.espdl)

if(EXISTS ${model_file})
    message(STATUS "🔧 DIRECT EMBED: Found model at ${model_file}")
    # Directly set the model file for embedding
    set(embed_files ${model_file})
    message(STATUS "🔧 DIRECT EMBED: Will embed ${model_file} as RODATA")
else()
    message(FATAL_ERROR "❌ Model file not found at ${model_file}")
endif()

idf_component_register(SRC_DIRS ${src_dirs} 
                       INCLUDE_DIRS ${include_dirs} 
                       REQUIRES ${requires} 
                       EMBED_FILES ${embed_files})
