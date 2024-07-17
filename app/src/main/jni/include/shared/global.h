#pragma once

#include <string>

typedef enum {
    REWOO_OUTPUT_NONE,
    REWOO_OUTPUT_YUV,
    REWOO_OUTPUT_JPG
} RewooDecodeOutputFileType;
extern const char* DSP_ENV_VAR;
extern const char* DEFAULT_DSP_LIB_DIR;
extern std::string g_output_dir;
extern std::string g_dsp_lib_dir;
extern RewooDecodeOutputFileType g_output_file_type;
extern int32_t g_video_width;
extern int32_t g_video_height;