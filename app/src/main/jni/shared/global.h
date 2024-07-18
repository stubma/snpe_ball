#pragma once

#include <string>
#include <vector>

typedef enum {
    REWOO_OUTPUT_NONE,
    REWOO_OUTPUT_YUV,
    REWOO_OUTPUT_JPG,
    REWOO_OUTPUT_RAW_RGB
} RewooDecodeOutputFileType;
typedef struct {
    int x, y;
} Point;
extern const char* DSP_ENV_VAR;
extern const char* DEFAULT_DSP_LIB_DIR;
extern std::string g_output_dir;
extern std::string g_dsp_lib_dir;
extern RewooDecodeOutputFileType g_output_file_type;
extern std::string g_video_path;
extern int32_t g_video_width;
extern int32_t g_video_height;
extern int32_t g_output_width;
extern int32_t g_output_height;
extern std::vector<Point> g_goalnet_points;
extern std::string g_dlc_dir;
extern std::string g_dlc_path;
extern std::string g_input_list_path;