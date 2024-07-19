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

// 环境变量名
extern const char* DSP_ENV_VAR;

// 默认环境变量路径和实际路径
extern const char* DEFAULT_DSP_LIB_DIR;
extern std::string g_dsp_lib_dir;

// 帧dump目录, dump类型, dump大小
extern std::string g_output_dir;
extern RewooDecodeOutputFileType g_output_file_type;
extern int32_t g_output_width;
extern int32_t g_output_height;

// 输入视频
extern std::string g_video_path;
extern int32_t g_video_width;
extern int32_t g_video_height;
extern std::string g_video_codec;

// 左球门坐标
extern std::vector<Point> g_goalnet_points;

// 模型路径
extern std::string g_dlc_dir;
extern std::string g_dlc_path;

// 输入文件列表路径
extern std::string g_input_list_path;

// 全局标志
extern bool g_decode_done;
extern bool g_dlc_done;