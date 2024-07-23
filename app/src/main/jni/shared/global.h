#pragma once

#include <string>
#include <vector>

typedef enum {
    REWOO_DUMP_NONE,
    REWOO_DUMP_YUV,
    REWOO_DUMP_JPG,
    REWOO_DUMP_RAW_RGB
} RewooDecodeOutputFileType;
typedef struct {
    int32_t x, y;
} Point;

std::string get_dump_type_str(RewooDecodeOutputFileType t);

// current directory
extern std::string g_cwd;

// 环境变量名
extern const char* DSP_ENV_VAR;

// 默认环境变量路径和实际路径
extern std::string g_dsp_lib_dir;

// 帧dump目录, dump类型, dump大小
extern std::string g_dump_dir;
extern RewooDecodeOutputFileType g_dump_file_type;
extern int32_t g_dump_from_frame;
extern int32_t g_dump_to_frame;

// 输入视频
extern std::string g_video_path;
extern int32_t g_video_width;
extern int32_t g_video_height;
extern std::string g_video_codec;

// 左球门坐标
extern std::vector<Point> g_goalnet_points;

// 模型路径
extern std::string g_goal_dlc_path;
extern std::string g_net_dlc_path;

// 全局标志
extern bool g_decode_done;
extern bool g_dlc_done;