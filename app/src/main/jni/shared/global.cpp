#include <string>
#include "global.h"

const char* DSP_ENV_VAR = "ADSP_LIBRARY_PATH";
const char* DEFAULT_DSP_LIB_DIR = "/vendor/lib";
std::string g_output_dir = "/data/local/tmp/decode_output";
std::string g_dsp_lib_dir = DEFAULT_DSP_LIB_DIR;
RewooDecodeOutputFileType g_output_file_type = REWOO_OUTPUT_NONE;
int32_t g_video_width = 7600;
int32_t g_video_height = 2160;
int32_t g_output_width = 640;
int32_t g_output_height = 384;
std::vector<Point> g_goalnet_points = {
        {1167, 438},
        {1386, 426},
        {1168, 543},
        {1381, 521}
};