#include <string>
#include "global.h"

std::string g_cwd = ".";
const char* DSP_ENV_VAR = "ADSP_LIBRARY_PATH";
std::string g_output_dir = "decode_output";
std::string g_dsp_lib_dir = "lib";
RewooDecodeOutputFileType g_output_file_type = REWOO_OUTPUT_NONE;
int32_t g_output_from_frame = 1;
int32_t g_output_to_frame = -1;
std::string g_video_path;
std::string g_video_codec = "c2.qti.avc.decoder";
int32_t g_video_width = 7600;
int32_t g_video_height = 2160;
std::vector<Point> g_goalnet_points = {
        {1167, 438},
        {1386, 426},
        {1168, 543},
        {1381, 521}
};
std::string g_dlc_dir;
std::string g_dlc_path;
bool g_decode_done = false;
bool g_dlc_done = false;

std::string get_output_type_str(RewooDecodeOutputFileType t) {
    switch(t) {
        case REWOO_OUTPUT_YUV:
            return "yuv";
        case REWOO_OUTPUT_JPG:
            return "jpg";
        case REWOO_OUTPUT_RAW_RGB:
            return "raw rgb";
        default:
            return "none";
    }
}