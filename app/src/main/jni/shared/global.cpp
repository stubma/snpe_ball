#include <string>
#include "global.h"

const char* DSP_ENV_VAR = "ADSP_LIBRARY_PATH";
const char* DEFAULT_DSP_LIB_DIR = "/vendor/lib";
std::string g_output_dir = "/data/local/tmp/decode_output";
std::string g_dsp_lib_dir = DEFAULT_DSP_LIB_DIR;
RewooDecodeOutputFileType g_output_file_type = REWOO_OUTPUT_NONE;
int32_t g_video_width = 7600;
int32_t g_video_height = 2160;