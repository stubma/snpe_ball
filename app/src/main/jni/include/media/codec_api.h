#pragma once

#include <string>

int decode_video(
        std::string file_path,
        std::string file_name,
        std::string stat_path,
        std::string codec,
        bool async,
        std::string out_dir);