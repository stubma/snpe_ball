#pragma once

#include "Decoder.h"

int decode_video(
        std::string file_path,
        std::string file_name,
        std::string stat_path,
        std::string codec,
        bool async,
        RewooDecoderCallback* cb = nullptr,
        void* cbUserData = nullptr);