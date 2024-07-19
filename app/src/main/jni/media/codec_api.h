#pragma once

#include "Decoder.h"

int decode_video(
        std::string video_path,
        std::string stat_path,
        std::string codec,
        bool async,
        RewooDecoderCallback* cb = nullptr,
        void* cbUserData = nullptr);