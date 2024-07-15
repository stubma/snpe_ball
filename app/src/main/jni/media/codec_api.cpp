/*
 * Copyright (C) 2019 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//#define LOG_NDEBUG 0
#define LOG_TAG "NativeDecoder"

#include <jni.h>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "codec_api.h"
#include <android/log.h>

#include "Decoder.h"
#include <media/NdkMediaFormat.h>
#include "Timers.h"

int decode_video(
        std::string file_path,
        std::string file_name,
        std::string stat_path,
        std::string codec,
        bool async,
        RewooDecoderCallback* cb,
        void* cbUserData) {
    string full_path = file_path + file_name;
    FILE *inputFp = fopen(full_path.c_str(), "rb");
    if (!inputFp) {
        ALOGE("Unable to open input file for reading");
        return -1;
    }
    ALOGD("start native decode file: %s", full_path.c_str());

    Decoder *decoder = new Decoder();
    Extractor *extractor = decoder->getExtractor();
    if (!extractor) {
        ALOGE("Extractor creation failed");
        return -1;
    }

    // Read file properties
    struct stat buf;
    stat(full_path.c_str(), &buf);
    size_t fileSize = buf.st_size;
    if (fileSize > kMaxBufferSize) {
        ALOGE("File size greater than maximum buffer size");
        return -1;
    }
    int32_t fd = fileno(inputFp);
    int32_t trackCount = extractor->initExtractor(fd, fileSize);
    if (trackCount <= 0) {
        ALOGE("initExtractor failed");
        return -1;
    }
    for (int curTrack = 0; curTrack < trackCount; curTrack++) {
        int32_t status = extractor->setupTrackFormat(curTrack);
        if (status != 0) {
            ALOGE("Track Format invalid");
            return -1;
        }
        AMediaFormat* format = extractor->getFormat();
        const char* mimeType = nullptr;
        AMediaFormat_getString(format, AMEDIAFORMAT_KEY_MIME, &mimeType);
        if(!strncmp(mimeType, "audio/", 6)) {
            ALOGD("can not decode audio now, skip audio track");
            continue;
        }

        uint8_t *inputBuffer = (uint8_t *) malloc(fileSize);
        if (!inputBuffer) {
            ALOGE("Insufficient memory");
            return -1;
        }

        vector<AMediaCodecBufferInfo> frameInfo;
        AMediaCodecBufferInfo info;
        uint32_t inputBufferOffset = 0;

        // Get frame data
        while (1) {
            status = extractor->getFrameSample(info);
            if (status || !info.size) break;
            // copy the meta data and buffer to be passed to decoder
            if (inputBufferOffset + info.size > kMaxBufferSize) {
                ALOGE("Memory allocated not sufficient");
                free(inputBuffer);
                return -1;
            }
            memcpy(inputBuffer + inputBufferOffset, extractor->getFrameBuf(), info.size);
            frameInfo.push_back(info);
            inputBufferOffset += info.size;
        }
        nsecs_t start = systemTime();

        decoder->setupDecoder();
        decoder->setCallback(cb, cbUserData);
        ALOGD("native decoder setup: codec: %s, input buffer size: %u, frame count: %zu",
                codec.c_str(), inputBufferOffset, frameInfo.size());
        status = decoder->decode(inputBuffer, frameInfo, codec, async);
        if (status != AMEDIA_OK) {
            ALOGE("Decode returned error: %d", status);
            free(inputBuffer);
            return -1;
        }

        nsecs_t end = systemTime();
        nsecs_t decodeTime = end - start;
        ALOGD("frame count: %zu, decode cost: %ldms(average: %ldms)",
              frameInfo.size(),
              nanoseconds_to_milliseconds(decodeTime),
              nanoseconds_to_milliseconds(decodeTime / frameInfo.size()));

        decoder->deInitCodec();
        decoder->dumpStatistics(file_name, codec, (async ? "async" : "sync"),
                                stat_path);
        if (inputBuffer) {
            free(inputBuffer);
            inputBuffer = nullptr;
        }
        decoder->resetDecoder();
    }
    if (inputFp) {
        fclose(inputFp);
        inputFp = nullptr;
    }
    extractor->deInitExtractor();
    delete decoder;
    ALOGD("native decode done");
    return 0;
}
