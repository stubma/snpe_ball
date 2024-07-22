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
#define LOG_TAG "extractor"

#include <iostream>

#include "Extractor.h"

int32_t Extractor::initExtractor(int32_t fd, size_t fileSize) {
    mExtractor = AMediaExtractor_new();
    if (!mExtractor) return AMEDIACODEC_ERROR_INSUFFICIENT_RESOURCE;
    media_status_t status = AMediaExtractor_setDataSourceFd(mExtractor, fd, 0, fileSize);
    if (status != AMEDIA_OK) return status;

    return AMediaExtractor_getTrackCount(mExtractor);
}

int32_t Extractor::getFrameSample(AMediaCodecBufferInfo &frameInfo, uint8_t* input, size_t bufSize) {
    int32_t size = AMediaExtractor_readSampleData(mExtractor, input, bufSize);
    if (size < 0) return -1;

    frameInfo.flags = AMediaExtractor_getSampleFlags(mExtractor);
    frameInfo.size = size;
    frameInfo.presentationTimeUs = AMediaExtractor_getSampleTime(mExtractor);
    AMediaExtractor_advance(mExtractor);

    return 0;
}

int32_t Extractor::setupTrackFormat(int32_t trackId) {
    AMediaExtractor_selectTrack(mExtractor, trackId);
    mFormat = AMediaExtractor_getTrackFormat(mExtractor, trackId);
    if (!mFormat) return AMEDIA_ERROR_INVALID_OBJECT;

    bool durationFound = AMediaFormat_getInt64(mFormat, AMEDIAFORMAT_KEY_DURATION, &mDurationUs);
    if (!durationFound) return AMEDIA_ERROR_INVALID_OBJECT;

    return AMEDIA_OK;
}

void Extractor::deInitExtractor() {
    if (mExtractor) {
        AMediaExtractor_delete(mExtractor);
        mExtractor = nullptr;
    }
}
