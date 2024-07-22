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

#ifndef __DECODER_H__
#define __DECODER_H__

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <string>

#include "Extractor.h"

class Decoder;

/**
 * Called when an input buffer becomes available.
 * The specified index is the index of the available input buffer.
 */
typedef void (*RewooDecoderOnInputAvailable)(
        AMediaCodec *codec,
        Decoder *decoder,
        int32_t index);

/**
 * Called when an output buffer becomes available.
 * The specified index is the index of the available output buffer.
 * The specified bufferInfo contains information regarding the available output buffer.
 */
typedef void (*RewooDecoderOnOutputAvailable)(
        AMediaCodec *codec,
        Decoder *decoder,
        int32_t index,
        AMediaCodecBufferInfo *bufferInfo);

/**
 * Called when the output format has changed.
 * The specified format contains the new output format.
 */
typedef void (*RewooDecoderOnFormatChanged)(
        AMediaCodec *codec,
        Decoder *decoder,
        AMediaFormat *format);

/**
 * Called when the MediaCodec encountered an error.
 * The specified actionCode indicates the possible actions that client can take,
 * and it can be checked by calling AMediaCodecActionCode_isRecoverable or
 * AMediaCodecActionCode_isTransient. If both AMediaCodecActionCode_isRecoverable()
 * and AMediaCodecActionCode_isTransient() return false, then the codec error is fatal
 * and the codec must be deleted.
 * The specified detail may contain more detailed messages about this error.
 */
typedef void (*RewooDecoderOnError)(
        AMediaCodec *codec,
        Decoder *decoder,
        media_status_t error,
        int32_t actionCode,
        const char *detail);

typedef struct {
    RewooDecoderOnInputAvailable onInputAvailable;
    RewooDecoderOnOutputAvailable onOutputAvailable;
    RewooDecoderOnFormatChanged onFormatChanged;
    RewooDecoderOnError onError;
} RewooDecoderCallback;

class Decoder {
public:
    Decoder()
            : mCodec(nullptr),
              mFormat(nullptr),
              mExtractor(nullptr),
              mNumInputFrame(0),
              mNumOutputFrame(0),
              mSawInputEOS(false),
              mSawOutputEOS(false),
              mErrorCode(AMEDIA_OK),
              _cb(nullptr),
              _cbUserData(nullptr) {
        mExtractor = new Extractor();
    }

    virtual ~Decoder() {
        if (mExtractor) delete mExtractor;
    }

    Extractor *getExtractor() { return mExtractor; }

    inline void setCallback(RewooDecoderCallback *cb, void *userData) {
        _cb = cb;
        _cbUserData = userData;
    }

    inline void *getCallbackUserData() { return _cbUserData; }

    inline int32_t getOuputFrameNum() { return mNumOutputFrame; }

    // Decoder related utilities
    void setupDecoder();

    void deInitCodec();

    void resetDecoder();

    AMediaFormat *getFormat();

    // Async callback APIs
    void onInputAvailable(AMediaCodec *codec, int32_t index);

    void onFormatChanged(AMediaCodec *codec, AMediaFormat *format);

    void onError(AMediaCodec *mediaCodec, media_status_t err);

    void onOutputAvailable(AMediaCodec *codec, int32_t index,
                           AMediaCodecBufferInfo *bufferInfo);

    // Process the frames and give decoded output
    int32_t decode(std::string &codecName);

private:
    // Utility to create and configure AMediaCodec
    AMediaCodec *createMediaCodec(AMediaFormat *format, const char *mime, std::string codecName);

private:
    AMediaCodec *mCodec;
    AMediaFormat *mFormat;

    Extractor *mExtractor;
    RewooDecoderCallback *_cb;
    void *_cbUserData;

    int32_t mNumInputFrame;
    int32_t mNumOutputFrame;

    bool mSawInputEOS;
    bool mSawOutputEOS;
    media_status_t mErrorCode;
    int32_t _tryAgainCount;
};

#endif  // __DECODER_H__
