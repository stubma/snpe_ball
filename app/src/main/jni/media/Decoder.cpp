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
#define LOG_TAG "decoder"

#include <iostream>

#include "Decoder.h"
#include "stdio.h"

void Decoder::onInputAvailable(AMediaCodec *mediaCodec, int32_t bufIdx) {
    if (mSawInputEOS || bufIdx < 0) return;

    size_t bufSize;
    uint8_t *buf = AMediaCodec_getInputBuffer(mCodec, bufIdx, &bufSize);
    if (!buf) {
        ALOGE("failed to get input buffer");
        mErrorCode = AMEDIA_ERROR_IO;
        return;
    }

    AMediaCodecBufferInfo frameInfo;
    mExtractor->getFrameSample(frameInfo, buf, bufSize);
    if (frameInfo.flags == AMEDIA_ERROR_MALFORMED) {
        ALOGE("failed to read frame sample, malformed");
        mErrorCode = (media_status_t) frameInfo.flags;
        return;
    }

    if ((frameInfo.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) != 0) mSawInputEOS = true;
//    ALOGV("%s bytesRead : %d presentationTimeUs : %" PRId64 " mSawInputEOS : %s", __FUNCTION__,
//          frameInfo.size, frameInfo.presentationTimeUs, mSawInputEOS ? "TRUE" : "FALSE");

    media_status_t status = AMediaCodec_queueInputBuffer(mCodec, bufIdx, 0 /* offset */,
                                                         frameInfo.size,
                                                         frameInfo.presentationTimeUs,
                                                         mSawInputEOS
                                                         ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM
                                                         : 0);
    if (AMEDIA_OK != status) {
        ALOGE("failed to queue input buffer: %d", status);
        mErrorCode = status;
        return;
    }
    mStats->addFrameSize(frameInfo.size);
    mNumInputFrame++;

    // forward
    if (_cb && _cb->onInputAvailable) {
        _cb->onInputAvailable(mediaCodec, this, bufIdx);
    }
}

void Decoder::onOutputAvailable(AMediaCodec *mediaCodec, int32_t bufIdx,
                                AMediaCodecBufferInfo *bufferInfo) {
    if (mSawOutputEOS || bufIdx < 0) return;

    mNumOutputFrame++;
//    ALOGD("%s index : %d  mSawOutputEOS : %s count : %u", __FUNCTION__, bufIdx,
//          mSawOutputEOS ? "TRUE" : "FALSE", mNumOutputFrame);

    // forward
    if (_cb && _cb->onOutputAvailable) {
        _cb->onOutputAvailable(mediaCodec, this, bufIdx, bufferInfo);
    }

    AMediaCodec_releaseOutputBuffer(mCodec, bufIdx, false);
}

void Decoder::onFormatChanged(AMediaCodec *mediaCodec, AMediaFormat *format) {
    ALOGV("%s { %s }", __FUNCTION__, AMediaFormat_toString(format));
    mFormat = format;

    // forward
    if (_cb && _cb->onFormatChanged) {
        _cb->onFormatChanged(mediaCodec, this, format);
    }
}

void Decoder::onError(AMediaCodec *mediaCodec, media_status_t err) {
    ALOGE("Received Error %d", err);
    mErrorCode = err;

    // forward
    if (_cb && _cb->onError) {
        _cb->onError(mediaCodec, this, err, 0, nullptr);
    }
}

void Decoder::setupDecoder() {
    if (!mFormat) mFormat = mExtractor->getFormat();
}

AMediaFormat *Decoder::getFormat() {
    return AMediaCodec_getOutputFormat(mCodec);
}

int32_t Decoder::decode(string &codecName) {
    mNumOutputFrame = 0;
    _tryAgainCount = 0;

    const char *mime = nullptr;
    AMediaFormat_getString(mFormat, AMEDIAFORMAT_KEY_MIME, &mime);
    if (!mime) return AMEDIA_ERROR_INVALID_OBJECT;

    int64_t sTime = mStats->getCurTime();
    mCodec = createMediaCodec(mFormat, mime, codecName, false /*isEncoder*/);
    if (!mCodec) return AMEDIA_ERROR_INVALID_OBJECT;

    media_status_t status = AMediaCodec_start(mCodec);
    if (status) {
        ALOGE("Error when start mediacodec decoder, return %d", status);
        return AMEDIA_ERROR_IO;
    }
    status = AMediaCodec_flush(mCodec);
    if (status != AMEDIA_OK) {
        ALOGE("Error when flush codec. return %d.", status);
        return AMEDIA_ERROR_IO;
    }

    int64_t eTime = mStats->getCurTime();
    int64_t timeTaken = mStats->getTimeDiff(sTime, eTime);
    mStats->setInitTime(timeTaken);

    mStats->setStartTime();
    while (true) {
        /* Queue input data */
        if (!mSawInputEOS) {
            ssize_t inIdx = AMediaCodec_dequeueInputBuffer(mCodec, kQueueDequeueTimeoutUs);
            if (inIdx < 0 && inIdx != AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
                ALOGE("AMediaCodec_dequeueInputBuffer returned invalid index %zd\n", inIdx);
                mErrorCode = (media_status_t) inIdx;
                onError(mCodec, mErrorCode);
                return mErrorCode;
            } else if (inIdx >= 0) {
                mStats->addInputTime();
                onInputAvailable(mCodec, inIdx);
            }
        }

        /* Dequeue output data */
        if (!mSawOutputEOS) {
            AMediaCodecBufferInfo info;
            ssize_t outIdx = AMediaCodec_dequeueOutputBuffer(mCodec, &info, kQueueDequeueTimeoutUs);
            if (outIdx >= 0) {
                if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                    mSawOutputEOS = true;
                    break;
                }
                mStats->addOutputTime();
                onOutputAvailable(mCodec, outIdx, &info);
            } else if (outIdx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
                mFormat = AMediaCodec_getOutputFormat(mCodec);
                _tryAgainCount = 0;
                ALOGD("Output format changed: %s", AMediaFormat_toString(mFormat));
            } else if (outIdx == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
                ALOGD("Output buffers changed.");
            } else if (outIdx == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
                ALOGD("Try again later. tryagaincnt = %d.", _tryAgainCount);
                _tryAgainCount++;
                if (_tryAgainCount > 20) {
                    ALOGE("Try again 20 times continously. consider it ended.");
                    break;
                }
            } else {
                ALOGD("dequeue output buffer got unexpected info code %zd", outIdx);
            }
        }
    }
    return AMEDIA_OK;
}

void Decoder::deInitCodec() {
    if (mFormat) {
        AMediaFormat_delete(mFormat);
        mFormat = nullptr;
    }
    if (!mCodec) return;
    int64_t sTime = mStats->getCurTime();
    AMediaCodec_stop(mCodec);
    AMediaCodec_delete(mCodec);
    int64_t eTime = mStats->getCurTime();
    int64_t timeTaken = mStats->getTimeDiff(sTime, eTime);
    mStats->setDeInitTime(timeTaken);
}

void Decoder::dumpStatistics(string inputReference, string componentName, string mode,
                             string statsFile) {
    int64_t durationUs = mExtractor->getClipDuration();
    string operation = "decode";
    mStats->dumpStatistics(operation, inputReference, durationUs, componentName, mode, statsFile);
}

void Decoder::resetDecoder() {
    if (mStats) mStats->reset();
}
