#include "tensor_producer.h"
#include "global.h"
#include "log.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "tensor_consumer.h"

TensorProducer::TensorProducer(TensorConsumer* c) {
    // init
    _decoder = nullptr;
    _video_track_idx = -1;
    _consumer = c;
    _batch_size = c->getBatchSize();

    // open video file
    _video_fp = fopen(g_video_path.c_str(), "rb");
    if (!_video_fp) {
        ALOGE("Unable to open video file: %s", g_video_path.c_str());
        return;
    }

    // create decoder
    _decoder = new Decoder();
    Extractor *extractor = _decoder->getExtractor();
    if (!extractor) {
        ALOGE("Extractor creation failed");
        return;
    }

    // get track count
    struct stat buf;
    stat(g_video_path.c_str(), &buf);
    size_t fileSize = buf.st_size;
    int32_t fd = fileno(_video_fp);
    int32_t trackCount = extractor->initExtractor(fd, fileSize);
    if (trackCount <= 0) {
        ALOGE("initExtractor failed");
        return;
    }

    // load frames into buffer
    for (int curTrack = 0; curTrack < trackCount; curTrack++) {
        // get track format
        int32_t status = extractor->setupTrackFormat(curTrack);
        if (status != 0) {
            ALOGE("Track Format invalid");
            return;
        }

        // find video track
        AMediaFormat *format = extractor->getFormat();
        const char *mimeType = nullptr;
        AMediaFormat_getString(format, AMEDIAFORMAT_KEY_MIME, &mimeType);
        if (!strncmp(mimeType, "video/", 6)) {
            _video_track_idx = curTrack;
            break;
        }
    }

    // if not found video
    if(_video_track_idx == -1) {
        ALOGD("no video track found! can not decode");
        if (_decoder) {
            _decoder->getExtractor()->deInitExtractor();
            delete _decoder;
            _decoder = nullptr;
        }
    }
}

TensorProducer::~TensorProducer() {
    stop();

    // close video file
    if (_video_fp) {
        fclose(_video_fp);
        _video_fp = nullptr;
    }

    // release
    if (_decoder) {
        _decoder->getExtractor()->deInitExtractor();
        delete _decoder;
        _decoder = nullptr;
    }
}

void TensorProducer::run() {
    _t = std::thread(
            [this] {
                loop();
            }
    );
}

void TensorProducer::stop() {
    _t.detach();
}

static void onOutputAvailable(
        AMediaCodec *codec,
        Decoder *decoder,
        int32_t index,
        AMediaCodecBufferInfo *bufferInfo) {
    TensorProducer *thiz = (TensorProducer *) decoder->getCallbackUserData();
    thiz->onOutputAvailable(codec, index, bufferInfo);
}

void TensorProducer::onOutputAvailable(AMediaCodec *codec,
                                       int32_t index,
                                       AMediaCodecBufferInfo *bufferInfo) {
    size_t bufSize;
    uint8_t *buf = AMediaCodec_getOutputBuffer(codec, index, &bufSize);
    if (buf && bufferInfo->size > 0) {
        // convert yuv to rgb
        cv::Mat matSrc = cv::Mat(g_video_height * 1.5, g_video_width, CV_8UC1, buf);
        cv::Mat matDst = cv::Mat(g_video_height, g_video_width, CV_8UC3);
        cv::cvtColor(matSrc, matDst, cv::COLOR_YUV2RGB_NV21);

        // crop by goal net position
        SNPEMeta& meta = _consumer->getMeta();
        int32_t cx1 = (g_goalnet_points[2].x + g_goalnet_points[3].x) / 2;
        int32_t cy1 = (g_goalnet_points[2].y + g_goalnet_points[3].y) / 2;
        int32_t lx = std::min(g_video_width - meta.input_width,
                              std::max(0, cx1 - meta.input_width / 2));
        int32_t ly = std::min(g_video_height - meta.input_height,
                              std::max(0, cy1 - meta.input_height / 2));
        cv::Rect roi(lx, ly, meta.input_width, meta.input_height);
        cv::Mat crop = matDst(roi);

        // normalization: convert rgb int to float
        cv::Mat floatCrop;
        crop.convertTo(floatCrop, CV_32F, 1 / 255.0);

        // write interleaved data in planar format
        std::vector<float> raw(meta.input_width * meta.input_height * meta.channels);
        memcpy_ex(raw.data(), floatCrop.data, sizeof(float32_t), floatCrop.total(),
                  0, floatCrop.elemSize());
        memcpy_ex(raw.data() + floatCrop.total(), floatCrop.data, sizeof(float32_t), floatCrop.total(),
                  sizeof(float32_t), floatCrop.elemSize());
        memcpy_ex(raw.data() + floatCrop.total() * 2, floatCrop.data, sizeof(float32_t), floatCrop.total(),
                  sizeof(float32_t) * 2, floatCrop.elemSize());

        // dump frame
        char path[512] = {0};
        int frameNum = _decoder->getOuputFrameNum();
        if(g_output_file_type != REWOO_OUTPUT_NONE &&
            frameNum >= g_output_from_frame &&
            (g_output_to_frame == -1 || frameNum <= g_output_to_frame)) {
            switch (g_output_file_type) {
                case REWOO_OUTPUT_YUV: {
                    sprintf(path, "%s/frame_%d.yuv", g_output_dir.c_str(), frameNum);
                    FILE *fp = fopen(path, "w+");
                    fwrite(buf, sizeof(char), bufferInfo->size, fp);
                    fflush(fp);
                    fclose(fp);
                    ALOGV("bytes(%d) written into file %s", bufferInfo->size, path);
                    break;
                }
                case REWOO_OUTPUT_JPG: {
                    sprintf(path, "%s/frame_%d.jpg", g_output_dir.c_str(), frameNum);
                    cv::Mat matSrc = cv::Mat(g_video_height * 1.5, g_video_width, CV_8UC1, buf);
                    cv::Mat matDst = cv::Mat(g_video_height, g_video_width, CV_8UC3);
                    cv::cvtColor(matSrc, matDst, cv::COLOR_YUV2RGB_NV21);
                    cv::imwrite(path, matDst);
                    ALOGV("JPG written into file %s", path);
                    break;
                }
                case REWOO_OUTPUT_RAW_RGB: {
                    sprintf(path, "%s/frame_%d.raw", g_output_dir.c_str(), frameNum);
                    FILE *fp = fopen(path, "w+");
                    fwrite(raw.data(), floatCrop.total() * floatCrop.elemSize(), 1, fp);
                    fflush(fp);
                    fclose(fp);
                    ALOGV("bytes(%lu) written into file %s", floatCrop.total() * floatCrop.elemSize(),
                          path);
                    break;
                }
                default:
                    break;
            }
        }

        // put to queue
        if(_pending_batch.size() >= _batch_size) {
            _consumer->push(_pending_batch);
            _pending_batch = std::vector<std::vector<float>>();
            _pending_batch.push_back(std::move(raw));
        } else {
            _pending_batch.push_back(std::move(raw));
        }
    }
}

void TensorProducer::loop() {
    if(_video_track_idx >= 0) {
        // setup decoder
        RewooDecoderCallback cb{
                nullptr,
                ::onOutputAvailable,
                nullptr,
                nullptr
        };
        _decoder->setupDecoder();
        _decoder->setCallback(&cb, this);

        // decode loop
        _decoder->decode(g_video_codec);

        // last batch
        if(!_pending_batch.empty()) {
            _consumer->push(_pending_batch);
        }

        // empty batch means no more
        _pending_batch = std::vector<std::vector<float>>();
        _consumer->push(_pending_batch);
    }

    // set flag
    g_decode_done = true;
}