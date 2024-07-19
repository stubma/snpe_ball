#pragma once

#include "Decoder.h"
#include <mutex>
#include <thread>
#include <media/NdkMediaFormat.h>
#include <vector>

class TensorProducer {
public:
    TensorProducer(int32_t batchSize);
    virtual ~TensorProducer();

    void run();
    void stop();
    void onOutputAvailable(AMediaCodec *codec,
                           int32_t index,
                           AMediaCodecBufferInfo *bufferInfo);

private:
    Decoder* _decoder;
    FILE* _video_fp;
    uint8_t* _buffer;
    std::vector<AMediaCodecBufferInfo> _frame_infos;
    int32_t _batch_size;
    std::vector<std::vector<float>> _pending_batch;
    std::vector<std::vector<std::vector<float>>> _batch_queue;

    std::thread _t;
    std::mutex _mutex;

private:
    void loop();
};