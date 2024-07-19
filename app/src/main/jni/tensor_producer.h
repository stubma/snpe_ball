#pragma once

#include "Decoder.h"
#include <mutex>
#include <thread>
#include <media/NdkMediaFormat.h>
#include <vector>
#include <deque>

class TensorConsumer;

class TensorProducer {
public:
    TensorProducer(TensorConsumer* c);
    virtual ~TensorProducer();

    void run();
    void stop();
    void onOutputAvailable(AMediaCodec *codec,
                           int32_t index,
                           AMediaCodecBufferInfo *bufferInfo);

private:
    TensorConsumer* _consumer;
    Decoder* _decoder;
    FILE* _video_fp;
    uint8_t* _buffer;
    std::vector<AMediaCodecBufferInfo> _frame_infos;
    int32_t _batch_size;
    std::vector<std::vector<float>> _pending_batch;

    std::thread _t;

private:
    void loop();
};