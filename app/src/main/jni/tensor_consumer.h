#pragma once

#include "SNPE.hpp"
#include "SNPEFactory.hpp"
#include <thread>
#include <mutex>
#include <deque>
#include <vector>
#include <condition_variable>
#include "utils.h"

class TensorConsumer {
public:
    TensorConsumer();
    virtual ~TensorConsumer();

    void run();
    void stop();
    void push(std::vector<std::vector<float>>& batch);

    inline size_t getBatchSize() { return _meta.batch_size; }
    inline SNPEMeta& getMeta() { return _meta; }

private:
    std::unique_ptr<SNPE::SNPE> _snpe;
    SNPEMeta _meta;
    std::deque<std::vector<std::vector<float>>> _batch_queue;

    std::thread _t;
    std::mutex _mutex;
    std::condition_variable _cond;
    bool _quit;

private:
    void loop();
};