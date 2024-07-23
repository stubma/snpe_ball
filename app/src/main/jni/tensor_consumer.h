#pragma once

#include "SNPE.hpp"
#include "SNPEFactory.hpp"
#include <thread>
#include <mutex>
#include <deque>
#include <vector>
#include <condition_variable>
#include "utils.h"
#include "global.h"
#include <opencv2/opencv.hpp>

class TensorConsumer {
public:
    TensorConsumer();
    virtual ~TensorConsumer();

    void run();
    void stop();
    void push(std::vector<std::vector<float>>& batch);
    std::vector<Point> detectNet(std::vector<float>& raw, cv::Mat& d2i);

    inline size_t getBatchSize() { return _meta_goal.batch_size; }
    inline SNPEMeta& getGoalMeta() { return _meta_goal; }
    inline SNPEMeta& getNetMeta() { return _meta_net; }

private:
    std::unique_ptr<SNPE::SNPE> _snpe_goal;
    std::unique_ptr<SNPE::SNPE> _snpe_net;
    SNPEMeta _meta_goal;
    SNPEMeta _meta_net;
    std::deque<std::vector<std::vector<float>>> _batch_queue;

    std::thread _t;
    std::mutex _mutex;
    std::condition_variable _cond;
    bool _quit;

private:
    void loop();
};