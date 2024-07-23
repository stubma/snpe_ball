#include "tensor_consumer.h"
#include "IDlContainer.hpp"
#include "SNPEBuilder.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadContainer.hpp"
#include "LoadInputTensor.hpp"
#include "global.h"
#include "log.h"
#include "utils.h"

TensorConsumer::TensorConsumer() {
    _quit = false;

    // print available runtime
    DlSystem::Runtime_t runtime = checkRuntime();
    switch (runtime) {
        case DlSystem::Runtime_t::GPU:
            ALOGD("Available runtime: GPU");
            break;
        case DlSystem::Runtime_t::CPU:
            ALOGD("Available runtime: CPU");
            break;
        case DlSystem::Runtime_t::DSP:
            ALOGD("Available runtime: DSP");
            break;
        default:
            ALOGD("Available runtime: Unsupported, can not proceed");
            break;
    }

    // load container
    std::unique_ptr<DlContainer::IDlContainer> goalContainer = loadContainerFromFile(
            g_goal_dlc_path);
    if (goalContainer == nullptr) {
        ALOGD("failed to load goal container, can not proceed");
        return;
    } else {
        ALOGD("goal container loaded, optimizing for runtime...");
    }
    std::unique_ptr<DlContainer::IDlContainer> netContainer = loadContainerFromFile(g_net_dlc_path);
    if (netContainer == nullptr) {
        ALOGD("failed to load net container, can not proceed");
        return;
    } else {
        ALOGD("net container loaded, optimizing for runtime...");
    }

    // set builder
    DlSystem::RuntimeList runtimeList;
    runtimeList.add(runtime);
    DlSystem::PlatformConfig platformConfig;
    bool usingInitCaching = true;
    _snpe_goal = setBuilderOptions(goalContainer, runtime, runtimeList,
                                   false, platformConfig,
                                   usingInitCaching);
    _snpe_net = setBuilderOptions(netContainer, runtime, runtimeList,
                                  false, platformConfig,
                                  usingInitCaching);

    // if caching enabled, save container
    if (usingInitCaching) {
        if (goalContainer->save(g_goal_dlc_path)) {
            ALOGD("Saved goal container into archive successfully");
        }
        if (netContainer->save(g_net_dlc_path)) {
            ALOGD("Saved net container into archive successfully");
        }
    }

    // get model metadata
    ALOGD("dump goal model metadata start ======>");
    dumpModel(_snpe_goal, _meta_goal);
    ALOGD("dump goal model metadata end   <======");
    ALOGD("dump net model metadata start  ======>");
    dumpModel(_snpe_net, _meta_net);
    ALOGD("dump net model metadata end    <======");
}

TensorConsumer::~TensorConsumer() {
    stop();
    _snpe_goal.reset();
    _snpe_net.reset();
}

void TensorConsumer::run() {
    _t = std::thread(
            [this] {
                loop();
            }
    );
}

void TensorConsumer::stop() {
    _quit = true;
    _t.detach();
}

void TensorConsumer::push(std::vector<std::vector<float>> &batch) {
    std::unique_lock<std::mutex> lock(_mutex);
    std::vector<std::vector<float>> item;
    item.insert(item.begin(), batch.begin(), batch.end());
    _batch_queue.push_back(item);
    lock.unlock();
    _cond.notify_one();
}

std::vector<Point> TensorConsumer::detectNet(std::vector<float>& raw, cv::Mat& d2i) {
    std::vector<Point> ret;

    // get input tensor names
    const auto &ref_input_tensor = _snpe_net->getInputTensorNames();
    if (!ref_input_tensor) throw std::runtime_error("Error obtaining net model input tensor names");
    const auto &input_tensor_names = *ref_input_tensor;

    // build tensor
    std::vector<std::vector<float>> batch;
    batch.push_back(std::move(raw));
    std::unique_ptr<DlSystem::ITensor> tensor = loadInputTensor(_snpe_net, batch,
                                                                input_tensor_names);

    // run network
    DlSystem::TensorMap output_tensor_map;
    bool execStatus = _snpe_net->execute(tensor.get(), output_tensor_map);
    if (execStatus) {
        // get output tensor
        const auto &name = _meta_net.output_names.at(0);
        DlSystem::ITensor *out_tensor = output_tensor_map.getTensor(name);

        // output tensor is [batch,height,width,channel] shape, reshape it to
        // [channel, width*height]
        DlSystem::TensorShape& tensor_shape = _meta_net.output_shapes[0];
        const size_t *dims = tensor_shape.getDimensions();
        size_t channels = dims[3];
        size_t output_width = dims[2];
        size_t output_height = dims[1];
        cv::Mat heatmap(channels, output_width * output_height, CV_32F);
        int32_t r = 0, c = 0;
        for (auto it = out_tensor->begin(); it != out_tensor->end(); it++) {
            heatmap.at<float>(r, c) = *it;
            r++;
            if(r >= channels) {
                r = 0;
                c++;
            }
        }

//        FILE *fp = fopen("/data/local/tmp/ball_v3/heatmap.raw", "w+");
//        fwrite(heatmap.data, heatmap.total() * heatmap.elemSize(), 1, fp);
//        fflush(fp);
//        fclose(fp);

        // get max point indices
        cv::Mat pred_index;
        cv::reduceArgMax(heatmap, pred_index, 1);

        // get points
        float ratio = _meta_net.input_width / output_width;
        for(int i = 0; i < pred_index.total(); i++) {
            ALOGD("pred index value: %d", pred_index.at<int32_t>(i));
            Point p;
            p.x = (pred_index.at<int32_t>(i) % output_width) * ratio * d2i.at<float>(0, 0) + d2i.at<float>(0, 2);
            p.y = pred_index.at<int32_t>(i) / output_width * ratio * d2i.at<float>(1, 1) + d2i.at<float>(1, 2);
            ret.push_back(p);
            ALOGD("detected goal net points[%d]: %d, %d", i, p.x, p.y);
        }
    } else {
        ALOGD("net model running - failed to detect net");
    }

    // return
    return ret;
}

void TensorConsumer::loop() {
    // get input tensor names
    const auto &ref_input_tensor = _snpe_goal->getInputTensorNames();
    if (!ref_input_tensor) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &input_tensor_names = *ref_input_tensor;

    // for statistics
    std::chrono::milliseconds network_cost = std::chrono::milliseconds(0);
    size_t total_frame = 0;
    size_t batch_num = 0;

    // network loop
    while (true) {
        // wait batch
        std::unique_lock<std::mutex> lock(_mutex);
        while (_batch_queue.empty() && !_quit) {
            _cond.wait(lock);
        }
        if (_quit) break;

        // pop first batch
        std::vector<std::vector<float>> &batch = _batch_queue.front();
        lock.unlock();

        // empty batch means no more
        if (batch.empty()) break;

        // statistics
        total_frame += batch.size();
        batch_num++;

        // build tensor
        std::unique_ptr<DlSystem::ITensor> tensor = loadInputTensor(_snpe_goal, batch,
                                                                    input_tensor_names);

        // execute this batch
        DlSystem::TensorMap output_tensor_map;
        const auto start = std::chrono::high_resolution_clock::now();
        bool execStatus = _snpe_goal->execute(tensor.get(), output_tensor_map);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::milliseconds cost = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start);
        network_cost += cost;

        // check result
        if (execStatus) {
            ALOGD("goal model running - batch %lu, frames: %lu, cost: %lldms(average: %lldms)",
                  batch_num, total_frame, network_cost.count(),
                  (network_cost.count() / total_frame));

            // print output
            int size = _meta_goal.output_names.size();
            for (int i = 0; i < size; i++) {
                const auto &name = _meta_goal.output_names.at(i);
                DlSystem::ITensor *out_tensor = output_tensor_map.getTensor(name);

                // print tensor
                printf("result tensor(%s): ", name);
                for (auto it = out_tensor->begin(); it != out_tensor->end(); it++) {
                    printf("%f, ", *it);
                }
                printf("\n");
            }
        } else {
            ALOGD("goal model running - failed for batch %lu", batch_num);
        }

        // pop front
        lock.lock();
        _batch_queue.pop_front();
        lock.unlock();
    }

    // set flag
    g_dlc_done = true;
}