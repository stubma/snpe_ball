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
    std::unique_ptr<DlContainer::IDlContainer> container = loadContainerFromFile(g_dlc_path);
    if (container == nullptr) {
        ALOGD("failed to load container, can not proceed");
        return;
    } else {
        ALOGD("container loaded: %p", container.get());
    }

    // set builder
    DlSystem::RuntimeList runtimeList;
    runtimeList.add(runtime);
    DlSystem::PlatformConfig platformConfig;
    bool usingInitCaching = true;
    _snpe = setBuilderOptions(container, runtime, runtimeList,
                             false, platformConfig,
                             usingInitCaching);

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    dumpModel(_snpe, _meta);
}

TensorConsumer::~TensorConsumer() {
    stop();
    _snpe.reset();
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

void TensorConsumer::push(std::vector<std::vector<float>>& batch) {
    std::unique_lock<std::mutex> lock(_mutex);
    std::vector<std::vector<float>> item;
    item.insert(item.begin(), batch.begin(), batch.end());
    _batch_queue.push_back(item);
    lock.unlock();
    _cond.notify_one();
}

void TensorConsumer::loop() {
    // get input tensor names
    const auto &ref_input_tensor = _snpe->getInputTensorNames();
    if (!ref_input_tensor) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &input_tensor_names = *ref_input_tensor;

    // for statistics
    std::chrono::milliseconds total_cost = std::chrono::milliseconds(0);
    size_t total_frame = 0;
    size_t batch_num = 0;

    // network loop
    while(true) {
        // wait batch
        std::unique_lock<std::mutex> lock(_mutex);
        while(_batch_queue.empty() && !_quit) {
            _cond.wait(lock);
        }
        if(_quit) break;

        // pop first batch
        std::vector<std::vector<float>>& batch = _batch_queue.front();
        lock.unlock();

        // empty batch means no more
        if(batch.empty()) break;

        // statistics
        total_frame += batch.size();
        batch_num++;

        // build tensor
        std::unique_ptr<DlSystem::ITensor> tensor = loadInputTensor(_snpe, batch, input_tensor_names);

        // execute this batch
        DlSystem::TensorMap output_tensor_map;
        const auto start = std::chrono::high_resolution_clock::now();
        bool execStatus = _snpe->execute(tensor.get(), output_tensor_map);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::milliseconds cost = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start);
        total_cost += cost;

        // check result
        if(execStatus) {
            ALOGD("model running - batch %lu, frames: %lu, cost: %lldms(average: %lldms)",
                  batch_num, total_frame, total_cost.count(), (total_cost.count() / total_frame));

            // print output
            int size = _meta.output_names.size();
            for(int i = 0; i < size; i++) {
                const auto& name = _meta.output_names.at(i);
                DlSystem::ITensor* out_tensor = output_tensor_map.getTensor(name);

                // print tensor
                printf("result tensor(%s): ", name);
                for(auto it = out_tensor->begin(); it != out_tensor->end(); it++) {
                    printf("%f, ", *it);
                }
                printf("\n");
            }
        } else {
            ALOGD("model running - failed for batch %lu", batch_num);
        }

        // pop front
        lock.lock();
        _batch_queue.pop_front();
        lock.unlock();
    }

    // set flag
    g_dlc_done = true;
}