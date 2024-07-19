#include <jni.h>
#include "DlEnums.hpp"
#include "DlVersion.hpp"
#include "SNPE.hpp"
#include "SNPEFactory.hpp"
#include "SNPEBuilder.hpp"
#include <iostream>
#include <string>
#include <memory>
#include "IDlContainer.hpp"
#include "log.h"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "PreprocessInput.hpp"
#include "CreateUserBuffer.hpp"
#include "LoadInputTensor.hpp"
#include "SaveOutputTensor.hpp"
#include "Util.hpp"
#include "dlc_runner.h"
#include "utils.h"

// constant
std::string DIR = "/data/local/tmp/ball_v2";
static std::string CONTAINER_PATH = DIR + "/ballspotting_woGSM_part1.dlc";
static std::string INPUT_FILE_PATH = DIR + "/target_raw_list.txt";
static std::string OUTPUT_DIR = DIR + "/output";

extern DlSystem::Runtime_t checkRuntime();

extern std::string getRuntimeStr();

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_hexagon_1test_Hexagon_checkRuntime(JNIEnv *env, jobject thiz) {
    return env->NewStringUTF(getRuntimeStr().c_str());
}

int run_dlc(InputProvider *provider) {
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
            return EXIT_FAILURE;
    }

    // load container
    std::unique_ptr<DlContainer::IDlContainer> container = loadContainerFromFile(CONTAINER_PATH);
    if (container == nullptr) {
        ALOGD("failed to load container, can not proceed");
        return EXIT_FAILURE;
    } else {
        ALOGD("container loaded: %p", container.get());
    }

    // set builder
    DlSystem::RuntimeList runtimeList;
    runtimeList.add(runtime);
    DlSystem::PlatformConfig platformConfig;
    bool usingInitCaching = true;
    std::unique_ptr<SNPE::SNPE> snpe = setBuilderOptions(container, runtime, runtimeList,
                                                         false, platformConfig,
                                                         usingInitCaching);

    // if caching enabled, save container
    if (usingInitCaching) {
        if (container->save(CONTAINER_PATH)) {
            ALOGD("Saved container into archive successfully");
        }
    }

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    SNPEMeta meta;
    dumpModel(snpe, meta);
    provider->setupProvider(meta.batch_size);

    // profile
    std::chrono::milliseconds networkCost = std::chrono::milliseconds(0);
    size_t frames = 0;

    // A tensor map for SNPE execution outputs
    DlSystem::TensorMap outputTensorMap;
    //Get input names and number
    const auto &inputTensorNamesRef = snpe->getInputTensorNames();
    if (!inputTensorNamesRef) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &inputTensorNames = *inputTensorNamesRef;

    bool execStatus = false;
    size_t tensorCount = provider->getTensorCount();
    for (size_t i = 0; i < tensorCount; i++) {
        // Load input/output buffers with ITensor
        if (inputTensorNames.size() == 1) {
            // Load input/output buffers with ITensor
            std::unique_ptr<DlSystem::ITensor> inputTensor = provider->getTensorAt(snpe, i);
            if (!inputTensor) {
                return EXIT_FAILURE;
            }

            // Execute the input tensor on the model with SNPE
            const auto start = std::chrono::high_resolution_clock::now();
            execStatus = snpe->execute(inputTensor.get(), outputTensorMap);
            const auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::milliseconds int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end - start);
            networkCost += int_ms;
            frames += meta.batch_size;
        } else {
            // Load input/output buffers with TensorMap
            DlSystem::TensorMap inputTensorMap = provider->getTensorMap(snpe, i);

            // Execute the multiple input tensorMap on the model with SNPE
            const auto start = std::chrono::high_resolution_clock::now();
            execStatus = snpe->execute(inputTensorMap, outputTensorMap);
            const auto end = std::chrono::high_resolution_clock::now();
            const std::chrono::milliseconds int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end - start);
            networkCost += int_ms;
            frames += meta.batch_size;
        }
        // Save the execution results if execution successful
        if (execStatus) {
            if (!saveOutput(outputTensorMap, OUTPUT_DIR, i * meta.batch_size, meta.batch_size)) {
                return EXIT_FAILURE;
            }
        } else {
            ALOGE("Error while executing the network.");
        }
    }

    // dump profile
    size_t avgFrameMs = networkCost.count() / frames;
    ALOGD("total cost: %llu ms, frames: %lu, average frame nn cost: %lu ms", networkCost.count(),
           frames, avgFrameMs);

    // Freeing of snpe object
    snpe.reset();

    // release provider
    delete provider;

    // ok
    return EXIT_SUCCESS;
}