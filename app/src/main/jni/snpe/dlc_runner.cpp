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
#include "shared/log.h"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "PreprocessInput.hpp"
#include "CreateUserBuffer.hpp"
#include "LoadInputTensor.hpp"
#include "SaveOutputTensor.hpp"
#include "Util.hpp"
#include "dlc_runner.h"

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

static void dumpModel(std::unique_ptr<SNPE::SNPE> &snpe, size_t *batchSize) {
    DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    const size_t *dims = tensorShape.getDimensions();
    printf("model input dimensions: ");
    for (int i = 0; i < tensorShape.rank(); i++) {
        if (i == 0) {
            *batchSize = dims[i];
        }
        printf("%d ", dims[i]);
    }
    printf("\n");
    printf("Batch size for the container is %ld\n", *batchSize);

    // dump input tensors
    const auto &inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const DlSystem::StringList &inputNames = *inputNamesOpt;
    for (const char *name: inputNames) {
        auto attrs = snpe->getInputOutputBufferAttributes(name);
        if (!attrs)
            throw std::runtime_error(
                    std::string("Error obtaining attributes for input tensor ") + name);

        printf("model input tensor(%s) dimensions: ", name);
        const DlSystem::TensorShape &bufferShape = (*attrs)->getDims();
        const size_t *dims = bufferShape.getDimensions();
        for (int i = 0; i < bufferShape.rank(); i++) {
            if (i == 0) {
                *batchSize = dims[i];
            }
            printf("%d ", dims[i]);
        }
        printf(", element size: %lu", (*attrs)->getElementSize());
        printf(", element type: %s\n", elementTypeStr((*attrs)->getEncodingType()).c_str());
    }

    // dump output tensors
    const auto &outputNamesOpt = snpe->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    const DlSystem::StringList &outputNames = *outputNamesOpt;
    for (const char *name: outputNames) {
        auto attrs = snpe->getInputOutputBufferAttributes(name);
        if (!attrs)
            throw std::runtime_error(
                    std::string("Error obtaining attributes for output tensor ") + name);

        printf("model output tensor(%s) dimensions: ", name);
        const DlSystem::TensorShape &bufferShape = (*attrs)->getDims();
        const size_t *dims = bufferShape.getDimensions();
        for (int i = 0; i < bufferShape.rank(); i++) {
            if (i == 0) {
                *batchSize = dims[i];
            }
            printf("%d ", dims[i]);
        }
        printf(", element size: %lu", (*attrs)->getElementSize());
        printf(", element type: %s\n", elementTypeStr((*attrs)->getEncodingType()).c_str());
    }
}

int run_dlc(InputProvider *provider) {
    // print available runtime
    DlSystem::Runtime_t runtime = checkRuntime();
    switch (runtime) {
        case DlSystem::Runtime_t::GPU:
            printf("Available runtime: GPU\n");
            break;
        case DlSystem::Runtime_t::CPU:
            printf("Available runtime: CPU\n");
            break;
        case DlSystem::Runtime_t::DSP:
            printf("Available runtime: DSP\n");
            break;
        default:
            printf("Available runtime: Unsupported, can not proceed\n");
            return EXIT_FAILURE;
    }

    // load container
    std::unique_ptr<DlContainer::IDlContainer> container = loadContainerFromFile(CONTAINER_PATH);
    if (container == nullptr) {
        printf("failed to load container, can not proceed\n");
        return EXIT_FAILURE;
    } else {
        printf("container loaded: %p\n", container.get());
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
            printf("Saved container into archive successfully\n");
        }
    }

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    size_t batchSize = 1;
    dumpModel(snpe, &batchSize);
    provider->setupProvider(batchSize);

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
            frames += batchSize;
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
            frames += batchSize;
        }
        // Save the execution results if execution successful
        if (execStatus) {
            if (!saveOutput(outputTensorMap, OUTPUT_DIR, i * batchSize, batchSize)) {
                return EXIT_FAILURE;
            }
        } else {
            std::cerr << "Error while executing the network." << std::endl;
        }
    }

    // dump profile
    size_t avgFrameMs = networkCost.count() / frames;
    printf("total cost: %llu ms, frames: %lu, average frame nn cost: %lu ms\n", networkCost.count(),
           frames, avgFrameMs);

    // Freeing of snpe object
    snpe.reset();

    // release provider
    delete provider;

    // ok
    return EXIT_SUCCESS;
}