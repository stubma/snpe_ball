#include <jni.h>
#include <stdio.h>
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

using namespace DlSystem;
using namespace DlContainer;

// java vm
static JavaVM *_vm;

// constant
//static std::string DIR = "/data/local/tmp/ball";
//static std::string CONTAINER_PATH = DIR + "/model3_1600_480_20230926_hp.dlc";
static std::string DIR = "/data/local/tmp/inception_v3";
static std::string CONTAINER_PATH = DIR + "/inception_v3.dlc";
static std::string INPUT_FILE_PATH = DIR + "/target_raw_list.txt";
static std::string OUTPUT_DIR = DIR + "/output";

Runtime_t checkRuntime() {
    Version_t Version = SNPE::SNPEFactory::getLibraryVersion();
    Runtime_t Runtime;
    ALOGD("Qualcomm (R) Neural Processing SDK Version: %s\n",
          Version.asString().c_str()); //Print Version number
    if (SNPE::SNPEFactory::isRuntimeAvailable(Runtime_t::DSP)) {
        Runtime = Runtime_t::DSP;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(Runtime_t::GPU)) {
        Runtime = Runtime_t::GPU;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(Runtime_t::GPU_FLOAT16)) {
        Runtime = Runtime_t::GPU;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(Runtime_t::CPU)) {
        Runtime = Runtime_t::CPU;
    } else {
        Runtime = Runtime_t::UNSET;
    }
    return Runtime;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_hexagon_1test_Hexagon_checkRuntime(JNIEnv *env, jobject thiz) {
    Runtime_t rt = checkRuntime();
    switch (rt) {
        case Runtime_t::GPU:
            return env->NewStringUTF("GPU");
        case Runtime_t::CPU:
            return env->NewStringUTF("CPU");
        case Runtime_t::DSP:
            return env->NewStringUTF("DSP");
        default:
            return env->NewStringUTF("Unsupported");
    }
}

int main(int argc, char *argv[]) {
    // print available runtime
    Runtime_t runtime = checkRuntime();
    switch (runtime) {
        case Runtime_t::GPU:
            printf("Available runtime: GPU\n");
            break;
        case Runtime_t::CPU:
            printf("Available runtime: CPU\n");
            break;
        case Runtime_t::DSP:
            printf("Available runtime: DSP\n");
            break;
        default:
            printf("Available runtime: Unsupported, can not proceed\n");
            return EXIT_FAILURE;
    }

    // load container
    std::unique_ptr<IDlContainer> container = loadContainerFromFile(CONTAINER_PATH);
    if (container == nullptr) {
        printf("failed to load container, can not proceed\n");
        return EXIT_FAILURE;
    } else {
        printf("container loaded: %p\n", container.get());
    }

    // set builder
    enum {
        UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16
    };
    enum {
        CPUBUFFER, GLBUFFER
    };
    int userBufferSourceType = CPUBUFFER;
    int bufferType = USERBUFFER_FLOAT;
    int bitWidth = 0;
    RuntimeList runtimeList;
    runtimeList.add(runtime);
    PlatformConfig platformConfig;
    bool usingInitCaching = false;
    bool staticQuantization = false;
    bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT ||
                                   bufferType == USERBUFFER_TF8 ||
                                   bufferType == USERBUFFER_TF16);
    if (bufferType == USERBUFFER_TF8) {
        bitWidth = 8;
    } else if (bufferType == USERBUFFER_TF16) {
        bitWidth = 16;
    }
    std::unique_ptr<SNPE::SNPE> snpe = setBuilderOptions(container, runtime, runtimeList,
                                                         useUserSuppliedBuffers, platformConfig,
                                                         usingInitCaching);

    // if caching enabled, save container
    if (usingInitCaching) {
        if (container->save(CONTAINER_PATH)) {
            printf("Saved container into archive successfully\n");
        } else {
            printf("Failed to save container into archive\n");
        }
    }

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];
    printf("Batch size for the container is %ld\n", batchSize);

    // Open the input file listing and group input files into batches
    std::vector<std::vector<std::string>> inputs = preprocessInput(INPUT_FILE_PATH, batchSize);

    // Load contents of input file batches ino a SNPE tensor or user buffer,
    // user buffer include cpu buffer and OpenGL buffer,
    // execute the network with the input and save each of the returned output to a file.
    if (useUserSuppliedBuffers) {
        // SNPE allows its input and output buffers that are fed to the network
        // to come from user-backed buffers. First, SNPE buffers are created from
        // user-backed storage. These SNPE buffers are then supplied to the network
        // and the results are stored in user-backed output buffers. This allows for
        // reusing the same buffers for multiple inputs and outputs.
        UserBufferMap inputMap, outputMap;
        std::vector<std::unique_ptr<IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
        std::unordered_map<std::string, std::vector<uint8_t>> applicationOutputBuffers;

        if (bufferType == USERBUFFER_TF8 || bufferType == USERBUFFER_TF16) {
            createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers,
                                  snpe, true, bitWidth);
            std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers;
            createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers,
                                 snpe, true, staticQuantization, bitWidth);

            for (size_t i = 0; i < inputs.size(); i++) {
                // Load input user buffer(s) with values from file(s)
                if (batchSize > 1)
                    std::cout << "Batch " << i << ":" << std::endl;
                if (!loadInputUserBufferTfN(applicationInputBuffers, snpe, inputs[i], inputMap,
                                            staticQuantization, bitWidth)) {
                    return EXIT_FAILURE;
                }

                // Execute the input buffer map on the model with SNPE
                bool execStatus = snpe->execute(inputMap, outputMap);

                // Save the execution results only if successful
                if (execStatus) {
                    if (!saveOutput(outputMap, applicationOutputBuffers, OUTPUT_DIR, i * batchSize,
                                    batchSize, true, bitWidth)) {
                        return EXIT_FAILURE;
                    }

                } else {
                    std::cerr << "Error while executing the network." << std::endl;
                }
            }
        } else if (bufferType == USERBUFFER_FLOAT) {
            createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers,
                                  snpe, false, bitWidth);

            if (userBufferSourceType == CPUBUFFER) {
                std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers;
                createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers,
                                     snpe, false, false, bitWidth);

                for (size_t i = 0; i < inputs.size(); i++) {
                    // Load input user buffer(s) with values from file(s)
                    if (!loadInputUserBufferFloat(applicationInputBuffers, snpe, inputs[i])) {
                        return EXIT_FAILURE;
                    }

                    // Execute the input buffer map on the model with SNPE
                    bool execStatus = snpe->execute(inputMap, outputMap);

                    // Save the execution results only if successful
                    if (execStatus) {
                        if (!saveOutput(outputMap, applicationOutputBuffers, OUTPUT_DIR,
                                        i * batchSize,
                                        batchSize, false, bitWidth)) {
                            return EXIT_FAILURE;
                        }
                    } else {
                        printf("Error while executing the network.\n");
                    }
                }
            }
        }
    } else if (bufferType == ITENSOR) {
        // A tensor map for SNPE execution outputs
        zdl::DlSystem::TensorMap outputTensorMap;
        //Get input names and number
        const auto &inputTensorNamesRef = snpe->getInputTensorNames();
        if (!inputTensorNamesRef) throw std::runtime_error("Error obtaining Input tensor names");
        const auto &inputTensorNames = *inputTensorNamesRef;

        bool execStatus = false;
        for (size_t i = 0; i < inputs.size(); i++) {
            // Load input/output buffers with ITensor
            if (batchSize > 1)
                std::cout << "Batch " << i << ":" << std::endl;
            if (inputTensorNames.size() == 1) {
                // Load input/output buffers with ITensor
                std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe,
                                                                                      inputs[i],
                                                                                      inputTensorNames);
                if (!inputTensor) {
                    return EXIT_FAILURE;
                }
                // Execute the input tensor on the model with SNPE
                execStatus = snpe->execute(inputTensor.get(), outputTensorMap);
            } else {
                std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> inputTensors(
                        inputTensorNames.size());
                zdl::DlSystem::TensorMap inputTensorMap;
                bool inputLoadStatus = false;
                // Load input/output buffers with TensorMap
                std::tie(inputTensorMap, inputLoadStatus) = loadMultipleInput(snpe, inputs[i],
                                                                              inputTensorNames,
                                                                              inputTensors);
                if (!inputLoadStatus) {
                    return EXIT_FAILURE;
                }
                // Execute the multiple input tensorMap on the model with SNPE
                execStatus = snpe->execute(inputTensorMap, outputTensorMap);
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
    }

    // Freeing of snpe object
    snpe.reset();

    // ok
    return EXIT_SUCCESS;
}