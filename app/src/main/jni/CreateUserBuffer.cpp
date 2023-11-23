#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <stdexcept>
#include <unordered_map>

#include "CreateUserBuffer.hpp"
#include "Util.hpp"

#include "SNPE.hpp"
#include "SNPEFactory.hpp"
#include "StringList.hpp"
#include "TensorShape.hpp"
#include "IUserBuffer.hpp"
#include "IUserBufferFactory.hpp"
#include "UserBufferMap.hpp"

void createUserBuffer(DlSystem::UserBufferMap &userBufferMap,
                      std::unordered_map <std::string, std::vector<uint8_t>> &applicationBuffers,
                      std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                      std::unique_ptr <SNPE::SNPE> &snpe,
                      const char *name,
                      const bool isTfNBuffer,
                      bool staticQuantization,
                      int bitWidth) {
    // get attributes of buffer by name
    printf("createUserBuffer, getInputOutputBufferAttributes: %s\n", name);
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt)
        throw std::runtime_error(
                std::string("Error obtaining attributes for input tensor ") + name);

    // calculate the size of buffer required by the input tensor
    const DlSystem::TensorShape &bufferShape = (*bufferAttributesOpt)->getDims();

    size_t bufferElementSize = 0;
    if (isTfNBuffer) {
        bufferElementSize = bitWidth / 8;
    } else {
        bufferElementSize = sizeof(float);
    }
    printf("createUserBuffer: bufferElementSize: %ld, bufferShape.rank(): %ld\n",
           bufferElementSize, bufferShape.rank());

    // Calculate the stride based on buffer strides.
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
    // Note: Buffer stride is usually known and does not need to be calculated.
    std::vector <size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = bufferElementSize;
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--) {
        (bufferShape[i] == 0) ? stride *= getResizableDim() : stride *= bufferShape[i];
        strides[i - 1] = stride;
    }

    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(),
                                      bufferElementSize);
    printf("createUserBuffer, calculated buffer size: %ld\n", bufSize);

    // set the buffer encoding type
    std::unique_ptr <DlSystem::UserBufferEncoding> userBufferEncoding;
    if (isTfNBuffer) {
        if ((*bufferAttributesOpt)->getEncodingType() ==
            DlSystem::UserBufferEncoding::ElementType_t::FLOAT && staticQuantization) {
            std::cerr << "ERROR: Quantization parameters not present in model" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        const DlSystem::UserBufferEncodingTfN *ubeTfN = dynamic_cast<const DlSystem::UserBufferEncodingTfN *>((*bufferAttributesOpt)->getEncoding());
        uint64_t stepEquivalentTo0 = ubeTfN->getStepExactly0();
        float quantizedStepSize = ubeTfN->getQuantizedStepSize();
        userBufferEncoding = std::unique_ptr<DlSystem::UserBufferEncodingTfN>(
                new DlSystem::UserBufferEncodingTfN(stepEquivalentTo0, quantizedStepSize,
                                                    bitWidth));
    } else {
        userBufferEncoding = std::unique_ptr<DlSystem::UserBufferEncodingFloat>(
                new DlSystem::UserBufferEncodingFloat());
    }

    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize));

    // create SNPE user buffer from the user-backed buffer
    DlSystem::IUserBufferFactory &ubFactory = SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(applicationBuffers.at(name).data(),
                                                               bufSize,
                                                               strides,
                                                               userBufferEncoding.get()));
    if (snpeUserBackedBuffers.back() == nullptr) {
        std::cerr << "Error while creating user buffer." << std::endl;
    }
    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

void createInputBufferMap(DlSystem::UserBufferMap &inputMap,
                          std::unordered_map <std::string, std::vector<uint8_t>> &applicationBuffers,
                          std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                          std::unique_ptr <SNPE::SNPE> &snpe,
                          bool isTfNBuffer,
                          bool staticQuantization,
                          int bitWidth) {
    // get input tensor names of the network that need to be populated
    const auto &inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const DlSystem::StringList &inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    // create SNPE user buffers for each application storage buffer
    for (const char *name: inputNames) {
        createUserBuffer(inputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name,
                         isTfNBuffer, staticQuantization, bitWidth);
    }
}

void createOutputBufferMap(DlSystem::UserBufferMap &outputMap,
                           std::unordered_map <std::string, std::vector<uint8_t>> &applicationBuffers,
                           std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                           std::unique_ptr <SNPE::SNPE> &snpe,
                           bool isTfNBuffer,
                           int bitWidth) {
    // get input tensor names of the network that need to be populated
    const auto &outputNamesOpt = snpe->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    const DlSystem::StringList &outputNames = *outputNamesOpt;

    // create SNPE user buffers for each application storage buffer
    for (const char *name: outputNames) {
        createUserBuffer(outputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name,
                         isTfNBuffer, false, bitWidth);
    }
}

void createUserBuffer(DlSystem::UserBufferMap &userBufferMap,
                      std::unordered_map <std::string, GLuint> &applicationBuffers,
                      std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                      std::unique_ptr <SNPE::SNPE> &snpe,
                      const char *name) {
    // get attributes of buffer by name
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt)
        throw std::runtime_error(
                std::string("Error obtaining attributes for input tensor ") + name);
    // calculate the size of buffer required by the input tensor
    const DlSystem::TensorShape &bufferShape = (*bufferAttributesOpt)->getDims();

    // calculate stride based on buffer strides
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
    std::vector <size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = sizeof(float);
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--) {
        stride *= bufferShape[i];
        strides[i - 1] = stride;
    }

    const size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(),
                                      bufferElementSize);

    // set the buffer encoding type
    DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
    DlSystem::UserBufferSourceGLBuffer userBufferSourceGLBuffer;

    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, GLuint(1));

    // create SNPE user buffer from the user-backed buffer
    DlSystem::IUserBufferFactory &ubFactory = SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(&applicationBuffers.at(name),
                                                               bufSize,
                                                               strides,
                                                               &userBufferEncodingFloat,
                                                               &userBufferSourceGLBuffer));

    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

void createInputBufferMap(DlSystem::UserBufferMap &inputMap,
                          std::unordered_map <std::string, GLuint> &applicationBuffers,
                          std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                          std::unique_ptr <SNPE::SNPE> &snpe) {
    // get input tensor names of the network that need to be populated
    const auto &inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const DlSystem::StringList &inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    // create SNPE user buffers for each application storage buffer
    for (const char *name: inputNames) {
        createUserBuffer(inputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name);
    }
}
