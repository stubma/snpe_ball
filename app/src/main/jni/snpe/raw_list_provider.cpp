#include "raw_list_provider.h"
#include "PreprocessInput.hpp"
#include "LoadInputTensor.hpp"
#include "log.h"

RawListProvider::RawListProvider(std::string rawListFilePath) {
    _rawListFilePath = rawListFilePath;
}

RawListProvider::~RawListProvider() {
}

size_t RawListProvider::getTensorCount() {
    return _inputs.size();
}

void RawListProvider::setupProvider(int32_t batchSize) {
    _inputs = preprocessInput(_rawListFilePath, batchSize);
}

std::unique_ptr<DlSystem::ITensor> RawListProvider::getTensorAt(std::unique_ptr<SNPE::SNPE>& snpe, int32_t idx) {
    const auto &inputTensorNamesRef = snpe->getInputTensorNames();
    const auto &inputTensorNames = *inputTensorNamesRef;
    return loadInputTensor(snpe, _inputs[idx], inputTensorNames);
}

DlSystem::TensorMap RawListProvider::getTensorMap(std::unique_ptr<SNPE::SNPE>& snpe, int32_t idx) {
    const auto &inputTensorNamesRef = snpe->getInputTensorNames();
    const auto &inputTensorNames = *inputTensorNamesRef;
    bool success = false;
    return loadMultipleInput(snpe, _inputs[idx], inputTensorNames, success);
}