#include "raw_list_provider.h"
#include "PreprocessInput.hpp"
#include "LoadInputTensor.hpp"

RawListProvider::RawListProvider(std::string rawListFilePath, int32_t batchSize) {
    _inputs = preprocessInput(rawListFilePath, batchSize);
}

RawListProvider::~RawListProvider() {
}

size_t RawListProvider::getTensorCount() {
    return _inputs.size();
}

std::unique_ptr<DlSystem::ITensor> RawListProvider::getTensorAt(std::unique_ptr<SNPE::SNPE>& snpe, int32_t idx) {
    const auto &inputTensorNamesRef = snpe->getInputTensorNames();
    const auto &inputTensorNames = *inputTensorNamesRef;
    return loadInputTensor(snpe, _inputs[idx], inputTensorNames);
}

DlSystem::TensorMap RawListProvider::getTensorMap(std::unique_ptr<SNPE::SNPE>& snpe) {
    const auto &inputTensorNamesRef = snpe->getInputTensorNames();
    const auto &inputTensorNames = *inputTensorNamesRef;
    bool success = false;
    return loadMultipleInput(snpe, _inputs, inputTensorNames, success);
}