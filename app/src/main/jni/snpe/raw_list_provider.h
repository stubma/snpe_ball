#pragma once

#include "input_provider.h"
#include <string>
#include <vector>

class RawListProvider: public InputProvider {
public:
    RawListProvider(std::string rawListFilePath, int32_t batchSize);
    ~RawListProvider();

    virtual size_t getTensorCount();
    virtual std::unique_ptr<DlSystem::ITensor> getTensorAt(std::unique_ptr<SNPE::SNPE>& snpe, int32_t idx);
    virtual DlSystem::TensorMap getTensorMap(std::unique_ptr<SNPE::SNPE>& snpe);

private:
    std::vector<std::vector<std::string>> _inputs;
};