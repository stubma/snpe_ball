#pragma once

#include "input_provider.h"
#include <string>
#include <vector>

class RawListProvider: public InputProvider {
public:
    RawListProvider(std::string rawListFilePath);
    virtual ~RawListProvider();

    virtual void setupProvider(int32_t batchSize);
    virtual size_t getTensorCount();
    virtual std::unique_ptr<DlSystem::ITensor> getTensorAt(std::unique_ptr<SNPE::SNPE>& snpe, int32_t idx);
    virtual DlSystem::TensorMap getTensorMap(std::unique_ptr<SNPE::SNPE>& snpe, int32_t idx);

private:
    std::string _rawListFilePath;
    std::vector<std::vector<std::string>> _inputs;
};