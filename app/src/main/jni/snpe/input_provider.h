#pragma once

#include "SNPE.hpp"
#include "SNPEFactory.hpp"

class InputProvider {
public:
    virtual ~InputProvider() {};
    virtual void setupProvider(int32_t batchSize) = 0;
    virtual size_t getTensorCount() = 0;
    virtual std::unique_ptr<DlSystem::ITensor> getTensorAt(std::unique_ptr<SNPE::SNPE>& snpe, int32_t idx) = 0;
    virtual DlSystem::TensorMap getTensorMap(std::unique_ptr<SNPE::SNPE>& snpe, int32_t idx) = 0;
};