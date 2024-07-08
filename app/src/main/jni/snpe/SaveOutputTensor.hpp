#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "SNPE.hpp"
#include "ITensor.hpp"
#include "UserBufferMap.hpp"

// Save output implementation of ITensor
bool saveOutput(DlSystem::TensorMap outputTensorMap,
                const std::string &outputDir,
                int num,
                size_t batchSize = 1);

// Save output USERBUFFER
bool saveOutput(DlSystem::UserBufferMap &outputMap,
                std::unordered_map <std::string, std::vector<uint8_t>> &applicationOutputBuffers,
                const std::string &outputDir,
                int num,
                size_t batchSize,
                bool isTfNBuffer,
                int bitWidth);
