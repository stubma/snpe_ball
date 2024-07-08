#pragma once

#include <unordered_map>
#include <string>
#include <vector>

#include "SNPE.hpp"
#include "ITensorFactory.hpp"
#include "TensorMap.hpp"

typedef unsigned int GLuint;

std::unique_ptr <DlSystem::ITensor> loadInputTensor(std::unique_ptr <SNPE::SNPE> &snpe,
                                                         std::vector <std::string> &fileLines,
                                                         const DlSystem::StringList &inputTensorNames);

std::tuple<DlSystem::TensorMap, bool>
loadMultipleInput(std::unique_ptr <SNPE::SNPE> &snpe,
                  std::vector <std::string> &fileLines,
                  const DlSystem::StringList &inputTensorNames,
                  std::vector <std::unique_ptr<DlSystem::ITensor>> &inputs);

bool
loadInputUserBufferFloat(std::unordered_map <std::string, std::vector<uint8_t>> &applicationBuffers,
                         std::unique_ptr <SNPE::SNPE> &snpe,
                         std::vector <std::string> &fileLines);

bool
loadInputUserBufferTfN(std::unordered_map <std::string, std::vector<uint8_t>> &applicationBuffers,
                       std::unique_ptr <SNPE::SNPE> &snpe,
                       std::vector <std::string> &fileLines,
                       DlSystem::UserBufferMap &inputMap,
                       bool staticQuantization,
                       int bitWidth);

void loadInputUserBuffer(std::unordered_map <std::string, GLuint> &applicationBuffers,
                         std::unique_ptr <SNPE::SNPE> &snpe,
                         const GLuint inputglbuffer);
