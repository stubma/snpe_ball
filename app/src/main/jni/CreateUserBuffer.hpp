#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include "SNPE.hpp"
#include "IUserBuffer.hpp"
#include "UserBufferMap.hpp"

typedef unsigned int GLuint;

// Helper function to fill a single entry of the UserBufferMap with the given user-backed buffer
void createUserBuffer(DlSystem::UserBufferMap &userBufferMap,
                      std::unordered_map <std::string, std::vector<uint8_t>> &applicationBuffers,
                      std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                      std::unique_ptr <SNPE::SNPE> &snpe,
                      const char *name,
                      const bool isTfNBuffer,
                      bool staticQuantization,
                      int bitWidth);

// Create a UserBufferMap of the SNPE network inputs
void createInputBufferMap(DlSystem::UserBufferMap &inputMap,
                          std::unordered_map <std::string, std::vector<uint8_t>> &applicationBuffers,
                          std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                          std::unique_ptr <SNPE::SNPE> &snpe,
                          const bool isTfNBuffer,
                          bool staticQuantization,
                          int bitWidth);

// Create a UserBufferMap of the SNPE network outputs
void createOutputBufferMap(DlSystem::UserBufferMap &outputMap,
                           std::unordered_map <std::string, std::vector<uint8_t>> &applicationBuffers,
                           std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                           std::unique_ptr <SNPE::SNPE> &snpe,
                           const bool isTfNBuffer,
                           int bitWidth);

void createUserBuffer(DlSystem::UserBufferMap &userBufferMap,
                      std::unordered_map <std::string, GLuint> &applicationBuffers,
                      std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                      std::unique_ptr <SNPE::SNPE> &snpe,
                      const char *name);

void createInputBufferMap(DlSystem::UserBufferMap &inputMap,
                          std::unordered_map <std::string, GLuint> &applicationBuffers,
                          std::vector <std::unique_ptr<DlSystem::IUserBuffer>> &snpeUserBackedBuffers,
                          std::unique_ptr <SNPE::SNPE> &snpe);
