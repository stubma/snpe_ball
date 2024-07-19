#pragma once

#include <stdio.h>
#include <string>
#include "DlEnums.hpp"
#include "DlVersion.hpp"
#include "SNPE.hpp"
#include "SNPEFactory.hpp"

bool is_file_exists(std::string path);
bool is_directory(std::string path);
void mkdirs(const char* buf);
DlSystem::Runtime_t checkRuntime();
std::string getRuntimeStr();
void dumpModel(std::unique_ptr<SNPE::SNPE> &snpe, size_t *batchSize);
size_t fwrite_ex(
        const void* ptr,
        size_t size,
        size_t nitems,
        size_t offset,
        size_t stride,
        FILE* stream);
std::string last_path_component(std::string p);
std::string remove_last_path_component(std::string p);
void memcpy_ex(void* dst, const void* src, size_t size, size_t nitems, size_t offset, size_t stride);