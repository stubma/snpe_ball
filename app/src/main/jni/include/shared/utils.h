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
size_t fwrite_ex(
        const void* ptr,
        size_t size,
        size_t nitems,
        size_t offset,
        size_t stride,
        FILE* stream);