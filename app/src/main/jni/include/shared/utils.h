#pragma once

#include <stdio.h>
#include <string>

bool is_file_exists(std::string path);
bool is_directory(std::string path);
void mkdirs(const char* buf);