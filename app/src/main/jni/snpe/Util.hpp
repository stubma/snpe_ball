#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "ITensorFactory.hpp"
#include "TensorShape.hpp"
#include "IUserBuffer.hpp"

template<typename Container>
Container &split(Container &result, const typename Container::value_type &s,
                 typename Container::value_type::value_type delimiter) {
    result.clear();
    std::istringstream ss(s);
    while (!ss.eof()) {
        typename Container::value_type field;
        getline(ss, field, delimiter);
        if (!field.empty() && field.back() == '\r') field.pop_back();
        if (field.empty()) continue;
        result.push_back(field);
    }
    return result;
}

size_t calcSizeFromDims(const DlSystem::Dimension *dims, size_t rank, size_t elementSize);

std::vector<float> loadFloatDataFile(const std::string &inputFile);

std::vector<unsigned char> loadByteDataFile(const std::string &inputFile);

std::vector<unsigned char> loadByteDataFileBatched(const std::string &inputFile);

template<typename T>
bool loadByteDataFile(const std::string &inputFile, std::vector <T> &loadVector) {
    std::ifstream in(inputFile, std::ifstream::binary);
    if (!in.is_open() || !in.good()) {
        std::cerr << "Failed to open input file: " << inputFile << "\n";
    }

    in.seekg(0, in.end);
    size_t length = in.tellg();
    in.seekg(0, in.beg);

    if (length % sizeof(T) != 0) {
        std::cerr << "Size of input file should be divisible by sizeof(dtype).\n";
        return false;
    }

    if (loadVector.size() == 0) {
        loadVector.resize(length / sizeof(T));
    } else if (loadVector.size() < length / sizeof(T)) {
        std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
        loadVector.resize(length / sizeof(T));
    }

    if (!in.read(reinterpret_cast<char *>(&loadVector[0]), length)) {
        std::cerr << "Failed to read the contents of: " << inputFile << "\n";
    }
    return true;
}

template<typename T>
bool
loadByteDataFileBatched(const std::string &inputFile, std::vector <T> &loadVector, size_t offset) {
    std::ifstream in(inputFile, std::ifstream::binary | std::ios::ate);
    if (!in.is_open() || !in.good()) {
        std::cerr << "Failed to open input file: " << inputFile << "\n";
    }

    size_t length = in.tellg();
    in.seekg(0, in.beg);

    if (length % sizeof(T) != 0) {
        std::cerr << "Size of input file should be divisible by sizeof(dtype).\n";
        return false;
    }

    if (loadVector.size() == 0) {
        loadVector.resize(length / sizeof(T));
    } else if (loadVector.size() < length / sizeof(T)) {
        printf("Vector is not large enough(%lu) to hold data of input file: %s(%lu)\n",
               loadVector.size(), inputFile.c_str(), length);
    }

    loadVector.resize((offset + 1) * length / sizeof(T));

    if (!in.read(reinterpret_cast<char *> (&loadVector[offset * length / sizeof(T)]), length)) {
        std::cerr << "Failed to read the contents of: " << inputFile << "\n";
    }
    return true;
}

bool loadByteDataFileBatchedTf8(const std::string &inputFile, std::vector <uint8_t> &loadVector,
                                size_t offset);

bool loadByteDataFileBatchedTfN(const std::string &inputFile, std::vector <uint8_t> &loadVector,
                                size_t offset,
                                unsigned char &stepEquivalentTo0, float &quantizedStepSize,
                                bool staticQuantization, int bitWidth);

bool
SaveITensorBatched(const std::string &path, const DlSystem::ITensor *tensor, size_t batchIndex = 0,
                   size_t batchChunk = 0);

bool SaveUserBufferBatched(const std::string &path, const std::vector <uint8_t> &buffer,
                           size_t batchIndex = 0, size_t batchChunk = 0);

bool EnsureDirectory(const std::string &dir);

void TfNToFloat(float *out, uint8_t *in, const unsigned char stepEquivalentTo0,
                const float quantizedStepSize, size_t numElement, int bitWidth);

bool FloatToTfN(uint8_t *out, unsigned char &stepEquivalentTo0, float &quantizedStepSize,
                bool staticQuantization, float *in, size_t numElement, int bitWidth);

void setResizableDim(size_t resizableDim);

size_t getResizableDim();

std::string elementTypeStr(DlSystem::UserBufferEncoding::ElementType_t t);
