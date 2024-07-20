#include "utils.h"
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include "log.h"
#include "Util.hpp"

bool starts_with(std::string src, std::string sub) {
    return src.rfind(sub, 0) == 0;
}

void mkdirs(const char* buf) {
    char tmp[256];
    char *p = NULL;
    size_t len;
    snprintf(tmp, sizeof(tmp), "%s", buf);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, S_IRWXU);
            *p = '/';
        }
    }
    mkdir(tmp, S_IRWXU);
}

bool is_directory(std::string path) {
    struct stat st;
    if(stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        return true;
    }
    return false;
}

bool is_file_exists(std::string path) {
    struct stat st;
    if(stat(path.c_str(), &st) == 0) {
        return true;
    }
    return false;
}

DlSystem::Runtime_t checkRuntime() {
    DlSystem::Version_t Version = SNPE::SNPEFactory::getLibraryVersion();
    DlSystem::Runtime_t Runtime;
    ALOGD("Qualcomm (R) Neural Processing SDK Version: %s\n",
          Version.asString().c_str()); //Print Version number
    if (SNPE::SNPEFactory::isRuntimeAvailable(DlSystem::Runtime_t::DSP)) {
        Runtime = DlSystem::Runtime_t::DSP;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(DlSystem::Runtime_t::GPU)) {
        Runtime = DlSystem::Runtime_t::GPU;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(DlSystem::Runtime_t::GPU_FLOAT16)) {
        Runtime = DlSystem::Runtime_t::GPU;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(DlSystem::Runtime_t::CPU)) {
        Runtime = DlSystem::Runtime_t::CPU;
    } else {
        Runtime = DlSystem::Runtime_t::UNSET;
    }
    return Runtime;
}

std::string getRuntimeStr() {
    DlSystem::Runtime_t rt = checkRuntime();
    switch (rt) {
        case DlSystem::Runtime_t::GPU:
            return "GPU";
        case DlSystem::Runtime_t::CPU:
            return "CPU";
        case DlSystem::Runtime_t::DSP:
            return "DSP";
        default:
            return "Unsupported";
    }
}

void dumpModel(std::unique_ptr<SNPE::SNPE> &snpe, SNPEMeta& meta) {
    DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    const size_t *dims = tensorShape.getDimensions();
    printf("model input dimensions: ");
    for (int i = 0; i < tensorShape.rank(); i++) {
        if (i == 0) {
            meta.batch_size = dims[i];
        } else if(i == 1) {
            meta.input_height = dims[i];
        } else if(i == 2) {
            meta.input_width = dims[i];
        } else if(i == 3) {
            meta.channels = dims[i];
        }
        printf("%d ", dims[i]);
    }
    printf("\n");
    printf("Batch size for the container is %ld\n", meta.batch_size);

    // dump input tensors
    const auto &inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const DlSystem::StringList &inputNames = *inputNamesOpt;
    for (const char *name: inputNames) {
        auto attrs = snpe->getInputOutputBufferAttributes(name);
        if (!attrs)
            throw std::runtime_error(
                    std::string("Error obtaining attributes for input tensor ") + name);

        printf("model input tensor(%s) dimensions: ", name);
        const DlSystem::TensorShape &bufferShape = (*attrs)->getDims();
        const size_t *dims = bufferShape.getDimensions();
        for (int i = 0; i < bufferShape.rank(); i++) {
            printf("%d ", dims[i]);
        }
        printf(", element size: %lu", (*attrs)->getElementSize());
        printf(", element type: %s\n", elementTypeStr((*attrs)->getEncodingType()).c_str());
    }

    // dump output tensors
    const auto &outputNamesOpt = snpe->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    const DlSystem::StringList &outputNames = *outputNamesOpt;
    meta.output_names = outputNames;
    for (const char *name: outputNames) {
        auto attrs = snpe->getInputOutputBufferAttributes(name);
        if (!attrs)
            throw std::runtime_error(
                    std::string("Error obtaining attributes for output tensor ") + name);

        printf("model output tensor(%s) dimensions: ", name);
        const DlSystem::TensorShape &bufferShape = (*attrs)->getDims();
        meta.output_shapes.push_back(bufferShape);
        const size_t *dims = bufferShape.getDimensions();
        for (int i = 0; i < bufferShape.rank(); i++) {
            printf("%d ", dims[i]);
        }
        meta.output_element_sizes.push_back((*attrs)->getElementSize());
        meta.output_element_types.push_back((*attrs)->getEncodingType());
        printf(", element size: %lu", (*attrs)->getElementSize());
        printf(", element type: %s\n", elementTypeStr((*attrs)->getEncodingType()).c_str());
    }
}

size_t fwrite_ex(
        const void* ptr,
        size_t size,
        size_t nitems,
        size_t offset,
        size_t stride,
        FILE* stream) {
    // if no stride, same as fwrite
    if(stride == 0) {
        return fwrite((char*)ptr + offset, size, nitems, stream);
    }

    // otherwise we need take care of stride
    char* buf = (char*)ptr;
    buf += offset;
    size_t ret = 0;
    for(size_t i = 0; i < nitems; i++) {
        size_t w = fwrite(buf, size, 1, stream);
        ret += w;
        if(w != 1) {
            break;
        } else {
            buf += stride;
        }
    }

    // success
    return ret;
}

void memcpy_ex(void* dst, const void* src, size_t size, size_t nitems, size_t offset, size_t stride) {
    if(stride == 0) {
        memcpy(dst, (uint8_t*)src + offset, size * nitems);
    } else {
        uint8_t* from = (uint8_t*)src;
        uint8_t* to = (uint8_t*)dst;
        from += offset;
        for(size_t i = 0; i < nitems; i++) {
            memcpy(to, from, size);
            from += stride;
            to += size;
        }
    }
}

std::string last_path_component(std::string p) {
    std::size_t pos = p.rfind("/");
    if(pos == std::string::npos) {
        return p;
    } else {
        return p.substr(pos + 1);
    }
}

std::string remove_last_path_component(std::string p) {
    std::size_t pos = p.rfind("/");
    if(pos == std::string::npos) {
        return p;
    } else {
        std::string d = p.substr(0, pos);
        if(d.empty()) {
            return p;
        } else {
            return d;
        }
    }
}