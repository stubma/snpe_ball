#include "utils.h"
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include "log.h"

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