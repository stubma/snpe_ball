#include <jni.h>
#include <stdio.h>
#include "DlEnums.hpp"
#include "DlVersion.hpp"
#include "SNPE.hpp"
#include "SNPEFactory.hpp"
#include "SNPEBuilder.hpp"
#include <iostream>
#include <string>
#include <memory>
#include "IDlContainer.hpp"
#include "log.h"

using namespace DlSystem;
using namespace DlContainer;

// java vm
static JavaVM* _vm;

Runtime_t checkRuntime() {
    Version_t Version = SNPE::SNPEFactory::getLibraryVersion();
    Runtime_t Runtime;
    ALOGD("Qualcomm (R) Neural Processing SDK Version: %s\n", Version.asString().c_str()); //Print Version number
    if (SNPE::SNPEFactory::isRuntimeAvailable(Runtime_t::DSP)) {
        Runtime = Runtime_t::DSP;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(Runtime_t::GPU)) {
        Runtime = Runtime_t::GPU;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(Runtime_t::GPU_FLOAT16)) {
        Runtime = Runtime_t::GPU;
    } else if (SNPE::SNPEFactory::isRuntimeAvailable(Runtime_t::CPU)) {
        Runtime = Runtime_t::CPU;
    } else {
        Runtime = Runtime_t::UNSET;
    }
    return Runtime;
}

std::unique_ptr<IDlContainer> loadContainerFromFile(std::string containerPath) {
    std::unique_ptr<IDlContainer> container;
    container = IDlContainer::open(containerPath);
    return container;
}

std::unique_ptr<SNPE::SNPE> setBuilderOptions(std::unique_ptr<IDlContainer>& container, RuntimeList runtimeList, bool useUserSuppliedBuffers) {
    std::unique_ptr<SNPE::SNPE> snpe;
    SNPE::SNPEBuilder snpeBuilder(container.get());
    snpe = snpeBuilder.setOutputLayers({})
            .setRuntimeProcessorOrder(runtimeList)
            .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
            .build();
    return snpe;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_hexagon_1test_Hexagon_checkRuntime(JNIEnv *env, jobject thiz) {
    Runtime_t rt = checkRuntime();
    switch(rt) {
        case Runtime_t::GPU:
            return env->NewStringUTF("GPU");
        case Runtime_t::CPU:
            return env->NewStringUTF("CPU");
        case Runtime_t::DSP:
            return env->NewStringUTF("DSP");
        default:
            return env->NewStringUTF("Unsupported");
    }
}

__attribute__ ((visibility ("default"))) jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    _vm = vm;
    setenv("LD_LIBRARY_PATH", "/vendor/lib64", 1);
    return JNI_VERSION_1_6;
}