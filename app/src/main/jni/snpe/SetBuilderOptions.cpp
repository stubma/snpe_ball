#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "log.h"

std::unique_ptr<SNPE::SNPE>
setBuilderOptions(std::unique_ptr<DlContainer::IDlContainer> &container,
                  DlSystem::Runtime_t runtime,
                  DlSystem::RuntimeList runtimeList,
                  bool useUserSuppliedBuffers,
                  DlSystem::PlatformConfig platformConfig,
                  bool useCaching) {
    ALOGD("1111");
    std::unique_ptr<SNPE::SNPE> snpe;
    SNPE::SNPEBuilder snpeBuilder(container.get());
    ALOGD("2222");
    if (runtimeList.empty()) {
        runtimeList.add(runtime);
    }
    ALOGD("33333");

    snpe = snpeBuilder.setOutputLayers({})
            .setRuntimeProcessorOrder(runtimeList)
            .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
            .setPlatformConfig(platformConfig)
            .setInitCacheMode(useCaching)
            .build();
    ALOGD("4444");
    return snpe;
}
