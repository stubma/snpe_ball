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
    std::unique_ptr<SNPE::SNPE> snpe;
    SNPE::SNPEBuilder snpeBuilder(container.get());
    if (runtimeList.empty()) {
        runtimeList.add(runtime);
    }

    std::chrono::milliseconds start = std::chrono::milliseconds(0);
    snpe = snpeBuilder.setOutputLayers({})
            .setRuntimeProcessorOrder(runtimeList)
            .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
            .setPlatformConfig(platformConfig)
            .setInitCacheMode(useCaching)
            .build();
    std::chrono::milliseconds end = std::chrono::milliseconds(0);
    ALOGD("build network for runtime cost %lldms", end.count() - start.count());
    return snpe;
}
