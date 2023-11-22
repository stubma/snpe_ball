#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"

std::unique_ptr <SNPE::SNPE>
setBuilderOptions(std::unique_ptr <DlContainer::IDlContainer> &container,
                  DlSystem::Runtime_t runtime,
                  DlSystem::RuntimeList runtimeList,
                  bool useUserSuppliedBuffers,
                  DlSystem::PlatformConfig platformConfig,
                  bool useCaching) {
    std::unique_ptr <SNPE::SNPE> snpe;
    SNPE::SNPEBuilder snpeBuilder(container.get());

    if (runtimeList.empty()) {
        runtimeList.add(runtime);
    }

    snpe = snpeBuilder.setOutputLayers({})
            .setRuntimeProcessorOrder(runtimeList)
            .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
            .setPlatformConfig(platformConfig)
            .setInitCacheMode(useCaching)
            .build();
    return snpe;
}
