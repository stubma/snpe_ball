#pragma once

#include "RuntimeList.hpp"
#include "SNPE.hpp"
#include "DlEnums.hpp"
#include "IDlContainer.hpp"
#include "PlatformConfig.hpp"

std::unique_ptr<SNPE::SNPE> setBuilderOptions(std::unique_ptr<DlContainer::IDlContainer> & container,
                                                   DlSystem::Runtime_t runtime,
                                                   DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching);

