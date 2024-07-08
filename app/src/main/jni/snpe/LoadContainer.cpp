#include <iostream>
#include <string>
#include "LoadContainer.hpp"
#include "DlContainer/IDlContainer.hpp"

std::unique_ptr<DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath) {
    std::unique_ptr<DlContainer::IDlContainer> container;
    container = DlContainer::IDlContainer::open(DlSystem::String(containerPath.c_str()));
    return container;
}
