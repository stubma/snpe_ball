#pragma once

#include <string>
#include "IDlContainer.hpp"

std::unique_ptr<DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath);