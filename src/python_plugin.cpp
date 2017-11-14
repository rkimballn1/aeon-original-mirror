#include "python_plugin.hpp"

std::vector<std::shared_ptr<nervana::module>> nervana::plugin::loaded_modules;
std::mutex                                    nervana::plugin::mtx;
