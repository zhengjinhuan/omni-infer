#ifndef DISTRIBUTED_OPS_H
#define DISTRIBUTED_OPS_H

#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

template<typename... Args>
std::string call_python_distributed_ops(const std::string& func_name, Args&&... args);

#endif