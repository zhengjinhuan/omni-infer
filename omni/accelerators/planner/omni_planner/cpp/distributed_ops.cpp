#include "distributed_ops.h"
#include <stdexcept>

namespace py = pybind11;

template<typename... Args>
std::string call_python_distributed_ops(const std::string& func_name, Args&&... args) {
    try {
        py::scoped_interpreter guard{}; // 初始化 Python 解释器

        // 设置 Python 模块搜索路径（根据需要调整）
        // py::module_ sys = py::module_::import("sys");
        // sys.attr("path").attr("append")("/path/to/omni_planner");

        // 导入 distributed_ops 模块
        py::module_ distributed_ops = py::module_::import("omni_planner.distributed_ops");

        // 调用指定函数
        py::object result = distributed_ops.attr(func_name.c_str())(std::forward<Args>(args)...);

        // 将结果转换为字符串返回
        return py::str(result).cast<std::string>();
    } catch (const py::error_already_set& e) {
        throw std::runtime_error("Python error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("C++ error: " + std::string(e.what()));
    }
}

// 显式实例化
template std::string call_python_distributed_ops(const std::string&, int&, int&);
