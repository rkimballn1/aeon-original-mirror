#pragma once

#include "python_utils.hpp"
#include "conversion.hpp"

namespace python
{
    namespace detail
    {
        template<typename H>
        void convert_arguments_aux(std::vector<PyObject*>& converted_args, H head)
        {
            converted_args.push_back(conversion::convert(head));
        }
 
        template<typename H, typename... T>
        void convert_arguments_aux(std::vector<PyObject*>& converted_args, H head, T... tail)
        {
            converted_args.push_back(conversion::convert(head));
            convert_arguments_aux<T...>(converted_args, tail...);
        }

        template<typename... Args>
        std::vector<PyObject*> convert_arguments(Args... args)
        {
            std::vector<PyObject*> converted_args;
            convert_arguments_aux<Args...>(converted_args, args...);

            return converted_args;
        }
    }

    template<typename Output, typename... Args>
    Output execute(std::string plugin_name, Args... args)
    {
        PyObject* plugin_module_name;
        PyObject* plugin_module;
        PyObject* plugin_func;
        PyObject* ret_val;

        python::ensure_gil l;

        plugin_module_name = PyString_FromString(plugin_name.c_str());
        plugin_module = PyImport_Import(plugin_module_name);

        if (!plugin_module)
        {
            PyErr_Print();
        }

        Output out;
        std::string plugin_func_name = "execute";
        
        if (plugin_module != NULL) {
            plugin_func = PyObject_GetAttrString(plugin_module, plugin_func_name.c_str());

            if (plugin_func != NULL)
                PyErr_Print();

            PyObject* arg_tuple = PyTuple_New(sizeof...(args));

            auto py_args = detail::convert_arguments<Args...>(args...);

            int idx = 0;

            for (auto a : py_args) {
                PyTuple_SetItem(arg_tuple, idx, a);
                idx++;
            }

            ret_val = PyObject_CallObject(plugin_func, arg_tuple);

            if (ret_val != NULL) {
                out = conversion::convert(ret_val);
            } 
        } else {
            PyErr_Print();
        }

        return out;
    }
}
