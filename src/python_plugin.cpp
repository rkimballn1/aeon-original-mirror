/*
 Copyright 2017 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include <Python.h>
#include <cstdlib>
#include "python_plugin.hpp"

std::vector<std::shared_ptr<nervana::module>> nervana::plugin::loaded_modules;
std::mutex                                    nervana::plugin::mtx;

namespace
{
    static nervana::call_initialize x;
}

namespace nervana
{
    module::module(std::string path)
        : filename(path)
    {
        name   = PyString_FromString(filename.c_str());
        handle = PyImport_Import(name);

        if (!handle)
        {
            PyErr_Print();
            throw std::runtime_error("python module not loaded");
        }

        klass = PyObject_GetAttrString(handle, "plugin");

        if (!klass)
        {
            PyErr_Print();
            throw std::runtime_error("python class not loaded");
        }
    }

    bool module::operator==(const module& other) { return filename == other.filename; }
    template <typename T>
    T plugin::augment(PyObject* methodname, const T& in_data)
    {
        std::lock_guard<std::mutex> lock(mtx);

        using convert = typename ::python::conversion::convert<T>;

        PyObject* arg = convert::to_pyobject(in_data);

        PyObject* ret_val = PyObject_CallMethodObjArgs(instance, methodname, arg, NULL);

        T out;
        if (ret_val != NULL)
        {
            out = convert::from_pyobject(ret_val);
        }
        else
        {
            PyObject *err_type, *err_value, *err_traceback;
            PyErr_Fetch(&err_type, &err_value, &err_traceback);
            char* err_msg = PyString_AsString(err_value);

            std::stringstream ss;
            ss << "Python has failed with error message: " << err_msg << std::endl;
            throw std::runtime_error(ss.str());
        }

        return out;
    }

    plugin::plugin(std::string filename, std::string params)
    {
        std::lock_guard<std::mutex> lock(mtx);

        auto it = std::find_if(
            loaded_modules.begin(),
            loaded_modules.end(),
            [&filename](const std::shared_ptr<module> obj) { return obj->filename == filename; });
        if (it == loaded_modules.end())
        {
            module_ptr = std::make_shared<module>(filename);
            loaded_modules.push_back(module_ptr);
        }
        else
        {
            module_ptr = *it;
        }

        PyObject* arg_tuple = PyTuple_New(1);

        PyTuple_SetItem(arg_tuple, 0, PyString_FromString(params.c_str()));
        instance = PyObject_Call(module_ptr->klass, arg_tuple, NULL);
        if (!instance)
        {
            PyErr_Print();
            throw std::runtime_error("python instance not loaded");
        }
    }

    plugin::~plugin()
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = std::find(loaded_modules.begin(), loaded_modules.end(), module_ptr);
        if (it != loaded_modules.end())
        {
            loaded_modules.erase(it);
        }
    }

    void plugin::prepare()
    {
        std::lock_guard<std::mutex> lock(mtx);

        PyObject_CallMethodObjArgs(instance, PyString_FromString("prepare"), NULL, NULL);
    }
    cv::Mat plugin::augment_image(const cv::Mat& m)

    {
        return augment(PyString_FromString("augment_image"), m);
    }

    std::vector<boundingbox::box>
        plugin::augment_boundingbox(const std::vector<boundingbox::box>& boxes)
    {
        return augment(PyString_FromString("augment_boundingbox"), boxes);
    }

    cv::Mat plugin::augment_pixel_mask(const cv::Mat& pixel_mask)
    {
        return augment(PyString_FromString("augment_pixel_mask"), pixel_mask);
    }

    cv::Mat plugin::augment_depthmap(const cv::Mat& depthmap)
    {
        return augment(PyString_FromString("augment_depthmap"), depthmap);
    }

    cv::Mat plugin::augment_audio(const cv::Mat& audio)
    {
        return augment(PyString_FromString("augment_audio"), audio);
    }

    call_initialize::call_initialize()
    {
        std::lock_guard<std::mutex> lock(nervana::plugin::mtx);
        Py_Initialize();
        PyRun_SimpleString("import threading");
        std::atexit(call_finalize);
    }

    void call_initialize::call_finalize()
    {
        std::lock_guard<std::mutex> lock(nervana::plugin::mtx);
        PyGILState_Ensure();
        Py_Finalize();
    }
};
