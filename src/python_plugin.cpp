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

std::mutex nervana::plugin::mtx;

namespace
{
    static nervana::call_initialize x;
}

namespace nervana
{
    PyThreadState* call_initialize::m_tstate{nullptr};

    call_initialize::call_initialize()
    {
        std::lock_guard<std::mutex> lock(nervana::plugin::mtx);
        Py_Initialize();
        std::cout << "Before init threads() " << PyGILState_GetThisThreadState() << std::endl;
        PyEval_InitThreads();
        std::cout << "After init threads() " << PyGILState_GetThisThreadState() << std::endl;

        //PyRun_SimpleString("import threading");
        std::atexit(call_finalize);
        m_tstate = PyEval_SaveThread();
    }

    void call_initialize::call_finalize()
    {
        std::lock_guard<std::mutex> lock(nervana::plugin::mtx);
        std::cout << "before finalize " << PyGILState_GetThisThreadState() << std::endl;

        PyEval_RestoreThread(m_tstate);
        //PyGILState_Ensure();
        Py_Finalize();
    }

    template <typename T>
    T plugin::augment(PyObject* methodname, const T& in_data)
    {
        std::lock_guard<std::mutex> lock(mtx);
        python::ensure_gil          gil;

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

    plugin::plugin(std::string fname, std::string params)
        : filename(fname)
    {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "plugin() " << PyGILState_GetThisThreadState() << std::endl;
        std::cout << "is initialized? " << Py_IsInitialized() << PyEval_ThreadsInitialized()
                  << std::endl;
        python::ensure_gil gil;
        std::cout << "plugin() after gil " << PyGILState_GetThisThreadState() << std::endl;

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

        PyObject* arg_tuple = PyTuple_New(1);
        PyTuple_SetItem(arg_tuple, 0, PyString_FromString(params.c_str()));

        instance = PyObject_CallObject(klass, arg_tuple);
        if (!instance)
        {
            PyErr_Print();
            throw std::runtime_error("python instance not loaded");
        }
    }

    plugin::~plugin() {}
    void plugin::prepare()
    {
        std::lock_guard<std::mutex> lock(mtx);
        python::ensure_gil          gil;
        PyObject_CallMethodObjArgs(instance, PyString_FromString("prepare"), NULL);
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
}
