#pragma once

#include "python_utils.hpp"
#include "conversion.hpp"

namespace nervana
{
    struct module
    {
        module(std::string path)
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
        bool operator==(const module& other) { return filename == other.filename; }
        std::string                   filename{""};
        PyObject*                     name{nullptr};
        PyObject*                     handle{nullptr};
        PyObject*                     klass{nullptr};
    };

    class plugin
    {
        static std::vector<std::shared_ptr<module>> loaded_modules;
        static std::mutex                           mtx;

        PyObject*               instance{nullptr};
        PyObject*               func_image{nullptr};
        PyObject*               func_pixel_mask{nullptr};
        PyObject*               func_depthmap{nullptr};
        PyObject*               func_audio{nullptr};
        PyObject*               func_boundingbox{nullptr};
        PyObject*               func_prepare{nullptr};
        std::shared_ptr<module> module_ptr;

        template<typename T>
        T augment(PyObject* func, const T& in_data)
        {
            using convert = typename ::python::conversion::convert<T>;

            std::lock_guard<std::mutex> lock(mtx);

            PyObject* arg_tuple = PyTuple_New(1);
            PyTuple_SetItem(arg_tuple, 0, convert::to_pyobject(in_data));

            PyObject* ret_val = PyObject_CallObject(func, arg_tuple);

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

    public:
        plugin()
        { }
        ~plugin()
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = std::find(loaded_modules.begin(), loaded_modules.end(), module_ptr);
            if (it != loaded_modules.end())
            {
                loaded_modules.erase(it);
            }
        }

        plugin(std::string filename, std::string params)
        {
            std::lock_guard<std::mutex> lock(mtx);

            auto it = std::find_if(loaded_modules.begin(),
                                   loaded_modules.end(),
                                   [&filename](const std::shared_ptr<module> obj) {
                                       return obj->filename == filename;
                                   });
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
            instance = PyInstance_New(module_ptr->klass, arg_tuple, NULL);
            if (!instance)
            {
                PyErr_Print();
                throw std::runtime_error("python instance not loaded");
            }

            func_image = PyObject_GetAttrString(instance, "augment_image");
            if (!func_image)
            {
                PyErr_Print();
                throw std::runtime_error("python augment_image function not loaded");
            }

            func_boundingbox = PyObject_GetAttrString(instance, "augment_boundingbox");
            if (!func_boundingbox)
            {
                PyErr_Print();
                throw std::runtime_error("python augment_boundingbox function not loaded");
            }

            func_pixel_mask = PyObject_GetAttrString(instance, "augment_pixel_mask");
            if (!func_pixel_mask)
            {
                PyErr_Print();
                throw std::runtime_error("python augment_pixel_mask function not loaded");
            }

            func_depthmap = PyObject_GetAttrString(instance, "augment_depthmap");
            if (!func_depthmap)
            {
                PyErr_Print();
                throw std::runtime_error("python augment_depthmap function not loaded");
            }

            func_audio = PyObject_GetAttrString(instance, "augment_audio");
            if (!func_audio)
            {
                PyErr_Print();
                throw std::runtime_error("python augment_audio function not loaded");
            }

            func_prepare = PyObject_GetAttrString(instance, "prepare");
            if (!func_prepare)
            {
                PyErr_Print();
                throw std::runtime_error("python prepare function not loaded");
            }
        }

        void prepare()
        {
            std::lock_guard<std::mutex> lock(mtx);

            PyObject* arg_tuple = PyTuple_New(0);
            PyObject_CallObject(func_prepare, arg_tuple);
        }

        cv::Mat augment_image(const cv::Mat& m)
        {
            return augment(func_image, m);
        }

        std::vector<boundingbox::box> augment_boundingbox(const std::vector<boundingbox::box>& boxes)
        {
            return augment(func_boundingbox, boxes);
        }
        cv::Mat augment_pixel_mask(cv::Mat& pixel_mask)
        {
            std::lock_guard<std::mutex> lock(mtx);

            PyObject* arg_tuple = PyTuple_New(1);
            PyTuple_SetItem(arg_tuple, 0, ::python::conversion::convert(pixel_mask));

            Py_XDECREF(ret_val);
            ret_val = NULL;
            ret_val = PyObject_CallObject(func_pixel_mask, arg_tuple);

            cv::Mat out;
            if (ret_val != NULL)
            {
                out = ::python::conversion::convert_to_mat(ret_val);
            }
            else
            {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                char*             pStrErrorMessage = PyString_AsString(pvalue);
                std::stringstream ss;
                ss << "Python has failed with error message: " << pStrErrorMessage << std::endl;
                throw std::runtime_error(ss.str());
            }
            return out;
        }

        cv::Mat augment_depthmap(cv::Mat& depthmap)
        {
            std::lock_guard<std::mutex> lock(mtx);

            PyObject* arg_tuple = PyTuple_New(1);
            PyTuple_SetItem(arg_tuple, 0, ::python::conversion::convert(depthmap));

            Py_XDECREF(ret_val);
            ret_val = NULL;
            ret_val = PyObject_CallObject(func_depthmap, arg_tuple);

            cv::Mat out;
            if (ret_val != NULL)
            {
                out = ::python::conversion::convert_to_mat(ret_val);
            }
            else
            {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                char*             pStrErrorMessage = PyString_AsString(pvalue);
                std::stringstream ss;
                ss << "Python has failed with error message: " << pStrErrorMessage << std::endl;
                throw std::runtime_error(ss.str());
            }
            return out;
        }

        cv::Mat augment_audio(cv::Mat& audio)
        {
            std::lock_guard<std::mutex> lock(mtx);

            PyObject* arg_tuple = PyTuple_New(1);
            PyTuple_SetItem(arg_tuple, 0, ::python::conversion::convert(audio));

            Py_XDECREF(ret_val);
            ret_val = NULL;
            ret_val = PyObject_CallObject(func_audio, arg_tuple);

            cv::Mat out;
            if (ret_val != NULL)
            {
                out = ::python::conversion::convert_to_mat(ret_val);
            }
            else
            {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                char*             pStrErrorMessage = PyString_AsString(pvalue);
                std::stringstream ss;
                ss << "Python has failed with error message: " << pStrErrorMessage << std::endl;
                throw std::runtime_error(ss.str());
            }
            return out;
        }
    };
}
