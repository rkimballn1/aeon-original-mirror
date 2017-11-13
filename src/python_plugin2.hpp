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
#ifdef DISPLAY
            std::cout << "module " << filename << " created" << std::endl;
#endif
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

        PyObject*               ret_val{nullptr};
        PyObject*               instance{nullptr};
        PyObject*               func_image{nullptr};
        PyObject*               func_boundingbox{nullptr};
        PyObject*               func_prepare{nullptr};
        std::shared_ptr<module> module_ptr;

    public:
        plugin()
        {
#ifdef DISPLAY
            std::cout << "constructor of empty plugin" << std::endl;
#endif
        }
        ~plugin()
        {
#ifdef DISPLAY
            std::cout << "destructor of plugin " << module_ptr->filename << " id "
                      << std::this_thread::get_id() << std::endl;
#endif
            /*
            std::lock_guard<std::mutex> lock(mtx);
            auto it = std::find(loaded_modules.begin(), loaded_modules.end(), module_ptr);
            if (it != loaded_modules.end())
            {
                loaded_modules.erase(it);
            }
            */
        }

        plugin(std::string filename, std::string params)
        {
            std::lock_guard<std::mutex> lock(mtx);
//python::ensure_gil gil;
//nervana::python::allow_threads;
#ifdef DISPLAY
            std::cout << "constructor of plugin " << filename << " id "
                      << std::this_thread::get_id() << std::endl;
#endif

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

#ifdef DISPLAY
            std::cout << "constructor succesfully returned " << instance << std::endl;
#endif

            if (instance != NULL)
            {
                func_image = PyObject_GetAttrString(instance, "augment_image");
                if (!func_image)
                {
                    PyErr_Print();
                    throw std::runtime_error("python function not loaded");
                }

                func_boundingbox = PyObject_GetAttrString(instance, "augment_boundingbox");
                if (!func_boundingbox)
                {
                    PyErr_Print();
                    throw std::runtime_error("python function not loaded");
                }

                func_prepare = PyObject_GetAttrString(instance, "prepare");
                if (!func_prepare)
                {
                    PyErr_Print();
                    throw std::runtime_error("python function not loaded");
                }
            }
#ifdef DISPLAY
            std::cout << "functions succesfully loaded " << instance << std::endl;
#endif
        }

        void prepare()
        {
            std::lock_guard<std::mutex> lock(mtx);
//nervana::python::ensure_gil gil;
//nervana::python::allow_threads;

#ifdef DISPLAY
            std::cout << "prepare " << module_ptr->filename << std::endl;
#endif

            PyObject* arg_tuple = PyTuple_New(0);

            PyObject_CallObject(func_prepare, arg_tuple);
        }
        cv::Mat augment_image(cv::Mat image)
        {
            std::lock_guard<std::mutex> lock(mtx);
//nervana::python::ensure_gil gil;
//nervana::python::allow_threads;

#ifdef DISPLAY
            std::cout << "augment image " << module_ptr->filename << std::endl;
#endif

            PyObject* arg_tuple = PyTuple_New(1);
            PyTuple_SetItem(arg_tuple, 0, ::python::conversion::convert(image));

            Py_XDECREF(ret_val);
            ret_val = NULL;
            ret_val = PyObject_CallObject(func_image, arg_tuple);

            cv::Mat out;
            if (ret_val != NULL)
            {
                out = ::python::conversion::convert_to_mat(ret_val);
            }
            else
            {
                PyErr_Print();
            }
            return out;
        }

        std::vector<boundingbox::box> augment_boundingbox(std::vector<boundingbox::box> boxes)
        {
            std::lock_guard<std::mutex> lock(mtx);
//nervana::python::ensure_gil gil;
//nervana::python::allow_threads;

#ifdef DISPLAY
            std::cout << "augment boundingbox " << module_ptr->filename << std::endl;
#endif

            PyObject* arg_tuple = PyTuple_New(1);
            PyTuple_SetItem(arg_tuple, 0, ::python::conversion::convert(boxes));

            Py_XDECREF(ret_val);
            ret_val = NULL;
            ret_val = PyObject_CallObject(func_boundingbox, arg_tuple);

            std::vector<boundingbox::box> out;
            if (ret_val != NULL)
            {
                out = ::python::conversion::convert_to_boxes(ret_val);
            }
            else
            {
                PyErr_Print();
            }
            return out;
        }
    };
}
