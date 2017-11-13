#pragma once

#include "python_utils.hpp"
#include "conversion.hpp"

namespace nervana
{
    class plugin
    {
        std::string m_filename{""};
        PyObject*   func_image{nullptr};
        PyObject*   func_boundingbox{nullptr};
        PyObject*   func_prepare{nullptr};
        PyObject*   ret_val{nullptr};
        PyObject*   module_name{nullptr};
        PyObject*   module{nullptr};
        PyObject*   instance{nullptr};
        PyObject*   klass{nullptr};

    public:
        plugin() { std::cout << "constructor of empty plugin" << std::endl; }
        ~plugin()
        {
            std::cout << "destructor of plugin " << m_filename << " id "
                      << std::this_thread::get_id() << std::endl;

            std::cout << func_image->ob_refcnt << std::endl;
            std::cout << func_boundingbox->ob_refcnt << std::endl;
            std::cout << func_prepare->ob_refcnt << std::endl;
            if (ret_val)
                std::cout << ret_val->ob_refcnt << std::endl;
        }
        plugin(std::string filename, std::string params)
            : m_filename(filename)
        {
            python::ensure_gil gil;
            //nervana::python::allow_threads;

            std::cout << "constructor of plugin " << m_filename << " id "
                      << std::this_thread::get_id() << std::endl;

            module_name = PyString_FromString(m_filename.c_str());
            module      = PyImport_Import(module_name);

            if (!module)
            {
                PyErr_Print();
                throw std::runtime_error("python module not loaded");
            }

            if (module != NULL)
            {
                klass = PyObject_GetAttrString(module, "plugin");

                if (!klass)
                {
                    PyErr_Print();
                    throw std::runtime_error("python class not loaded");
                }

                if (klass != NULL)
                {
                    PyObject* arg_tuple = PyTuple_New(1);

                    PyTuple_SetItem(arg_tuple, 0, PyString_FromString(params.c_str()));
                    instance = PyInstance_New(klass, arg_tuple, NULL);
                    if (!instance)
                    {
                        PyErr_Print();
                        throw std::runtime_error("python instance not loaded");
                    }

                    std::cout << "constructor succesfully returned " << instance << std::endl;

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
                    std::cout << "functions succesfully loaded " << instance << std::endl;
                }
            }
        }

        void prepare()
        {
            nervana::python::ensure_gil gil;
            //nervana::python::allow_threads;

            std::cout << "prepare " << m_filename << std::endl;

            PyObject* arg_tuple = PyTuple_New(0);
            //PyTuple_SetItem(arg_tuple, 0, nullptr);

            std::cout << func_image->ob_refcnt << std::endl;
            std::cout << func_boundingbox->ob_refcnt << std::endl;
            std::cout << func_prepare->ob_refcnt << std::endl;
            if (ret_val)
                std::cout << ret_val->ob_refcnt << std::endl;

            PyObject_CallObject(func_prepare, arg_tuple);
        }
        cv::Mat augment_image(cv::Mat image)
        {
            nervana::python::ensure_gil gil;
            //nervana::python::allow_threads;

            std::cout << "augment image " << m_filename << std::endl;

            PyObject* arg_tuple = PyTuple_New(1);
            PyTuple_SetItem(arg_tuple, 0, ::python::conversion::convert(image));

            Py_XDECREF(ret_val);
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
            nervana::python::ensure_gil gil;
            //nervana::python::allow_threads;

            std::cout << "augment boundingbox " << m_filename << std::endl;

            PyObject* arg_tuple = PyTuple_New(1);
            PyTuple_SetItem(arg_tuple, 0, ::python::conversion::convert(boxes));

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
