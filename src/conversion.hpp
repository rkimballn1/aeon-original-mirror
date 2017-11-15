#pragma once

#include <Python.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "numpy/ndarrayobject.h"
#include "boundingbox.hpp"

#define NUMPY_IMPORT_ARRAY_RETVAL

namespace python
{
    void import_numpy();

    namespace conversion
    {
        namespace detail
        {
            cv::Mat to_mat(const PyObject* o);
            std::vector<nervana::boundingbox::box> to_boxes(const PyObject* o);
            PyObject* to_ndarray(const cv::Mat& mat);
            PyObject* to_list(const std::vector<nervana::boundingbox::box>& boxes);
        }

        cv::Mat convert_to_mat(const PyObject* o);
        std::vector<nervana::boundingbox::box> convert_to_boxes(const PyObject* o);

        template<typename T>
        T convert_to(const PyObject* o)
        {
            if constexpr (std::is_same<T, cv::Mat>::value)
                return convert_to_mat(o);
            else if constexpr (std::is_same<T, std::vector<nervana::boundingbox::box>>::value)
                return convert_to_boxes(o);
        }

        PyObject* convert(const int& a);
        PyObject* convert(const cv::Mat& img);
        PyObject* convert(const std::vector<nervana::boundingbox::box>& boxes);

        template<typename T>
        struct convert
        {
            static T from_pyobject(const PyObject* from);
            static PyObject* to_pyobject(const T& t);
        };

        template<>
        struct convert<cv::Mat>
        {
            static cv::Mat from_pyobject(const PyObject* from)
            {
                return detail::to_mat(from);      
            }

            static PyObject* to_pyobject(const cv::Mat& from)
            {
                return detail::to_ndarray(from);
            }
        };

        template<>
        struct convert<std::vector<nervana::boundingbox>
        {
            static std::vector<nervana::boundingbox> from_pyobject(const PyObject* from)
            {
                return detail::to_mat(from);
            }

            static PyObject* to_pyobject(const cv::Mat& from)
            {
                return detail::to_ndarray(from);
            }
        };
    }
}
