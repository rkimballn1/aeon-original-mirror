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

        template <typename T>
        struct convert
        {
            static T from_pyobject(const PyObject* from);
            static PyObject* to_pyobject(const T& from);
        };

        template <>
        struct convert<cv::Mat>
        {
            static cv::Mat from_pyobject(const PyObject* from) { return detail::to_mat(from); }
            static PyObject* to_pyobject(const cv::Mat& from)
            {
                cv::Mat to_convert = from.clone();
                python::import_numpy();
                return detail::to_ndarray(from);
            }
        };

        template <>
        struct convert<std::vector<nervana::boundingbox::box>>
        {
            static std::vector<nervana::boundingbox::box> from_pyobject(const PyObject* from)
            {
                return detail::to_boxes(from);
            }

            static PyObject* to_pyobject(const std::vector<nervana::boundingbox::box>& from)
            {
                python::import_numpy();
                return detail::to_list(from);
            }
        };
    }
}
