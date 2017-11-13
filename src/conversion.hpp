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
            PyObject* to_ndarray(const cv::Mat& mat);
            std::vector<nervana::boundingbox::box> to_boxes(const PyObject* o);
        }

        cv::Mat convert_to_mat(PyObject* o);
        std::vector<nervana::boundingbox::box> convert_to_boxes(PyObject* o);
        PyObject* convert(int& a);
        PyObject* convert(cv::Mat& img);
        PyObject* convert(std::vector<nervana::boundingbox::box> boxes);
    }
}