#pragma once

#include <Python.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "numpy/ndarrayobject.h"

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
    }

    cv::Mat convert(PyObject* o);
    PyObject* convert(int& a);
    PyObject* convert(cv::Mat& img);
}}
