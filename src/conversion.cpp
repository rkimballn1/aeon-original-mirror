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
#include "conversion.hpp"
#include "python_utils.hpp"

/*
 * The following conversion functions are taken/adapted from OpenCV's cv2.cpp file
 * inside modules/python/src2 folder.
 */

void python::import_numpy()
{
    import_array();
}

namespace
{
    static PyObject* opencv_error = 0;

    static int failmsg(const char* fmt, ...);

    static int failmsg(const char* fmt, ...)
    {
        char str[1000];

        va_list ap;
        va_start(ap, fmt);
        vsnprintf(str, sizeof(str), fmt, ap);
        va_end(ap);

        PyErr_SetString(PyExc_TypeError, str);
        return 0;
    }

    static PyObject* failmsgp(const char* fmt, ...)
    {
        char str[1000];

        va_list ap;
        va_start(ap, fmt);
        vsnprintf(str, sizeof(str), fmt, ap);
        va_end(ap);

        PyErr_SetString(PyExc_TypeError, str);
        return 0;
    }

    static size_t REFCOUNT_OFFSET =
        (size_t) &
        (((PyObject*)0)->ob_refcnt) +
            (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0") * sizeof(int);

    static inline PyObject* pyObjectFromRefcount(const int* refcount)
    {
        return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
    }

    static inline int* refcountFromPyObject(const PyObject* obj)
    {
        return (int*)((size_t)obj + REFCOUNT_OFFSET);
    }

    class NumpyAllocator : public cv::MatAllocator
    {
    public:
        NumpyAllocator() {}
        ~NumpyAllocator() {}
        void allocate(int        dims,
                      const int* sizes,
                      int        type,
                      int*&      refcount,
                      uchar*&    datastart,
                      uchar*&    data,
                      size_t*    step)
        {
            int       depth = CV_MAT_DEPTH(type);
            int       cn    = CV_MAT_CN(type);
            const int f     = (int)(sizeof(size_t) / 8);
            int       typenum =
                depth == CV_8U
                    ? NPY_UBYTE
                    : depth == CV_8S
                          ? NPY_BYTE
                          : depth == CV_16U
                                ? NPY_USHORT
                                : depth == CV_16S
                                      ? NPY_SHORT
                                      : depth == CV_32S
                                            ? NPY_INT
                                            : depth == CV_32F
                                                  ? NPY_FLOAT
                                                  : depth == CV_64F
                                                        ? NPY_DOUBLE
                                                        : f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
            int      i;
            npy_intp _sizes[CV_MAX_DIM + 1];
            for (i = 0; i < dims; i++)
            {
                _sizes[i] = sizes[i];
            }

            if (cn > 1)
            {
                _sizes[dims++] = cn;
            }

            PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);

            if (!o)
            {
                CV_Error_(
                    CV_StsError,
                    ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
            }

            refcount = refcountFromPyObject(o);

            npy_intp* _strides = PyArray_STRIDES(o);
            for (i      = 0; i < dims - (cn > 1); i++)
                step[i] = (size_t)_strides[i];
            datastart = data = (uchar*)PyArray_DATA(o);
        }

        void deallocate(int* refcount, uchar*, uchar*)
        {
            if (!refcount)
                return;
            PyObject* o = pyObjectFromRefcount(refcount);
            Py_INCREF(o);
            Py_DECREF(o);
        }
    };

    NumpyAllocator g_numpyAllocator;
}
cv::Mat python::conversion::detail::to_mat(const PyObject* o)
{
    cv::Mat m;

    if (!o || o == Py_None)
    {
        if (!m.data)
            m.allocator = &g_numpyAllocator;
    }

    if (!PyArray_Check(o))
    {
        failmsg("toMat: Object is not a numpy array");
    }

    int typenum = PyArray_TYPE(o);
    int type    = typenum == NPY_UBYTE
                   ? CV_8U
                   : typenum == NPY_BYTE
                         ? CV_8S
                         : typenum == NPY_USHORT
                               ? CV_16U
                               : typenum == NPY_SHORT
                                     ? CV_16S
                                     : typenum == NPY_INT || typenum == NPY_LONG
                                           ? CV_32S
                                           : typenum == NPY_FLOAT ? CV_32F : typenum == NPY_DOUBLE
                                                                                 ? CV_64F
                                                                                 : -1;

    if (type < 0)
    {
        failmsg("toMat: Data type = %d is not supported", typenum);
    }

    int ndims = PyArray_NDIM(o);

    if (ndims >= CV_MAX_DIM)
    {
        failmsg("toMat: Dimensionality (=%d) is too high", ndims);
    }

    int             size[CV_MAX_DIM + 1];
    size_t          step[CV_MAX_DIM + 1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes     = PyArray_DIMS(o);
    const npy_intp* _strides   = PyArray_STRIDES(o);
    bool            transposed = false;

    for (int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    if (ndims == 0 || step[ndims - 1] > elemsize)
    {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if (ndims >= 2 && step[0] < step[1])
    {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }

    if (ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize * size[2])
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if (ndims > 2)
    {
        failmsg("toMat: Object has more than 2 dimensions");
    }

    m = cv::Mat(ndims, size, type, PyArray_DATA(o), step);

    if (m.data)
    {
        m.refcount = refcountFromPyObject(o);
        m.addref(); // protect the original numpy array from deallocation
                    // (since Mat destructor will decrement the reference counter)
    };
    m.allocator = &g_numpyAllocator;

    if (transposed)
    {
        cv::Mat tmp;
        tmp.allocator = &g_numpyAllocator;
        transpose(m, tmp);
        m = tmp;
    }
    return m;
}

std::vector<nervana::boundingbox::box> python::conversion::detail::to_boxes(const PyObject* o)
{
    std::vector<nervana::boundingbox::box> boxes;
    size_t                                 size = PyList_Size(const_cast<PyObject*>(o));
    boxes.reserve(size);
    for (size_t i = 0; i < size; i++)
    {
        PyObject* dict      = PyList_GetItem(const_cast<PyObject*>(o), i);
        float     xmin      = PyFloat_AsDouble(PyDict_GetItemString(dict, "xmin"));
        float     ymin      = PyFloat_AsDouble(PyDict_GetItemString(dict, "ymin"));
        float     xmax      = PyFloat_AsDouble(PyDict_GetItemString(dict, "xmax"));
        float     ymax      = PyFloat_AsDouble(PyDict_GetItemString(dict, "ymax"));
        int       label     = PyInt_AsLong(PyDict_GetItemString(dict, "label"));
        bool      difficult = PyDict_Contains(dict, PyString_FromString("difficult"))
                             ? PyBool_Check(PyDict_GetItemString(dict, "difficult"))
                             : false;
        bool truncated = PyDict_Contains(dict, PyString_FromString("difficult"))
                             ? PyBool_Check(PyDict_GetItemString(dict, "truncated"))
                             : false;
        boxes.emplace_back(xmin, ymin, xmax, ymax, label, difficult, truncated);
    }
    return boxes;
}

PyObject* python::conversion::detail::to_ndarray(const cv::Mat& m)
{
    if (!m.data)
        Py_RETURN_NONE;

    cv::Mat temp, *p = (cv::Mat *)&m;
    if (!p->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        m.copyTo(temp);
        p = &temp;
    }
    p->addref();

    return pyObjectFromRefcount(p->refcount);
}

PyObject* python::conversion::detail::to_list(const std::vector<nervana::boundingbox::box>& boxes)
{
    if (boxes.empty())
        Py_RETURN_NONE;

    PyObject* list = PyList_New(boxes.size());
    for (size_t i = 0; i < boxes.size(); i++)
    {
        PyObject* item = PyDict_New();
        assert(PyDict_SetItemString(item, "xmin", PyFloat_FromDouble(boxes[i].xmin())) == 0);
        assert(PyDict_SetItemString(item, "ymin", PyFloat_FromDouble(boxes[i].ymin())) == 0);
        assert(PyDict_SetItemString(item, "xmax", PyFloat_FromDouble(boxes[i].xmax())) == 0);
        assert(PyDict_SetItemString(item, "ymax", PyFloat_FromDouble(boxes[i].ymax())) == 0);
        assert(PyDict_SetItemString(item, "label", PyInt_FromLong(boxes[i].label())) == 0);
        assert(PyDict_SetItemString(item, "difficult", PyBool_FromLong(boxes[i].difficult())) == 0);
        assert(PyDict_SetItemString(item, "truncated", PyBool_FromLong(boxes[i].truncated())) == 0);
        PyList_SetItem(list, i, item);
    }

    return list;
}
