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

#include <iostream>
#include <sstream>

#include "api.hpp"
#include "json_parser.hpp"
#include <numpy/arrayobject.h>
#include "structmember.h"
using namespace nervana;
using namespace std;

extern "C" {

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#ifdef Py_TPFLAGS_HAVE_FINALIZE
#define PYOBJ_TAIL_INIT NULL
#else
#define PYOBJ_TAIL_INIT
#endif

#define DL_get_loader(v) (((aeon_DataLoader*)(v))->m_loader)

struct aeon_state
{
    PyObject* error;
};

#ifdef IS_PY3K
#define INITERROR return NULL
#define GETSTATE(m) ((struct aeon_state*)PyModule_GetState(m))
#define Py_TPFLAGS_HAVE_ITER 0
#else
#define INITERROR return
#define GETSTATE(m) (&_state)
static struct aeon_state _state;
#endif

static PyObject* error_out(PyObject* m)
{
    struct aeon_state* st = GETSTATE(m);
    PyErr_SetString(st->error, "aeon module level error");
    return NULL;
}

/*
 * This method dumps dictionary to a json string
 */
static PyObject* dict2json(PyObject* self, PyObject* dictionary)
{
    try
    {
        return PyUnicode_FromString(JsonParser().parse(dictionary).dump().c_str());
    }
    catch (std::exception& e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject* wrap_buffer_as_np_array(const buffer_fixed_size_elements* buf);

typedef struct
{
    PyObject_HEAD PyObject* ndata;
    PyObject*               batch_size;
    PyObject*               axes_info;
    PyObject*               config;
    loader*                 m_loader;
    uint32_t                m_i;
    bool                    m_first_iteration;
} aeon_DataLoader;

static PyMethodDef aeon_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {"dict2json", (PyCFunction)dict2json, METH_O, "Dump dict to json string"},
    {NULL, NULL, NULL, NULL}};

static PyObject* DataLoader_iter(PyObject* self)
{
#ifdef AEON_DEBUG
    INFO << " aeon_DataLoader_iter";
#endif
    Py_INCREF(self);
    DL_get_loader(self)->reset();
    ((aeon_DataLoader*)(self))->m_first_iteration = true;
    return self;
}

static PyObject* DataLoader_iternext(PyObject* self)
{
#ifdef AEON_DEBUG
    INFO << " aeon_DataLoader_iternext";
#endif
    PyObject* result = NULL;

    if (!((aeon_DataLoader*)(self))->m_first_iteration)
        DL_get_loader(self)->get_current_iter()++;
    else
        ((aeon_DataLoader*)(self))->m_first_iteration = false;

    if (DL_get_loader(self)->get_current_iter() != DL_get_loader(self)->get_end_iter())
    {
        // d will be const fixed_buffer_map&
        const fixed_buffer_map& d     = *(DL_get_loader(self)->get_current_iter());
        auto                    names = DL_get_loader(self)->get_buffer_names();

        result = PyTuple_New(names.size());
        cout << "tuple_len: " << names.size() << endl;
        int buf_tuple_len = 2;
        int tuple_pos = 0;
        for (auto&& nm : names)
        {
            PyObject* wrapped_buf = wrap_buffer_as_np_array(d[nm]);
            PyObject* buf_name = Py_BuildValue("s", nm.c_str());
            PyObject* named_buf_tuple = PyTuple_New(buf_tuple_len);

            // build tuple of (name, buffer) ex: ('image', buf)
            PyTuple_SetItem(named_buf_tuple, 0, buf_name);
            PyTuple_SetItem(named_buf_tuple, 1, wrapped_buf);

            int set_status = PyTuple_SetItem(result, tuple_pos, named_buf_tuple);
            tuple_pos++;
 
            // Fix me: do i need call Py_DECREF on named_buf_tuple?
            // Note: PyTuple_SetItem steals the reference.
            if (set_status < 0)
            {
                ERR << "Error building shape string";
                PyErr_SetString(PyExc_RuntimeError, "Error building shape dict");
            }
        }
    }
    else
    {
        /* Raising of standard StopIteration exception with empty value. */
        PyErr_SetNone(PyExc_StopIteration);
    }

    return result;
}

static Py_ssize_t aeon_DataLoader_length(PyObject* self)
{
#ifdef AEON_DEBUG
    INFO << " aeon_DataLoader_length " << DL_get_loader(self)->record_count();
#endif
    return DL_get_loader(self)->record_count();
}

static PySequenceMethods DataLoader_sequence_methods = {aeon_DataLoader_length, /* sq_length */
                                                        0,                      /* sq_length */
                                                        0,                      /* sq_concat */
                                                        0,                      /* sq_repeat */
                                                        0,                      /* sq_item */
                                                        0,                      /* sq_ass_item */
                                                        0,                      /* sq_contains */
                                                        0, /* sq_inplace_concat */
                                                        0, /* sq_inplace_repeat */
                                                        0 /* sq_inplace_repeat */};


static PyObject* wrap_buffer_as_np_array(const buffer_fixed_size_elements* buf)
{
    std::vector<npy_intp> dims;
    dims.push_back(buf->get_item_count());
    auto shape = buf->get_shape_type().get_shape();
    dims.insert(dims.end(), shape.begin(), shape.end());

    int nptype = buf->get_shape_type().get_otype().get_np_type();

    PyObject* p_array =
        PyArray_SimpleNewFromData(dims.size(), &dims[0], nptype, const_cast<char*>(buf->data()));

    if (p_array == NULL)
    {
        ERR << "Unable to wrap buffer as npy array";
        PyErr_SetString(PyExc_RuntimeError, "Unable to wrap buffer as npy array");
    }

    return p_array;
}

static void DataLoader_dealloc(aeon_DataLoader* self)
{
#ifdef AEON_DEBUG
    INFO << " DataLoader_dealloc";
#endif
    if (self->m_loader != nullptr)
    {
        delete self->m_loader;
    }
    Py_XDECREF(self->ndata);
    Py_XDECREF(self->batch_size);
    Py_XDECREF(self->axes_info);
    Py_XDECREF(self->config);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* DataLoader_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
#ifdef AEON_DEBUG
    INFO << " DataLoader_new";
#endif
    aeon_DataLoader* self = nullptr;

    static const char* keyword_list[] = {"config", nullptr};

    PyObject* dict;
    auto      rc = PyArg_ParseTupleAndKeywords(
        args, kwds, "O!", const_cast<char**>(keyword_list), &PyDict_Type, &dict);

    if (rc)
    {
        nlohmann::json json_config;
        try
        {
            json_config = JsonParser().parse(dict);
        }
        catch (const std::exception& e)
        {
            std::stringstream ss;
            ss << "Unable to parse config: " << e.what();
            ERR << ss.str();
            PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
            return NULL;
        }
#ifdef AEON_DEBUG
        INFO << " config " << json_config.dump(4);
#endif
        self = (aeon_DataLoader*)type->tp_alloc(type, 0);
        if (!self)
        {
            return NULL;
        }

        try
        {
            self->m_loader          = new loader(json_config);
            self->m_i               = 0;
            self->m_first_iteration = true;
            self->ndata             = Py_BuildValue("i", self->m_loader->record_count());
            self->batch_size        = Py_BuildValue("i", self->m_loader->batch_size());
            self->axes_info         = PyDict_New();
            self->config            = PyDict_Copy(dict);

            auto name_shape_list = self->m_loader->get_names_and_shapes();

            for (auto&& name_shape : name_shape_list)
            {
                auto datum_name   = name_shape.first;
                auto axes_lengths = name_shape.second.get_shape();
                auto axes_names   = name_shape.second.get_names();

                // using tuple instead of dict to preserve the order
                // python 2.x doesnt have the support for ordereddict obj, its supported in python 3.x onwards

                PyObject* py_axis_tuple = PyTuple_New(axes_lengths.size());
                int axis_tuple_len = 2;
                for (size_t i = 0; i < axes_lengths.size(); ++i)
                {
                    PyObject* tmp_length = Py_BuildValue("i", axes_lengths[i]);
                    PyObject* axes_name = Py_BuildValue("s", axes_names[i].c_str());
                    PyObject* py_temp_tuple = PyTuple_New(axis_tuple_len);

                    if (py_temp_tuple == NULL){
                        ERR << "Error creating new tuple";
                        PyErr_SetString(PyExc_RuntimeError, "Error creating new tuple");
                        return NULL;
                    }

                    // create a tuple of (axis_name, axis_length)
                    PyTuple_SetItem(py_temp_tuple, 0 , axes_name);
                    PyTuple_SetItem(py_temp_tuple, 1 , tmp_length);

                    //add it to a tuple list
                    int tuple_status = PyTuple_SetItem(py_axis_tuple, i, py_temp_tuple);
                    if (tuple_status != 0){
                        ERR << "Error building tuple of (axis_name, axis_length)";
                        PyErr_SetString(PyExc_RuntimeError, "Error building shape tuple");
                        return NULL;
                    }
                }

                int set_status =
                    PyDict_SetItemString(self->axes_info, datum_name.c_str(), py_axis_tuple);
                // NOTE: py_axis_tuple is tuple of tuples, ex: (("channel", 3), ("height", 28), ("width", 28))
                //       so single Py_DECREF on py_axis_tuple will call Py_DECREF on all contained objects.
                Py_DECREF(py_axis_tuple);

                if (set_status < 0)
                {
                    ERR << "Error building shape string";
                    PyErr_SetString(PyExc_RuntimeError, "Error building shape dict");
                    return NULL;
                }
            }
        }
        catch (std::exception& e)
        {
            // Some kind of problem with creating the internal loader object
            ERR << "Unable to create internal loader object";
            std::stringstream ss;
            ss << "Unable to create internal loader object: " << e.what() << endl;
            ss << "config is: " << json_config << endl;
            PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
            return NULL;
        }
    }

    return (PyObject*)self;
}

static int DataLoader_init(aeon_DataLoader* self, PyObject* args, PyObject* kwds)
{
    return 0;
}

static PyObject* aeon_reset(PyObject* self, PyObject*)
{
#ifdef AEON_DEBUG
    INFO << " aeon_reset";
#endif
    DL_get_loader(self)->reset();
    return Py_None;
}

static PyMethodDef DataLoader_methods[] = {
    //    {"DataLoader",  aeon_myiter, METH_VARARGS, "Iterate from i=0 while i<m."},
    // {"shapes", aeon_shapes, METH_NOARGS, "Get output shapes"},
    {"reset", aeon_reset, METH_NOARGS, "Reset iterator"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyMemberDef DataLoader_members[] = {
    {(char*)"ndata",
     T_OBJECT_EX,
     offsetof(aeon_DataLoader, ndata),
     0,
     (char*)"number of records in dataset"},
    {(char*)"batch_size",
     T_OBJECT_EX,
     offsetof(aeon_DataLoader, batch_size),
     0,
     (char*)"batch size"},
    {(char*)"axes_info",
     T_OBJECT_EX,
     offsetof(aeon_DataLoader, axes_info),
     0,
     (char*)"axes names and lengths"},
    {(char*)"config",
     T_OBJECT_EX,
     offsetof(aeon_DataLoader, config),
     0,
     (char*)"config passed to DataLoader object"},
    {NULL, NULL, 0, 0, NULL} /* Sentinel */
};

static PyTypeObject aeon_DataLoaderType = {
#ifdef IS_PY3K
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL) 0,
#endif
        "aeon.DataLoader",                                           /*tp_name*/
    sizeof(aeon_DataLoader),                                         /*tp_basicsize*/
    0,                                                               /*tp_itemsize*/
    (destructor)DataLoader_dealloc,                                  /*tp_dealloc*/
    0,                                                               /*tp_print*/
    0,                                                               /*tp_getattr*/
    0,                                                               /*tp_setattr*/
    0,                                                               /*tp_compare*/
    0,                                                               /*tp_repr*/
    0,                                                               /*tp_as_number*/
    &DataLoader_sequence_methods,                                    /*tp_as_sequence*/
    0,                                                               /*tp_as_mapping*/
    0,                                                               /*tp_hash */
    0,                                                               /*tp_call*/
    0,                                                               /*tp_str*/
    0,                                                               /*tp_getattro*/
    0,                                                               /*tp_setattro*/
    0,                                                               /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_ITER, /*tp_flags*/
    "Internal myiter iterator object.",                              /* tp_doc */
    0,                                                               /* tp_traverse */
    0,                                                               /* tp_clear */
    0,                                                               /* tp_richcompare */
    0,                                                               /* tp_weaklistoffset */
    DataLoader_iter,                                                 /* tp_iter */
    DataLoader_iternext,                                             /* tp_iternext */
    DataLoader_methods,                                              /* tp_methods */
    DataLoader_members,                                              /* tp_members */
    0,                                                               /* tp_getset */
    0,                                                               /* tp_base */
    0,                                                               /* tp_dict */
    0,                                                               /* tp_descr_get */
    0,                                                               /* tp_descr_set */
    0,                                                               /* tp_dictoffset */
    (initproc)DataLoader_init,                                       /* tp_init */
    0,                                                               /* tp_alloc */
    DataLoader_new,                                                  /* tp_new */
    0,                                                               /* tp_free */
    0,                                                               /* tp_is_gc */
    0,                                                               /* tp_bases */
    0,                                                               /* tp_mro */
    0,                                                               /* tp_cache */
    0,                                                               /* tp_subclasses */
    0,                                                               /* tp_weaklist */
    0,                                                               /* tp_del */
    0,                                                               /* tp_version_tag */
    PYOBJ_TAIL_INIT                                                  /* tp_finalize */
};

#ifdef IS_PY3K
static int aeon_traverse(PyObject* m, visitproc visit, void* arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}
static int aeon_clear(PyObject* m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef aeonmodule = {PyModuleDef_HEAD_INIT,
                                        "aeon",
                                        "DataLoader containing module",
                                        sizeof(struct aeon_state),
                                        aeon_methods,
                                        NULL,
                                        aeon_traverse,
                                        aeon_clear,
                                        NULL};

PyMODINIT_FUNC PyInit_aeon(void)
#else
PyMODINIT_FUNC initaeon(void)
#endif
{
#ifdef AEON_DEBUG
    INFO << " initaeon";
#endif

    PyObject* m;
    if (PyType_Ready(&aeon_DataLoaderType) < 0)
    {
        INITERROR;
    }

    if (_import_array() < 0)
    {
        INITERROR;
    }

#ifdef IS_PY3K
    m = PyModule_Create(&aeonmodule);
#else
    m = Py_InitModule3("aeon", aeon_methods, "DataLoader containing module");
#endif
    if (m == NULL)
    {
        INITERROR;
    }

    Py_INCREF(&aeon_DataLoaderType);
    PyModule_AddObject(m, "DataLoader", (PyObject*)&aeon_DataLoaderType);

#ifdef IS_PY3K
    return m;
#endif
}
}
