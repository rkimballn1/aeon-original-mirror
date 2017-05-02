// /*
//  Copyright 2016 Nervana Systems Inc.
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//       http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// */

#include <iostream>
#include <sstream>
// #include <memory>

#include "api.hpp"
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

#define DL_get_loader(v) (((aeon_Dataloader*)(v))->m_loader)

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

static PyObject* wrap_buffer_as_np_array(const buffer_fixed_size_elements* buf);

typedef struct
{
    PyObject_HEAD PyObject* ndata;
    PyObject*               batch_size;
    PyObject*               axes_info;
    loader*                 m_loader;
    uint32_t                m_i;
} aeon_Dataloader;

static PyMethodDef aeon_methods[] = {{"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
                                     {NULL, NULL, NULL, NULL}};

static PyObject* Dataloader_iter(PyObject* self)
{
#ifdef AEON_DEBUG
    INFO << " aeon_Dataloader_iter";
#endif
    Py_INCREF(self);
    DL_get_loader(self)->reset();
    return self;
}

static PyObject* Dataloader_iternext(PyObject* self)
{
#ifdef AEON_DEBUG
    INFO << " aeon_Dataloader_iternext";
#endif
    PyObject* result = NULL;
    if (DL_get_loader(self)->get_current_iter() != DL_get_loader(self)->get_end_iter())
    {
        // d will be const fixed_buffer_map&
        const fixed_buffer_map& d     = *(DL_get_loader(self)->get_current_iter());
        auto                    names = DL_get_loader(self)->get_buffer_names();

        result = PyDict_New();

        for (auto&& nm : names)
        {
            PyObject* wrapped_buf = wrap_buffer_as_np_array(d[nm]);

            int set_status = PyDict_SetItemString(result, nm.c_str(), wrapped_buf);
            Py_DECREF(wrapped_buf); // DECREF is because SetItemString increments

            if (set_status < 0)
            {
                ERR << "Error building shape string";
                PyErr_SetString(PyExc_RuntimeError, "Error building shape dict");
            }
        }
        DL_get_loader(self)->get_current_iter()++;
    }
    else
    {
        /* Raising of standard StopIteration exception with empty value. */
        PyErr_SetNone(PyExc_StopIteration);
    }

    return result;
}

static Py_ssize_t aeon_Dataloader_length(PyObject* self)
{
#ifdef AEON_DEBUG
    INFO << " aeon_Dataloader_length " << DL_get_loader(self)->record_count();
#endif
    return DL_get_loader(self)->record_count();
}

static PySequenceMethods Dataloader_sequence_methods = {aeon_Dataloader_length, /* sq_length */
                                                        0,                      /* sq_length */
                                                        0,                      /* sq_concat */
                                                        0,                      /* sq_repeat */
                                                        0,                      /* sq_item */
                                                        0,                      /* sq_ass_item */
                                                        0,                      /* sq_contains */
                                                        0, /* sq_inplace_concat */
                                                        0, /* sq_inplace_repeat */
                                                        0 /* sq_inplace_repeat */};

/* This function handles py2 and py3 independent unpacking of string object (bytes or unicode)
 * as an ascii std::string
 */
static std::string py23_string_to_string(PyObject* py_str)
{
    PyObject*         s = NULL;
    std::stringstream ss;

    if (PyUnicode_Check(py_str))
    {
        s = PyUnicode_AsUTF8String(py_str);
    }
    else if (PyBytes_Check(py_str))
    {
        s = PyObject_Bytes(py_str);
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Unexpected key type");
    }

    if (s != NULL)
    {
        ss << PyBytes_AsString(s);
        Py_XDECREF(s);
    }
    return ss.str();
}

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

static void Dataloader_dealloc(aeon_Dataloader* self)
{
#ifdef AEON_DEBUG
    INFO << " Dataloader_dealloc";
#endif
    if (self->m_loader != nullptr)
    {
        delete self->m_loader;
    }
    Py_XDECREF(self->ndata);
    Py_XDECREF(self->batch_size);
    Py_XDECREF(self->axes_info);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Dataloader_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
#ifdef AEON_DEBUG
    INFO << " Dataloader_new";
#endif
    aeon_Dataloader* self = nullptr;

    static const char* keyword_list[] = {"config", nullptr};

    PyObject* dict;
    auto rc = PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(keyword_list), &dict);

    if (rc)
    {
        std::string dict_string = py23_string_to_string(dict);
        nlohmann::json json_config = nlohmann::json::parse(dict_string);
#ifdef AEON_DEBUG
        INFO << " config " << json_config.dump(4);
#endif
        self = (aeon_Dataloader*)type->tp_alloc(type, 0);
        if (!self)
        {
            return NULL;
        }

        try
        {
            self->m_loader   = new loader(json_config);
            self->m_i        = 0;
            self->ndata      = Py_BuildValue("i", self->m_loader->record_count());
            self->batch_size = Py_BuildValue("i", self->m_loader->batch_size());
            self->axes_info  = PyDict_New();

            auto name_shape_list = self->m_loader->get_names_and_shapes();

            for (auto&& name_shape : name_shape_list)
            {
                auto datum_name   = name_shape.first;
                auto axes_lengths = name_shape.second.get_shape();
                auto axes_names   = name_shape.second.get_names();

                PyObject* py_axis_dict = PyDict_New();

                for (size_t i = 0; i < axes_lengths.size(); ++i)
                {
                    PyObject* tmp_length = Py_BuildValue("i", axes_lengths[i]);
                    PyDict_SetItemString(py_axis_dict, axes_names[i].c_str(), tmp_length);
                    Py_DECREF(tmp_length);
                }

                int set_status =
                    PyDict_SetItemString(self->axes_info, datum_name.c_str(), py_axis_dict);
                Py_DECREF(py_axis_dict);

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
            ss << "Unable to create internal loader object: " << e.what();
            PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
            return NULL;
        }
    }

    return (PyObject*)self;
}

static int Dataloader_init(aeon_Dataloader* self, PyObject* args, PyObject* kwds)
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

static PyMethodDef Dataloader_methods[] = {
    //    {"Dataloader",  aeon_myiter, METH_VARARGS, "Iterate from i=0 while i<m."},
    // {"shapes", aeon_shapes, METH_NOARGS, "Get output shapes"},
    {"reset", aeon_reset, METH_NOARGS, "Reset iterator"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyMemberDef Dataloader_members[] = {
    {(char*)"ndata", T_OBJECT_EX, offsetof(aeon_Dataloader, ndata), 0, (char*)"number of records in dataset"},
    {(char*)"batch_size", T_OBJECT_EX, offsetof(aeon_Dataloader, batch_size), 0, (char*)"mini-batch size"},
    {(char*)"axes_info", T_OBJECT_EX, offsetof(aeon_Dataloader, axes_info), 0, (char*)"axes names and lengths"},
    {NULL, NULL, 0, 0, NULL} /* Sentinel */
};

static PyTypeObject aeon_DataloaderType = {
#ifdef IS_PY3K
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL) 0,
#endif
        "aeon.Dataloader",                                           /*tp_name*/
    sizeof(aeon_Dataloader),                                         /*tp_basicsize*/
    0,                                                               /*tp_itemsize*/
    (destructor)Dataloader_dealloc,                                  /*tp_dealloc*/
    0,                                                               /*tp_print*/
    0,                                                               /*tp_getattr*/
    0,                                                               /*tp_setattr*/
    0,                                                               /*tp_compare*/
    0,                                                               /*tp_repr*/
    0,                                                               /*tp_as_number*/
    &Dataloader_sequence_methods,                                    /*tp_as_sequence*/
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
    Dataloader_iter,                                                 /* tp_iter */
    Dataloader_iternext,                                             /* tp_iternext */
    Dataloader_methods,                                              /* tp_methods */
    Dataloader_members,                                              /* tp_members */
    0,                                                               /* tp_getset */
    0,                                                               /* tp_base */
    0,                                                               /* tp_dict */
    0,                                                               /* tp_descr_get */
    0,                                                               /* tp_descr_set */
    0,                                                               /* tp_dictoffset */
    (initproc)Dataloader_init,                                       /* tp_init */
    0,                                                               /* tp_alloc */
    Dataloader_new,                                                  /* tp_new */
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
                                        "Dataloader containing module",
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
    if (PyType_Ready(&aeon_DataloaderType) < 0)
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
    m = Py_InitModule3("aeon", aeon_methods, "Dataloader containing module");
#endif
    if (m == NULL)
    {
        INITERROR;
    }

    Py_INCREF(&aeon_DataloaderType);
    PyModule_AddObject(m, "Dataloader", (PyObject*)&aeon_DataloaderType);

#ifdef IS_PY3K
    return m;
#endif
}
}
