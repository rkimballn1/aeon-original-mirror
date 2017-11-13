#include <Python.h>
#include <cstdlib>

#define NP_NO_DEPRACATED_API NPY_1_7_API_VERSION

namespace {
    void call_finalize()
    {
        PyGILState_Ensure();
        Py_Finalize();
    }

    struct call_initialize
    {
        call_initialize()
        {
            Py_Initialize();
            PyRun_SimpleString("import threading");
            std::atexit(call_finalize);
        }
    };

    static call_initialize x;
}

