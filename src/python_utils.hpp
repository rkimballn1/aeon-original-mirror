#pragma once

namespace python
{
/*
struct gil_lock
{
private:
    PyGILState_STATE gstate;
    static int gil_init;

public:
    gil_lock()
    {
        if (!gil_lock::gil_init) {
            gil_lock::gil_init = 1;
            PyEval_InitThreads();
            PyEval_SaveThread();
        }
        gstate = PyGILState_Ensure();
    }

    ~gil_lock()
    {
        PyGILState_Release(gstate);
    }
};

int gil_lock::gil_init = 0;
*/

class allow_threads
{
public:
    allow_threads() : _state(PyEval_SaveThread()) 
    {
    }

    ~allow_threads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class ensure_gil
{
public:
    ensure_gil() : _state(PyGILState_Ensure())
    {
    }

    ~ensure_gil()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};
}
