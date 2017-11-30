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
#pragma once
#include <Python.h>
#include <iostream>

namespace nervana
{
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
            allow_threads(PyThreadState* m_state) { PyEval_RestoreThread(m_state); }
            ~allow_threads() { PyEval_SaveThread(); }
        };

        class ensure_gil
        {
        public:
            ensure_gil()
            {
                _state = PyGILState_Ensure();
            }

            ~ensure_gil()
            {
                PyGILState_Release(_state);
            }

        private:
            PyGILState_STATE _state;
        };
    }
}
