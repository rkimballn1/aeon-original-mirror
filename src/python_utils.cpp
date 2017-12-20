#include "python_utils.hpp"

namespace nervana
{
    namespace python
    {
        static_initialization::static_initialization()
        {
            if (!Py_IsInitialized())
            {
                Py_Initialize();
                PyEval_InitThreads();
                PyEval_ReleaseLock();
            }
        }

#ifdef PYTHON_PLUGIN
        allow_threads::allow_threads()
            : _state{PyEval_SaveThread()}
        {
        }

        allow_threads::~allow_threads() { PyEval_RestoreThread(_state); }
        block_threads::block_threads(allow_threads& a)
            : _parent{a}
        {
            std::swap(_state, _parent._state);
            PyEval_RestoreThread(_state);
        }

        block_threads::~block_threads()
        {
            PyEval_SaveThread();
            std::swap(_parent._state, _state);
        }
#endif
    }
}
