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
    }
}
