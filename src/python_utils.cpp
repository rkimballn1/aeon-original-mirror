#include "python_utils.hpp"

namespace nervana
{
    namespace python
    {
        static_initialization::static_initialization()
        {
            Py_Initialize();
            PyEval_InitThreads();
            PyEval_ReleaseLock();
        }
    }
}
