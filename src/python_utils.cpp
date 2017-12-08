#include "python_utils.hpp"

namespace nervana
{
    namespace python
    {
        PyThreadState* main_thread;

        void atexit_cleanup()
        {
            std::cout << "happy" << std::endl;
            PyEval_RestoreThread(main_thread);
            Py_Finalize();
        }

        static_initialization::static_initialization()
        {
            Py_Initialize();
            PyEval_InitThreads();
            main_thread = PyEval_SaveThread();
            std::atexit(atexit_cleanup);
        }
    }
}

namespace
{
    //static nervana::python::static_initialization init;
}
