#include <iostream>
#include <string>
#include <Python.h>

using namespace std;

int main(int argc, char** argv)
{
	Py_Initialize();
	if(!Py_IsInitialized())
	{
		return -1;
	}

	PyObject *pModule,*pModule_convert, *pDict, *pDict_convert, *pFunc, *pArgs;

	pModule = PyImport_ImportModule("get_result");
	pModule_convert = PyImport_ImportModule("convert");
	if(!pModule && !pModule_convert)
	{
		cout << "[error]:failed to load python module" << endl;
		//getchar();
		return -1;
	}

	pDict = PyModule_GetDict(pModule);
	pDict_convert = PyModule_GetDict(pModule_convert);
	if(!pDict && !pDict_convert)
	{
		cout << "[error]:failed to load python dict" << endl;
		return -1;
	}

	pFunc = PyDict_GetItemString(pDict, "get_segemation3");
	if(!pFunc || !PyCallable_Check(pFunc))
	{
		cout << "[error]:failed to load python function" << endl;
		return -1;
	}

	PyEval_CallObject(pFunc, NULL);

	pFunc = PyDict_GetItemString(pDict_convert, "loadImageFromNpy");
	if(!pFunc || !PyCallable_Check(pFunc))
	{
		cout << "[error] load function failed" << endl;
		return -1;
	}

	PyEval_CallObject(pFunc, NULL);

	Py_Finalize();

	//getchar();
	return 0;
}
