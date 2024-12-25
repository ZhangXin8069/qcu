#include <stdio.h>
#include <Python.h>

int main() {
    Py_Initialize();  // 初始化 Python 解释器
    PyRun_SimpleString("print('Hello from Python via ctypes!')");
    Py_Finalize();  // 清理 Python 解释器
    return 0;
}
